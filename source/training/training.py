from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data, discrete_mixup_data, graphon_mixup, graphon_mixup_for_regression, renn_mixup, get_batch_kde_mixup_batch, drop_node, drop_edge, c_r_mixup
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(
            f'#model params: {count_params(self.model)}, sample size: {cfg.dataset.sample_size}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        if cfg.dataset.regression:
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph
        if cfg.preprocess.graphon:
            if cfg.dataset.regression:
                self.sampler = graphon_mixup_for_regression()
            else:
                self.sampler = graphon_mixup()

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        for time_series, node_feature, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            if self.config.preprocess.discrete:
                node_feature, label = discrete_mixup_data(
                    node_feature, label)
            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(
                    time_series, node_feature, y=label)
            if self.config.preprocess.graphon:
                new_node_feature, new_label = self.sampler(
                    sample_num=label.shape[0], y=label)
                node_feature = torch.concat((node_feature, new_node_feature))
                label = torch.concat((label, new_label))
            if self.config.preprocess.renn_mixup and (not self.config.preprocess.c_mixup):
                node_feature, label = renn_mixup(
                    node_feature, label, self.config.preprocess.alpha)
            if self.config.preprocess.c_mixup and (not self.config.preprocess.renn_mixup):
                node_feature, label = get_batch_kde_mixup_batch(
                    node_feature, label)
            if self.config.preprocess.drop_edge:
                node_feature = drop_edge(node_feature)
            if self.config.preprocess.drop_node:
                node_feature = drop_node(node_feature)

            if self.config.preprocess.c_mixup and self.config.preprocess.renn_mixup:
                node_feature, label = c_r_mixup(
                    node_feature, label, self.config.preprocess.alpha)

            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()

            predict = self.model(time_series, node_feature)

            loss = self.loss_fn(predict, label)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not self.config.dataset.regression:
                top1 = accuracy(predict, label[:, 1])[0]
                self.train_accuracy.update_with_weight(top1, label.shape[0])
            # wandb.log({"LR": lr_scheduler.lr,
            #            "Iter loss": loss.item()})

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for time_series, node_feature, label in dataloader:
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            output = self.model(time_series, node_feature)

            label = label.float()

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            if not self.config.dataset.regression:
                top1 = accuracy(output, label[:, 1])[0]
                acc_meter.update_with_weight(top1, label.shape[0])

                if self.config.dataset.num_classes == 2:
                    result += F.softmax(output, dim=1)[:, 1].tolist()
                    labels += label[:, 1].tolist()
                else:
                    result.append(output.detach().cpu())
                    labels.append(label.detach().cpu())
            else:
                result+=output.detach().cpu().tolist()
                labels+=label.detach().cpu().tolist()

        if self.config.dataset.regression:
            result, labels = np.array(result), np.array(labels)
            result = result.reshape(-1)
            labels = labels.reshape(-1)
            mse = mean_squared_error(labels, result)
            mae = mean_absolute_error(labels, result)
            return {'mse': mse, 'mae': mae}

        if self.config.dataset.num_classes == 2:

            auc = roc_auc_score(labels, result)
            result, labels = np.array(result), np.array(labels)
            result[result > 0.5] = 1
            result[result <= 0.5] = 0
            metric = precision_recall_fscore_support(
                labels, result, average='micro')

            report = classification_report(
                labels, result, output_dict=True, zero_division=0)

            recall = [0, 0]
            for k in report:
                if isfloat(k):
                    recall[int(float(k))] = report[k]['recall']
            return [auc] + list(metric) + recall

        else:

            result, labels = torch.vstack(result), torch.vstack(labels)
            labels = torch.argmax(labels, dim=1)
            result = torch.argmax(result, dim=1)
            # top1, top3, top5 = accuracy(result, labels, top_k=(1, 3, 5))
            metric = precision_recall_fscore_support(
                labels, result, average='macro')
            return list(metric)

    def generate_save_learnable_matrix(self):

        # wandb.log({'heatmap_with_text': wandb.plots.HeatMap(x_labels, y_labels, matrix_values, show_text=False)})
        learable_matrixs = []

        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            _, learable_matrix, _ = self.model(time_series, node_feature)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        self.current_step = 0
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            if self.config.dataset.regression:

                self.logger.info(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .3f}',
                    f'Val Loss:{self.val_loss.avg: .3f}',
                    f'Test Loss:{self.test_loss.avg: .3f}',
                    f'Test MSE:{test_result["mse"]:.4f}',
                    f'Test MAE:{test_result["mae"]:.4f}',
                ]))
                wandb.log({
                    "Train Loss": self.train_loss.avg,
                    "Val Loss": self.val_loss.avg,
                    "Test Loss": self.test_loss.avg,
                    "Test MSE": test_result["mse"],
                    "Test MAE": test_result["mae"],
                })
                training_process.append({
                    "Epoch": epoch,
                    "Train Loss": self.train_loss.avg,
                    "Test Loss": self.test_loss.avg,
                    "Val Loss": self.val_loss.avg,
                })

            elif self.config.dataset.num_classes == 2:

                self.logger.info(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .3f}',
                    f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                    f'Test Loss:{self.test_loss.avg: .3f}',
                    f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                    f'Val AUC:{val_result[0]:.4f}',
                    f'Test AUC:{test_result[0]:.4f}',
                    f'Test Sen:{test_result[-1]:.4f}',
                    f'LR:{self.lr_schedulers[0].lr:.4f}'
                ]))

                wandb.log({
                    "Train Loss": self.train_loss.avg,
                    "Train Accuracy": self.train_accuracy.avg,
                    "Test Loss": self.test_loss.avg,
                    "Test Accuracy": self.test_accuracy.avg,
                    "Val AUC": val_result[0],
                    "Test AUC": test_result[0],
                    'Test Sensitivity': test_result[-1],
                    'Test Specificity': test_result[-2],
                    'micro F1': test_result[-4],
                    'micro recall': test_result[-5],
                    'micro precision': test_result[-6],
                })

                training_process.append({
                    "Epoch": epoch,
                    "Train Loss": self.train_loss.avg,
                    "Train Accuracy": self.train_accuracy.avg,
                    "Test Loss": self.test_loss.avg,
                    "Test Accuracy": self.test_accuracy.avg,
                    "Test AUC": test_result[0],
                    'Test Sensitivity': test_result[-1],
                    'Test Specificity': test_result[-2],
                    'micro F1': test_result[-4],
                    'micro recall': test_result[-5],
                    'micro precision': test_result[-6],
                    "Val AUC": val_result[0],
                    "Val Loss": self.val_loss.avg,
                })

            else:

                self.logger.info(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .3f}',
                    f'Val Loss:{self.val_loss.avg: .3f}',
                    f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                    f'Test Loss:{self.test_loss.avg: .3f}',
                    f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                    f'LR:{self.lr_schedulers[0].lr:.4f}'
                ]))

                wandb.log({
                    "Train Loss": self.train_loss.avg,
                    "Val Loss": self.val_loss.avg,
                    "Train Accuracy": self.train_accuracy.avg,
                    "Test Loss": self.test_loss.avg,
                    "Test Accuracy": self.test_accuracy.avg,
                    # 'Top 1 Acc': test_result[-3],
                    # 'Top 3 Acc': test_result[-2],
                    # 'Top 5 Acc': test_result[-1],
                    'macro recall': test_result[1],
                    'macro precision': test_result[0],
                })

                training_process.append({
                    "Epoch": epoch,
                    "Train Loss": self.train_loss.avg,
                    "Train Accuracy": self.train_accuracy.avg,
                    "Val Loss": self.val_loss.avg,
                    "Test Loss": self.test_loss.avg,
                    "Test Accuracy": self.test_accuracy.avg,
                    "Val Loss": self.val_loss.avg,
                })

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)
