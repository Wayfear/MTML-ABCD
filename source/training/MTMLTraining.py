from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data, discrete_mixup_data, graphon_mixup, graphon_mixup_for_regression, renn_mixup, get_batch_kde_mixup_batch, drop_node, drop_edge, c_r_mixup
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
import matplotlib.pyplot as plt
import pandas as pd
from source.dataset import StandardScaler
from captum.attr import IntegratedGradients


class MTMLTrain:

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
        self.learnable_masks = []
        self.avg_test_network = None

        node = pd.read_csv(cfg.training.node_file)
        node_name = node['AAc-6']
        self.index_list = node_name.unique()
        self.idx = []
        self.divide_line_pos = [0]
        # region_counter = {}
        for group_name in self.index_list:
            tmp = [i for i, r in enumerate(node_name) if r == group_name]
            self.idx += tmp
            # region_counter[group_name] = 2*len(tmp)*node_name.shape[0]
            self.divide_line_pos.append(len(tmp)+self.divide_line_pos[-1])
        self.label_pos = []
        for i in range(1, len(self.divide_line_pos)):
            self.label_pos.append((self.divide_line_pos[i]+self.divide_line_pos[i-1])/2)
        self.label_pos[-1] += 1

        self.regreloss_fn = torch.nn.MSELoss(reduction='mean')
        self.classloss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_mask = cfg.save_learnable_mask
        if self.config.training.standard_scaler:
            self.standard_scaler = StandardScaler()
            self.standard_scaler.to_cuda()

        self.init_meters()


    def init_meters(self):
        self.total_train_loss, self.total_val_loss,\
            self.total_test_loss = [
                TotalMeter() for _ in range(3)]
    
        self.train_loss_for_each_task, self.val_loss_for_each_task,\
            self.test_loss_for_each_task = {}, {}, {}
        
        self.sparse_loss = TotalMeter()

        for task in self.config.dataset.tasks:
            self.train_loss_for_each_task[task.column] = TotalMeter()
            self.val_loss_for_each_task[task.column] = TotalMeter()
            self.test_loss_for_each_task[task.column] = TotalMeter()


    def reset_meters(self):
        for meter in [self.total_train_loss, self.total_val_loss, self.total_test_loss]:
            meter.reset()
        for meters in [self.train_loss_for_each_task, self.val_loss_for_each_task, \
                       self.test_loss_for_each_task]:
            for meter in meters.values():
                meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        for time_series, node_feature, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(
                    time_series, node_feature, y=label)

            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()

            predict = self.model(node_feature)

            loss = 0

            for i, task in enumerate(self.config.dataset.tasks):
                check_nan = torch.isnan(label[:, i])
                task_predict = predict[~check_nan, i].unsqueeze(1)
                task_label = label[~check_nan, i].unsqueeze(1)
                if task.regression:
                    task_loss = self.regreloss_fn(task_predict, task_label)
                else:
                    task_loss = self.classloss_fn(task_predict, task_label)
                if self.config.dataset.use_balance_weight:
                    loss += task_loss * 1.0/(task_loss.detach() + 1e-6)
                else:
                    loss += task_loss * task.weight
                self.train_loss_for_each_task[task.column].update_with_weight(task_loss.item(), label.shape[0])

            if self.config.model.mask and self.config.model.sparse_loss:
                mask = self.model.get_mask()
                sparse_loss = torch.norm(mask, p=1)
                loss += (sparse_loss * self.config.model.sparse_loss_weight /(sparse_loss.detach() + 1e-6))
                self.sparse_loss.update_with_weight(sparse_loss.item(), label.shape[0])

            self.total_train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_per_epoch(self, dataloader, loss_meters, total_loss_meter):

        self.model.eval()

        labels_cls, preds_cls = {}, {}
        labels_reg, preds_reg = {}, {}

        for task in self.config.dataset.tasks:
            if not task.regression:
                labels_cls[task.column] = []
                preds_cls[task.column] = []
            else:
                labels_reg[task.column] = []
                preds_reg[task.column] = []
        
        networks = []

        for time_series, node_feature, label in dataloader:
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            predict= self.model(node_feature)
            if self.config.training.standard_scaler:
                predict = self.standard_scaler.inverse_transform(predict)
            if self.avg_test_network is None:
                networks.append(torch.mean(node_feature, dim=0))

            loss = 0

            for i, task in enumerate(self.config.dataset.tasks):
                check_nan = torch.isnan(label[:, i])
                task_predict = predict[~check_nan, i].unsqueeze(1)
                task_label = label[~check_nan, i].unsqueeze(1)
                if task.regression:
                    task_loss = self.regreloss_fn(task_predict, task_label)
                    preds_reg[task.column] += task_predict.tolist()
                    labels_reg[task.column] += task_label.tolist()
                else:
                    task_loss = self.classloss_fn(task_predict, task_label)
                    preds_cls[task.column] += torch.sigmoid(task_predict).tolist()
                    labels_cls[task.column] += task_label.tolist()
                loss += task_loss * task.weight
                loss_meters[task.column].update_with_weight(task_loss.item(), task_label.shape[0])

            total_loss_meter.update_with_weight(loss.item(), label.shape[0])

        if self.avg_test_network is None:
            self.avg_test_network = torch.mean(torch.stack(networks), dim=0, keepdim=True)

        auc = {}
        for k in preds_cls:
            auc[k] = roc_auc_score(labels_cls[k], preds_cls[k])

        R_squared = {}
        for k in preds_reg:
            R_squared[k] = r2_score(labels_reg[k], preds_reg[k])

        return auc, R_squared


    def calculate_ig(self, dataloader, sample_size=64):
        self.model.eval()
        ig_model = IntegratedGradients(self.model)
        ig_result = {task.name: 0 for task in self.config.dataset.tasks}
        batch_num = sample_size//self.config.dataset.batch_size
        for i, (time_series, node_feature, label) in enumerate(dataloader):
            if i >= batch_num:
                break
            node_feature = node_feature.cuda()
            for j, task in enumerate(self.config.dataset.tasks):
                ig = ig_model.attribute(node_feature, target=j, n_steps=10)
                ig_result[task.name] += np.mean(ig.cpu().detach().numpy(), axis=0)

        for k, v in ig_result.items():
            ig_result[k] = v/batch_num

        return ig_result
    

    def draw_mask(self, mask, title=None):
        plt.figure(dpi=300)
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)

        im = ax.imshow(mask, cmap='coolwarm')
        # im = ax.imshow(mask, cmap='coolwarm')

        for pos in self.divide_line_pos[1:-1]:
            ax.axvline(x=pos, color='k', linestyle=':', linewidth=1)
            ax.axhline(y=pos, color='k', linestyle=':', linewidth=1)

        ax.set_xticks([])
        ax.set_yticks(self.label_pos)

        ax.set_yticklabels(self.index_list)

        if title:
            ax.set_title(title)

        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig

    def visualize_mask(self):
        mask = self.model.get_masked_network(self.avg_test_network)
        mask = mask[:, :, self.idx]
        mask = mask[:, self.idx, :]

        plots = {}
        task_name = [task.name for task in self.config.dataset.tasks]
        node_sz = self.config.dataset.node_sz
        for i, task in enumerate(self.config.dataset.tasks):
            plots[task.column] = self.draw_mask(mask[i])
        

        if len(self.config.dataset.tasks) > 1:
            # para_mask = self.model.get_mask()
            para_mask = mask.reshape(len(task_name), -1)
            correltaion = np.corrcoef(para_mask)

            plots["correlation"] = wandb.plots.HeatMap(
                    x_labels=task_name, y_labels=task_name, matrix_values=correltaion, show_text=False)
            
        return plots

    def draw_correlation(self, correlation, task_list):
        plt.figure(dpi=300)
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)

        im = ax.imshow(correlation, cmap='coolwarm', vmax=1, vmin=-1)

        ax.set_xticks(np.arange(len(task_list)))
        ax.set_yticks(np.arange(len(task_list)))
        ax.set_xticklabels(task_list, rotation=90, ha="right")
        ax.set_yticklabels(task_list)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig

    def visualize_for_each_ig_result(self, ig_result, name='Train'):
        plots = {}
        corr = []
        tasks = []
        result = {}
        for k, v in ig_result.items():
            for i, view in enumerate(self.config.dataset.views):
                plots[f'{name}_{k}_{view.name}'] = self.draw_mask(v[i], title=f'{name}_{k}_{view.name}')
                result[f'{name}_{k}_{view.name}'] = v[i]

            corr.append(v.reshape(-1))
            tasks.append(k)

        corr = np.array(corr)
        corr = np.corrcoef(corr)
        plots[f'{name}_correlation'] = self.draw_correlation(corr, tasks)
        
        
        return plots, result

    def visualize_ig(self):

        ig_result_test = self.calculate_ig(self.test_dataloader)
        test_plots, test_result = self.visualize_for_each_ig_result(ig_result_test, name='Test')

        ig_result_train = self.calculate_ig(self.train_dataloader)

        train_plots, train_result = self.visualize_for_each_ig_result(ig_result_train, name='Train')

        test_result.update(train_result)

        test_plots.update(train_plots)
        
        return test_plots, test_result

    

    def save_learnable_mask(self):
        masks = np.array(self.learnable_masks)
        np.save(self.save_path/"learnable_mask.npy", masks)


    def train(self):
        # training_process = []
        self.current_step = 0
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_auc, val_r2 = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss_for_each_task,
                                             self.total_val_loss)

            test_auc, test_r2 = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss_for_each_task,
                                              self.total_test_loss)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Total Train Loss:{self.total_train_loss.avg: .3f}',
                f'Total Test Loss:{self.total_test_loss.avg: .3f}',
                f'Total Val Loss:{self.total_val_loss.avg: .3f}',
                f'Sparse Loss:{self.sparse_loss.avg:.3f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}',
            ] + [f'Train {task.column} Loss: {self.train_loss_for_each_task[task.column].avg:.3f}' for task in self.config.dataset.tasks]
            + [f'Val {task.column} Loss: {self.val_loss_for_each_task[task.column].avg:.3f}' for task in self.config.dataset.tasks]
            + [f'Test {task.column} Loss: {self.test_loss_for_each_task[task.column].avg:.3f}' for task in self.config.dataset.tasks]
            + [f'Val {task} AUC: {val_auc[task]:.3f}' for task in val_auc]
            + [f'Test {task} AUC: {test_auc[task]:.3f}' for task in test_auc]
            ))

            wandb_log = {
                "Total Train Loss": self.total_train_loss.avg,
                "Total Val Loss": self.total_val_loss.avg,
                "Total Test Loss": self.total_test_loss.avg,
                "Sparse Loss": self.sparse_loss.avg,
                "LR": self.lr_schedulers[0].lr,
            }
            
            wandb_log.update({f"Train {task.column} Loss": self.train_loss_for_each_task[task.column].avg for task in self.config.dataset.tasks})
            wandb_log.update({f"Val {task.column} Loss": self.val_loss_for_each_task[task.column].avg for task in self.config.dataset.tasks})
            wandb_log.update({f"Test {task.column} Loss": self.test_loss_for_each_task[task.column].avg for task in self.config.dataset.tasks})
            wandb_log.update({f"Val {task} AUC": val_auc[task] for task in val_auc})
            wandb_log.update({f"Test {task} AUC": test_auc[task] for task in test_auc})
            wandb_log.update({f"Val {task} R2": val_r2[task] for task in val_r2})
            wandb_log.update({f"Test {task} R2": test_r2[task] for task in test_r2})

            if epoch % 50 == 0 and self.config.model.mask:
                visual_log = self.visualize_mask()
                wandb_log.update(visual_log)
                plt.close('all')

            if epoch % 50 == 0 and self.config.model.mask:
                self.learnable_masks.append(self.model.get_mask())

            if epoch % self.config.training.save_per_epoch == 0 and self.config.training.ig_visualize:
                ig_plots, ig_result = self.visualize_ig()
                wandb_log.update(ig_plots)
                plt.close('all')
                if self.config.training.save_ig:
                    np.save(self.save_path/f'ig_result_{epoch}.npy', ig_result, allow_pickle=True)
                    self.logger.info(f"Save ig result to {self.save_path/f'ig_result_{epoch}.npy'}")

            wandb.log(wandb_log, step=epoch)

        if self.save_learnable_mask and self.config.model.mask:
            self.save_learnable_mask()