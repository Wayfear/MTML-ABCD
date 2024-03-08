from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_recall_fscore_support, classification_report
from omegaconf import DictConfig
import logging
from typing import List
import torch.utils.data as utils
from source.utils import isfloat
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
import wandb


class ShallowTraining:

    def __init__(self, cfg: DictConfig,
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger, **kargs) -> None:

        assert cfg.model.name in ["LogisticRegression", "SVC",
                                  "RandomForestClassifier", "ElasticNet", 'GradientBoostingRegressor']
        self.model_name = cfg.model.name
        self.logger = logger
        self.cfg = cfg
        index = np.triu_indices(cfg.dataset.node_sz, k=1)
        train, val, test = [self.dataloader2numpy(d) for d in dataloaders]

        if not cfg.dataset.regression:
            pearson_train, labels_train = train[0][:,
                                                   index[0], index[1]], train[1][:, 1]
            pearson_val, labels_val = val[0][:,
                                             index[0], index[1]], val[1][:, 1]
            self.pearson_test, self.labels_test = test[0][:, index[0], index[1]],\
                test[1][:, 1]
        else:
            pearson_train, labels_train = train[0][:,
                                                   index[0], index[1]], train[1][:, 0]
            pearson_val, labels_val = val[0][:,
                                             index[0], index[1]], val[1][:, 0]
            self.pearson_test, self.labels_test = test[0][:, index[0], index[1]],\
                test[1][:, 0]

        self.train_index = list(
            range(0, pearson_train.shape[0])
        )
        self.val_index = list(
            range(pearson_train.shape[0],
                  pearson_train.shape[0]+pearson_val.shape[0])
        )

        self.pearson_train, self.labels_train = np.concatenate(
            [pearson_train, pearson_val]), np.concatenate([labels_train, labels_val])

        self.grid = {k: v for k, v in cfg.model.grid.items()}

        for k in ['C', 'l1_ratio', 'alpha']:
            if k in self.grid:
                self.grid[k] = eval(self.grid[k])

        self.model = eval(self.model_name)()

    @staticmethod
    def index_generator(index):
        yield [index, index]

    @staticmethod
    def dataloader2numpy(dataloader):
        pearsons, labels = [], []
        for _, pearson, label in dataloader:
            pearsons.append(pearson)
            labels.append(label)

        return np.concatenate(pearsons), np.concatenate(labels)

    def train(self):
        self.logger.info("Start search")
        baseline_cv = GridSearchCV(
            self.model, self.grid, cv=self.index_generator(self.val_index), n_jobs=8)
        baseline_cv.fit(self.pearson_train, self.labels_train)
        self.logger.info("End search")

        self.logger.info(f"Tuned hyperparameters: {baseline_cv.best_params_}")
        self.logger.info(f"Score : {baseline_cv.best_score_}")

        if self.model_name == 'SVC':
            baseline_cv = eval(self.model_name)(
                **baseline_cv.best_params_, probability=True)
        else:
            baseline_cv = eval(self.model_name)(**baseline_cv.best_params_)

        baseline_cv.fit(
            self.pearson_train[self.train_index], self.labels_train[self.train_index])

        if not self.cfg.dataset.regression:

            pred = baseline_cv.predict_proba(self.pearson_test)

            pred = pred[:, 1]

            auc = roc_auc_score(self.labels_test, pred)
            result = np.array(pred)
            result[result > 0.5] = 1
            result[result <= 0.5] = 0
            metrics = precision_recall_fscore_support(
                self.labels_test, result, average='micro')

            report = classification_report(
                self.labels_test, result, output_dict=True, zero_division=0)

            recall = [0, 0]
            for k in report:
                if isfloat(k):
                    recall[int(float(k))] = report[k]['recall']

            self.logger.info(" | ".join([
                f"Test AUC:{auc: .3f}",
                f'Test Sensitivity:{recall[1]: .3f}',
                f'Test Specificity:{recall[0]: .3f}',
                f'micro F1:{metrics[2]: .3f}',
                f'micro recall:{metrics[1]: .3f}',
                f'micro precision:{metrics[0]: .3f}',
            ]))

            wandb.log({
                "Test AUC": auc,
                'Test Sensitivity': recall[1],
                'Test Specificity': recall[0],
                'micro F1': metrics[2],
                'micro recall': metrics[1],
                'micro precision': metrics[0],
            })
        else:
            pred = baseline_cv.predict(self.pearson_test)
            mae = mean_absolute_error(self.labels_test, pred)
            mse = mean_squared_error(self.labels_test, pred)
            cof = baseline_cv.score(self.pearson_test, self.labels_test)

            self.logger.info(" | ".join([
                f"Test MSE:{mse: .3f}",
                f'Test MAE:{mae: .3f}',
                f'Test R2:{cof: .3f}',
            ]))

            wandb.log({
                "Test MSE": mse,
                'Test MAE': mae,
                'Test R2': cof,
            })
