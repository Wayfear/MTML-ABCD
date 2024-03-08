import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict
from scipy.io import loadmat


def load_emory_data(cfg: DictConfig):

    ts_data = loadmat(cfg.dataset.time_seires)['atlas_data']
    ts_data = ts_data.transpose(2, 0, 1)

    feature = loadmat(cfg.dataset.feature)['brain_volume_hcp']

    # feature = feature[:, :, :2]
    # feature_std = np.std(feature, axis=0)
    # feature_std = np.where(np.isnan(feature_std), 1, feature_std)
    # feature_std = np.where(feature_std == 0, 1, feature_std)
    # stand_scaler = StandardScaler(
    #     np.mean(feature, axis=0), feature_std)
    # feature = stand_scaler.transform(feature)

    # if feature.shape[2] % 4 != 0:
    #     addon_dim = 4 - feature.shape[2] % 4
    #     addon = np.zeros((feature.shape[0], feature.shape[1], addon_dim))
    #     feature = np.concatenate((feature, addon), axis=2)

    all_sample_pearson = []
    for d in ts_data:
        m = np.corrcoef(d)
        all_sample_pearson.append(m)

    label_df = np.loadtxt(cfg.dataset.label)

    non_value = -1
    if cfg.dataset.column == "gender":
        non_value = 0

    final_timeseires, final_label, final_pearson = [], [], []

    # for ts, p, l, f in zip(ts_data, all_sample_pearson, label_df, feature):
    #     if l != non_value:
    #         if np.any(np.isnan(p)) == False and np.any(np.isnan(ts)) == False and np.any(np.isnan(f)) == False:
    #             final_timeseires.append(ts)
    #             final_label.append(l)
    #             final_pearson.append(np.concatenate((p, f), axis=1))

    for ts, p, l, f in zip(ts_data, all_sample_pearson, label_df, feature):
        if l != non_value:
            if np.any(np.isnan(p)) == False and np.any(np.isnan(ts)) == False and np.any(np.isnan(f)) == False:
                final_timeseires.append(ts)
                final_label.append(l)
                final_pearson.append(p)

    final_timeseires, final_pearson, labels = [np.array(
        data) for data in (final_timeseires, final_pearson, final_label)]

    if cfg.dataset.column == "gender":
        labels = labels - 1

    if (not cfg.dataset.regression) and cfg.dataset.column == "alzheimer":
        labels = np.where(labels <= 0.25, 0, 1)
    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]
        cfg.dataset.num_classes = labels.unique().shape[0]

    if "stratified" in cfg.dataset and cfg.dataset.stratified:
        return final_timeseires, final_pearson, labels, labels
    return final_timeseires, final_pearson, labels
