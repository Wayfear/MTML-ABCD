import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_aurora_data(cfg: DictConfig):


    pearson_data = np.load(cfg.dataset.node_feature)
    label_df = pd.read_csv(cfg.dataset.label)

    label_df = label_df[['PID', cfg.dataset.column]]
    label_df[cfg.dataset.column] = label_df[cfg.dataset.column].replace(".", np.nan)
    label_df = label_df.dropna()

    with open(cfg.dataset.node_id, 'r') as f:
        lines = f.readlines()
        pearson_id = [int(line) for line in lines]

    id2pearson = dict(zip(pearson_id, pearson_data))

    id2label = dict(zip(label_df['PID'], label_df[cfg.dataset.column]))

    final_timeseires, final_label, final_pearson = [], [], []

    for i, m in id2pearson.items():
        if i in id2label:
            if np.any(np.isnan(m)) == False:
                final_timeseires.append(0)
                final_label.append(id2label[i])
                final_pearson.append(m)

    if not cfg.dataset.regression:

        encoder = preprocessing.LabelEncoder()

        encoder.fit(label_df[cfg.dataset.column])

        labels = encoder.transform(final_label)

    else:
        labels = final_label

    final_timeseires, final_pearson, labels = [np.array(
        data) for data in (final_timeseires, final_pearson, labels)]
    
    labels = labels.astype(np.float32)

    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = None
        cfg.dataset.num_classes = labels.unique().shape[0]
    if "stratified" in cfg.dataset and cfg.dataset.stratified:
        return final_timeseires, final_pearson, labels, labels
    return final_timeseires, final_pearson, labels