import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict
import functools as ft


def find_top_and_bottom(data, label, top=0.1, bottom=0.1):
    top, bottom = int(data.shape[0] * top), int(data.shape[0] * bottom)
    top_idx = np.argsort(label)[-top:]
    bottom_idx = np.argsort(label)[:bottom]
    top = np.average(data[top_idx], axis=0)
    bottom = np.average(data[bottom_idx], axis=0)
    top_label = np.average(label[top_idx])
    bottom_label = np.average(label[bottom_idx])
    return (top, top_label), (bottom, bottom_label)


def load_abcd_mtml_data(cfg: DictConfig):

    ids2data = []

    for view in cfg.dataset.views:
        with open(view.node_id, 'r') as f:
            lines = f.readlines()
            ids = [line[:-1] for line in lines]

        pearson_data = np.load(view.node_feature)

        ids2data.append(dict(zip(ids, pearson_data)))

    sample_id = list(ids2data[0].keys())

    all_pearson = []

    ids = []

    for i in sample_id:
        data = []
        flag = True
        for d in ids2data:
            if i not in d:
                flag = False
                break
            data.append(d[i])

        if flag:
            all_pearson.append(data)
            ids.append(i)

    all_pearson = np.array(all_pearson)

    labels = []

    columns = []

    for task in cfg.dataset.tasks:
        label_df = pd.read_csv(task.label)

        label_df = label_df[['id', task.column]]

        if not task.regression:
            le = preprocessing.LabelEncoder()
            label_df[task.column] = le.fit_transform(label_df[task.column])

        labels.append(label_df)

        columns.append(task.column)
    
    labels = ft.reduce(lambda left, right: pd.merge(left, right, on='id', how='outer'), labels)

    labels = labels.dropna(subset=columns, how='all')
    print("label shape", labels.shape)

    id2label = dict(zip(labels['id'], labels[columns].to_numpy()))

    final_label, final_pearson = [], []

    for i, pearson in zip(ids, all_pearson):
        if i not in id2label:
            continue
        final_label.append(id2label[i])
        final_pearson.append(pearson)

    # shape: b, 3, 360, 360; b, 35 
    final_pearson, labels = [np.array(
        data).astype(float) for data in (final_pearson, final_label)]
    
    np.save("final_label.npy", labels)
    
    # if cfg.training.ig_visualize:
    #     task_names = [task.name for task in cfg.dataset.tasks]
    #     top_bottom_sample = {}
    #     for i, name in enumerate(task_names):
    #         top, bottom = find_top_and_bottom(final_pearson, labels[:, i])
    #         top_bottom_sample[name] = (top, bottom)


    final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_pearson, labels)]

    assert final_pearson.shape[0] == labels.shape[0]

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[2:]
        cfg.dataset.num_classes = labels.unique().shape[0]
    if "stratified" in cfg.dataset and cfg.dataset.stratified:
        return torch.zeros(final_pearson.shape[0]), final_pearson, labels, labels
    return torch.zeros(final_pearson.shape[0]), final_pearson, labels
