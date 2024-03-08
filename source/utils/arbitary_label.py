import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.stats import ttest_ind


def tensor_log(t):
    s, u = torch.linalg.eigh(t)
    s[s <= 0] = 1e-8
    return u @ torch.diag_embed(torch.log(s)) @ u.permute(1, 0)


def tensor_exp(t):
    # condition: t is symmetric!
    s, u = torch.linalg.eigh(t)
    return u @ torch.diag_embed(torch.exp(s)) @ u.permute(1, 0)


pearson_data = np.load(
    "/home/Anonymous/dataset/ABCD/abcd_rest-pearson-HCP2016.npy", allow_pickle=True)
label_df = pd.read_csv("/home/Anonymous/dataset/ABCD/abcd_tbss01_baseline.csv")

column = "nihtbx_totalcomp_uncorrected"

label_df = label_df[['id', column]]
label_df = label_df[~label_df[column].isnull()]

with open("/home/Anonymous/dataset/ABCD/ids_HCP2016.txt", 'r') as f:
    lines = f.readlines()
    pearson_id = [line[:-1] for line in lines]


id2pearson = dict(zip(pearson_id, pearson_data))

id2label = dict(zip(label_df['id'], label_df[column]))

final_label, final_pearson = [], []

for label_id, l in id2label.items():
    if label_id in id2pearson:
        if np.any(np.isnan(id2pearson[label_id])) == False:
            final_label.append(l)
            final_pearson.append(id2pearson[label_id])

final_pearson = torch.from_numpy(np.array(final_pearson))
final_label = np.array(final_label)

sample_num = final_pearson.shape[0]


v_diss = []
r_diss = []

for i in range(1000):

    index = np.random.choice(sample_num, size=3, replace=False)

    index = sorted(index, key=lambda x: final_label[x])
    # print(index)
    print(final_label[index[0]], final_label[index[1]], final_label[index[2]])

    weight = (final_label[index[1]] - final_label[index[2]]) / \
        (final_label[index[0]] - final_label[index[2]])

    print(weight*final_label[index[0]] + (1-weight) *
          final_label[index[2]], final_label[index[1]])

    pearson1 = final_pearson[index[0]]
    pearson2 = final_pearson[index[1]]
    pearson3 = final_pearson[index[2]]

    vmixup = weight*pearson1 + (1-weight)*pearson3

    v_dis = torch.norm(pearson2 - vmixup, p=1).item()

    x1 = tensor_log(pearson1)
    x2 = tensor_log(pearson3)
    x = weight * x1 + (1 - weight) * x2
    rmixup = tensor_exp(x)

    r_dis = torch.norm(pearson2 - rmixup, p=1).item()

    v_diss.append(v_dis)
    r_diss.append(r_dis)

v_diss = np.array(v_diss)
r_diss = np.array(r_diss)
print("v_diss", np.mean(v_diss), np.std(v_diss))
print("r_diss", np.mean(r_diss), np.std(r_diss))


_, pval = ttest_ind(v_diss, r_diss)

print("pval", pval)