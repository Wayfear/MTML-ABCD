import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import torch
from prepossess import log_euclidean_distance

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

node_num = final_pearson.shape[0]


xs = []
e_diss = []
le_diss = []

for i in range(1000):

    node1 = np.random.randint(node_num)
    node2 = np.random.randint(node_num)

    if node1 == node2:
        continue

    pearson1 = final_pearson[node1].unsqueeze(0)
    pearson2 = final_pearson[node2].unsqueeze(0)

    label1 = final_label[node1]
    label2 = final_label[node2]

    e_distance = torch.norm(pearson1 - pearson2, p=1).item()

    le_dis = log_euclidean_distance(pearson1, pearson2).item()

    x = np.abs(label1 - label2)

    e_diss.append(e_distance)
    le_diss.append(le_dis)
    xs.append(x)
    print(i)


# plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(nrows=1, ncols=2)


ax[0].scatter(xs, e_diss)
ax[1].scatter(xs, le_diss)

plt.savefig('abcd_distance.png', dpi=300)

# g = sns.histplot(distribuion, x="Eig Value", hue="TS Length", palette=[
#     '#4477AA', '#EE6677', '#228833', "#CAD93F", "#9E9EA2"], log_scale=(False, True), fill=True)

# g.legend_.set_title(None)
# plt.yticks([10, 100, 1000])
# plt.tight_layout()

# plt.savefig('abcd_distribution.png', dpi=300)
