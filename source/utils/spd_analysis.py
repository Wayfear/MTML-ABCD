import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
# eig_val = {}

# for i in (64, 128, 256, 512, 1024):
#     data = np.load(
#         Path(
#             f"/local/home/Anonymous/ABCD/ABCD/abcd_rest-pearson-HCP2016-{i}.npy"),
#         allow_pickle=True)
#     samples = []
#     for d in data:
#         if np.isnan(d).any():
#             continue
#         samples.append(d)

#     data = np.array(samples)

#     data = torch.from_numpy(data)

#     L, Q = torch.linalg.eigh(data)
#     eig_val[i] = L.numpy()


# tem_eig_val = np.save(Path(
#     "/local/home/Anonymous/ABCD/ABCD/abcd_rest-eigval-HCP2016.npy"), eig_val, allow_pickle=True)

tem_eig_val = np.load(Path(
    "/local/home/Anonymous/ABCD/ABCD/abcd_rest-eigval-HCP2016.npy"), allow_pickle=True).item()
eig_val = {}
for k, v in tem_eig_val.items():
    temp = []
    for i in v:
        if np.iscomplex(i).any():
            continue
        temp.append(i)

    v = np.array(temp)
    # v[v < 1e-3] = 0
    eig_val[k] = v.flatten()

flat_eig_val = []
keys = []

for k, v in eig_val.items():
    flat_eig_val.append(v)
    keys.append(np.array([k] * v.shape[0]))

flat_eig_val = np.concatenate(flat_eig_val, axis=0)
keys = np.concatenate(keys, axis=0)


distribuion = {"Eig Value": flat_eig_val, "TS Length": keys}

data = pd.DataFrame.from_dict(distribuion)

g = sns.histplot(distribuion, x="Eig Value", hue="TS Length", palette=[
    '#4477AA', '#EE6677', '#228833', "#CAD93F", "#9E9EA2"], log_scale=(False, True), fill=True)

g.legend_.set_title(None)
plt.yticks([10, 100, 1000])
plt.tight_layout()

plt.savefig('abcd_distribution.png', dpi=300)
