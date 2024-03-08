import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig


def cal_se(m):
    s, u = torch.linalg.eigh(m)
    s[torch.abs(s) < 1] = 1
    return torch.prod(s)


def tensor_log(t):
    s, u = torch.linalg.eigh(t)
    s[s <= 0] = 1e-8
    return u @ torch.diag_embed(torch.log(s)) @ u.permute(1, 0)


def tensor_exp(t):
    # condition: t is symmetric!
    s, u = torch.linalg.eigh(t)
    return u @ torch.diag_embed(torch.exp(s)) @ u.permute(1, 0)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    if cfg.dataset.name == "abcd":
        pearson_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
    elif cfg.dataset.name == "pnc":
        pearson_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
        pearson_data = pearson_data.item()
        pearson_data = pearson_data['data'][:, :120, :120]
    elif cfg.dataset.name == "abide":
        data = np.load(cfg.dataset.path, allow_pickle=True).item()
        pearson_data = data["corr"][:, :100, :100]
    elif cfg.dataset.name == "tcga":
        loaded_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
        loaded_data = loaded_data.item()
        pearson_data = loaded_data['data']
        pearson_data = pearson_data[:, :15, :15]
    pearson_data = torch.from_numpy(pearson_data).float()
    final_pearson = torch.from_numpy(np.array(pearson_data))
    sample_num = final_pearson.shape[0]

    v_diss = []
    r_diss = []
    o_diss = []
    for i in range(1000):

        index = np.random.choice(sample_num, size=2, replace=False)

        pearson1 = final_pearson[index[0]]
        pearson2 = final_pearson[index[1]]

        weight = np.random.uniform(0, 1)

        vmixup = weight*pearson1 + (1-weight)*pearson2

        x1 = tensor_log(pearson1)
        x2 = tensor_log(pearson2)
        x = weight * x1 + (1 - weight) * x2
        rmixup = tensor_exp(x)

        v_dis = cal_se(vmixup)

        r_dis = cal_se(rmixup)

        o_dis = cal_se(pearson2)

        if torch.isnan(v_dis) or torch.isnan(r_dis) or torch.isnan(o_dis):
            print("nan")
            continue

        v_diss.append(v_dis.item())
        r_diss.append(r_dis.item())
        o_diss.append(o_dis.item())

    v_diss = np.abs(np.array(v_diss))
    r_diss = np.abs(np.array(r_diss))
    o_diss = np.abs(np.array(o_diss))
    print("v_diss", f"{np.mean(v_diss):.2E}+-{np.std(v_diss):.2E}")
    print("r_diss", f"{np.mean(r_diss):.2E}+-{np.std(r_diss):.2E}")
    print("o_diss", f"{np.mean(o_diss):.2E}+-{np.std(o_diss):.2E}")
# print(r_diss)
# print(v_diss)
# print(o_diss)


# _, pval = ttest_ind(v_diss, r_diss)

# print("pval", pval)
if __name__ == '__main__':
    main()
