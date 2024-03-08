import time
import hydra
import torch
from omegaconf import DictConfig
import numpy as np


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
    print('shape', pearson_data.shape)

    start = time.time()
    s, u = torch.linalg.eigh(pearson_data)
    end = time.time()
    print(f"Time elapsed: {end - start}")


if __name__ == '__main__':
    main()
