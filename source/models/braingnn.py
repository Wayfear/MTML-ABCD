import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from .base import BaseModel
import torch.nn as nn


class BrainGNN(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()
        inner_dim = config.dataset.node_sz
        self.roi_num = config.dataset.node_sz
        self.gcn = nn.Sequential(
            nn.Linear(config.dataset.node_feature_sz, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        output_dim = 1 if config.dataset.regression else config.dataset.num_classes

        self.fcn = nn.Sequential(
            nn.Linear(8*self.roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):
        bz = node_feature.shape[0]

        m = torch.abs(node_feature)

        degree = torch.sum(m, dim=-1)

        degree = torch.bmm(torch.unsqueeze(degree, dim=-1),
                           torch.unsqueeze(degree, dim=-2))

        degree = torch.sqrt(degree)
        degree = torch.where(degree == 0, torch.ones_like(degree), degree)

        m = torch.div(m, degree)

        x = torch.einsum('ijt,itk->ijk', m, node_feature)

        x = self.gcn(x)



        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijt,itk->ijk', m, x)

        x = self.gcn1(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijt,itk->ijk', m, x)

        x = self.gcn2(x)

        x = self.bn3(x)

        x = x.view(bz, -1)

        return self.fcn(x)
