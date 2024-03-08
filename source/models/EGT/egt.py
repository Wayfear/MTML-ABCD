
import torch
from torch import nn
import torch.nn.functional as F
from .egt_layers import EGT_Layer
from omegaconf import DictConfig
from ..base import BaseModel

"""
The EGT model implementation revised from https://github.com/shamim-hussain/egt_pytorch
"""


class EGT(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.EGT_layers = nn.ModuleList([EGT_Layer(config)
                                         for _ in range(config.model.model_height)])

        self.dim_reduction = nn.Sequential(
            nn.Linear(config.model.node_width, 8),
            nn.LeakyReLU()
        )
        final_dim = 8 * config.model.node_width

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def final_embedding(self, nodes):
        outputs = self.dim_reduction(nodes)
        outputs = outputs.reshape((nodes.shape[0], -1))
        return self.fc(outputs)

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):
        nodes, edges = node_feature, node_feature.unsqueeze(-1)

        for layer in self.EGT_layers:
            nodes, edges = layer(nodes, edges)

        outputs = self.final_embedding(nodes)

        return outputs
