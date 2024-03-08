import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.functional as F
from omegaconf import DictConfig
from .base import BaseModel
import source.dataset as dataset


class PartitionLayer(nn.Module):
    def __init__(self, input_shape, partitions, bias: bool = True):
        super().__init__()
        m, n = input_shape
        self.weight = nn.Parameter(torch.zeros((n, m), requires_grad=True))
        self.partitions = partitions.cuda()

        if bias:
            self.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        t = self.weight * self.partitions
        return F.linear(x, t, self.bias)


class SparseNN(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()
        inner_channels = 64
        self.roi_num = config.dataset.node_sz
        self.triu_indexs = torch.triu_indices(
            self.roi_num, self.roi_num, offset=1)
        self.part_layer = PartitionLayer(
            (config.dataset.node_sz*(config.dataset.node_sz-1)//2,
             config.dataset.node_feature_sz), dataset.masks_from_data)

        self.block1 = nn.Sequential(
            nn.Dropout(config.model.dropout),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.block2 = nn.Linear(self.roi_num, inner_channels)
        self.block3 = nn.Sequential(
            nn.Dropout(config.model.dropout),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(inner_channels, 32),
            nn.Dropout(config.model.dropout),
            nn.LeakyReLU(negative_slope=0.2)
        )

        output_dim = 1 if config.dataset.regression else config.dataset.num_classes

        self.final_block = nn.Sequential(
            nn.Linear(32, 16),
            nn.Dropout(config.model.dropout),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(16, output_dim),
        )

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor) -> torch.tensor:

        pearson = node_feature[:, self.triu_indexs[1], self.triu_indexs[0]]
        x = self.part_layer(pearson)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        result = self.final_block(x)
        return result
