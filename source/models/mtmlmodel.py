import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from omegaconf import DictConfig
from .base import BaseModel
from .BNT import BrainNetworkTransformer


class MTMLBNTWGaitM(BaseModel):

    def __init__(self, cfg: DictConfig):

        super().__init__()

        self.config = cfg

        self.view_num = len(cfg.dataset.views)

        task_cfg = cfg.dataset.tasks
        task_type = ["regression" if task.regression else "classification"for task in task_cfg]

        self.task_num = len(task_type)
        self.bnt = BrainNetworkTransformer(cfg)

        if cfg.model.mask:
            # self.mask = nn.Parameter(torch.ones(len(task_type), cfg.dataset.node_sz, cfg.dataset.node_sz), requires_grad=True)
            self.mask = nn.Parameter(torch.zeros(len(task_type), cfg.dataset.node_sz, cfg.dataset.node_sz), requires_grad=True)
            self.mask = nn.init.kaiming_uniform_(self.mask, a=0, mode='fan_in', nonlinearity='leaky_relu')
            self.register_parameter('mask', self.mask)

        self.fcs = nn.ModuleList()

        # self.task_fusion_layer = nn.MultiheadAttention(cfg.model.preprocess_dim, num_heads=4, batch_first=True)

        for i, task in enumerate(task_type):

            self.fcs.append(nn.Sequential(
                nn.Linear(cfg.model.preprocess_dim * self.view_num, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1)
            ))
        

    def forward(self, node_feature):
        bz, view_num, node_num, feature_dim = node_feature.shape

        time_seires = torch.ones(bz, 1).cuda()

        output = []
        mask = None
        if self.config.model.mask:
            node_feature = node_feature.unsqueeze(dim=1)

            # masked_feature = node_feature * self.mask

            # (b x 1 x n x n) * (t x n x n) = (b x t x n x n)
            masked_feature = torch.matmul(node_feature, self.mask)

            masked_feature_after_sigmoid = torch.sigmoid(masked_feature)

            mask = masked_feature_after_sigmoid

            masked_feature = masked_feature_after_sigmoid * node_feature

            masked_feature = masked_feature.view(-1, node_num, feature_dim)

            graph_embedding = self.bnt(time_seires, masked_feature)

            graph_embedding = graph_embedding.view(bz, self.task_num, -1)

            # fusion_embedding = self.task_fusion_layer(graph_embedding, graph_embedding, graph_embedding, need_weights=False)[0]

            # graph_embedding += fusion_embedding

            for i, fc in enumerate(self.fcs):
                output.append(fc(graph_embedding[:, i]))
        else:

            node_feature = node_feature.view(-1, node_num, feature_dim)
            graph_embedding = self.bnt(time_seires, node_feature)
            graph_embedding = graph_embedding.view(bz, -1)
            for i, fc in enumerate(self.fcs):
                output.append(fc(graph_embedding))

        output = torch.cat(output, dim=1)
        return output
    

    def get_mask(self):
        return self.mask.detach().cpu().numpy()
    
    def get_masked_network(self, node_feature):
        node_feature = node_feature.unsqueeze(dim=1)
        masked_feature = torch.matmul(node_feature, self.mask)
        masked_feature_after_sigmoid = torch.sigmoid(masked_feature)
        # masked_feature = masked_feature_after_sigmoid * node_feature
        return masked_feature_after_sigmoid[0].detach().cpu().numpy()

    # def get_attention_weights(self):
    #     return [atten.get_attention_weights() for atten in self.attention_list]
