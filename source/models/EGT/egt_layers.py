
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from omegaconf import DictConfig


class EGT_Layer(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()

        assert not (config.model.node_width % config.model.num_heads)

        self.config = config
        self.dot_dim = config.model.node_width//config.model.num_heads

        self.mha_ln_h = nn.LayerNorm(config.model.node_width)
        self.mha_ln_e = nn.LayerNorm(config.model.edge_width)
        self.lin_E = nn.Linear(config.model.edge_width, config.model.num_heads)
        if config.model.node_update:
            self.lin_QKV = nn.Linear(
                config.model.node_width, config.model.node_width*3)
            self.lin_G = nn.Linear(
                config.model.edge_width, config.model.num_heads)
        else:
            self.lin_QKV = nn.Linear(
                config.model.node_width, config.model.node_width*2)

        self.ffn_fn = getattr(F, config.model.activation)
        if config.model.node_update:
            self.lin_O_h = nn.Linear(
                config.model.node_width, config.model.node_width)
            if config.model.node_mha_dropout > 0:
                self.mha_drp_h = nn.Dropout(config.model.node_mha_dropout)

            node_inner_dim = round(
                config.model.node_width*config.model.node_ffn_multiplier)
            self.ffn_ln_h = nn.LayerNorm(config.model.node_width)
            self.lin_W_h_1 = nn.Linear(config.model.node_width, node_inner_dim)
            self.lin_W_h_2 = nn.Linear(node_inner_dim, config.model.node_width)
            if config.model.node_ffn_dropout > 0:
                self.ffn_drp_h = nn.Dropout(config.model.node_ffn_dropout)

        if config.model.edge_update:
            self.lin_O_e = nn.Linear(
                config.model.num_heads, config.model.edge_width)
            if config.model.edge_mha_dropout > 0:
                self.mha_drp_e = nn.Dropout(config.model.edge_mha_dropout)

            edge_inner_dim = round(
                config.model.edge_width*config.model.edge_ffn_multiplier)
            self.ffn_ln_e = nn.LayerNorm(config.model.edge_width)
            self.lin_W_e_1 = nn.Linear(config.model.edge_width, edge_inner_dim)
            self.lin_W_e_2 = nn.Linear(edge_inner_dim, config.model.edge_width)
            if config.model.edge_ffn_dropout > 0:
                self.ffn_drp_e = nn.Dropout(config.model.edge_ffn_dropout)

    def _egt(self,
             QKV: torch.Tensor,
             E: torch.Tensor,
             G: torch.Tensor,
             mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        shp = QKV.shape
        Q, K, V = QKV.view(
            shp[0], shp[1], -1, self.config.model.num_heads).split(self.dot_dim, dim=2)

        A_hat = torch.einsum('bldh,bmdh->blmh', Q, K)
        if self.config.model.scale_dot:
            A_hat = A_hat * (self.dot_dim ** -0.5)

        H_hat = A_hat.clamp(
            *self.config.model.clip_logits_value) + E

        if mask is None:
            if self.config.model.attn_maskout > 0 and self.training:
                rmask = torch.empty_like(H_hat).bernoulli_(
                    self.config.model.attn_maskout) * -1e9
                gates = torch.sigmoid(G)  # +rmask
                A_tild = F.softmax(H_hat+rmask, dim=2) * gates
            else:
                gates = torch.sigmoid(G)
                A_tild = F.softmax(H_hat, dim=2) * gates
        else:
            if self.config.model.attn_maskout > 0 and self.training:
                rmask = torch.empty_like(H_hat).bernoulli_(
                    self.config.model.attn_maskout) * -1e9
                gates = torch.sigmoid(G+mask)
                A_tild = F.softmax(H_hat+mask+rmask, dim=2) * gates
            else:
                gates = torch.sigmoid(G+mask)
                A_tild = F.softmax(H_hat+mask, dim=2) * gates

        if self.config.model.attn_dropout > 0:
            A_tild = F.dropout(
                A_tild,
                p=self.config.model.attn_dropout,
                training=self.training)

        V_att = torch.einsum('blmh,bmkh->blkh', A_tild, V)

        if self.config.model.scale_degree:
            degrees = torch.sum(gates, dim=2, keepdim=True)
            degree_scalers = torch.log(1+degrees)
            degree_scalers[:, :0] = 1.
            V_att = V_att * degree_scalers

        V_att = V_att.reshape(
            shp[0], shp[1], self.config.model.num_heads*self.dot_dim)
        return V_att, H_hat

    def _egt_edge(self, QK: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        shp = QK.shape
        Q, K = QK.view(
            shp[0], shp[1], -1, self.config.model.num_heads).split(self.dot_dim, dim=2)

        A_hat = torch.einsum('bldh,bmdh->blmh', Q, K)
        if self.config.model.scale_dot:
            A_hat = A_hat * (self.dot_dim ** -0.5)
        H_hat = A_hat.clamp(*self.config.model.clip_logits_value) + E
        return H_hat

    def forward(self, h, e):
        h_r1 = h
        e_r1 = e

        h_ln = self.mha_ln_h(h)
        e_ln = self.mha_ln_e(e)

        QKV = self.lin_QKV(h_ln)
        E = self.lin_E(e_ln)

        if self.config.model.node_update:
            G = self.lin_G(e_ln)
            V_att, H_hat = self._egt(QKV, E, G, None)

            h = self.lin_O_h(V_att)
            if self.config.model.node_mha_dropout > 0:
                h = self.mha_drp_h(h)
            h.add_(h_r1)

            h_r2 = h
            h_ln = self.ffn_ln_h(h)
            h = self.lin_W_h_2(self.ffn_fn(self.lin_W_h_1(h_ln)))
            if self.config.model.node_ffn_dropout > 0:
                h = self.ffn_drp_h(h)
            h.add_(h_r2)
        else:
            H_hat = self._egt_edge(QKV, E)

        if self.config.model.edge_update:
            e = self.lin_O_e(H_hat)
            if self.config.model.edge_mha_dropout > 0:
                e = self.mha_drp_e(e)
            e.add_(e_r1)

            e_r2 = e
            e_ln = self.ffn_ln_e(e)
            e = self.lin_W_e_2(self.ffn_fn(self.lin_W_e_1(e_ln)))
            if self.config.model.edge_ffn_dropout > 0:
                e = self.ffn_drp_e(e)
            e.add_(e_r2)

        return h, e
