import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEdgePredictor(nn.Module):
    def __init__(self, in_dim=6, hidden=64, emb_dim=64, dropout=0.2):
        super().__init__()
        # 使用MLP作为节点编码器，因为边结构未知
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 边解码器：基于节点嵌入预测边存在性
        self.edge_decoder = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

        # leader分类器
        self.leader_head = nn.Sequential(
            nn.Linear(emb_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x, pairs=None):
        # 节点编码
        h = self.node_encoder(x)  # [N, emb_dim]

        # leader预测
        leader_logits = self.leader_head(h).squeeze(-1)  # [N]

        # 边预测
        if pairs is None:
            return None, leader_logits

        src = pairs[0]
        dst = pairs[1]
        h_src = h[src]
        h_dst = h[dst]

        # 拼接源节点和目标节点特征
        h_cat = torch.cat([h_src, h_dst], dim=1)
        edge_logits = self.edge_decoder(h_cat).squeeze(-1)

        return edge_logits, leader_logits