import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import networkx as nx

# ----------------------------
# 配置
# ----------------------------
DATA_DIR = "../dataset_output_01_binary_0.020"
EPISODES = 20
GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if GPU else 'cpu')
BATCH_SIZE = 8
EPOCHS = 20
MASK_RATIO = 0.2  # 训练中隐藏一部分边用于测试

# ----------------------------
# 读取数据
# ----------------------------
start_time = time.time()
all_data = []

for ep in range(EPISODES):
    file_path = os.path.join(DATA_DIR, f"episode_{ep:03d}.jsonl")
    with open(file_path, 'r') as f:
        for line in f:
            frame = json.loads(line)
            node_feat = frame['node_features']
            # 拼接节点特征: pos(2)+vel(2)+speed(1)+heading(1) = 6维
            positions = np.array(node_feat['positions'])
            velocities = np.array(node_feat['velocities'])
            speeds = np.array(node_feat['speeds']).reshape(-1, 1)
            headings = np.array(node_feat['headings']).reshape(-1, 1)
            x = np.concatenate([positions, velocities, speeds, headings], axis=1)
            x = torch.tensor(x, dtype=torch.float)

            adj = np.array(frame['edge_features']['adjacency_matrix'])
            # edge_index 和 edge_label
            src, dst = np.where(adj == 1)
            edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
            edge_label = torch.ones(edge_index.shape[1], dtype=torch.float)  # 正边

            # 对训练隐藏一部分边用于测试
            num_edges = edge_index.shape[1]
            mask_num = int(num_edges * MASK_RATIO)
            perm = np.random.permutation(num_edges)
            mask_idx = perm[:mask_num]
            train_mask_idx = perm[mask_num:]

            train_edge_index = edge_index[:, train_mask_idx]
            train_edge_label = edge_label[train_mask_idx]

            test_edge_index = edge_index[:, mask_idx]
            test_edge_label = edge_label[mask_idx]

            data = Data(x=x,
                        edge_index=train_edge_index,
                        edge_label=train_edge_label,
                        test_edge_index=test_edge_index,
                        test_edge_label=test_edge_label)
            all_data.append(data)

read_time = time.time() - start_time
print(f"数据读取耗时: {read_time:.2f}s")

# ----------------------------
# 数据加载器
# ----------------------------
loader = DataLoader(all_data, batch_size=BATCH_SIZE, shuffle=True)


# ----------------------------
# 模型定义
# ----------------------------
class GCNLinkPredictor(nn.Module):
    def __init__(self, in_feats, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_pair=None):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        if edge_pair is not None:
            # edge_pair: 2xE
            src, dst = edge_pair
            prob = torch.sigmoid((h[src] * h[dst]).sum(dim=1))
            return prob
        else:
            return h


# ----------------------------
# 训练
# ----------------------------
model = GCNLinkPredictor(in_feats=6, hidden_dim=64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

train_start = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_index)
        loss = criterion(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss={total_loss / len(loader):.4f}")
train_time = time.time() - train_start
print(f"训练耗时: {train_time:.2f}s")

# ----------------------------
# 评估
# ----------------------------
model.eval()
with torch.no_grad():
    all_correct = 0
    all_total = 0
    eval_start = time.time()
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.test_edge_index)
        pred_label = (pred > 0.5).float()
        correct = (pred_label == batch.test_edge_label).sum().item()
        all_correct += correct
        all_total += batch.test_edge_label.numel()
    eval_time = time.time() - eval_start
accuracy = all_correct / all_total
print(f"边补全准确率: {accuracy * 100:.2f}%")
print(f"评估耗时: {eval_time:.2f}s")


# ----------------------------
# 可视化几帧
# ----------------------------
def plot_frame(data, title=""):
    G_true = nx.Graph()
    G_pred = nx.Graph()
    x = data.x.cpu().numpy()
    pos = {i: x[i, :2] for i in range(x.shape[0])}

    # 真边
    src, dst = data.edge_index.cpu().numpy()
    for s, d in zip(src, dst):
        G_true.add_edge(s, d)
    # 预测边
    model.eval()
    with torch.no_grad():
        pred_prob = model(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.edge_index.to(DEVICE))
        pred_edges = data.edge_index[:, (pred_prob > 0.5).cpu().numpy()]
        src_p, dst_p = pred_edges.cpu().numpy()
        for s, d in zip(src_p, dst_p):
            G_pred.add_edge(s, d)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    nx.draw(G_true, pos, with_labels=True, node_color='skyblue', edge_color='black')
    plt.title(title + "真实边")
    plt.subplot(1, 2, 2)
    nx.draw(G_pred, pos, with_labels=True, node_color='lightgreen', edge_color='gray')
    plt.title(title + "预测边")
    plt.show()


# 可视化前5帧
for i in range(5):
    plot_frame(all_data[i], title=f"Episode {i}")
