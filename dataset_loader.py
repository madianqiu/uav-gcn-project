import json
import os
import time
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

# Assumptions:
# - Each JSONL line is one frame as you posted.
# - Node features: positions(15,2), velocities(15,2), speeds(15,), headings(15,)
# - adjacency_matrix present in edge_features

class EpisodesDataset(Dataset):
    def __init__(self, jsonl_paths, feature_mode='pos_vel_speed_head', transform=None):
        super().__init__()
        self.paths = jsonl_paths
        self.transform = transform
        self.feature_mode = feature_mode
        # We'll parse all frames into memory (small per-frame graphs)
        self.frames = []
        t0 = time.perf_counter()
        for p in self.paths:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    j = json.loads(line)
                    self.frames.append(j)
        self.load_time = time.perf_counter() - t0

    def len(self):
        return len(self.frames)

    def get(self, idx):
        # Build a torch_geometric Data for a single frame
        f = self.frames[idx]
        node = f['node_features']
        positions = np.asarray(node['positions'], dtype=float)  # (N,2)
        velocities = np.asarray(node['velocities'], dtype=float)  # (N,2)
        speeds = np.asarray(node.get('speeds', [np.linalg.norm(v) for v in velocities]), dtype=float).reshape(-1,1)
        headings = np.asarray(node.get('headings', [0.0]*len(positions)), dtype=float).reshape(-1,1)

        # Compose features according to feature_mode (we'll use pos+vel+speed+heading by default)
        x = np.concatenate([positions, velocities, speeds, headings], axis=1).astype(np.float32)  # (N,6)
        x = torch.from_numpy(x)

        adj = np.asarray(f['edge_features']['adjacency_matrix'], dtype=np.int64)
        N = x.shape[0]

        # decide if adjacency is symmetric (undirected) or directed
        is_symmetric = np.all(adj == adj.T)

        # prepare edge_label_index and edge_label according to symmetry
        pairs = []
        labels = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if is_symmetric and j < i:
                    continue  # for undirected, only keep i<j pairs
                pairs.append([i, j])
                labels.append(int(adj[i][j]))
        if len(pairs) == 0:
            edge_label_index = torch.zeros((2,0), dtype=torch.long)
            edge_label = torch.zeros((0,), dtype=torch.float32)
        else:
            edge_label_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()  # (2, E)
            edge_label = torch.tensor(labels, dtype=torch.float32)

        data = Data(x=x, num_nodes=N)
        # store labels and meta
        data.edge_label_index = edge_label_index
        data.edge_label = edge_label
        data.is_undirected = bool(is_symmetric)
        # leader: infer leader index? We don't have explicit label in JSON. We'll assume leader is node 0 if present.
        # If you have explicit leader label in metadata, replace below.
        leader_idx = f.get('metadata', {}).get('leader_index', None)
        if leader_idx is None:
            # fallback heuristic: the node closest to cluster_centroid in global_features
            centroid = np.asarray(f.get('global_features', {}).get('cluster_centroid', [0.0,0.0]), dtype=float)
            d = np.linalg.norm(positions - centroid.reshape(1,2), axis=1)
            leader_idx = int(np.argmin(d))
        data.leader_label = torch.tensor(leader_idx, dtype=torch.long)

        # save extra for visualization
        data.positions = torch.from_numpy(positions.astype(np.float32))
        data.raw_adj = torch.from_numpy(adj.astype(np.int64))

        return data