import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from dataset_loader import EpisodesDataset
from models.gcn_edge_predictor import GCNEdgePredictor
from torch_geometric.loader import DataLoader
from utils import edge_completion_accuracy
import json


def visualize_frame(data, pred_adj_binary, out_path, title=''):
    pos = data.positions.numpy()
    N = pos.shape[0]
    plt.figure(figsize=(8, 8))

    # 绘制节点
    plt.scatter(pos[:, 0], pos[:, 1], c='blue', s=100, alpha=0.7, edgecolors='black')

    # 标记节点编号
    for i in range(N):
        plt.text(pos[i, 0] + 0.5, pos[i, 1] + 0.5, str(i), color='black', fontsize=12, fontweight='bold')

    # 真实边（黑色实线）
    gt = data.raw_adj.numpy()
    for i in range(N):
        for j in range(i + 1, N):
            if gt[i, j] == 1:
                plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                         linestyle='-', color='black', linewidth=2, alpha=0.7)

    # 预测边（红色虚线）
    for i in range(N):
        for j in range(i + 1, N):
            if pred_adj_binary[i, j] == 1:
                plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                         linestyle='--', color='red', linewidth=2, alpha=0.8)

    # 标记leader节点
    leader_idx = data.leader_label.item()
    plt.scatter(pos[leader_idx, 0], pos[leader_idx, 1],
                c='gold', s=200, marker='*', edgecolors='black', label=f'Leader {leader_idx}')

    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_and_visualize(model_path, data_root, out_dir, hidden=128, emb_dim=128, dropout=0.2, device='cpu',
                           sample_frames=10):
    # 加载数据
    paths = [os.path.join(data_root, f) for f in sorted(os.listdir(data_root))
             if f.endswith('.jsonl')]

    if not paths:
        print(f"No JSONL files found in {data_root}")
        return

    ds = EpisodesDataset(paths)
    print(f'Evaluating on {len(ds)} frames')

    # 加载模型 - 使用与训练时相同的参数
    model = GCNEdgePredictor(in_dim=6, hidden=hidden, emb_dim=emb_dim, dropout=dropout)
    print(f"Loading model with hidden={hidden}, emb_dim={emb_dim}, dropout={dropout}")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Trying to load with strict=False...")
        # 尝试非严格加载
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print("✓ Model loaded with strict=False")

    model.to(device)
    model.eval()

    loader = DataLoader(ds, batch_size=1, shuffle=False)

    os.makedirs(out_dir, exist_ok=True)

    # 评估指标
    overall = {'TP': 0, 'FP': 0, 'FN': 0}
    leader_correct = 0
    total_graphs = 0

    sampled = 0
    results = []

    for i, data in enumerate(loader):
        data = data.to(device)
        with torch.no_grad():
            logits, leader_logits = model(data.x, pairs=data.edge_label_index)
            probs = torch.sigmoid(logits)
            pred_bin = (probs > 0.5).long().cpu().numpy()
            labels = data.edge_label.cpu().numpy()

            # 边预测指标
            TP = int(((pred_bin == 1) & (labels == 1)).sum())
            FP = int(((pred_bin == 1) & (labels == 0)).sum())
            FN = int(((pred_bin == 0) & (labels == 1)).sum())

            overall['TP'] += TP
            overall['FP'] += FP
            overall['FN'] += FN

            # leader预测指标
            batch = data.batch
            for g in range(int(batch.max().item()) + 1):
                mask = (batch == g)
                if mask.sum() == 0:
                    continue

                scores = leader_logits[mask]
                pred_leader = torch.argmax(scores).item()
                true_leader = data.leader_label[g].item()

                if pred_leader == true_leader:
                    leader_correct += 1
                total_graphs += 1

            # 重建预测邻接矩阵用于可视化
            N = data.num_nodes
            is_undirected = data.is_undirected
            pred_adj = np.zeros((N, N), dtype=int)

            idx = 0
            if is_undirected:
                for a in range(N):
                    for b in range(a + 1, N):
                        if idx < len(pred_bin) and pred_bin[idx] == 1:
                            pred_adj[a, b] = 1
                            pred_adj[b, a] = 1
                        idx += 1
            else:
                for a in range(N):
                    for b in range(N):
                        if a == b:
                            continue
                        if idx < len(pred_bin) and pred_bin[idx] == 1:
                            pred_adj[a, b] = 1
                        idx += 1

            # 保存结果
            frame_result = {
                'frame_id': i,
                'TP': TP,
                'FP': FP,
                'FN': FN,
                'edge_accuracy': TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0,
                'leader_correct': pred_leader == true_leader
            }
            results.append(frame_result)

            # 可视化采样帧
            if sampled < sample_frames:
                out_path = os.path.join(out_dir, f'frame_{i:04d}.png')
                title = f'Frame {i}\nTP={TP}, FP={FP}, FN={FN}'
                visualize_frame(data.cpu(), pred_adj, out_path, title=title)
                sampled += 1

    # 计算总体指标
    TP, FP, FN = overall['TP'], overall['FP'], overall['FN']
    edge_accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    leader_accuracy = leader_correct / total_graphs if total_graphs > 0 else 0.0

    print('=' * 50)
    print('EVALUATION RESULTS')
    print('=' * 50)
    print(f'Edge Completion:')
    print(f'  TP: {TP}, FP: {FP}, FN: {FN}')
    print(f'  Accuracy: {edge_accuracy:.4f}')
    print(f'Leader Prediction:')
    print(f'  Correct: {leader_correct}/{total_graphs}')
    print(f'  Accuracy: {leader_accuracy:.4f}')
    print('=' * 50)

    # 保存详细结果
    results_summary = {
        'edge_completion': {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'accuracy': edge_accuracy
        },
        'leader_prediction': {
            'correct': leader_correct,
            'total': total_graphs,
            'accuracy': leader_accuracy
        },
        'per_frame_results': results
    }

    with open(os.path.join(out_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f'Detailed results saved to {os.path.join(out_dir, "evaluation_results.json")}')
    print(f'Visualizations saved to {out_dir}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--data_root', default='dataset_output_01_binary_0.020', help='Root directory of dataset')
    parser.add_argument('--out_dir', default='results_visuals', help='Output directory for visualizations')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension (must match training)')
    parser.add_argument('--emb_dim', type=int, default=128, help='Embedding dimension (must match training)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (must match training)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--sample_frames', type=int, default=10, help='Number of frames to visualize')
    args = parser.parse_args()

    evaluate_and_visualize(
        model_path=args.model,
        data_root=args.data_root,
        out_dir=args.out_dir,
        hidden=args.hidden,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        device=args.device,
        sample_frames=args.sample_frames
    )