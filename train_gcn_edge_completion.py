import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from dataset_loader import EpisodesDataset
from models.gcn_edge_predictor import GCNEdgePredictor
from utils import Timer, edge_completion_accuracy
import time
from tqdm import tqdm


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')

    # 收集数据路径
    paths = []
    for i in range(args.episode_start, args.episode_end + 1):
        p = os.path.join(args.data_root, f'episode_{i:03d}.jsonl')
        if os.path.exists(p):
            paths.append(p)

    if not paths:
        raise ValueError(
            f"No data files found in {args.data_root} with episode range {args.episode_start}-{args.episode_end}")

    print(f'Found {len(paths)} data files')
    ds = EpisodesDataset(paths)
    print('Loaded frames:', len(ds), 'data load time: {:.3f}s'.format(ds.load_time))

    # 分割训练集和验证集
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GCNEdgePredictor(in_dim=6, hidden=args.hidden, emb_dim=args.emb_dim, dropout=args.dropout)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器 - 移除verbose参数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    # 训练记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    total_training_time = 0.0
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        t0 = time.perf_counter()
        total_loss_epoch = 0.0
        total_edge_loss = 0.0
        total_leader_loss = 0.0

        for data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}'):
            data = data.to(device)

            if data.edge_label_index.numel() == 0:
                continue

            opt.zero_grad()
            edge_logits, leader_logits = model(data.x, pairs=data.edge_label_index)

            # 边预测损失
            edge_loss = bce_loss(edge_logits, data.edge_label.to(device))

            # leader预测损失
            batch = data.batch
            batch_size = int(batch.max().item()) + 1
            leader_loss = 0.0

            for g in range(batch_size):
                mask = (batch == g)
                if mask.sum() == 0:
                    continue

                scores = leader_logits[mask]
                target_idx = data.leader_label[g].item()

                # 确保目标索引在有效范围内
                if 0 <= target_idx < len(scores):
                    local_target = torch.tensor([target_idx], dtype=torch.long, device=device)
                    leader_loss += ce_loss(scores.unsqueeze(0), local_target)

            leader_loss = leader_loss / max(1, batch_size)

            # 总损失
            loss = edge_loss + args.alpha * leader_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss_epoch += loss.item()
            total_edge_loss += edge_loss.item()
            total_leader_loss += leader_loss.item()

        epoch_time = time.perf_counter() - t0
        total_training_time += epoch_time

        avg_loss = total_loss_epoch / len(train_loader)
        avg_edge_loss = total_edge_loss / len(train_loader)
        avg_leader_loss = total_leader_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                if data.edge_label_index.numel() == 0:
                    continue

                edge_logits, leader_logits = model(data.x, pairs=data.edge_label_index)

                edge_loss = bce_loss(edge_logits, data.edge_label.to(device))

                # leader损失计算
                batch = data.batch
                batch_size = int(batch.max().item()) + 1
                batch_leader_loss = 0.0

                for g in range(batch_size):
                    mask = (batch == g)
                    if mask.sum() == 0:
                        continue

                    scores = leader_logits[mask]
                    target_idx = data.leader_label[g].item()

                    if 0 <= target_idx < len(scores):
                        local_target = torch.tensor([target_idx], dtype=torch.long, device=device)
                        batch_leader_loss += ce_loss(scores.unsqueeze(0), local_target)

                batch_leader_loss = batch_leader_loss / max(1, batch_size)
                val_loss += (edge_loss + args.alpha * batch_leader_loss).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f'New best model saved with val_loss: {val_loss:.4f}')

        print(f'Epoch {epoch + 1}/{args.epochs}:')
        print(f'  Train Loss: {avg_loss:.4f} (Edge: {avg_edge_loss:.4f}, Leader: {avg_leader_loss:.4f})')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Time: {epoch_time:.2f}s')
        print(f'  LR: {opt.param_groups[0]["lr"]:.6f}')

    print('Training finished. Total training time: {:.2f}s'.format(total_training_time))

    # 绘制损失曲线
    if args.plot_loss:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
            plt.close()
            print('Training loss plot saved as training_loss.png')
        except ImportError:
            print('matplotlib not available, skipping loss plot')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GCN Edge Completion Model')
    parser.add_argument('--data_root', type=str, default='dataset_output_01_binary_0.020')
    parser.add_argument('--episode_start', type=int, default=0)
    parser.add_argument('--episode_end', type=int, default=19)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='gcn_edge_model.pth')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--plot_loss', action='store_true', help='Plot training loss curve')
    args = parser.parse_args()
    train(args)