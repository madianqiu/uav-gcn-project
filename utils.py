import time
import torch

class Timer:
    def __init__(self):
        self.start = None
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

def edge_completion_accuracy(pred_mask, pred_binary, labels):
    pred = pred_binary.long()
    labels = labels.long()
    TP = int(((pred==1) & (labels==1)).sum().item())
    FP = int(((pred==1) & (labels==0)).sum().item())
    FN = int(((pred==0) & (labels==1)).sum().item())
    denom = TP + FP + FN
    acc = TP / denom if denom>0 else 0.0
    return {
        'TP':TP, 'FP':FP, 'FN':FN, 'accuracy':acc
    }