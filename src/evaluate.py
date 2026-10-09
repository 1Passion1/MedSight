"""评估入口。

用法:
    python -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data.dataset import ChestXrayDataset
from src.models.classifier import build_model
from src.train import build_dataloader, validate
from torch.utils.data import DataLoader


METRIC_NAMES = ["auroc", "auprc", "f1", "sensitivity", "specificity", "accuracy"]


def compute_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {}
    preds = (probs >= threshold).astype(int)
    try:
        metrics["auroc"] = float(roc_auc_score(targets, probs, average="macro"))
    except ValueError:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = float(average_precision_score(targets, probs, average="macro"))
    except ValueError:
        metrics["auprc"] = float("nan")
    metrics["f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))
    metrics["sensitivity"] = float(recall_score(targets, preds, average="macro", zero_division=0))
    metrics["specificity"] = _specificity(targets, preds)
    metrics["accuracy"] = float((preds == targets).mean())
    return metrics


def _specificity(targets: np.ndarray, preds: np.ndarray) -> float:
    tn = ((preds == 0) & (targets == 0)).sum(axis=0)
    fp = ((preds == 1) & (targets == 0)).sum(axis=0)
    denom = tn + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.where(denom > 0, tn / denom, np.nan)
    return float(np.nanmean(spec))


def find_best_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.1, 0.91, 0.05):
        f1 = f1_score(targets, (probs >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser(description="MedSight 评估")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    labels = cfg["data"]["labels"]
    model = build_model(cfg, num_classes=len(labels)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = build_dataloader(cfg, "test", train=False)
    print("正在评估测试集...")
    probs, targets = _predict(model, loader, device)

    threshold = cfg.get("eval", {}).get("threshold", 0.5)
    metrics = compute_metrics(probs, targets, threshold)
    best_t = find_best_threshold(probs, targets)

    print("=" * 50)
    print(f"测试集评估 (阈值={threshold}):")
    for name in METRIC_NAMES:
        print(f"  {name:>12}: {metrics[name]:.4f}")
    print(f"  最佳阈值约: {best_t:.2f}")
    print("=" * 50)

    print("各病种 AUROC:")
    for i, label in enumerate(labels):
        if targets[:, i].sum() == 0:
            continue
        try:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(targets[:, i], probs[:, i])
            print(f"  {label:>20}: {auc:.4f}")
        except ValueError:
            continue


@torch.no_grad()
def _predict(model, loader, device):
    all_probs, all_targets = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


if __name__ == "__main__":
    main()
