"""训练入口。

用法:
    python -m src.train --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import ChestXrayDataset
from src.models.classifier import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(cfg: dict, split: str, train: bool) -> DataLoader:
    data_cfg = cfg["data"]
    root = data_cfg[f"{split}_dir"]
    csv_path = data_cfg.get(f"{split}_csv")
    dataset = ChestXrayDataset(
        root=root,
        labels=data_cfg["labels"],
        image_size=data_cfg["image_size"],
        mean=data_cfg.get("mean", (0.5, 0.5, 0.5)),
        std=data_cfg.get("std", (0.25, 0.25, 0.25)),
        csv_path=csv_path,
        train=train,
    )
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"] if train else cfg["train"]["batch_size"],
        shuffle=train,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=train,
    )


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg.get("lr", 1e-4)
    wd = train_cfg.get("weight_decay", 1e-4)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=cfg["train"].get("mixed_precision", True)):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg["train"].get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg["train"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())
    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_labels).numpy()
    return total_loss / len(loader.dataset), probs, targets


def compute_auroc(probs, targets) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(targets, probs, average="macro"))
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="MedSight 训练")
    parser.add_argument("--config", required=True, help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader = build_dataloader(cfg, "train", train=True)
    val_loader = build_dataloader(cfg, "val", train=False)

    model = build_model(cfg).to(device)

    pos_weight = None
    if cfg["train"].get("pos_weight") == "auto":
        labels = torch.stack([train_loader.dataset[i][1] for i in range(min(len(train_loader.dataset), 2000))])
        pos = labels.sum(0)
        neg = labels.shape[0] - pos
        pos_weight = (neg / pos.clamp(min=1)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    use_amp = cfg["train"].get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    output_dir = Path(cfg["train"].get("output_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    best_auroc = -1.0
    patience = cfg["train"].get("early_stopping_patience", 8)
    epochs_no_improve = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg)
        scheduler.step()
        val_loss, probs, targets = validate(model, val_loader, criterion, device)
        auroc = compute_auroc(probs, targets)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} AUROC={auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch},
                       output_dir / "best.pt")
            epochs_no_improve = 0
            print(f"  → 保存最佳模型 (AUROC={auroc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("早停触发，训练结束。")
                break

    print(f"训练完成，最佳 AUROC={best_auroc:.4f}")


if __name__ == "__main__":
    main()
