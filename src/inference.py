"""单张/批量推理入口。

用法:
    python -m src.inference --checkpoint checkpoints/best.pt --image path/to/xray.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data.preprocess import preprocess
from src.models.classifier import build_model


def load_model(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    labels = cfg["data"]["labels"]
    model = build_model(cfg, num_classes=len(labels)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, labels


@torch.no_grad()
def predict_image(model, cfg, image_path: str, device: torch.device, tta: bool = False):
    data_cfg = cfg["data"]
    mean = torch.tensor(data_cfg.get("mean", [0.5] * 3)).view(3, 1, 1)
    std = torch.tensor(data_cfg.get("std", [0.25] * 3)).view(3, 1, 1)

    arr = preprocess(image_path, image_size=data_cfg["image_size"])
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits)
    if tta:
        logits_flip = model(torch.flip(tensor, dims=[3]))
        probs = (probs + torch.sigmoid(logits_flip)) / 2
    return probs.squeeze(0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="MedSight 推理")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="影像路径或目录")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--config", default=None, help="可选，覆盖 checkpoint 内配置")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, labels = load_model(args.checkpoint, device)
    if args.config:
        cfg = load_config(args.config)
    tta = cfg.get("inference", {}).get("tta", False)
    threshold = args.threshold or cfg.get("eval", {}).get("threshold", 0.5)

    image_path = Path(args.image)
    paths = [image_path] if image_path.is_file() else sorted(image_path.rglob("*.png"))

    for p in paths:
        probs = predict_image(model, cfg, str(p), device, tta=tta)
        print(f"\n影像: {p}")
        print("-" * 40)
        for label, prob in sorted(zip(labels, probs), key=lambda x: -x[1]):
            flag = "⚠" if prob >= threshold else " "
            print(f"  {flag} {label:>20}: {prob:.4f}")
        preds = [labels[i] for i in np.argsort(-probs) if probs[i] >= threshold]
        print(f"  → 提示病种: {', '.join(preds) if preds else '无明显异常'}")
        print("  (结果仅供研究参考，不作为诊断依据)")


if __name__ == "__main__":
    main()
