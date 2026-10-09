"""数据集定义。

支持两种组织方式：
1. 目录分类：data/<split>/<class_name>/*.png（单标签）；
2. CSV 清单：多标签，image 列 + 若干 0/1 标签列。

为简化依赖，这里不强制使用 albumentations：若已安装则启用增强。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocess import preprocess

try:  # 可选增强
    import albumentations as A
except Exception:  # pragma: no cover
    A = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".nii"}


def build_transforms(image_size: int, train: bool):
    """返回增强函数或 None。"""
    if A is None:
        return None
    if train:
        return A.Compose(
            [
                A.RandomResizedCrop(image_size, image_size, scale=(0.85, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            ]
        )
    return A.Compose([A.Resize(image_size, image_size)])


class ChestXrayDataset(Dataset):
    """胸部影像数据集，返回 (image_tensor, label_tensor)。"""

    def __init__(
        self,
        root: str | Path,
        labels: Sequence[str],
        image_size: int = 512,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.25, 0.25, 0.25),
        csv_path: Optional[str | Path] = None,
        train: bool = False,
    ) -> None:
        self.root = Path(root)
        self.labels = list(labels)
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.transform = build_transforms(image_size, train)
        self.samples: List[tuple[Path, np.ndarray]] = []

        if csv_path is not None:
            self._load_csv(Path(csv_path))
        else:
            self._load_dir()

        if not self.samples:
            raise RuntimeError(f"在 {self.root} 未找到任何影像样本")

    def _load_csv(self, csv_path: Path) -> None:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get("image") or row.get("Image") or row.get("path")
                if not img_name:
                    continue
                img_path = (self.root / img_name) if not Path(img_name).is_absolute() else Path(img_name)
                label = np.array([float(row.get(l, 0) or 0) for l in self.labels], dtype=np.float32)
                self.samples.append((img_path, label))

    def _load_dir(self) -> None:
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if not class_dirs:
            files = [p for p in self.root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
            for p in files:
                self.samples.append((p, np.zeros(len(self.labels), dtype=np.float32)))
            return
        for idx, cls_dir in enumerate(class_dirs):
            for p in cls_dir.rglob("*"):
                if p.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label = np.zeros(len(self.labels), dtype=np.float32)
                if idx < len(self.labels):
                    label[idx] = 1.0
                self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        arr = preprocess(img_path, image_size=self.image_size)
        if self.transform is not None:
            augmented = self.transform(image=(arr * 255).astype(np.uint8))
            arr = augmented["image"].astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        tensor = (tensor - self.mean) / self.std
        return tensor, torch.from_numpy(label)
