"""医学影像预处理工具。

支持 DICOM / NIfTI / 普通图像，统一为 numpy 灰度或 RGB 数组。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def read_image(path: str | Path) -> np.ndarray:
    """读取影像为 float32 数组 (H, W) 或 (H, W, C)，值域 [0, 1]。"""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".dcm", ".dicom"}:
        return _read_dicom(path)
    if suffix in {".nii", ".gz"}:
        return _read_nifti(path)
    return _read_standard(path)


def _read_dicom(path: Path) -> np.ndarray:
    import pydicom  # 延迟导入，避免无依赖时报错

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    return _normalize(arr)


def _read_nifti(path: Path) -> np.ndarray:
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    if arr.ndim == 3:  # 取中间层作为 2D 示意
        arr = arr[arr.shape[0] // 2]
    return _normalize(arr)


def _read_standard(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        arr = np.asarray(img).astype(np.float32)
    return _normalize(arr)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """按百分位裁剪后归一化到 [0, 1]。"""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def apply_clahe(arr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """对 X 光影像做 CLAHE 对比度增强。输入 [0,1] 浮点数组。"""
    import cv2

    img = (arr * 255).astype(np.uint8)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    out = clahe.apply(gray)
    return out.astype(np.float32) / 255.0


def crop_black_border(arr: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """裁剪四周黑色背景。"""
    mask = arr > threshold
    if mask.ndim == 3:
        mask = mask.any(axis=-1)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return arr
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return arr[rmin : rmax + 1, cmin : cmax + 1]


def to_three_channel(arr: np.ndarray) -> np.ndarray:
    """灰度复制为 3 通道，以复用 ImageNet 预训练骨干。"""
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1)
    return arr


def preprocess(path: str | Path, image_size: int = 512, use_clahe: bool = True) -> np.ndarray:
    """完整预处理：读取 → 裁剪黑边 → (CLAHE) → 缩放 → 3 通道。"""
    arr = read_image(path)
    arr = crop_black_border(arr)
    if use_clahe and arr.ndim == 2:
        arr = apply_clahe(arr)
    arr = to_three_channel(arr)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0
