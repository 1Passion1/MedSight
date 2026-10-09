"""基础测试（无需真实数据）。"""

from __future__ import annotations

import numpy as np


def test_import_modules():
    import src  # noqa: F401
    from src.data import preprocess  # noqa: F401
    from src.config import load_config  # noqa: F401


def test_normalize_range():
    from src.data.preprocess import _normalize

    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    out = _normalize(arr)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_to_three_channel():
    from src.data.preprocess import to_three_channel

    gray = np.zeros((16, 16), dtype=np.float32)
    out = to_three_channel(gray)
    assert out.shape == (16, 16, 3)


def test_crop_black_border():
    from src.data.preprocess import crop_black_border

    arr = np.zeros((32, 32), dtype=np.float32)
    arr[8:24, 8:24] = 1.0
    cropped = crop_black_border(arr)
    assert cropped.shape == (16, 16)
