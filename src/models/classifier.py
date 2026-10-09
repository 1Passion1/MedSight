"""影像分类模型。

优先使用 timm 加载预训练骨干；未安装 timm 时退化为基础实现。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import timm
except Exception:  # pragma: no cover
    timm = None

try:
    import torchvision.models as tvm
except Exception:  # pragma: no cover
    tvm = None


class ImageClassifier(nn.Module):
    """通用影像分类器：骨干 + Dropout + 线性头。"""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.3,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.feature_dim = self._build_backbone(backbone, pretrained, in_channels)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def _build_backbone(self, backbone: str, pretrained: bool, in_channels: int) -> int:
        if timm is not None:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0,
                in_chans=in_channels,
            )
            return self.backbone.num_features
        if tvm is not None:
            net = getattr(tvm, backbone, None) or tvm.resnet50(pretrained=pretrained)
            self.feature_dim_attr = getattr(net, "fc", None)
            feat = net.fc.in_features if hasattr(net, "fc") else 2048
            net.fc = nn.Identity()
            self.backbone = net
            return feat
        raise RuntimeError("需要安装 timm 或 torchvision 才能构建骨干网络")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim > 2:
            features = features.flatten(1)
        return self.head(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """返回 sigmoid 概率（多标签）。"""
        return torch.sigmoid(self.forward(x))


def build_model(cfg: dict, num_classes: Optional[int] = None) -> ImageClassifier:
    """根据配置构建模型。"""
    model_cfg = cfg.get("model", {})
    return ImageClassifier(
        backbone=model_cfg.get("backbone", "convnext_tiny"),
        num_classes=num_classes or model_cfg.get("num_classes", 14),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
    )
