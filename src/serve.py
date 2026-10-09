"""推理服务 (FastAPI)。

用法:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000

环境变量:
    MEDSIGHT_CHECKPOINT  模型权重路径 (默认 checkpoints/best.pt)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.inference import load_model, predict_image

app = FastAPI(title="MedSight API", version="0.1.0", description="医学影像 AI 辅助识别（科研用途）")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = os.environ.get("MEDSIGHT_CHECKPOINT", "checkpoints/best.pt")

_model = None
_cfg = None
_labels = None


def get_model():
    global _model, _cfg, _labels
    if _model is None:
        if not Path(CHECKPOINT).exists():
            raise FileNotFoundError(f"未找到模型权重: {CHECKPOINT}")
        _model, _cfg, _labels = load_model(CHECKPOINT, DEVICE)
    return _model, _cfg, _labels


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(DEVICE), "model_loaded": _model is not None}


@app.post("/v1/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    model, cfg, labels = get_model()
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        probs = predict_image(model, cfg, tmp_path, DEVICE, tta=cfg.get("inference", {}).get("tta", False))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    threshold = cfg.get("eval", {}).get("threshold", 0.5)
    predictions = [
        {"label": label, "probability": float(prob)}
        for label, prob in sorted(zip(labels, probs), key=lambda x: -x[1])
    ]
    return JSONResponse(
        {
            "model_version": "0.1.0",
            "threshold": threshold,
            "predictions": predictions,
            "disclaimer": "仅供科研参考，不构成诊断意见。",
        }
    )
