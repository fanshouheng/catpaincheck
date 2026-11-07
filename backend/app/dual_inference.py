"""One local photograph, independent YOLO and automatic FGS predictions.

The legacy ai envelope deliberately contains no fabricated binary score.
New reports consume machines; older persisted reports keep their old fields.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import kittycheck, localization, yolo_inference
from .schemas import AutoFgsResult, YoloResult

LOGGER = logging.getLogger(__name__)
MODEL_VERSION = "yolo-kittycheck.dual-v1"


def get_model_status() -> dict:
    engines = {"localization": localization.get_localization_status(),
               "yolo": yolo_inference.get_model_status(), "auto_fgs": kittycheck.get_model_status()}
    ready = all(item["ready"] for item in engines.values())
    return {"ready": ready, "detail": "猫脸定位、YOLO 与自动 FGS 已就绪。" if ready else "部分模型尚未就绪；仍可完成人工 FGS。",
            "version": MODEL_VERSION, "engines": engines}


def analyze_image(image_path: Path, crop_path: Path) -> dict:
    from PIL import Image, ImageOps

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    geometry = localization.localize_image(image)
    if geometry["status"] == "ok":
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop(tuple(int(v) for v in geometry["bbox"])).save(crop_path, "JPEG", quality=94)
    # Isolate each engine's failure so the other result remains available.
    try:
        yolo = YoloResult.model_validate(yolo_inference.analyze(image, geometry)).model_dump(mode="json")
    except Exception:
        LOGGER.exception("YOLO engine failed")
        yolo = {"status": "error", "note": "YOLO 未能完成。", "model_version": yolo_inference.MODEL_VERSION,
                "preprocess_version": yolo_inference.PREPROCESS_VERSION}
    try:
        automatic = AutoFgsResult.model_validate(kittycheck.analyze(image, geometry)).model_dump(mode="json")
    except Exception:
        LOGGER.exception("Automatic FGS engine failed")
        automatic = {"status": "error", "note": "自动 FGS 未能完成。", "model_version": kittycheck.MODEL_VERSION,
                     "preprocess_version": kittycheck.PREPROCESS_VERSION}
    return {"status": "unavailable", "label": None, "raw_score": None, "model_version": MODEL_VERSION,
            "note": "本记录分别保存 YOLO 与自动 FGS；无单一合并模型分值。", "bbox": geometry.get("bbox"),
            "localization": geometry, "machines": {
                "prediction_id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
                "yolo": yolo, "auto_fgs": automatic}}
