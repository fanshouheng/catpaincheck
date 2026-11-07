"""Lightweight ONNX inference for Vercel; no torch import at runtime.

The detector and classification threshold are shared with the local adapter.
Float-mode Pillow resampling follows the local tensor antialias filter closely,
but floating-point accumulation can change a quantized channel by 1/255. This
preprocessing adapter therefore has its own version and numerical validation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any

from .inference import THRESHOLD, UPSTREAM_COMMIT, WEIGHTS_SHA256, _result

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "catpain-rn18.onnx"
MANIFEST_NAME = "cloud-model.json"
ARTIFACT_VERSION = "rn18-b48eeee4.cloud-haar-v1.onnx-pillowfloat-v1"
MODEL_VERSION = "rn18-b48eeee4.cloud-catflw-v2.onnx-pillowfloat-v1"
_LOCK = threading.RLock()
_NET: Any = None
_NET_FINGERPRINT: tuple[str, int, int] | None = None
_VERIFIED_FINGERPRINT: tuple[str, int, int, int, int] | None = None


def _models_dir() -> Path:
    return Path(os.environ.get("CATPAIN_MODELS_DIR", PROJECT_ROOT / "models"))


def _check_model() -> tuple[bool, str]:
    global _VERIFIED_FINGERPRINT
    model_path = _models_dir() / MODEL_NAME
    manifest_path = _models_dir() / MANIFEST_NAME
    if not model_path.is_file() or not manifest_path.is_file():
        return False, "云端模型文件尚未准备就绪，仍可完成 FGS 人工评分。"
    model_stat, manifest_stat = model_path.stat(), manifest_path.stat()
    fingerprint = (str(model_path.resolve()), model_stat.st_size, model_stat.st_mtime_ns,
                   manifest_stat.st_size, manifest_stat.st_mtime_ns)
    with _LOCK:
        if fingerprint != _VERIFIED_FINGERPRINT:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("source_weights_sha256") != WEIGHTS_SHA256
                    or manifest.get("source_commit") != UPSTREAM_COMMIT
                    or manifest.get("model_version") != ARTIFACT_VERSION
                    or manifest.get("size") != model_stat.st_size):
                return False, "云端模型版本校验未通过，未生成 AI 判断。"
            with model_path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != manifest.get("sha256"):
                return False, "云端模型文件校验未通过，未生成 AI 判断。"
            _VERIFIED_FINGERPRINT = fingerprint
    return True, "云端模型已通过校验。"


def get_model_status() -> dict[str, Any]:
    try:
        missing = [name for name in ("cv2", "PIL", "numpy") if importlib.util.find_spec(name) is None]
        if missing:
            return {"ready": False, "detail": "云端推理依赖尚未安装。", "version": MODEL_VERSION}
        valid, detail = _check_model()
        if not valid:
            return {"ready": False, "detail": detail, "version": MODEL_VERSION}
        from .localization import get_localization_status

        localization = get_localization_status()
        if not localization["ready"]:
            return {"ready": False, "detail": localization["detail"], "version": MODEL_VERSION}
        return {"ready": True, "detail": "猫检测、48 点定位与分类模型已就绪。", "version": MODEL_VERSION}
    except Exception:
        LOGGER.exception("Cloud model readiness check failed")
        return {"ready": False, "detail": "云端模型暂不可用，可继续人工评分。", "version": MODEL_VERSION}


def preprocess(crop: Any) -> Any:
    """Return RGB [0,1], NCHW float32 with a black 224x224 letterbox.

    Resample float channels, rather than an 8-bit RGB Pillow image, to retain
    the local adapter's round-down quantization after antialiased resizing.
    """
    import numpy as np
    from PIL import Image

    aspect_ratio = crop.width / crop.height
    if crop.width > crop.height:
        target_w, target_h = 224, max(1, int(224 / aspect_ratio))
    else:
        target_h, target_w = 224, max(1, int(224 * aspect_ratio))
    pixels = np.asarray(crop.convert("RGB"), dtype=np.float32) / np.float32(255)
    resized = np.stack([
        np.asarray(Image.fromarray(pixels[:, :, channel]).resize(
            (target_w, target_h), Image.Resampling.BILINEAR), dtype=np.float32)
        for channel in range(3)
    ], axis=2)
    quantized = np.clip(resized * np.float32(255), 0, 255).astype(np.uint8)
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    top, left = (224 - target_h) // 2, (224 - target_w) // 2
    canvas[top:top + target_h, left:left + target_w] = quantized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)


def _get_net() -> Any:
    global _NET, _NET_FINGERPRINT
    import cv2
    import numpy as np

    path = _models_dir() / MODEL_NAME
    stat = path.stat()
    fingerprint = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if _NET is None or fingerprint != _NET_FINGERPRINT:
        cv2.setNumThreads(max(1, min(2, os.cpu_count() or 1)))
        # Byte loading also supports the Windows project's non-ASCII path.
        net = cv2.dnn.readNetFromONNX(np.frombuffer(path.read_bytes(), dtype=np.uint8))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _NET, _NET_FINGERPRINT = net, fingerprint
    return _NET


def predict_tensor(tensor: Any) -> float:
    with _LOCK:
        net = _get_net()
        net.setInput(tensor)
        score = float(net.forward().reshape(-1)[0])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model returned an invalid scalar")
    return score


def _cloud_result(status: str, note: str, **extra: Any) -> dict[str, Any]:
    return _result(status, note, model_version=MODEL_VERSION, **extra)


def analyze_image(image_path: Path, crop_path: Path) -> dict[str, Any]:
    localization = None
    try:
        from PIL import Image, ImageOps
        from .localization import localize_image

        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        localization = localize_image(image)
        if localization["status"] != "ok":
            return _cloud_result("unavailable", localization["note"], localization=localization)
        status = get_model_status()
        if not status["ready"]:
            return _cloud_result("unavailable", status["detail"], localization=localization)
        bbox = tuple(int(v) for v in localization["bbox"])
        crop = image.crop(bbox)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path, format="JPEG", quality=94)
        score = predict_tensor(preprocess(crop))
        return _cloud_result(
            "ok", "模型分值并非疼痛概率；该研究模型的诊断准确性尚未验证。",
            label="pain" if score >= THRESHOLD else "no_pain", raw_score=score, bbox=list(bbox), localization=localization,
        )
    except Exception:
        LOGGER.exception("Cloud pain inference failed")
        return _cloud_result("error", "本次模型分析未完成，没有生成疼痛判断；可保留人工 FGS 评分后重试。", localization=localization)
