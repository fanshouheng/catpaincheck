"""Local research-model inference; never sends photographs to another service.

The upstream implementation sends photos to a landmarks API and uses its bbox.
This adapter uses a local COCO cat detector and CatFLW face localization. The
resulting crop distribution has not been validated against the training set.
The classifier scalar is NOT a calibrated probability of pain.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "b48eeee42596e26537fe4cc15bdf7a7954d2dac6"
WEIGHTS_NAME = "cat-pr-rn18-adamW-FL.pth"
WEIGHTS_SHA256 = "4ace5922b92e8c89c14b4d275cd41d4b1b9884de14bf8c14bac41e32c8c73581"
WEIGHTS_SIZE = 112_376_644
THRESHOLD = 0.4
MODEL_VERSION = "rn18-b48eeee4.local-catflw-v2.letterbox224-rgb01"

_LOCK = threading.RLock()
_MODEL: Any = None
_MODEL_FINGERPRINT: tuple[str, int, int] | None = None
_VERIFIED_FINGERPRINT: tuple[str, int, int] | None = None


def _models_dir() -> Path:
    return Path(os.environ.get("CATPAIN_MODELS_DIR", PROJECT_ROOT / "models"))


def _weights_path() -> Path:
    return _models_dir() / WEIGHTS_NAME


def _check_weights() -> tuple[bool, str]:
    global _VERIFIED_FINGERPRINT
    path = _weights_path()
    if not path.is_file():
        return False, "尚未安装疼痛模型。请运行项目中的模型安装脚本。"
    stat = path.stat()
    if stat.st_size != WEIGHTS_SIZE:
        return False, "模型文件不完整或版本不符，请重新运行模型安装脚本。"
    fingerprint = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        if fingerprint != _VERIFIED_FINGERPRINT:
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != WEIGHTS_SHA256:
                return False, "模型文件校验未通过，初筛已停用，请重新安装模型。"
            _VERIFIED_FINGERPRINT = fingerprint
    return True, "模型文件已通过校验。"


def get_model_status() -> dict[str, Any]:
    """Quick readiness check. Loading the neural network is deferred to use."""
    try:
        missing = [
            name
            for name in ("torch", "torchvision", "cv2", "PIL", "numpy")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            return {
                "ready": False,
                "detail": "本地初筛依赖尚未安装：" + "、".join(missing),
                "version": MODEL_VERSION,
            }
        valid, detail = _check_weights()
        if not valid:
            return {"ready": False, "detail": detail, "version": MODEL_VERSION}
        from .localization import get_localization_status

        localization = get_localization_status()
        if not localization["ready"]:
            return {"ready": False, "detail": localization["detail"], "version": MODEL_VERSION}
        return {
            "ready": True,
            "detail": "本地模型已就绪，使用猫检测与 48 点模型定位单张猫脸。",
            "version": MODEL_VERSION,
        }
    except Exception:
        LOGGER.exception("Model readiness check failed")
        return {
            "ready": False,
            "detail": "本地模型暂不可用，请查看服务器日志。",
            "version": MODEL_VERSION,
        }


def _get_model() -> Any:
    global _MODEL, _MODEL_FINGERPRINT
    import torch
    from torch import nn
    from torchvision.models import resnet18

    path = _weights_path()
    stat = path.stat()
    fingerprint = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if _MODEL is not None and fingerprint == _MODEL_FINGERPRINT:
        return _MODEL
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )
    # Restricted loading: do not execute arbitrary pickle objects from a file.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.cpu().eval()
    _MODEL = model
    _MODEL_FINGERPRINT = fingerprint
    return model


def _result(status: str, note: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "label": None,
        "raw_score": None,
        "threshold": THRESHOLD,
        "model_version": MODEL_VERSION,
        "note": note,
        "bbox": None,
        **extra,
    }


def analyze_image(image_path: Path, crop_path: Path) -> dict[str, Any]:
    """Classify one detectable cat face, or explicitly abstain.

    Preserve the upstream final classifier input contract: RGB, aspect-ratio
    preserving black letterbox at 224x224, tensor values in [0,1]. The upstream
    code normalizes and then denormalizes before classification; applying
    ImageNet normalization here would silently change that contract.
    """
    localization = None
    try:
        from PIL import Image, ImageOps
        from .localization import localize_image

        with Image.open(image_path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        localization = localize_image(image)
        if localization["status"] != "ok":
            return _result("unavailable", localization["note"], localization=localization)
        status = get_model_status()
        if not status["ready"]:
            return _result("unavailable", status["detail"], localization=localization)
        import torch
        from torchvision.transforms.functional import resize, to_tensor

        bbox = tuple(int(v) for v in localization["bbox"])
        crop = image.crop(bbox)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path, format="JPEG", quality=94)

        # Match upstream torchvision resize / PIL quantization / black padding.
        aspect_ratio = crop.width / crop.height
        if crop.width > crop.height:
            target_w, target_h = 224, max(1, int(224 / aspect_ratio))
        else:
            target_h, target_w = 224, max(1, int(224 * aspect_ratio))
        resized = resize(to_tensor(crop), [target_h, target_w], antialias=True)
        from torchvision.transforms.functional import to_pil_image

        letterbox = Image.new("RGB", (224, 224), "black")
        letterbox.paste(to_pil_image(resized), ((224 - target_w) // 2, (224 - target_h) // 2))
        tensor = to_tensor(letterbox).unsqueeze(0).cpu()

        # Serialize lazy initialization and model calls for predictable CPU use.
        with _LOCK, torch.inference_mode():
            model = _get_model()
            score = float(model(tensor).reshape(-1)[0].item())
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Model returned an invalid scalar")
        return _result(
            "ok",
            "仅反映本次照片的模型判断；模型分值并非疼痛概率。该研究模型在本网站中的准确性尚未验证。",
            label="pain" if score >= THRESHOLD else "no_pain",
            raw_score=score,
            bbox=list(bbox),
            localization=localization,
        )
    except Exception:
        LOGGER.exception("Local pain inference failed")
        return _result(
            "error",
            "本次模型分析未完成，请重试或先查看 FGS 人工评分；没有生成疼痛预测。",
            localization=localization,
        )
