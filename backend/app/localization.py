"""Local COCO cat gate -> CatFLW face box -> 48 predicted landmarks.

No photograph leaves the site's runtime. Coordinates refer to the sanitized,
EXIF-oriented image, not its browser display size. CatFLW is a regressor: it
does not supply visibility or per-landmark confidence and cannot score FGS.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import threading
import zipfile
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_VERSION = "efficientdet-lite0.catflw-8404a7a9.roi-v1"
CAT_THRESHOLD = 0.35  # Object detector score, never a pain probability.
FILES = {
    "coco-efficientdet-lite0.tflite": "2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b",
    "cat_face_localizer.tflite": "a9621320ba8ca39b706c82649737c3cb72ef10ba37200ee91e84346600bee0c7",
    "cat_face_landmarks_full.tflite": "1a8b09c4b30a8211de89979244ab90e595935c27fdba1e64b433990f8a3a5b66",
}
_LOCK = threading.RLock()
_INTERPRETERS: dict[tuple, Any] = {}
_VERIFIED: set[tuple] = set()


def _models_dir() -> Path:
    return Path(os.environ.get("CATPAIN_MODELS_DIR", PROJECT_ROOT / "models"))


def _fingerprint(path: Path) -> tuple:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


def get_localization_status() -> dict:
    try:
        if importlib.util.find_spec("ai_edge_litert") is None:
            return {"ready": False, "detail": "猫脸定位依赖尚未安装。", "version": MODEL_VERSION}
        with _LOCK:
            for name, expected in FILES.items():
                path = _models_dir() / name
                if not path.is_file():
                    return {"ready": False, "detail": "猫脸定位模型尚未安装。", "version": MODEL_VERSION}
                key = _fingerprint(path)
                if key not in _VERIFIED:
                    with path.open("rb") as source:
                        digest = hashlib.file_digest(source, "sha256").hexdigest()
                    if digest != expected:
                        return {"ready": False, "detail": "猫脸定位模型校验失败。", "version": MODEL_VERSION}
                    _VERIFIED.add(key)
        return {"ready": True, "detail": "猫检测与 48 点定位模型已就绪。", "version": MODEL_VERSION}
    except Exception:
        LOGGER.exception("Localization readiness check failed")
        return {"ready": False, "detail": "猫脸定位服务暂不可用。", "version": MODEL_VERSION}


def _run(name: str, tensor: Any) -> list[Any]:
    from ai_edge_litert.interpreter import Interpreter

    path = _models_dir() / name
    key = _fingerprint(path)
    with _LOCK:
        interpreter = _INTERPRETERS.get(key)
        if interpreter is None:
            # Native filename loaders on Windows can reject the Chinese
            # Desktop path; Python's bytes loader preserves the exact file.
            interpreter = Interpreter(model_content=path.read_bytes(), num_threads=2)
            interpreter.allocate_tensors()
            _INTERPRETERS[key] = interpreter
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], tensor)
        interpreter.invoke()
        # get_tensor copies values before another call can reuse the arena.
        return [interpreter.get_tensor(item["index"]) for item in interpreter.get_output_details()]


def _iou(a: tuple, b: tuple) -> float:
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0.0


def _detect_cats(image: Any) -> list[tuple[float, tuple[float, float, float, float]]]:
    import numpy as np
    from PIL import Image

    name = "coco-efficientdet-lite0.tflite"
    # Read the embedded labels rather than assuming a COCO category offset.
    with zipfile.ZipFile(_models_dir() / name) as archive:
        cat_id = archive.read("labelmap.txt").decode().splitlines().index("cat")
    tensor = np.asarray(image.resize((320, 320), Image.Resampling.BILINEAR), dtype=np.uint8)[None]
    boxes, classes, scores, count = _run(name, tensor)
    candidates = []
    for box, category, score in zip(boxes[0][:int(count[0])], classes[0], scores[0]):
        if int(category) != cat_id or float(score) < CAT_THRESHOLD:
            continue
        if not np.isfinite(box).all() or not np.isfinite(score):
            continue
        y1, x1, y2, x2 = np.clip(box, 0, 1)
        mapped = (float(x1 * image.width), float(y1 * image.height), float(x2 * image.width), float(y2 * image.height))
        if mapped[2] > mapped[0] and mapped[3] > mapped[1]:
            candidates.append((float(score), mapped))
    # Built-in detection NMS is followed by cross-candidate duplicate removal.
    selected = []
    for score, box in sorted(candidates, reverse=True):
        if all(_iou(box, old) < 0.45 for _, old in selected):
            selected.append((score, box))
    return selected


def _expanded_roi(box: tuple, size: tuple[int, int], margin: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    left = max(0, int(x1 - width * margin))
    top = max(0, int(y1 - height * margin))
    right = min(size[0], int(x2 + width * margin))
    bottom = min(size[1], int(y2 + height * margin))
    return left, top, right, bottom


def _face_box(image: Any, region: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    import numpy as np
    from PIL import Image

    roi = image.crop(region)
    scale = min(224 / roi.width, 224 / roi.height)
    width, height = round(roi.width * scale), round(roi.height * scale)
    left, top = (224 - width) // 2, (224 - height) // 2
    canvas = Image.new("RGB", (224, 224), "black")
    canvas.paste(roi.resize((width, height), Image.Resampling.BILINEAR), (left, top))
    tensor = np.asarray(canvas, dtype=np.float32)[None] / np.float32(255)
    box = _run("cat_face_localizer.tflite", tensor)[0].reshape(4)
    if not np.isfinite(box).all():
        raise ValueError("Non-finite face box")
    x1, y1, x2, y2 = np.clip(box, 0, 1) * 224
    # The exact resize/padding above is shared by the inverse transform.
    scale_x, scale_y = width / roi.width, height / roi.height
    x1, x2 = sorted(np.clip([(x1 - left) / scale_x, (x2 - left) / scale_x], 0, roi.width))
    y1, y2 = sorted(np.clip([(y1 - top) / scale_y, (y2 - top) / scale_y], 0, roi.height))
    return float(region[0] + x1), float(region[1] + y1), float(region[0] + x2), float(region[1] + y2)


def _result(status: str, reason: str, note: str, **kwargs: Any) -> dict:
    return {"status": status, "reason": reason, "note": note, "model_version": MODEL_VERSION,
            "coordinate_space": "image_pixels", "bbox": None, "cat_bbox": None,
            "detector_confidence": None, "points": [], **kwargs}


def localize_image(image: Any) -> dict:
    """Return model geometry or abstain; never substitute a template face."""
    try:
        import numpy as np
        from PIL import Image

        status = get_localization_status()
        if not status["ready"]:
            return _result("unavailable", "model_unavailable", status["detail"])
        image = image.convert("RGB")
        cats = _detect_cats(image)
        if not cats:
            return _result("unavailable", "no_cat", "未检测到猫，未生成猫脸点位。请上传只包含一只猫的照片。")
        if len(cats) > 1:
            return _result("unavailable", "multiple_cats", "检测到多只猫，请裁剪为一只猫后重新上传。")
        confidence, cat_box = cats[0]
        extra = {"cat_bbox": list(cat_box), "detector_confidence": confidence}
        region = _expanded_roi(cat_box, image.size, 0.10)
        if min(region[2] - region[0], region[3] - region[1]) < 80:
            return _result("unavailable", "cat_too_small", "猫在画面中太小，请靠近拍摄。", **extra)
        box = _face_box(image, region)
        width, height = box[2] - box[0], box[3] - box[1]
        if min(width, height) < 80 or not 0.35 < width / height < 2.8:
            return _result("unavailable", "face_not_localized", "已检测到猫，但未能定位适合分析的猫脸。请拍到完整双耳、双眼和口鼻。", **extra)
        roi = _expanded_roi(box, image.size, 0.10)
        crop = image.crop(roi)
        tensor = np.asarray(crop.resize((384, 384), Image.Resampling.BILINEAR), dtype=np.float32)[None] / np.float32(255)
        xy = _run("cat_face_landmarks_full.tflite", tensor)[0].reshape(48, 2)
        if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > 1):
            return _result("unavailable", "invalid_landmarks", "猫脸点位预测异常，本次不显示点位。", **extra)
        xy = xy * [crop.width, crop.height] + [roi[0], roi[1]]
        # Reject collapsed outputs. This is a geometric sanity check, not an
        # accuracy estimate, visibility detector or calibrated confidence.
        eye_distance = float(np.linalg.norm(xy[4] - xy[8]))
        if eye_distance < 12 or eye_distance < crop.width * 0.12:
            return _result("unavailable", "invalid_landmarks", "猫脸点位未通过基本几何检查，请换一张正面照片。", **extra)
        return _result("ok", "located", "48 个模型预测点；遮挡点仍可能被估计，逐点可见性与误差尚未判定。",
                       bbox=list(roi), points=[{"id": i + 1, "x": round(float(x), 3), "y": round(float(y), 3), "confidence": None} for i, (x, y) in enumerate(xy)], **extra)
    except Exception:
        LOGGER.exception("Cat face localization failed")
        return _result("error", "runtime_error", "猫脸定位运行失败，未生成点位。请重试。")
