"""Local port of KittyCheck's five FGS scorers (research prototype).

The upstream adapter is pinned below. This keeps its region geometry, RGB
normalization, eye alignment and head-image contract; it does not use CatFLW
coordinates as substitute inputs. Missing/invalid regions are NA, never zero.
No visibility head or calibrated per-item confidence is present in the models.
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
ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "818d4f5943f707e9fd3ec4ec420461c54e882832"
MODEL_VERSION = "kittycheck-818d4f59.fgs-five-v1"
PREPROCESS_VERSION = "kittycheck-dart-roi.python-linear-v1"
KEYS = ("ears", "eyes", "muzzle", "whiskers", "head")
FILES = {
    "ears_scorer.tflite": (16689496, "e2edca443ed5c928dbf6b1328472b98aa88686a9d82f3e7a0d4af7355b918825"),
    "eyes_scorer.tflite": (16689496, "ae8932ca4404db5885d34b4cca5a70e58421e9127f844794aea53f46bc34a56f"),
    "face_detector.tflite": (16689756, "95f3ef0042a5f3c5d63c7bab82dbfe2072bd9a338ea8b5db34a0a1fe006ea96a"),
    "head_scorer.tflite": (16689496, "4b02ed832f7beb7e10d31b85abc14ccdbdd061442a80b7f9019cc29bf0a06f05"),
    "muzzle_scorer.tflite": (16689496, "88ad0785bb46e840b7458c0cf4a70ee46562fdb8937c474eb8b9d78731ef7355"),
    "region_detector.tflite": (16691320, "39dd520796984887326ac692b2b2fd76a82a3b00d6e59c9e6c63a35fa9461d13"),
    "whiskers_scorer.tflite": (16689500, "a82f4980f66f8d840e43f180fd6eed69e67b306ef72b7f0f6c8452193f110ef1"),
}
_LOCK = threading.RLock()
_VERIFIED: set[tuple] = set()
_MODELS: dict[tuple, Any] = {}


def _directory() -> Path:
    return Path(os.environ.get("CATPAIN_MODELS_DIR", ROOT / "models")) / "kittycheck"


def _file_key(path: Path) -> tuple:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


def _verified(name: str) -> bool:
    path = _directory() / name
    if not path.is_file() or path.stat().st_size != FILES[name][0]:
        return False
    key = _file_key(path)
    with _LOCK:
        if key not in _VERIFIED:
            with path.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != FILES[name][1]:
                    return False
            _VERIFIED.add(key)
    return True


def get_model_status() -> dict:
    try:
        dependency = importlib.util.find_spec("ai_edge_litert") is not None
        present = {name: _verified(name) for name in FILES}
        ready = dependency and all(present.values())
        return {"ready": ready, "version": MODEL_VERSION,
                "detail": "自动 FGS 五项模型已就绪。" if ready else "自动 FGS 模型或依赖尚未准备就绪。",
                "files": present}
    except Exception:
        LOGGER.exception("KittyCheck readiness lookup failed")
        return {"ready": False, "version": MODEL_VERSION, "detail": "自动 FGS 模型暂不可用。"}


def _run(name: str, tensor: Any) -> Any:
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    if not _verified(name):
        raise FileNotFoundError("KittyCheck model missing or failed hash check: " + name)
    path = _directory() / name
    key = _file_key(path)
    with _LOCK:
        model = _MODELS.get(key)
        if model is None:
            model = Interpreter(model_content=path.read_bytes(), num_threads=2)
            model.allocate_tensors()
            _MODELS[key] = model
        model.set_tensor(model.get_input_details()[0]["index"], np.ascontiguousarray(tensor, dtype=np.float32))
        model.invoke()
        return model.get_tensor(model.get_output_details()[0]["index"]).reshape(-1)


def _round(value: float) -> int:
    """Dart round: nearest integer, ties away from zero."""
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _resize(pixels: Any, width: int, height: int) -> Any:
    """Dart image.copyResize linear coordinates, with uint8 truncation.

    Pillow's half-pixel centers produce different samples, particularly on
    narrow eye crops. The upstream uses x*src_width/dst_width coordinates.
    """
    import numpy as np

    if width <= 0 or height <= 0:
        raise ValueError("Invalid resize dimensions")
    source = np.asarray(pixels, dtype=np.float64)
    source_h, source_w = source.shape[:2]
    if (source_w, source_h) == (width, height):
        return source.astype(np.uint8)
    x = np.minimum(np.arange(width, dtype=np.float64) * source_w / width, source_w - 1)
    y = np.minimum(np.arange(height, dtype=np.float64) * source_h / height, source_h - 1)
    ix, iy = x.astype(int), y.astype(int)
    nx, ny = np.minimum(ix + 1, source_w - 1), np.minimum(iy + 1, source_h - 1)
    kx, ky = (x - ix)[None, :, None], (y - iy)[:, None, None]
    p00, p10 = source[iy[:, None], ix], source[iy[:, None], nx]
    p01, p11 = source[ny[:, None], ix], source[ny[:, None], nx]
    result = p00 + kx * (p10 - p00 + ky * (p00 + p11 - p01 - p10)) + ky * (p01 - p00)
    return np.clip(result, 0, 255).astype(np.uint8)


def _letterbox(pixels: Any, width: int = 224, height: int = 224) -> Any:
    import numpy as np

    source_h, source_w = pixels.shape[:2]
    scale = min(width / source_w, height / source_h)
    out_w, out_h = max(1, int(source_w * scale)), max(1, int(source_h * scale))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left, top = (width - out_w) // 2, (height - out_h) // 2
    canvas[top:top + out_h, left:left + out_w] = _resize(pixels, out_w, out_h)
    return canvas


def _tensor(pixels: Any, normalize: bool) -> Any:
    import numpy as np

    values = np.asarray(pixels, dtype=np.float32) / np.float32(255)
    if normalize:
        values = (values - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(values[None], dtype=np.float32)


def _detect_face(pixels: Any) -> tuple[int, int, int, int]:
    import numpy as np

    height, width = pixels.shape[:2]
    box = _run("face_detector.tflite", _tensor(_letterbox(pixels), False))
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Invalid face regression output")
    # Upstream targets are interpreted as normalized ORIGINAL-image x/y,
    # even though its input is letterboxed. Preserve that model contract.
    x1, y1, x2, y2 = [_round(float(v) * dim) for v, dim in zip(box, (width, height, width, height))]
    box = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if box[2] - box[0] < 40 or box[3] - box[1] < 40:
        raise ValueError("Predicted face is invalid or too small")
    return box


def _expanded_box(box: tuple, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx, dy, extra_top = _round(bw * 0.25), _round(bh * 0.25), _round(bh * 0.30)
    return max(0, x1 - dx), max(0, y1 - dy - extra_top), min(width, x2 + dx), min(height, y2 + dy)


def _consistent_faces(box: tuple, reference: Any, width: int, height: int) -> bool:
    """Require the two independent localizers to refer to the same face.

    Intersection over the smaller box permits the different margins used by
    the models. The .35 gate is an adapter sanity check, not calibrated
    confidence or an estimate of landmark/pain accuracy.
    """
    try:
        if reference is None or len(reference) != 4:
            return False
        ref = tuple(float(value) for value in reference)
        if not all(math.isfinite(value) for value in ref):
            return False
        if not (0 <= ref[0] < ref[2] <= width and 0 <= ref[1] < ref[3] <= height):
            return False
        overlap = max(0.0, min(box[2], ref[2]) - max(box[0], ref[0])) * max(
            0.0, min(box[3], ref[3]) - max(box[1], ref[1]))
        smaller = min((box[2] - box[0]) * (box[3] - box[1]),
                      (ref[2] - ref[0]) * (ref[3] - ref[1]))
        return smaller > 0 and overlap / smaller >= 0.35
    except (TypeError, ValueError, OverflowError):
        return False


def _detect_centers(face: Any) -> dict[str, list[float]]:
    import numpy as np

    values = _run("region_detector.tflite", _tensor(face, False))
    if values.shape != (10,) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Invalid region-center output")
    names = ("left_eye", "right_eye", "nose", "left_ear", "right_ear")
    return {name: [float(values[2 * i] * 224), float(values[2 * i + 1] * 224)] for i, name in enumerate(names)}


def _rotation_angle(centers: dict) -> float:
    left, right = centers["left_eye"], centers["right_eye"]
    dx, dy = right[0] - left[0], right[1] - left[1]
    if math.hypot(dx, dy) < 5:
        raise ValueError("Region centers collapse to one position")
    if abs(dx) > 1e-6 and abs(dy) / (abs(dx) + 1e-6) < 0.02:
        return 0.0
    angle = -math.atan2(dy, dx) * 180 / math.pi
    return 0.0 if abs(angle) < 5 else max(-30.0, min(30.0, angle))


def _rotate(face: Any, centers: dict, angle: float) -> tuple[Any, dict]:
    import numpy as np

    if abs(angle) < 0.1:
        return face, centers
    cx = (centers["left_eye"][0] + centers["right_eye"][0]) / 2
    cy = (centers["left_eye"][1] + centers["right_eye"][1]) / 2
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    height, width = face.shape[:2]
    new_w = _round(height * abs(sine) + width * abs(cosine))
    new_h = _round(height * abs(cosine) + width * abs(sine))
    yy, xx = np.mgrid[:new_h, :new_w].astype(np.float64)
    dx, dy = xx - new_w / 2, yy - new_h / 2
    src_x, src_y = cosine * dx + sine * dy + cx, -sine * dx + cosine * dy + cy
    ix, iy = np.floor(src_x).astype(int), np.floor(src_y).astype(int)
    valid = (ix >= 0) & (ix < width - 1) & (iy >= 0) & (iy < height - 1)
    canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    vx, vy = ix[valid], iy[valid]
    fx, fy = (src_x[valid] - vx)[:, None], (src_y[valid] - vy)[:, None]
    values = face.astype(np.float64)
    top = values[vy, vx] * (1 - fx) + values[vy, vx + 1] * fx
    bottom = values[vy + 1, vx] * (1 - fx) + values[vy + 1, vx + 1] * fx
    canvas[valid] = np.clip(np.floor(top * (1 - fy) + bottom * fy + 0.5), 0, 255).astype(np.uint8)
    mapped = {name: [cosine * (x - cx) - sine * (y - cy) + new_w / 2,
                     sine * (x - cx) + cosine * (y - cy) + new_h / 2]
              for name, (x, y) in centers.items()}
    return canvas, mapped


def _crop_regions(face: Any, centers: dict) -> dict[str, Any]:
    height, width = face.shape[:2]
    result = {}

    def crop(key: str, box: tuple, mode: str) -> None:
        x1, y1, x2, y2 = box
        left, top = max(0, _round(x1)), max(0, _round(y1))
        right, bottom = min(width, _round(x2)), min(height, _round(y2))
        if right - left < 4 or bottom - top < 4:
            return
        region = face[top:bottom, left:right]
        result[key] = _letterbox(region) if mode == "pad" else _resize(region, 224, 150)

    le, re = centers["left_ear"], centers["right_ear"]
    left, right, nose = centers["left_eye"], centers["right_eye"], centers["nose"]
    crop("ears", (min(le[0], re[0]) - 45, min(le[1], re[1]) - 70,
                  max(le[0], re[0]) + 45, max(le[1], re[1]) + 40), "stretch")
    crop("eyes", (min(left[0], right[0]) - 35, min(left[1], right[1]) - 25,
                  max(left[0], right[0]) + 35, max(left[1], right[1]) + 25), "stretch")
    eye_distance = math.hypot(right[0] - left[0], right[1] - left[1])
    muzzle_w = max(100.0, eye_distance * 0.6)
    nose_eye = left[1] - nose[1]
    crop("muzzle", (nose[0] - muzzle_w / 2, nose[1] - max(35, nose_eye * 0.5),
                    nose[0] + muzzle_w / 2, nose[1] + max(45, nose_eye * 0.5)), "pad")
    ear_distance = math.hypot(re[0] - le[0], re[1] - le[1])
    whiskers_w = max(160.0, ear_distance * 2)
    nose_eye = (left[1] + right[1]) / 2 - nose[1]
    crop("whiskers", (nose[0] - whiskers_w / 2, nose[1] - max(25, nose_eye * 0.3),
                      nose[0] + whiskers_w / 2, nose[1] + max(70, nose_eye * 1.2)), "pad")
    return result


def _head_image(pixels: Any, box: tuple) -> Any:
    """Head scorer sees full context with the upstream red face rectangle."""
    canvas = pixels.copy()
    x1, y1, x2, y2 = box
    height, width = canvas.shape[:2]
    # Dart drawRect(thickness:3) uses centered horizontal strokes and inset
    # vertical strokes. Clip each stroke rather than changing the box itself.
    for left, top, right, bottom in ((x1, y1 - 1, x2 + 1, y1 + 2),
                                     (x1, y2 - 1, x2 + 1, y2 + 2),
                                     (x1, y1 + 2, x1 + 3, y2 - 1),
                                     (x2 - 2, y1 + 2, x2 + 1, y2 - 1)):
        canvas[max(0, top):min(height, bottom), max(0, left):min(width, right)] = (255, 0, 0)
    return _letterbox(canvas)


def _result(status: str, note: str, scores: dict | None = None, raw: dict | None = None) -> dict:
    scores = scores or {key: None for key in KEYS}
    observed = [v for v in scores.values() if v is not None]
    total, count = sum(observed), len(observed)
    complete = count == 5
    return {"status": status, "model_version": MODEL_VERSION, "preprocess_version": PREPROCESS_VERSION,
            "note": note, "scores": scores, "raw_outputs": raw or {}, "regions": {},
            "total": total, "max": count * 2, "assessable_count": count, "complete": complete,
            "flag": ("positive" if total >= 4 else "negative") if complete else "incomplete"}


def analyze(image: Any, localization: dict) -> dict:
    """Predict five AU scores only after the independent cat gate succeeds."""
    if not localization or localization.get("status") != "ok":
        return _result("unavailable", "未获得可用的单猫定位，本次未生成自动 FGS。")
    try:
        import numpy as np

        if not importlib.util.find_spec("ai_edge_litert"):
            return _result("unavailable", "自动 FGS 推理依赖尚未安装。")
        if not _verified("face_detector.tflite") or not _verified("region_detector.tflite"):
            return _result("unavailable", "自动 FGS 定位模型缺失或校验失败，未生成评分。")
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = pixels.shape[:2]
        face_box = _detect_face(pixels)
        if not _consistent_faces(face_box, localization.get("bbox"), width, height):
            return _result("unavailable", "两个模型的猫脸定位不一致或定位框无效，本次自动 FGS 五项均保留 NA。")
        expanded = _expanded_box(face_box, width, height)
        x1, y1, x2, y2 = expanded
        face = _resize(pixels[y1:y2, x1:x2], 224, 224)
        centers = _detect_centers(face)
        rotated, centers = _rotate(face, centers, _rotation_angle(centers))
        regions = _crop_regions(rotated, centers)
        regions["head"] = _head_image(pixels, face_box)
        scores, raw = {key: None for key in KEYS}, {}
        for key in KEYS:
            if key not in regions or not _verified(key + "_scorer.tflite"):
                continue
            try:
                output = _run(key + "_scorer.tflite", _tensor(_resize(regions[key], 224, 224), True))
                if output.shape != (3,) or not np.isfinite(output).all():
                    continue
                scores[key] = int(np.argmax(output))
                raw[key] = [float(v) for v in output]
            except Exception:
                LOGGER.exception("KittyCheck scorer failed for %s", key)
        count = sum(value is not None for value in scores.values())
        status = "ok" if count == 5 else "partial" if count else "unavailable"
        return _result(status, "研究模型预测；不具备逐项可见性判断，遮挡和异常姿态可能导致误分。缺失或失败项目记为 NA。", scores, raw)
    except ValueError:
        LOGGER.info("KittyCheck rejected invalid face/region geometry", exc_info=True)
        return _result("unavailable", "自动 FGS 未获得有效的猫脸或区域定位，五项均保留 NA。")
    except Exception:
        LOGGER.exception("KittyCheck inference failed")
        return _result("error", "自动 FGS 分析未能完成，未生成评分。")
