"""Pinned YOLOv8n cat pain detector, ONNX/OpenCV CPU inference.

The detector's score is not a calibrated probability of pain. A separate
species/face gate must succeed before this model's output is accepted.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "dd3144e38ac414cc720d34c81f711ad231d67467"
SOURCE_SHA256 = "c1e73b7d141e3017c032096beb7a8491ec295b5bc221b6d5a4d4a508704be36d"
MODEL_VERSION = "yolov8n-catpain-dd3144e3.onnx-v1"
PREPROCESS_VERSION = "rgb-letterbox768-fill114-v1"
INPUT_SIZE = 768
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
_LOCK = threading.RLock()
_NET: Any = None
_KEY: tuple | None = None
_VERIFIED: tuple | None = None


def _directory() -> Path:
    return Path(os.environ.get("CATPAIN_MODELS_DIR", ROOT / "models"))


def get_model_status() -> dict:
    global _VERIFIED
    try:
        model, manifest = _directory() / "yolo-catpain.onnx", _directory() / "yolo-model.json"
        if not model.is_file() or not manifest.is_file():
            return {"ready": False, "detail": "YOLO 模型尚未安装。", "version": MODEL_VERSION}
        stat, meta_stat = model.stat(), manifest.stat()
        key = (str(model.resolve()), stat.st_size, stat.st_mtime_ns, meta_stat.st_mtime_ns)
        with _LOCK:
            if key != _VERIFIED:
                meta = json.loads(manifest.read_text(encoding="utf-8"))
                if (meta.get("source_sha256") != SOURCE_SHA256 or meta.get("source_commit") != SOURCE_COMMIT
                        or meta.get("model_version") != MODEL_VERSION or meta.get("input_size") != INPUT_SIZE
                        or meta.get("size") != stat.st_size or meta.get("names") != {"0": "no_pain", "1": "pain"}):
                    raise ValueError("YOLO manifest mismatch")
                with model.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != meta["sha256"]:
                        raise ValueError("YOLO checksum mismatch")
                _VERIFIED = key
        return {"ready": True, "detail": "YOLO 模型已就绪。", "version": MODEL_VERSION}
    except Exception:
        LOGGER.exception("YOLO readiness failed")
        return {"ready": False, "detail": "YOLO 模型校验失败。", "version": MODEL_VERSION}


def preprocess(image: Any) -> tuple[Any, tuple]:
    import cv2
    import numpy as np

    width, height = image.size
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    # Match Ultralytics LetterBox with auto=False and scaleup=True.
    left = round((INPUT_SIZE - resized_w) / 2 - 0.1)
    top = round((INPUT_SIZE - resized_h) / 2 - 0.1)
    pixels = cv2.resize(np.asarray(image.convert("RGB")), (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = pixels
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
    return tensor, (scale, left, top, width, height)


def predict_tensor(tensor: Any) -> Any:
    global _NET, _KEY
    import cv2
    import numpy as np

    path = _directory() / "yolo-catpain.onnx"
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        if _NET is None or key != _KEY:
            cv2.setNumThreads(2)
            _NET = cv2.dnn.readNetFromONNX(np.frombuffer(path.read_bytes(), dtype=np.uint8))
            _NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            _NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            _KEY = key
        _NET.setInput(tensor)
        return _NET.forward().copy()


def postprocess(output: Any, transform: tuple) -> list[dict]:
    import cv2
    import numpy as np

    array = np.asarray(output)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[1] != 6 or not np.isfinite(array).all():
        raise ValueError("Unexpected YOLO output")
    rows = array[0].T
    # Two class scores, no separate objectness channel in YOLOv8.
    classes = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), classes + 4]
    mask = (scores >= CONFIDENCE_THRESHOLD) & (scores <= 1) & (rows[:, 2] > 0) & (rows[:, 3] > 0)
    rows, classes, scores = rows[mask], classes[mask], scores[mask]
    scale, left, top, width, height = transform
    candidates = []
    for row, category, score in zip(rows, classes, scores):
        cx, cy, bw, bh = row[:4]
        x1, x2 = np.clip([(cx - bw / 2 - left) / scale, (cx + bw / 2 - left) / scale], 0, width)
        y1, y2 = np.clip([(cy - bh / 2 - top) / scale, (cy + bh / 2 - top) / scale], 0, height)
        if x2 > x1 and y2 > y1:
            candidates.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                               "label": "pain" if category == 1 else "no_pain", "confidence": float(score)})
    # Suppress overlapping alternative class assignments to the same face.
    boxes = [[d["bbox"][0], d["bbox"][1], d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]] for d in candidates]
    indices = cv2.dnn.NMSBoxes(boxes, [d["confidence"] for d in candidates], CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
    return sorted([candidates[int(i)] for i in np.asarray(indices).reshape(-1)], key=lambda d: d["confidence"], reverse=True)


def _result(status: str, note: str, **extra: Any) -> dict:
    return {"status": status, "label": None, "detection_confidence": None, "bbox": None, "detections": [],
            "model_version": MODEL_VERSION, "preprocess_version": PREPROCESS_VERSION,
            "confidence_threshold": CONFIDENCE_THRESHOLD, "note": note, **extra}


def analyze(image: Any, localization: dict) -> dict:
    try:
        if localization.get("status") != "ok":
            return _result("unavailable", "未通过单猫猫脸定位，本次 YOLO 不作疼痛判断。")
        status = get_model_status()
        if not status["ready"]:
            return _result("unavailable", status["detail"])
        tensor, transform = preprocess(image)
        detections = postprocess(predict_tensor(tensor), transform)
        if not detections:
            return _result("unavailable", "YOLO 未检测到达到阈值的猫脸，未生成疼痛判断。")
        if len(detections) > 1:
            return _result("unavailable", "YOLO 检测到多个猫脸候选，请裁剪为单只猫后重试。", detections=detections)
        detection = detections[0]
        # Require the proposed pain face to overlap the independent face gate.
        a, b = detection["bbox"], localization["bbox"]
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        if smaller <= 0 or intersection / smaller < 0.35:
            return _result("unavailable", "YOLO 与猫脸定位区域不一致，本次不作判断。", detections=detections)
        return _result("ok", "实验性猫脸二分类；检测分数不是疼痛概率。", label=detection["label"],
                       detection_confidence=detection["confidence"], bbox=detection["bbox"], detections=detections)
    except Exception:
        LOGGER.exception("YOLO inference failed")
        return _result("error", "YOLO 运行失败，未生成疼痛判断。")
