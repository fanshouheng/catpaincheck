"""Download the pinned YOLO research weight and safely export a CPU ONNX model.

Export-only dependencies: torch, ultralytics==8.3.221, onnx. The website only
needs OpenCV; it never unpickles the upstream checkpoint during inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "work" / "yolo-config"))
from app import yolo_inference as runtime


def load_source(path: Path):
    import torch

    # Only allow known nn.Module types, never arbitrary checkpoint functions.
    allowed_names = {
        "ultralytics.nn.tasks.DetectionModel", "ultralytics.nn.modules.block.SPPF",
        "ultralytics.nn.modules.conv.Conv", "ultralytics.nn.modules.conv.Concat",
        "ultralytics.nn.modules.block.C2f", "ultralytics.nn.modules.head.Detect",
        "ultralytics.nn.modules.block.DFL", "ultralytics.nn.modules.block.Bottleneck",
        "torch.nn.modules.conv.Conv2d", "torch.nn.modules.container.ModuleList",
        "torch.nn.modules.batchnorm.BatchNorm2d", "torch.nn.modules.activation.SiLU",
        "torch.nn.modules.upsampling.Upsample", "torch.nn.modules.container.Sequential",
        "torch.nn.modules.pooling.MaxPool2d",
    }
    globals_found = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if not globals_found <= allowed_names:
        raise ValueError("Unexpected checkpoint global; export stopped")
    allowed = []
    for name in sorted(globals_found):
        module, attribute = name.rsplit(".", 1)
        value = getattr(importlib.import_module(module), attribute)
        if not isinstance(value, type) or not issubclass(value, torch.nn.Module):
            raise ValueError("Checkpoint global is not a known model class")
        allowed.append(value)
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        result = runtime.get_model_status()
        print(json.dumps(result, ensure_ascii=True))
        raise SystemExit(0 if result["ready"] else 1)
    import numpy as np
    import onnx
    import torch

    directory = runtime._directory()
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "yolo-catpain.pt"
    if not source.is_file():
        temporary = source.with_suffix(".download")
        try:
            url = f"https://raw.githubusercontent.com/sophiapyx/IAT360-Computer-Vision-Project/{runtime.SOURCE_COMMIT}/best.pt"
            urllib.request.urlretrieve(url, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != runtime.SOURCE_SHA256:
                raise ValueError("Upstream YOLO checksum mismatch")
            temporary.replace(source)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != runtime.SOURCE_SHA256:
        raise ValueError("Upstream YOLO checksum mismatch")
    checkpoint = load_source(source)
    model = checkpoint["model"].float().eval()
    if model.names != {0: "no_pain", 1: "pain"} or checkpoint["train_args"]["imgsz"] != runtime.INPUT_SIZE:
        raise ValueError("Unexpected class mapping or image size")
    torch.set_num_threads(2)
    model.fuse(verbose=False)
    for module in model.modules():
        if module.__class__.__name__ == "Detect":
            module.export, module.format, module.dynamic = True, "onnx", False
    example = torch.zeros((1, 3, runtime.INPUT_SIZE, runtime.INPUT_SIZE), dtype=torch.float32)
    with torch.inference_mode():
        model(example)
    target = directory / "yolo-catpain.onnx"
    pending = directory / "yolo-catpain.pending.onnx"
    try:
        torch.onnx.export(model, example, str(pending), export_params=True, opset_version=12,
                          do_constant_folding=True, input_names=["images"], output_names=["detections"], dynamo=False)
        onnx.checker.check_model(str(pending))
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)
    manifest = {"schema_version": 1, "model_version": runtime.MODEL_VERSION,
                "preprocess_version": runtime.PREPROCESS_VERSION, "input_size": runtime.INPUT_SIZE,
                "source_commit": runtime.SOURCE_COMMIT, "source_sha256": runtime.SOURCE_SHA256,
                "source_repository": "https://github.com/sophiapyx/IAT360-Computer-Vision-Project",
                "checkpoint_ultralytics_version": checkpoint.get("version"),
                "checkpoint_license_metadata": checkpoint.get("license"),
                "repository_license": "No separate LICENSE file found; metadata is not a warranty of all rights",
                "size": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "names": {"0": "no_pain", "1": "pain"}, "opset": 12,
                "note": "Scores are object detector outputs, not calibrated pain probabilities."}
    (directory / "yolo-model.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from PIL import Image

    comparisons = []
    for path in list((ROOT / "data" / "uploads").glob("*.jpg"))[:3]:
        with Image.open(path) as photo:
            tensor, transform = runtime.preprocess(photo.convert("RGB"))
        with torch.inference_mode():
            expected = model(torch.from_numpy(tensor)).cpu().numpy()
        actual = runtime.predict_tensor(tensor)
        difference = np.abs(expected - actual)
        torch_detections = runtime.postprocess(expected, transform)
        onnx_detections = runtime.postprocess(actual, transform)
        if [d["label"] for d in torch_detections] != [d["label"] for d in onnx_detections]:
            raise ValueError("ONNX export changed sample labels")
        comparisons.append({"sample_index": len(comparisons), "output_shape": list(actual.shape),
                            "max_abs_difference": float(difference.max()), "mean_abs_difference": float(difference.mean()),
                            "labels_equal": True, "detections": len(onnx_detections)})
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "yolo-export-validation.json").write_text(json.dumps({
        "model_version": runtime.MODEL_VERSION, "comparisons": comparisons,
        "scope": "Numerical/runtime verification only. No independently labeled pain evaluation."}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exported": target.name, "size": target.stat().st_size, "comparisons": comparisons}, ensure_ascii=True))


if __name__ == "__main__":
    main()
