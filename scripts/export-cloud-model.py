"""Export the pinned research model to ONNX and validate the cloud adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, help="Optional real cat image for end-to-end validation")
    args = parser.parse_args()

    import cv2
    import numpy as np
    import onnx
    import torch
    from PIL import Image
    from torchvision.transforms.functional import resize, to_pil_image, to_tensor
    from app import cloud_inference as cloud
    from app import inference as local

    valid, reason = local._check_weights()
    if not valid:
        raise SystemExit(reason)
    model = local._get_model()
    destination = ROOT / "models" / cloud.MODEL_NAME
    pending = destination.with_suffix(".pending.onnx")
    try:
        torch.onnx.export(
            model, torch.zeros((1, 3, 224, 224), dtype=torch.float32), str(pending),
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=["image"], output_names=["score"], dynamo=False,
        )
        onnx.checker.check_model(str(pending))
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    manifest = {
        "schema_version": 1, "model_version": cloud.MODEL_VERSION,
        "filename": cloud.MODEL_NAME, "size": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_commit": local.UPSTREAM_COMMIT, "source_weights_sha256": local.WEIGHTS_SHA256,
        "opset": 17, "input": "float32 RGB NCHW 1x3x224x224, [0,1], black letterbox",
        "preprocessing": "Pillow float-channel antialiased bilinear, then uint8 floor; versioned separately from torchvision tensor resize",
        "threshold": local.THRESHOLD,
    }
    (ROOT / "models" / cloud.MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def local_preprocess(image: Image.Image) -> np.ndarray:
        ratio = image.width / image.height
        target_w, target_h = ((224, max(1, int(224 / ratio))) if image.width > image.height
                              else (max(1, int(224 * ratio)), 224))
        scaled = resize(to_tensor(image), [target_h, target_w], antialias=True)
        letterbox = Image.new("RGB", (224, 224), "black")
        letterbox.paste(to_pil_image(scaled), ((224 - target_w) // 2, (224 - target_h) // 2))
        return to_tensor(letterbox).unsqueeze(0).numpy()

    rng = np.random.default_rng(8)
    comparisons = []
    for height, width in ((733, 1151), (224, 224), (80, 91), (300, 120), (1598, 1600)):
        image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        baseline, adapted = local_preprocess(image), cloud.preprocess(image)
        with torch.inference_mode():
            torch_score = float(model(torch.from_numpy(baseline)).item())
        same_input_score = cloud.predict_tensor(baseline)
        adapted_score = cloud.predict_tensor(adapted)
        delta = np.abs(baseline - adapted)
        comparisons.append({
            "image_size": [width, height], "input_max_abs_difference": float(delta.max()),
            "input_differing_fraction": float(np.mean(delta != 0)),
            "torch_score": torch_score, "onnx_same_input_score": same_input_score,
            "onnx_same_input_abs_difference": abs(torch_score - same_input_score),
            "cloud_adapter_score": adapted_score,
            "end_to_end_abs_difference": abs(torch_score - adapted_score),
        })
    if max(row["onnx_same_input_abs_difference"] for row in comparisons) > 1e-5:
        raise RuntimeError("ONNX numerical comparison failed")
    if max(row["input_max_abs_difference"] for row in comparisons) > 1 / 255 + 1e-6:
        raise RuntimeError("Preprocessing differs by more than one channel quantization step")

    blank_path = ROOT / "work" / "cloud-test-blank.png"
    Image.new("RGB", (640, 640), "white").save(blank_path)
    blank = cloud.analyze_image(blank_path, ROOT / "work" / "cloud-blank-crop.jpg")
    blank_path.unlink(missing_ok=True)
    if blank["status"] != "unavailable" or blank["raw_score"] is not None:
        raise RuntimeError("Blank image must not produce a score")
    evidence = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "versions": {"torch": torch.__version__, "opencv": cv2.__version__, "onnx": onnx.__version__},
        "model": manifest, "comparisons": comparisons, "blank_image": blank,
        "limitations": "Numerical equivalence checks and a sample output do not establish diagnostic accuracy. Pillow float-channel resize has bounded quantization differences from the original tensor adapter.",
    }
    if args.sample:
        local_result = local.analyze_image(args.sample, ROOT / "work" / "cloud-test-local-crop.jpg")
        cloud_result = cloud.analyze_image(args.sample, ROOT / "work" / "cloud-test-onnx-crop.jpg")
        evidence["sample"] = {"path": args.sample.name, "local": local_result, "cloud": cloud_result}
        if local_result["status"] != "ok" or cloud_result["status"] != "ok":
            raise RuntimeError("Real cat sample did not run successfully")
        evidence["sample"]["abs_difference"] = abs(local_result["raw_score"] - cloud_result["raw_score"])
    output = ROOT / "outputs" / "cloud-inference-validation.json"
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_bytes": manifest["size"], "model_sha256": manifest["sha256"],
                      "max_same_input_difference": max(row["onnx_same_input_abs_difference"] for row in comparisons),
                      "max_adapter_difference": max(row["end_to_end_abs_difference"] for row in comparisons),
                      "evidence": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
