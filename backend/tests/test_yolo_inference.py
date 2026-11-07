from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import yolo_inference as yolo


def test_letterbox_and_inverse_coordinates_preserve_landscape_photo():
    tensor, transform = yolo.preprocess(Image.new("RGB", (1000, 500), "white"))
    assert tensor.shape == (1, 3, 768, 768)
    assert tensor[0, 0, 0, 0] == pytest.approx(114 / 255)
    assert tensor[0, 0, 300, 300] == 1
    scale, left, top, _, _ = transform
    row = [200 * scale + left, 175 * scale + top, 200 * scale, 150 * scale, 0.1, 0.8]
    result = yolo.postprocess(np.array(row, dtype=np.float32).reshape(1, 6, 1), transform)
    assert len(result) == 1 and result[0]["label"] == "pain"
    assert result[0]["bbox"] == pytest.approx([100, 100, 300, 250], abs=1e-4)


def test_alternative_labels_for_same_face_are_suppressed_but_second_face_remains():
    rows = [[200, 200, 100, 100, .1, .9], [202, 202, 100, 100, .8, .1],
            [600, 600, 100, 100, .7, .1]]
    result = yolo.postprocess(np.asarray(rows, dtype=np.float32).T[None], (1, 0, 0, 768, 768))
    assert len(result) == 2
    assert [item["label"] for item in result] == ["pain", "no_pain"]


@pytest.mark.parametrize("output", [np.zeros((1, 7, 1)), np.full((1, 6, 1), np.nan)])
def test_invalid_output_is_not_a_prediction(output):
    with pytest.raises(ValueError):
        yolo.postprocess(output, (1, 0, 0, 768, 768))


def test_no_confident_detection_is_empty():
    assert yolo.postprocess(np.zeros((1, 6, 4)), (1, 0, 0, 768, 768)) == []


def test_cat_gate_failure_never_runs_pain_model(monkeypatch):
    monkeypatch.setattr(yolo, "predict_tensor", lambda *_: pytest.fail("Unvalidated image reached model"))
    result = yolo.analyze(Image.new("RGB", (200, 200)), {"status": "unavailable"})
    assert result["status"] == "unavailable" and result["label"] is None


def test_no_model_does_not_fabricate_prediction(monkeypatch, tmp_path):
    monkeypatch.setenv("CATPAIN_MODELS_DIR", str(tmp_path))
    result = yolo.analyze(Image.new("RGB", (200, 200)), {"status": "ok", "bbox": [0, 0, 200, 200]})
    assert result["status"] == "unavailable"
    assert result["label"] is None and result["detection_confidence"] is None


def test_geometrically_unrelated_detection_is_rejected(monkeypatch):
    monkeypatch.setattr(yolo, "get_model_status", lambda: {"ready": True})
    monkeypatch.setattr(yolo, "predict_tensor", lambda *_: None)
    monkeypatch.setattr(yolo, "postprocess", lambda *_: [{"bbox": [150, 150, 190, 190], "label": "pain", "confidence": .9}])
    result = yolo.analyze(Image.new("RGB", (200, 200)), {"status": "ok", "bbox": [0, 0, 80, 80]})
    assert result["status"] == "unavailable" and result["label"] is None


def test_one_engine_failure_preserves_other_prediction(monkeypatch, tmp_path):
    from app import dual_inference as dual
    from app.schemas import AiResult

    source = tmp_path / "photo.jpg"
    Image.new("RGB", (200, 200)).save(source)
    geometry = {"status": "ok", "bbox": [0, 0, 200, 200],
                "points": [{"id": i + 1, "x": i, "y": i} for i in range(48)]}
    monkeypatch.setattr(dual.localization, "localize_image", lambda *_: geometry)
    monkeypatch.setattr(dual.yolo_inference, "analyze", lambda *_: {"status": "ok", "label": "pain", "detection_confidence": .8})

    def failed(*_):
        raise RuntimeError("engine failed")

    monkeypatch.setattr(dual.kittycheck, "analyze", failed)
    result = AiResult.model_validate(dual.analyze_image(source, tmp_path / "new" / "crop.jpg"))
    assert result.machines.yolo.label == "pain"
    assert result.machines.auto_fgs.status == "error"
    assert result.raw_score is None and result.label is None


@pytest.mark.parametrize("invalid_engine", ["yolo", "auto_fgs"])
def test_invalid_engine_payload_preserves_other_prediction(monkeypatch, tmp_path, invalid_engine):
    from app import dual_inference as dual
    from app.schemas import AiResult

    source = tmp_path / "photo.jpg"
    Image.new("RGB", (200, 200)).save(source)
    geometry = {"status": "ok", "bbox": [0, 0, 200, 200],
                "points": [{"id": i + 1, "x": i, "y": i} for i in range(48)]}
    monkeypatch.setattr(dual.localization, "localize_image", lambda *_: geometry)
    detection = {"status": "ok", "label": "pain", "detection_confidence": .8}
    automatic = {"status": "ok", "scores": dict.fromkeys(("ears", "eyes", "muzzle", "whiskers", "head"), 1)}
    if invalid_engine == "yolo":
        detection["detection_confidence"] = float("nan")
    else:
        automatic["raw_outputs"] = {"eyes": [0, float("nan"), 1]}
    monkeypatch.setattr(dual.yolo_inference, "analyze", lambda *_: detection)
    monkeypatch.setattr(dual.kittycheck, "analyze", lambda *_: automatic)
    result = AiResult.model_validate(dual.analyze_image(source, tmp_path / "crop.jpg"))
    if invalid_engine == "yolo":
        assert result.machines.yolo.status == "error"
        assert result.machines.auto_fgs.total == 5
    else:
        assert result.machines.auto_fgs.status == "error"
        assert result.machines.yolo.label == "pain"
