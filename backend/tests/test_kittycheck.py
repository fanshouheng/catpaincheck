from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app import kittycheck as model

LOCALIZATION = {"status": "ok", "bbox": [20, 20, 180, 180]}


@pytest.fixture
def centers():
    return {"left_eye": [70.0, 110.0], "right_eye": [154.0, 110.0], "nose": [112.0, 150.0],
            "left_ear": [45.0, 45.0], "right_ear": [179.0, 45.0]}


@pytest.fixture
def working_pipeline(monkeypatch, centers):
    monkeypatch.setattr(model, "_verified", lambda name: True)
    monkeypatch.setattr(model, "_detect_face", lambda pixels: (20, 20, 180, 180))
    monkeypatch.setattr(model, "_detect_centers", lambda image: centers)
    monkeypatch.setattr(model, "_run", lambda name, tensor: np.array([0.1, -0.2, 1.4], dtype=np.float32))
    return Image.new("RGB", (200, 200), (145, 120, 110))


def test_classifier_input_uses_rgb_imagenet_but_regression_does_not():
    pixels = np.array([[[255, 0, 128]]], dtype=np.uint8)
    regression = model._tensor(pixels, False)
    scoring = model._tensor(pixels, True)
    assert regression.shape == scoring.shape == (1, 1, 1, 3)
    np.testing.assert_allclose(regression[0, 0, 0], [1, 0, 128 / 255], atol=1e-7)
    np.testing.assert_allclose(scoring[0, 0, 0], [(1 - .485) / .229, - .456 / .224,
                                               (128 / 255 - .406) / .225], atol=1e-6)


def test_resize_uses_dart_origin_aligned_sampling_and_uint8_truncation():
    pixels = np.repeat(np.array([[0, 100], [200, 255]], dtype=np.uint8)[:, :, None], 3, axis=2)
    resized = model._resize(pixels, 4, 4)
    assert resized[0, 1, 0] == 50
    assert resized[1, 0, 0] == 100
    assert resized[1, 1, 0] == 138
    assert resized[3, 3, 0] == 255


def test_letterbox_preserves_context_and_black_padding():
    source = np.full((100, 200, 3), 120, dtype=np.uint8)
    output = model._letterbox(source)
    assert output.shape == (224, 224, 3)
    assert not output[:56].any()
    assert np.all(output[56:168] == 120)
    assert not output[168:].any()


def test_face_regressor_uses_upstream_original_image_coordinate_target(monkeypatch):
    monkeypatch.setattr(model, "_run", lambda name, tensor: np.array([.1, .2, .8, .9], dtype=np.float32))
    box = model._detect_face(np.zeros((300, 600, 3), dtype=np.uint8))
    assert box == (60, 60, 480, 270)


def test_eye_alignment_maps_both_eye_centers_to_one_row(centers):
    centers["right_eye"][1] += 20
    angle = model._rotation_angle(centers)
    image, aligned = model._rotate(np.full((224, 224, 3), 100, dtype=np.uint8), centers, angle)
    assert aligned["left_eye"][1] == pytest.approx(aligned["right_eye"][1])
    assert image.shape[0] > 224
    assert np.any(image == 0)  # Expanded rotation canvas retains padding.


def test_head_input_keeps_full_image_with_red_rectangle():
    source = np.full((100, 200, 3), 120, dtype=np.uint8)
    head = model._head_image(source, (40, 20, 160, 80))
    assert not head[:56].any()
    assert np.array_equal(head[100, 10], [120, 120, 120])
    assert ((head[:, :, 0] > 200) & (head[:, :, 1] < 30) & (head[:, :, 2] < 30)).any()
    assert (source == 120).all()  # Does not draw into the uploaded image.


def test_no_cat_gate_cannot_produce_scores(monkeypatch):
    monkeypatch.setattr(model, "_verified", lambda name: pytest.fail("Models must not run before the cat gate"))
    result = model.analyze(Image.new("RGB", (200, 200)), {"status": "unavailable"})
    assert result["status"] == "unavailable"
    assert all(v is None for v in result["scores"].values())
    assert result["raw_outputs"] == {}


def test_raw_logits_argmax_gives_five_scores_and_complete_sum(working_pipeline):
    result = model.analyze(working_pipeline, LOCALIZATION)
    assert result["status"] == "ok"
    assert result["scores"] == {key: 2 for key in model.KEYS}
    assert result["total"] == result["max"] == 10
    assert result["complete"] and result["flag"] == "positive"
    assert result["raw_outputs"]["eyes"][1] < 0  # Raw logits are not probabilities.


def test_missing_head_model_is_na_and_not_zero(monkeypatch, working_pipeline):
    monkeypatch.setattr(model, "_verified", lambda name: name != "head_scorer.tflite")
    result = model.analyze(working_pipeline, LOCALIZATION)
    assert result["status"] == "partial"
    assert result["scores"]["head"] is None
    assert result["total"] == result["max"] == 8
    assert result["assessable_count"] == 4
    assert not result["complete"] and result["flag"] == "incomplete"
    assert "head" not in result["raw_outputs"]


def test_bad_logits_do_not_fabricate_fgs_scores(monkeypatch, working_pipeline):
    monkeypatch.setattr(model, "_run", lambda name, tensor: np.array([np.nan, 1, 2]))
    result = model.analyze(working_pipeline, LOCALIZATION)
    assert result["status"] == "unavailable"
    assert result["total"] == result["max"] == result["assessable_count"] == 0
    assert all(v is None for v in result["scores"].values())


def test_out_of_range_region_centers_are_rejected(monkeypatch):
    monkeypatch.setattr(model, "_run", lambda name, tensor: np.full((10,), 1.5))
    with pytest.raises(ValueError, match="Invalid region"):
        model._detect_centers(np.zeros((224, 224, 3), dtype=np.uint8))


def test_scorer_runtime_failure_only_removes_that_item(monkeypatch, working_pipeline):
    def inference(name, tensor):
        if name == "muzzle_scorer.tflite":
            raise RuntimeError("broken scorer")
        return np.array([0, 2, 1], dtype=np.float32)
    monkeypatch.setattr(model, "_run", inference)
    result = model.analyze(working_pipeline, LOCALIZATION)
    assert result["status"] == "partial"
    assert result["scores"]["muzzle"] is None
    assert result["scores"]["eyes"] == 1
    assert result["total"] == 4 and result["max"] == 8


@pytest.mark.parametrize("box", [None, [], [20, 20, 20, 80], [0, 0, 201, 200],
                                 [-1, 0, 100, 100], [0, 0, float("nan"), 100]])
def test_invalid_independent_face_box_cannot_authorize_scores(monkeypatch, working_pipeline, box):
    monkeypatch.setattr(model, "_detect_centers", lambda face: pytest.fail("Must reject before region inference"))
    result = model.analyze(working_pipeline, {"status": "ok", "bbox": box})
    assert result["status"] == "unavailable"
    assert all(score is None for score in result["scores"].values())
    assert result["raw_outputs"] == {}
    assert "定位不一致" in result["note"]


def test_nonoverlapping_face_cannot_score_another_region(monkeypatch, working_pipeline):
    monkeypatch.setattr(model, "_detect_face", lambda pixels: (120, 120, 180, 180))
    monkeypatch.setattr(model, "_detect_centers", lambda face: pytest.fail("Must reject before region inference"))
    result = model.analyze(working_pipeline, {"status": "ok", "bbox": [0, 0, 80, 80]})
    assert result["status"] == "unavailable"
    assert result["assessable_count"] == 0
    assert result["raw_outputs"] == {}


def test_independent_face_gate_allows_different_margins_and_checks_overlap():
    assert model._consistent_faces((40, 40, 160, 160), [0, 0, 200, 200], 200, 200)
    assert model._consistent_faces((0, 0, 100, 100), [65, 0, 165, 100], 200, 200)
    assert not model._consistent_faces((0, 0, 100, 100), [66, 0, 166, 100], 200, 200)
