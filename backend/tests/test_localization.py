"""Geometry and abstention tests with synthetic tensors, never real model outputs."""

import zipfile

import numpy as np
import pytest
from PIL import Image

from app import localization
from app.schemas import Localization


@pytest.mark.parametrize("roi_size", [(501, 333), (333, 501)])
def test_face_box_inverts_actual_resized_letterbox_on_each_axis(monkeypatch, roi_size):
    width, height = roi_size
    image = Image.new("RGB", (800, 800), (255, 128, 64))
    region = (30, 50, 30 + width, 50 + height)
    # Source-image offsets that should survive a round trip through the
    # localizer's normalized box, including the non-leading rounded dimension.
    source_box = np.array([60, 50, 280, 300], dtype=np.float64)
    scale = min(224 / width, 224 / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    left, top = (224 - resized_width) // 2, (224 - resized_height) // 2
    normalized_box = np.array([
        (60 * resized_width / width + left) / 224,
        (50 * resized_height / height + top) / 224,
        (280 * resized_width / width + left) / 224,
        (300 * resized_height / height + top) / 224,
    ], dtype=np.float32)

    def run(name, tensor):
        assert name == "cat_face_localizer.tflite"
        assert tensor.shape == (1, 224, 224, 3)
        assert tensor.dtype == np.float32
        np.testing.assert_allclose(tensor[0, top + 2, left + 2], [1, 128 / 255, 64 / 255])
        if top:
            assert not tensor[0, :top].any()
        if left:
            assert not tensor[0, :, :left].any()
        return [normalized_box[None]]

    monkeypatch.setattr(localization, "_run", run)
    expected = source_box + [30, 50, 30, 50]
    np.testing.assert_allclose(localization._face_box(image, region), expected, atol=0.0001, rtol=0)


def test_cat_detector_uses_embedded_label_and_deduplicates_but_preserves_two_cats(monkeypatch, tmp_path):
    # Deliberately put cat at a synthetic non-COCO offset: hardcoding category
    # 16 instead of reading this model's labels would fail the test.
    with zipfile.ZipFile(tmp_path / "coco-efficientdet-lite0.tflite", "w") as archive:
        archive.writestr("labelmap.txt", "background\nperson\ndog\ncat\n")
    monkeypatch.setattr(localization, "_models_dir", lambda: tmp_path)
    boxes = np.array([[
        [0.10, 0.10, 0.60, 0.40],  # First cat.
        [0.11, 0.11, 0.59, 0.39],  # Duplicate of first cat.
        [0.20, 0.65, 0.80, 0.95],  # Second cat.
        [0.10, 0.50, 0.30, 0.70],  # Wrong class, despite high score.
        [0.00, 0.50, 0.10, 0.60],  # Below score threshold.
        [np.nan, 0.0, 0.3, 0.3],  # Invalid coordinates.
        [0.50, 0.5, 0.10, 0.6],  # Reversed box.
        [0.00, 0.7, 0.10, 0.8],  # Beyond the model's count output.
    ]], dtype=np.float32)
    classes = np.array([[3, 3, 3, 16, 3, 3, 3, 3]], dtype=np.float32)
    scores = np.array([[0.95, 0.85, 0.81, 0.99, 0.34, 0.9, 0.9, 0.99]], dtype=np.float32)

    def run(name, tensor):
        assert name == "coco-efficientdet-lite0.tflite"
        assert tensor.shape == (1, 320, 320, 3)
        assert tensor.dtype == np.uint8
        np.testing.assert_array_equal(tensor[0, 0, 0], [64, 128, 255])
        return [boxes, classes, scores, np.array([7], dtype=np.float32)]

    monkeypatch.setattr(localization, "_run", run)
    detected = localization._detect_cats(Image.new("RGB", (800, 400), (64, 128, 255)))
    assert len(detected) == 2
    np.testing.assert_allclose([score for score, _ in detected], [0.95, 0.81])
    np.testing.assert_allclose([box for _, box in detected], [[80, 40, 320, 240], [520, 80, 760, 320]], atol=0.0001)


@pytest.mark.parametrize(
    "cats,reason",
    [([], "no_cat"), ([(0.9, (10, 10, 300, 300)), (0.8, (350, 10, 600, 300))], "multiple_cats")],
)
def test_no_or_multiple_cats_never_call_face_regressors(monkeypatch, cats, reason):
    monkeypatch.setattr(localization, "get_localization_status", lambda: {"ready": True})
    monkeypatch.setattr(localization, "_detect_cats", lambda _: cats)

    def unexpected(*_):
        pytest.fail("No face regressor may run without exactly one detected cat")

    monkeypatch.setattr(localization, "_face_box", unexpected)
    monkeypatch.setattr(localization, "_run", unexpected)
    result = localization.localize_image(Image.new("RGB", (640, 480)))
    assert result["status"] == "unavailable"
    assert result["reason"] == reason
    assert result["points"] == []
    assert result["bbox"] is None


def test_landmarks_map_back_through_same_integer_roi_for_non_square_photo(monkeypatch):
    monkeypatch.setattr(localization, "get_localization_status", lambda: {"ready": True})
    monkeypatch.setattr(localization, "_detect_cats", lambda _: [(0.9, (80.5, 30.25, 650.75, 470.75))])

    def face_box(image, region):
        assert image.size == (721, 503)
        assert region == (23, 0, 707, 503)
        return 145.8, 80.6, 580.3, 390.2

    monkeypatch.setattr(localization, "_face_box", face_box)
    xy = np.full((48, 2), 0.5, dtype=np.float32)
    xy[0] = [0.125, 0.25]
    xy[4], xy[8] = [0.2, 0.4], [0.8, 0.4]
    xy[-1] = [0.875, 0.75]

    def run(name, tensor):
        assert name == "cat_face_landmarks_full.tflite"
        assert tensor.shape == (1, 384, 384, 3)
        assert tensor.dtype == np.float32
        np.testing.assert_allclose(tensor[0, 0, 0], [1, 128 / 255, 64 / 255])
        return [xy.reshape(1, 96)]

    monkeypatch.setattr(localization, "_run", run)
    result = localization.localize_image(Image.new("RGB", (721, 503), (255, 128, 64)))
    assert result["status"] == "ok"
    assert result["bbox"] == [102, 49, 623, 421]
    # 521x372 integer crop is used both for model input and coordinate recovery.
    assert result["points"][0] == {"id": 1, "x": 167.125, "y": 142.0, "confidence": None}
    assert result["points"][-1] == {"id": 48, "x": 557.875, "y": 328.0, "confidence": None}
    assert [point["id"] for point in result["points"]] == list(range(1, 49))
    assert all(point["confidence"] is None for point in result["points"])
    Localization.model_validate(result)


def test_missing_models_abstains_without_running_any_model_or_creating_points(monkeypatch, tmp_path):
    monkeypatch.setattr(localization, "_models_dir", lambda: tmp_path)

    def unexpected(*_):
        pytest.fail("Missing model files must stop the pipeline")

    monkeypatch.setattr(localization, "_detect_cats", unexpected)
    monkeypatch.setattr(localization, "_run", unexpected)
    result = localization.localize_image(Image.new("RGB", (640, 480)))
    assert result["status"] == "unavailable"
    assert result["reason"] == "model_unavailable"
    assert result["points"] == []
    assert result["bbox"] is None


@pytest.mark.parametrize("failure", ["non_finite", "out_of_bounds", "collapsed", "exception"])
def test_failed_landmark_predictions_never_substitute_template_points(monkeypatch, failure):
    monkeypatch.setattr(localization, "get_localization_status", lambda: {"ready": True})
    monkeypatch.setattr(localization, "_detect_cats", lambda _: [(0.9, (0, 0, 600, 400))])
    monkeypatch.setattr(localization, "_face_box", lambda *_: (100, 50, 500, 350))
    xy = np.full((48, 2), 0.5, dtype=np.float32)
    if failure == "non_finite":
        xy[0, 0] = np.nan
    elif failure == "out_of_bounds":
        xy[0, 0] = 1.1

    def run(*_):
        if failure == "exception":
            raise RuntimeError("synthetic native runtime failure")
        return [xy.reshape(1, 96)]

    monkeypatch.setattr(localization, "_run", run)
    result = localization.localize_image(Image.new("RGB", (640, 480)))
    assert result["status"] == ("error" if failure == "exception" else "unavailable")
    assert result["reason"] == ("runtime_error" if failure == "exception" else "invalid_landmarks")
    assert result["points"] == []
    assert result["bbox"] is None
