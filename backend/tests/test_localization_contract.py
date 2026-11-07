import copy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app
from app.schemas import Localization


SCORES = {"ears": 0, "eyes": 0, "muzzle": 0, "whiskers": 0, "head": 0}


def fixture_geometry():
    # Interface fixture only; these are not claimed to be detected coordinates.
    return {
        "status": "ok", "reason": "localized", "note": "接口测试几何",
        "model_version": "catflw-test", "coordinate_space": "image_pixels",
        "bbox": [10, 10, 200, 200], "cat_bbox": [0, 0, 400, 400],
        "detector_confidence": 0.9,
        "points": [{"id": i, "x": 20 + i, "y": 30 + i, "confidence": None} for i in range(1, 49)],
    }


@pytest.mark.parametrize("synchronous", [False, True])
def test_geometry_is_public_but_pain_prediction_stays_hidden(tmp_path, photo, synchronous):
    geometry = fixture_geometry()

    def analyze(*_):
        return {
            "status": "ok", "label": "pain", "raw_score": 0.8,
            "model_version": "classifier-test", "localization": geometry,
        }

    app = create_app(data_dir=tmp_path, inference_fn=analyze, synchronous_inference=synchronous)
    with TestClient(app) as client:
        response = client.post("/api/analyses", files={"image": ("fixture.png", photo, "image/png")})
        assert response.status_code == 202, response.text
        if synchronous:
            assert response.json()["localization"] == geometry
        else:
            assert response.json()["localization"]["status"] == "processing"
        id = response.json()["id"]
        meta_response = client.get(f"/api/analyses/{id}")
        meta = meta_response.json()
        assert meta["localization"] == geometry
        for hidden_key in ("ai", "label", "raw_score", "threshold"):
            assert f'"{hidden_key}"' not in meta_response.text
            assert f'"{hidden_key}"' not in response.text
        assert client.get(f"/api/analyses/{id}/report").status_code == 404

        result = client.post(f"/api/analyses/{id}/scores", json={"scores": SCORES})
        assert result.status_code == 200, result.text
        report = result.json()
        assert report["localization"] == geometry
        assert report["ai"]["label"] == "pain"
        assert report["ai"]["raw_score"] == 0.8
        assert client.get(f"/api/analyses/{id}/report").json() == report


def test_reports_created_before_localization_remain_readable(client, photo):
    upload = client.post("/api/analyses", files={"image": ("fixture.png", photo, "image/png")}).json()
    id = upload["id"]
    report = client.post(f"/api/analyses/{id}/scores", json={"scores": SCORES}).json()
    report.pop("localization")
    report["ai"].pop("localization")
    client.app.state.store.set_report(id, report, "fgs-comparison-v1")
    restored = client.get(f"/api/analyses/{id}/report")
    assert restored.status_code == 200
    assert restored.json()["localization"] is None
    assert restored.json()["ai"]["localization"] is None
    assert restored.json()["fgs"] == report["fgs"]
    assert client.get(f"/api/analyses/{id}").json()["localization"] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda g: g["points"].pop(),
        lambda g: g["points"][1].update(id=1),
        lambda g: g["points"][0].update(x=float("nan")),
        lambda g: g["points"][0].update(confidence=float("inf")),
        lambda g: g.update(detector_confidence=1.1),
        lambda g: g.update(bbox=[200, 10, 10, 200]),
        lambda g: g.update(cat_bbox=[0, 0, float("inf"), 400]),
        lambda g: g.update(status="unavailable"),
        lambda g: g.update(label="pain"),
    ],
)
def test_invalid_or_unconfirmed_geometry_is_rejected(change):
    geometry = copy.deepcopy(fixture_geometry())
    change(geometry)
    with pytest.raises(ValidationError):
        Localization.model_validate(geometry)


def test_missing_point_confidence_stays_unknown():
    geometry = fixture_geometry()
    for point in geometry["points"]:
        point.pop("confidence")
    validated = Localization.model_validate(geometry)
    assert all(point.confidence is None for point in validated.points)
