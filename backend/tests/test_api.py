import io
import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import MAX_UPLOAD, PROJECT_ROOT, create_app, now


ZERO_SCORES = {"ears": 0, "eyes": 0, "muzzle": 0, "whiskers": 0, "head": 0}


def test_project_root_is_the_project_folder():
    assert PROJECT_ROOT == Path(__file__).resolve().parents[2]


def upload(client, photo):
    response = client.post("/api/analyses", files={"image": ("cat.png", photo, "image/png")}, data={"cat_name": "小橘"})
    assert response.status_code == 202, response.text
    return response.json()["id"]


def test_upload_ai_hidden_then_score_report_and_persistence(client, photo):
    id = upload(client, photo)
    before = client.get(f"/api/analyses/{id}")
    assert before.status_code == 200
    meta = before.json()
    assert meta["status"] == "ready"
    assert meta["cat_name"] == "小橘"
    assert not ({"ai", "label", "prediction", "raw_score", "bbox"} & set(meta))
    assert "pain" not in before.text
    assert client.get(f"/api/analyses/{id}/report").status_code == 404
    assert client.get(f"/api/analyses/{id}/crop").status_code == 404
    image = client.get(meta["image_url"])
    assert image.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(image.content)).getexif() == {}

    response = client.post(f"/api/analyses/{id}/scores", json={"scores": ZERO_SCORES, "notes": "今天稍安静"})
    assert response.status_code == 200
    report = response.json()
    assert report["ai"]["label"] == "pain"
    assert report["ai"]["model_version"] == "test-fixture"
    assert report["fgs"]["total"] == 0
    assert report["conclusion"]["code"] == "discordant"
    assert report["notes"] == "今天稍安静"
    assert client.get(f"/api/analyses/{id}/report").json() == report
    assert client.get(report["ai"]["crop_url"]).status_code == 200
    row = client.app.state.store.get(id)
    assert json.loads(row["ai_json"])["raw_score"] == 0.8
    assert row["rule_version"] == "fgs-comparison-v1"


def test_incomplete_choice_stays_null(client, photo):
    id = upload(client, photo)
    response = client.post(f"/api/analyses/{id}/scores", json={"scores": {**ZERO_SCORES, "head": None}})
    assert response.status_code == 200
    report = response.json()
    assert report["fgs"]["scores"]["head"] is None
    assert report["fgs"]["assessable_count"] == 4
    assert report["conclusion"]["code"] == "incomplete"


def test_invalid_scores_rejected_without_revealing_ai(client, photo):
    id = upload(client, photo)
    for invalid in [True, "0", 0.0, 3]:
        response = client.post(f"/api/analyses/{id}/scores", json={"scores": {**ZERO_SCORES, "head": invalid}})
        assert response.status_code == 422
        assert "raw_score" not in response.text
    assert client.get(f"/api/analyses/{id}/report").status_code == 404


def test_missing_model_still_allows_fgs_without_fabricated_ai(tmp_path, photo):
    application = create_app(
        data_dir=tmp_path,
        inference_fn=lambda *_: {"status": "unavailable", "note": "缺少权重", "model_version": "missing"},
        model_status_fn=lambda: {"ready": False, "detail": "缺少权重", "version": "missing"},
    )
    with TestClient(application) as client:
        assert client.get("/api/health").json()["model"]["ready"] is False
        id = upload(client, photo)
        report = client.post(f"/api/analyses/{id}/scores", json={"scores": {**ZERO_SCORES, "ears": 2, "eyes": 2}}).json()
        assert report["ai"]["label"] is None
        assert report["ai"]["raw_score"] is None
        assert report["conclusion"]["code"] == "ai_unavailable"
        assert "达到关注参考值" in report["conclusion"]["title"]


def test_model_failure_becomes_explicit_unavailable_report(tmp_path, photo):
    def fail(*_):
        raise RuntimeError("fixture model crash")

    with TestClient(create_app(data_dir=tmp_path, inference_fn=fail)) as client:
        id = upload(client, photo)
        report = client.post(f"/api/analyses/{id}/scores", json={"scores": ZERO_SCORES}).json()
        assert report["ai"]["status"] == "error"
        assert report["ai"]["label"] is None
        assert report["conclusion"]["code"] == "ai_unavailable"


@pytest.mark.parametrize("content", [b"", b"not an image", b"<svg xmlns='http://www.w3.org/2000/svg'/>"])
def test_corrupt_images_rejected(client, content):
    response = client.post("/api/analyses", files={"image": ("fake.png", content, "image/png")})
    assert response.status_code == 422


def test_upload_size_limit(client):
    response = client.post("/api/analyses", files={"image": ("large.jpg", b"x" * (MAX_UPLOAD + 1), "image/jpeg")})
    assert response.status_code == 413


def test_tiny_image_rejected(client):
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buffer, "PNG")
    assert client.post("/api/analyses", files={"image": ("tiny.png", buffer.getvalue(), "image/png")}).status_code == 422


def test_pending_analysis_cannot_be_scored(client):
    id = uuid.uuid4().hex
    client.app.state.store.insert(id, now(), "等待中", 640, 480)
    response = client.post(f"/api/analyses/{id}/scores", json={"scores": ZERO_SCORES})
    assert response.status_code == 409


def test_resource_errors_do_not_fall_into_spa(client):
    assert client.get("/api").status_code == 404
    assert client.get("/api/no-such-api").status_code == 404
    assert client.get("/api/analyses/not-a-valid-id").status_code == 404
    assert client.get(f"/api/analyses/{uuid.uuid4().hex}").status_code == 404
    response = client.get("/api/health")
    assert response.json()["service"] == "catpaincheck"
    assert response.headers["cache-control"] == "no-store"


def test_interrupted_analysis_recovers_as_unavailable(tmp_path):
    old = create_app(data_dir=tmp_path, model_status_fn=lambda: {})
    id = uuid.uuid4().hex
    old.state.store.insert(id, now(), "旧记录", 640, 480)
    restarted = create_app(data_dir=tmp_path, model_status_fn=lambda: {})
    with TestClient(restarted) as client:
        meta = client.get(f"/api/analyses/{id}").json()
        assert meta["status"] == "ready"
        assert "ai" not in meta
        report = client.post(f"/api/analyses/{id}/scores", json={"scores": ZERO_SCORES}).json()
        assert report["ai"]["status"] == "unavailable"
