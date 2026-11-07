import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.cloud_storage import BlobStore
from app.main import create_app
from test_cloud_storage import fake_blob  # noqa: F401 - shared pytest fixture


LIMIT = 4 * 1024 * 1024
SCORES = {"ears": 0, "eyes": 0, "muzzle": 0, "whiskers": 0, "head": 0}


def cloud_app(data_dir: Path):
    def analyze(source, crop):
        with Image.open(source) as photo:
            photo.crop((10, 10, 200, 200)).save(crop)
        return {
            "status": "ok", "label": "pain", "raw_score": 0.8,
            "threshold": 0.4, "model_version": "cloud-test-fixture",
            "note": "仅用于云端接口流程测试的固定结果", "bbox": [10, 10, 200, 200],
        }

    return create_app(
        data_dir=data_dir,
        inference_fn=analyze,
        model_status_fn=lambda: {"ready": True, "detail": "Test fixture", "version": "cloud-test-fixture"},
        store_factory=BlobStore,
        deployment="cloud",
        max_upload=LIMIT,
        synchronous_inference=True,
    )


def test_cloud_upload_hidden_ai_then_report_restores_on_fresh_instance(tmp_path, photo, fake_blob):
    app = cloud_app(tmp_path / "first")
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["deployment"] == "cloud"
        assert health["max_upload_bytes"] == LIMIT
        response = client.post(
            "/api/analyses",
            files={"image": ("cat.png", photo, "image/png")},
            data={"cat_name": "测试猫"},
        )
        assert response.status_code == 202, response.text
        meta = response.json()
        id = meta["id"]
        assert meta["status"] == "ready"
        assert not ({"ai", "label", "prediction", "raw_score", "bbox"} & set(meta))
        assert "raw_score" not in response.text
        assert not list(app.state.store.uploads.iterdir())
        assert not list(app.state.store.crops.iterdir())
        assert client.get(f"/api/analyses/{id}/report").status_code == 404
        assert client.get(f"/api/analyses/{id}/crop").status_code == 404
        original = client.get(meta["image_url"])
        assert original.status_code == 200
        assert original.headers["cache-control"] == "no-store"
        assert Image.open(io.BytesIO(original.content)).getexif() == {}
        scored = client.post(
            f"/api/analyses/{id}/scores",
            json={"scores": SCORES, "notes": "固定测试输入"},
        )
        assert scored.status_code == 200, scored.text
        report = scored.json()
        assert report["ai"]["raw_score"] == 0.8
        assert report["conclusion"]["code"] == "discordant"

    fresh = cloud_app(tmp_path / "second")
    with TestClient(fresh) as client:
        assert client.get(f"/api/analyses/{id}/report").json() == report
        assert client.get(report["image_url"]).content == original.content
        crop = client.get(report["ai"]["crop_url"])
        assert crop.status_code == 200
        assert Image.open(io.BytesIO(crop.content)).size == (190, 190)
        assert not list(fresh.state.store.uploads.iterdir())
        assert not list(fresh.state.store.crops.iterdir())


def test_cloud_rejects_over_four_mib_without_persisting_or_revealing_ai(tmp_path, fake_blob):
    with TestClient(cloud_app(tmp_path)) as client:
        response = client.post(
            "/api/analyses",
            files={"image": ("too-large.jpg", b"x" * (LIMIT + 1), "image/jpeg")},
        )
        assert response.status_code == 413
        assert "4 MB" in response.json()["detail"]
        assert "raw_score" not in response.text
        assert fake_blob.writes == []
