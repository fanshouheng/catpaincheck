import json
import importlib.util
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.cloud_storage import BlobStore
from app.main import create_app
from app.schemas import AutoFgsResult
from app.storage import Store
from test_cloud_storage import fake_blob  # noqa: F401


ZERO = dict.fromkeys(("ears", "eyes", "muzzle", "whiskers", "head"), 0)


def machine_prediction(auto_scores=None, yolo_label="pain"):
    return {
        "status": "unavailable", "note": "Legacy scalar is unused", "model_version": "dual-test",
        "machines": {
            "prediction_id": uuid.uuid4().hex, "created_at": "2026-11-07T01:00:00Z",
            "yolo": {"status": "ok" if yolo_label else "unavailable", "label": yolo_label,
                     "detection_confidence": 0.8 if yolo_label else None, "model_version": "yolo-test"},
            "auto_fgs": {"status": "ok", "scores": auto_scores or ZERO, "model_version": "fgs-test"},
        },
    }


def app_for(tmp_path, result=None, **kwargs):
    return create_app(data_dir=tmp_path, inference_fn=lambda image, crop: result or machine_prediction(),
                      model_status_fn=lambda: {"ready": True, "version": "test", "detail": "Test fixture"},
                      **kwargs)


def upload(client, photo, **data):
    response = client.post("/api/analyses", files={"image": ("cat.png", photo, "image/png")}, data=data)
    assert response.status_code == 202, response.text
    return response.json()


def score(client, id, scores=None):
    response = client.post(f"/api/analyses/{id}/scores", json={"scores": scores or ZERO, "notes": "private note"})
    assert response.status_code == 200, response.text
    return response.json()


def test_dual_predictions_hidden_until_manual_submission(tmp_path, photo):
    with TestClient(app_for(tmp_path)) as client:
        meta = upload(client, photo)
        meta = client.get(f"/api/analyses/{meta['id']}").json()
        assert "machines" not in meta
        assert not any(text in json.dumps(meta) for text in ("raw_outputs", "yolo-test", "fgs-test", "detection_confidence"))
        assert meta["collection"]["training_allowed"] is False
        result = score(client, meta["id"])
        assert result["machines"]["yolo"]["label"] == "pain"
        assert result["machines"]["auto_fgs"]["total"] == 0
        assert result["annotation"]["saw_ai_before_scoring"] is False
        assert result["conclusion"]["code"] == "dual_discordant"
        assert "YOLO" in result["conclusion"]["summary"]


def test_na_and_failed_model_do_not_become_negative_votes(tmp_path, photo):
    prediction = machine_prediction(auto_scores={**ZERO, "ears": 2, "eyes": 2, "head": None}, yolo_label=None)
    with TestClient(app_for(tmp_path, prediction)) as client:
        id = upload(client, photo)["id"]
        report = score(client, id, {**ZERO, "head": None})
        automatic = report["machines"]["auto_fgs"]
        assert automatic["status"] == "partial"
        assert automatic["scores"]["head"] is None
        assert (automatic["total"], automatic["max"], automatic["complete"], automatic["flag"]) == (4, 8, False, "incomplete")
        assert report["conclusion"]["code"] == "dual_incomplete"
        assert "不能因缺项而忽略" in report["conclusion"]["summary"]


@pytest.mark.parametrize("output", [[0, float("nan"), 1], [0, 1], [0, float("inf"), 1]])
def test_invalid_model_outputs_rejected(output):
    with pytest.raises(ValidationError):
        AutoFgsResult(status="ok", scores=ZERO, raw_outputs={"ears": output})


def test_automatic_totals_are_derived_not_trusted():
    result = AutoFgsResult(status="ok", scores={**ZERO, "ears": 2}, total=999, max=0, flag="positive")
    assert (result.total, result.max, result.assessable_count, result.flag) == (2, 10, 5, "negative")
    with pytest.raises(ValidationError):
        AutoFgsResult(status="error", scores=ZERO)


def test_first_manual_annotation_and_prediction_history_are_immutable(tmp_path, photo):
    application = app_for(tmp_path)
    with TestClient(application) as client:
        id = upload(client, photo, training_allowed="true", cat_id=str(uuid.uuid4()))["id"]
        first = score(client, id, {**ZERO, "ears": 1})
        second = score(client, id, {**ZERO, "ears": 2})
        assert second["annotation"]["saw_ai_before_scoring"] is True
        assert second["annotation"]["supersedes_annotation_id"] == first["annotation"]["annotation_id"]
        assert first["annotation"]["annotation_id"] != second["annotation"]["annotation_id"]
        store = application.state.store
        annotations = store.annotation_history(id)
        assert [item["scores"]["ears"] for item in annotations] == [1, 2]
        assert annotations[0]["saw_ai_before_scoring"] is False
        candidates = store.training_candidates()
        assert len(candidates) == 1
        assert candidates[0]["annotation"]["scores"]["ears"] == 1
        assert candidates[0]["weak_label"] is True and candidates[0]["split"] == "unassigned"
        assert "private note" not in json.dumps(candidates)
        assert "cat_name" not in json.dumps(candidates)
        store.set_ai(id, machine_prediction(), "later")
        with store.connection() as db:
            assert db.execute("SELECT count(*) FROM predictions WHERE analysis_id=?", (id,)).fetchone()[0] == 2


def test_training_permission_default_and_withdrawal_filter_export(tmp_path, photo):
    app = app_for(tmp_path)
    with TestClient(app) as client:
        default_id = upload(client, photo)["id"]
        score(client, default_id)
        assert app.state.store.training_candidates() == []
        id = upload(client, photo, training_allowed="true")["id"]
        score(client, id)
        assert len(app.state.store.training_candidates()) == 1
        response = client.post(f"/api/analyses/{id}/training-permission", json={"allowed": False})
        assert response.status_code == 200
        assert response.json()["training_allowed"] is False
        assert app.state.store.training_candidates() == []
        assert client.get(f"/api/analyses/{id}/report").json()["collection"]["training_allowed"] is False
        assert client.get(f"/api/analyses/{id}/image").status_code == 200
        assert client.post(f"/api/analyses/{id}/training-permission", json={"allowed": "false"}).status_code == 422


def test_all_na_has_no_training_target_and_bad_cat_id_rejected(tmp_path, photo):
    app = app_for(tmp_path)
    with TestClient(app) as client:
        assert client.post("/api/analyses", files={"image": ("cat.png", photo, "image/png")}, data={"cat_id": "../bad"}).status_code == 422
        id = upload(client, photo, training_allowed="true")["id"]
        score(client, id, dict.fromkeys(ZERO, None))
        assert app.state.store.training_candidates() == []


def test_blob_preserves_first_blind_record_and_revocation_on_fresh_instance(tmp_path, photo, fake_blob):
    app = app_for(tmp_path / "one", store_factory=BlobStore, synchronous_inference=True, deployment="cloud")
    with TestClient(app) as client:
        id = upload(client, photo, training_allowed="true")["id"]
        first = score(client, id, {**ZERO, "eyes": 1})
        first_path = f"catpain/annotations/{id}/first.json"
        first_bytes = fake_blob.objects[first_path]
        second = score(client, id, {**ZERO, "eyes": 2})
        assert first_bytes == fake_blob.objects[first_path]
        assert second["annotation"]["saw_ai_before_scoring"] is True
        assert len([path for path in fake_blob.objects if path.startswith(f"catpain/annotations/{id}/")]) == 3
        assert len([path for path in fake_blob.objects if path.startswith(f"catpain/predictions/{id}/")]) == 1
        client.post(f"/api/analyses/{id}/training-permission", json={"allowed": False})
    fresh = app_for(tmp_path / "two", store_factory=BlobStore, synchronous_inference=True, deployment="cloud")
    with TestClient(fresh) as client:
        assert client.get(f"/api/analyses/{id}").json()["collection"]["training_allowed"] is False
        assert client.get(f"/api/analyses/{id}/report").json()["collection"]["training_allowed"] is False
        assert client.get(f"/api/analyses/{id}/report").json()["machines"] == first["machines"]
    immutable = [options for path, options in fake_blob.writes if "/annotations/" in path or "/predictions/" in path or "/training-permissions/" in path]
    assert immutable and all(options["allow_overwrite"] is False for options in immutable)


def test_old_sqlite_migration_keeps_records_without_implicit_permission(tmp_path):
    path = tmp_path / "catpain.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE analyses (id TEXT PRIMARY KEY, created_at TEXT, cat_name TEXT, width INTEGER, height INTEGER, status TEXT, ai_json TEXT, report_json TEXT, rule_version TEXT, updated_at TEXT)")
        db.execute("INSERT INTO analyses VALUES ('old','before','name',640,480,'ready',NULL,NULL,NULL,'before')")
    store = Store(tmp_path)
    row = store.get("old")
    assert row["cat_name"] == "name" and row["collection_json"] is None
    assert store.training_candidates() == []


def test_concurrent_sqlite_scores_have_only_one_first_blind_annotation(tmp_path, photo):
    app = app_for(tmp_path)
    with TestClient(app) as client:
        id = upload(client, photo)["id"]
    store = app.state.store
    barrier = Barrier(2)

    def submit(value):
        report = {"completed_at": f"2026-11-07T01:00:0{value}Z", "fgs": {"scores": {**ZERO, "eyes": value}}}
        barrier.wait(timeout=5)
        return store.set_report(id, report, "concurrency-test")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(submit, (1, 2)))
    assert sum(result["annotation"]["saw_ai_before_scoring"] is False for result in results) == 1
    history = store.annotation_history(id)
    assert len(history) == 2
    assert [item["saw_ai_before_scoring"] for item in history] == [False, True]
    assert history[1]["supersedes_annotation_id"] == history[0]["annotation_id"]


def test_blob_stale_submit_cannot_replace_atomic_first_annotation(tmp_path, photo, fake_blob, monkeypatch):
    app = app_for(tmp_path, store_factory=BlobStore, synchronous_inference=True, deployment="cloud")
    with TestClient(app) as client:
        id = upload(client, photo)["id"]
    store = app.state.store
    stale_row = dict(store.get(id))
    original_get = store.get
    monkeypatch.setattr(store, "get", lambda record_id: dict(stale_row))
    first = store.set_report(id, {"completed_at": "one", "fgs": {"scores": {**ZERO, "eyes": 1}}}, "test")
    second = store.set_report(id, {"completed_at": "two", "fgs": {"scores": {**ZERO, "eyes": 2}}}, "test")
    first_payload = json.loads(fake_blob.objects[f"catpain/annotations/{id}/first.json"])
    assert first_payload["annotation_id"] == first["annotation"]["annotation_id"]
    assert first_payload["scores"]["eyes"] == 1
    assert second["annotation"]["saw_ai_before_scoring"] is True
    assert second["annotation"]["supersedes_annotation_id"] == first["annotation"]["annotation_id"]
    monkeypatch.setattr(store, "get", original_get)


def test_export_script_uses_current_permission_and_only_first_weak_label(tmp_path, photo):
    data = tmp_path / "private-data"
    app = app_for(data)
    with TestClient(app) as client:
        id = upload(client, photo, cat_name="Never export this name", training_allowed="true")["id"]
        score(client, id, {**ZERO, "ears": 1})
        score(client, id, {**ZERO, "ears": 2})
        output = tmp_path / "export-one"
        script = Path(__file__).resolve().parents[2] / "scripts" / "export-training-data.py"
        result = subprocess.run([sys.executable, str(script), "--data-dir", str(data), "--output", str(output), "--copy-images"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        dataset = json.loads((output / "dataset.json").read_text())
        text = (output / "candidates.jsonl").read_text(encoding="utf-8")
        candidate = json.loads(text)
        assert dataset["count"] == 1 and dataset["images_copied"] is True
        assert candidate["annotation"]["scores"]["ears"] == 1
        assert candidate["weak_label"] is True and candidate["split"] == "unassigned"
        assert "private note" not in text and "Never export this name" not in text
        assert (output / "images" / f"{id}.jpg").is_file()
        client.post(f"/api/analyses/{id}/training-permission", json={"allowed": False})
        fresh = tmp_path / "export-two"
        result = subprocess.run([sys.executable, str(script), "--data-dir", str(data), "--output", str(fresh)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads((fresh / "dataset.json").read_text())["count"] == 0


def test_private_cloud_export_filters_all_pages_without_reading_rejected_photos(tmp_path, photo, fake_blob, monkeypatch):
    app = app_for(tmp_path / "cloud", store_factory=BlobStore, synchronous_inference=True, deployment="cloud")
    with TestClient(app) as client:
        denied = upload(client, photo)["id"]
        score(client, denied)
        withdrawn = upload(client, photo, training_allowed="true")["id"]
        score(client, withdrawn)
        client.post(f"/api/analyses/{withdrawn}/training-permission", json={"allowed": False})
        missing_label = upload(client, photo, training_allowed="true")["id"]
        accepted = upload(client, photo, training_allowed="true")["id"]
        score(client, accepted, {**ZERO, "ears": 1})
    pages_seen = []

    def paged_query(*, prefix, page_size):
        assert prefix == "catpain/analyses/" and page_size == 100
        for number, ids in enumerate(((denied, withdrawn), (missing_label, accepted))):
            pages_seen.append(number)
            for id in ids:
                yield SimpleNamespace(pathname=f"catpain/analyses/{id}.json")

    monkeypatch.setattr(fake_blob, "query_blobs", paged_query, raising=False)
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "export-training-data.py"
    spec = importlib.util.spec_from_file_location("training_export_test", script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setenv("BLOB_READ_WRITE_TOKEN", "fake-test-only-token")
    metadata_output = tmp_path / "metadata-export"
    monkeypatch.setattr(sys, "argv", [str(script_path), "--cloud", "--output", str(metadata_output)])
    fake_blob.reads.clear()
    assert script.main() == 0
    assert pages_seen == [0, 1]
    assert not any("/images/" in path for path in fake_blob.reads)
    records = [json.loads(line) for line in (metadata_output / "candidates.jsonl").read_text().splitlines()]
    assert [item["analysis_id"] for item in records] == [accepted]
    assert records[0]["weak_label"] is True
    image_output = tmp_path / "photo-export"
    monkeypatch.setattr(sys, "argv", [str(script_path), "--cloud", "--output", str(image_output), "--copy-images"])
    fake_blob.reads.clear()
    assert script.main() == 0
    image_reads = [path for path in fake_blob.reads if "/images/" in path]
    assert len(image_reads) == 1 and accepted in image_reads[0]
    assert len(list((image_output / "images").glob("*.jpg"))) == 1
    assert json.loads((image_output / "dataset.json").read_text())["source"] == "private_blob"


def test_export_rechecks_withdrawal_after_candidate_snapshot(tmp_path, photo, monkeypatch):
    data = tmp_path / "private-data"
    app = app_for(data)
    with TestClient(app) as client:
        id = upload(client, photo, training_allowed="true")["id"]
        score(client, id)
    store = app.state.store
    original_candidates = store.training_candidates

    def stale_candidates():
        candidates = original_candidates()
        assert len(candidates) == 1
        store.set_training_permission(id, False, "withdrawn-during-enumeration")
        return candidates

    monkeypatch.setattr(store, "training_candidates", stale_candidates)
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "export-training-data.py"
    spec = importlib.util.spec_from_file_location("training_export_withdrawal_test", script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "Store", lambda *_: store)
    output = tmp_path / "export"
    monkeypatch.setattr(sys, "argv", [str(script_path), "--data-dir", str(data), "--output", str(output), "--copy-images"])
    assert script.main() == 0
    assert json.loads((output / "dataset.json").read_text())["count"] == 0
    assert (output / "candidates.jsonl").read_text() == ""
    assert list((output / "images").iterdir()) == []
