import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import pytest
from vercel.blob import BlobError, BlobNotFoundError, BlobServiceNotAvailable

from app import cloud_storage
from app.cloud_storage import BlobStore


RECORD_ID = "a" * 32


class FakeBlob:
    """A stale CDN copy makes missing cache bypasses fail in tests."""

    def __init__(self):
        self.objects = {}
        self.cached = {}
        self.reads = []
        self.writes = []
        self.failure = None

    def put_blob(self, path, data, **options):
        if self.failure:
            raise self.failure
        assert options["access"] == "private"
        if not options["allow_overwrite"] and path in self.objects:
            raise BlobError("Object already exists")
        assert options["cache_control_max_age"] >= 60
        self.objects[path] = bytes(data)
        self.writes.append((path, options))
        return SimpleNamespace(url=f"https://test.private.blob.vercel-storage.com/{path}")

    def get_blob(self, target, **options):
        if self.failure:
            raise self.failure
        assert options["access"] == "private"
        self.reads.append(target)
        parsed = urlsplit(target)
        path = unquote(parsed.path.lstrip("/")) if parsed.scheme else target
        if path not in self.objects:
            raise BlobNotFoundError()
        if parsed.query == "cache=0":
            data = self.objects[path]
        else:
            data = self.cached.setdefault(path, self.objects[path])
        return SimpleNamespace(body=data)


@pytest.fixture
def fake_blob(monkeypatch):
    fake = FakeBlob()
    monkeypatch.setattr(cloud_storage, "blob", fake)
    monkeypatch.setattr(cloud_storage, "default_sync_credentials", lambda: SimpleNamespace(store_id="test"))
    return fake


def insert(store, id=RECORD_ID):
    (store.uploads / f"{id}.jpg").write_bytes(b"clean original jpeg")
    store.insert(id, "2026-11-06T00:00:00Z", "小橘", 800, 600)


def test_persists_across_instances_without_local_history(tmp_path, fake_blob):
    first = BlobStore(tmp_path / "first")
    insert(first)
    first.uploads.joinpath(f"{RECORD_ID}.jpg").unlink()
    fresh = BlobStore(tmp_path / "second")
    row = fresh.get(RECORD_ID)
    assert row["cat_name"] == "小橘"
    assert row["status"] == "processing"
    assert row["ai_json"] is None and row["report_json"] is None
    assert fresh.read_image(RECORD_ID, "image") == b"clean original jpeg"
    assert not list(fresh.uploads.iterdir())
    assert not list(fresh.crops.iterdir())
    assert not list(tmp_path.rglob("*.sqlite3"))


def test_ai_report_and_crop_survive_new_instance_and_stale_cdn(tmp_path, fake_blob):
    first = BlobStore(tmp_path / "first")
    insert(first)
    record_path = f"catpain/analyses/{RECORD_ID}.json"
    fake_blob.cached[record_path] = fake_blob.objects[record_path]
    first.crops.joinpath(f"{RECORD_ID}.jpg").write_bytes(b"cropped jpeg")
    ai = {"status": "ok", "label": "no_pain", "raw_score": 0.2}
    first.set_ai(RECORD_ID, ai, "2026-11-06T00:01:00Z")
    report = {"completed_at": "2026-11-06T00:02:00Z", "notes": "测试"}
    fresh = BlobStore(tmp_path / "second")
    fresh.set_report(RECORD_ID, report, "rules-v1")
    row = BlobStore(tmp_path / "third").get(RECORD_ID)
    assert row["status"] == "ready"
    assert json.loads(row["ai_json"]) == ai
    assert json.loads(row["report_json"]) == report
    assert row["rule_version"] == "rules-v1"
    assert row["updated_at"] == report["completed_at"]
    assert fresh.read_image(RECORD_ID, "crop") == b"cropped jpeg"
    assert all(target.endswith("?cache=0") for target in fake_blob.reads if ".json" in target)


def test_only_missing_objects_map_to_none(tmp_path, fake_blob):
    store = BlobStore(tmp_path)
    assert store.get(RECORD_ID) is None
    assert store.read_image(RECORD_ID, "image") is None
    fake_blob.failure = BlobServiceNotAvailable()
    with pytest.raises(BlobServiceNotAvailable):
        store.get(RECORD_ID)
    with pytest.raises(BlobServiceNotAvailable):
        store.read_image(RECORD_ID, "crop")
    with pytest.raises(BlobServiceNotAvailable):
        insert(store)


def test_cold_start_does_not_invalidate_other_instance_work(tmp_path, fake_blob):
    first = BlobStore(tmp_path / "first")
    insert(first)
    before = dict(fake_blob.objects)
    fresh = BlobStore(tmp_path / "new")
    fresh.recover_pending({"status": "unavailable"}, "later")
    assert fake_blob.objects == before
    assert fake_blob.reads == []


def test_invalid_object_targets_and_corrupt_record_are_errors(tmp_path, fake_blob):
    store = BlobStore(tmp_path)
    with pytest.raises(ValueError):
        store.get("../other")
    with pytest.raises(ValueError):
        store.read_image(RECORD_ID, "unknown")
    fake_blob.objects[f"catpain/analyses/{RECORD_ID}.json"] = b"not JSON"
    with pytest.raises(json.JSONDecodeError):
        store.get(RECORD_ID)
