"""Private Blob persistence for stateless Vercel function instances."""

from __future__ import annotations

import json
import hashlib
import uuid
import re
import threading
from pathlib import Path
from typing import Any, Literal

from vercel.blob import BlobNotFoundError, BlobError
from vercel.blob import sync as blob
from vercel.blob._internal.credentials import default_sync_credentials
from vercel.blob._internal.validation import construct_delivery_url

from .storage import training_candidate


_ID = re.compile(r"^[0-9a-f]{32}$")
_PREFIX = "catpain"


class BlobStore:
    """Keep records in private Blob; local directories are temporary workspaces.

    vercel-blob 0.5.1 has no public ``use_cache`` argument. Its URL reader does
    preserve query parameters, so record reads use the official Node SDK's
    ``useCache: false`` equivalent, ``?cache=0``. The pinned SDK credential
    resolver supplies only the store ID here; the normal SDK handles auth.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.uploads = self.data_dir / "uploads"
        self.crops = self.data_dir / "crops"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.crops.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _id(id: str) -> str:
        if not isinstance(id, str) or not _ID.fullmatch(id):
            raise ValueError("Invalid analysis ID")
        return id

    def _record_path(self, id: str) -> str:
        return f"{_PREFIX}/analyses/{self._id(id)}.json"

    def _image_path(self, id: str, kind: Literal["image", "crop"]) -> str:
        if kind not in ("image", "crop"):
            raise ValueError("Invalid image kind")
        directory = "images" if kind == "image" else "crops"
        return f"{_PREFIX}/{directory}/{self._id(id)}.jpg"

    @staticmethod
    def _json(value: dict) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    @staticmethod
    def _put(path: str, data: bytes, content_type: str, *, immutable: bool = False) -> None:
        blob.put_blob(
            path,
            data,
            access="private",
            allow_overwrite=not immutable,
            cache_control_max_age=60,
            content_type=content_type,
        )

    def _write_record(self, row: dict) -> None:
        self._put(self._record_path(row["id"]), self._json(row).encode("utf-8"), "application/json")

    def _read_json(self, path: str) -> dict | None:
        credentials = default_sync_credentials()
        url = construct_delivery_url(credentials.store_id, path, "private") + "?cache=0"
        try:
            data = blob.get_blob(url, access="private").body
        except BlobNotFoundError:
            return None
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("Invalid persisted JSON object")
        return result

    def _history(self, kind: str, id: str, payload: dict, *, key: str | None = None) -> str:
        event_id = key or uuid.uuid4().hex
        path = f"{_PREFIX}/{kind}/{self._id(id)}/{event_id}.json"
        self._put(path, self._json(payload).encode("utf-8"), "application/json", immutable=True)
        return path

    def _collection_path(self, id: str) -> str:
        return f"{_PREFIX}/collections/{self._id(id)}.json"

    def recover_pending(self, result: dict, updated_at: str) -> None:
        # Other instances can still be processing their own records. Never
        # enumerate or invalidate shared pending work during a cold start.
        return None

    def insert(self, id: str, created_at: str, cat_name: str, width: int, height: int, collection: dict | None = None) -> None:
        self._id(id)
        image = (self.uploads / f"{id}.jpg").read_bytes()
        row = {
            "id": id,
            "created_at": created_at,
            "cat_name": cat_name,
            "width": width,
            "height": height,
            "status": "processing",
            "ai_json": None,
            "report_json": None,
            "rule_version": None,
            "updated_at": created_at,
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "collection_json": self._json(collection) if collection is not None else None,
        }
        with self._lock:
            # Publish the record only after its image is durably available.
            self._put(self._image_path(id, "image"), image, "image/jpeg")
            if collection is not None:
                self._history("training-permissions", id, collection)
                self._put(self._collection_path(id), self._json(collection).encode("utf-8"), "application/json")
            self._write_record(row)

    def get(self, id: str) -> dict[str, Any] | None:
        row = self._read_json(self._record_path(id))
        if row is None:
            return None
        if row.get("id") != id:
            raise ValueError("Invalid persisted analysis record")
        # Consent has its own authoritative object: a concurrent report write
        # must not re-enable training after a withdrawal.
        collection = self._read_json(self._collection_path(id))
        if collection is not None:
            row["collection_json"] = self._json(collection)
        return row

    def set_ai(self, id: str, result: dict, updated_at: str) -> None:
        with self._lock:
            row = self.get(id)
            if row is None:
                raise FileNotFoundError("Analysis record does not exist")
            crop = self.crops / f"{self._id(id)}.jpg"
            if crop.is_file():
                self._put(self._image_path(id, "crop"), crop.read_bytes(), "image/jpeg")
            self._history("predictions", id, {"created_at": updated_at, "result": result})
            row.update(ai_json=self._json(result), status="ready", updated_at=updated_at)
            self._write_record(row)

    def set_report(self, id: str, report: dict, rule_version: str) -> dict:
        with self._lock:
            row = self.get(id)
            if row is None:
                raise FileNotFoundError("Analysis record does not exist")
            if "fgs" in report:
                previous = json.loads(row["report_json"]).get("annotation") if row.get("report_json") else None
                annotation = {
                    "annotation_id": uuid.uuid4().hex, "submitted_at": report["completed_at"],
                    "label_source": "user_fgs", "rubric_version": "fgs-five-au-v1",
                    "saw_ai_before_scoring": bool(row.get("report_json")),
                    "supersedes_annotation_id": previous.get("annotation_id") if previous else None,
                }
                payload = {**annotation, "scores": report["fgs"]["scores"]}
                if not row.get("report_json"):
                    first_path = f"{_PREFIX}/annotations/{self._id(id)}/first.json"
                    try:
                        self._put(first_path, self._json(payload).encode("utf-8"), "application/json", immutable=True)
                    except BlobError:
                        # A different function may have atomically claimed the
                        # first slot. Never replace its blind annotation.
                        first = self._read_json(first_path)
                        if first is None:
                            raise
                        annotation["saw_ai_before_scoring"] = True
                        annotation["supersedes_annotation_id"] = first["annotation_id"]
                        payload = {**annotation, "scores": report["fgs"]["scores"]}
                self._history("annotations", id, payload, key=annotation["annotation_id"])
                report = {**report, "annotation": annotation,
                          "collection": json.loads(row["collection_json"]) if row.get("collection_json") else None}
            row.update(report_json=self._json(report), rule_version=rule_version, updated_at=report["completed_at"])
            self._write_record(row)
        return report

    def set_training_permission(self, id: str, allowed: bool, updated_at: str) -> dict:
        with self._lock:
            row = self.get(id)
            if row is None:
                raise FileNotFoundError("Analysis record does not exist")
            collection = json.loads(row["collection_json"]) if row.get("collection_json") else {
                "cat_id": None, "permission_version": "photo-fgs-training-v1",
            }
            collection.update(training_allowed=allowed, permission_at=updated_at)
            self._history("training-permissions", id, collection)
            # This object is intentionally mutable; immutable permission events
            # preserve history while current eligibility is revocable.
            self._put(self._collection_path(id), self._json(collection).encode("utf-8"), "application/json")
            row.update(collection_json=self._json(collection), updated_at=updated_at)
            self._write_record(row)
        return collection

    def training_candidates(self):
        """Operator-only iterator; SDK follows catalog pagination lazily.

        This is deliberately not reachable through a public HTTP route. Images
        are never read while listing/filtering candidates.
        """
        prefix = f"{_PREFIX}/analyses/"
        for entry in blob.query_blobs(prefix=prefix, page_size=100):
            path = entry.pathname
            if not path.startswith(prefix) or not path.endswith(".json"):
                continue
            id = path[len(prefix):-5]
            if not _ID.fullmatch(id):
                continue
            row = self.get(id)
            if row is None or not row.get("collection_json"):
                continue
            collection = json.loads(row["collection_json"])
            if collection.get("training_allowed") is not True:
                continue
            first = self._read_json(f"{_PREFIX}/annotations/{id}/first.json")
            candidate = training_candidate(row, collection, first)
            if candidate is not None:
                yield candidate

    def read_image(self, id: str, kind: Literal["image", "crop"]) -> bytes | None:
        try:
            return blob.get_blob(self._image_path(id, kind), access="private").body
        except BlobNotFoundError:
            return None
