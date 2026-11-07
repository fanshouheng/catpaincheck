from __future__ import annotations

import json
import hashlib
import uuid
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def training_candidate(row: dict, collection: dict, annotation: dict | None) -> dict | None:
    """Build a weak-label candidate without names, notes or model-as-truth labels."""
    if collection.get("training_allowed") is not True or annotation is None:
        return None
    if annotation.get("saw_ai_before_scoring") is not False or all(value is None for value in annotation.get("scores", {}).values()):
        return None
    ai = json.loads(row["ai_json"]) if row.get("ai_json") else {}
    machines = ai.get("machines") or {}
    # Explicitly whitelist annotation fields; unrelated private metadata cannot
    # leak into a training export if the report format later grows.
    label = {key: annotation.get(key) for key in (
        "annotation_id", "submitted_at", "label_source", "rubric_version",
        "saw_ai_before_scoring", "supersedes_annotation_id", "scores",
    )}
    return {
        "analysis_id": row["id"], "image_sha256": row.get("image_sha256"),
        "image_filename": f"{row['id']}.jpg", "cat_id": collection.get("cat_id"),
        "grouping_status": "browser_provided" if collection.get("cat_id") else "unknown",
        "permission": {key: collection.get(key) for key in ("cat_id", "training_allowed", "permission_version", "permission_at")},
        "annotation": label,
        "weak_label": True, "label_source": "user_fgs", "review_status": "unreviewed", "split": "unassigned",
        "quality": {"localization_status": (ai.get("localization") or {}).get("status", "unknown"),
                    "localization_reason": (ai.get("localization") or {}).get("reason", "unknown")},
        "model_versions": {key: value.get("model_version", "unknown") for key, value in machines.items() if isinstance(value, dict)},
        "preprocess_versions": {key: value.get("preprocess_version", "unknown") for key, value in machines.items() if isinstance(value, dict)},
    }


class Store:
    """Small local SQLite store; every worker gets a short-lived connection."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.uploads = data_dir / "uploads"
        self.crops = data_dir / "crops"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.crops.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "catpain.sqlite3"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    cat_name TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    ai_json TEXT,
                    report_json TEXT,
                    rule_version TEXT,
                    updated_at TEXT NOT NULL
                )
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(analyses)")}
            for name in ("collection_json", "image_sha256"):
                if name not in columns:
                    db.execute(f"ALTER TABLE analyses ADD COLUMN {name} TEXT")
            db.execute("""CREATE TABLE IF NOT EXISTS annotations (
                annotation_id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL,
                submitted_at TEXT NOT NULL, payload_json TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS predictions (
                event_id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL,
                created_at TEXT NOT NULL, payload_json TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS training_permissions (
                event_id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL,
                created_at TEXT NOT NULL, payload_json TEXT NOT NULL
            )""")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def recover_pending(self, result: dict, updated_at: str) -> None:
        """A server restart cannot resume its old in-process background tasks."""
        with self.connection() as db:
            db.execute(
                "UPDATE analyses SET ai_json=?,status='ready',updated_at=? WHERE status='processing'",
                (json.dumps(result, ensure_ascii=False, allow_nan=False), updated_at),
            )

    def insert(self, id: str, created_at: str, cat_name: str, width: int, height: int, collection: dict | None = None) -> None:
        with self.connection() as db:
            path = self.uploads / f"{id}.jpg"
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            db.execute(
                "INSERT INTO analyses (id,created_at,cat_name,width,height,status,updated_at,collection_json,image_sha256) VALUES (?,?,?,?,?,?,?,?,?)",
                (id, created_at, cat_name, width, height, "processing", created_at,
                 self._json(collection) if collection is not None else None, digest),
            )
            if collection is not None:
                db.execute("INSERT INTO training_permissions VALUES (?,?,?,?)",
                           (uuid.uuid4().hex, id, created_at, self._json(collection)))

    def get(self, id: str) -> dict[str, Any] | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM analyses WHERE id=?", (id,)).fetchone()
        return dict(row) if row else None

    def set_ai(self, id: str, result: dict, updated_at: str) -> None:
        with self.connection() as db:
            db.execute("INSERT INTO predictions VALUES (?,?,?,?)",
                       (uuid.uuid4().hex, id, updated_at, self._json(result)))
            db.execute(
                "UPDATE analyses SET ai_json=?,status='ready',updated_at=? WHERE id=?",
                (self._json(result), updated_at, id),
            )

    @staticmethod
    def _json(value: dict) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    def set_report(self, id: str, report: dict, rule_version: str) -> dict:
        with self.connection() as db:
            # Acquire the write transaction before checking for first/revised
            # annotations. The original blind score must never be overwritten.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT report_json,collection_json FROM analyses WHERE id=?", (id,)).fetchone()
            if row is None:
                raise FileNotFoundError("Analysis record does not exist")
            if "fgs" in report:
                previous = db.execute("SELECT annotation_id FROM annotations WHERE analysis_id=? ORDER BY rowid DESC LIMIT 1", (id,)).fetchone()
                annotation = {
                    "annotation_id": uuid.uuid4().hex,
                    "submitted_at": report["completed_at"],
                    "label_source": "user_fgs", "rubric_version": "fgs-five-au-v1",
                    "saw_ai_before_scoring": bool(previous or row["report_json"]),
                    "supersedes_annotation_id": previous["annotation_id"] if previous else None,
                }
                payload = {**annotation, "scores": report["fgs"]["scores"]}
                db.execute("INSERT INTO annotations VALUES (?,?,?,?)",
                           (annotation["annotation_id"], id, annotation["submitted_at"], self._json(payload)))
                report = {**report, "annotation": annotation,
                          "collection": json.loads(row["collection_json"]) if row["collection_json"] else None}
            db.execute(
                "UPDATE analyses SET report_json=?,rule_version=?,updated_at=? WHERE id=?",
                (self._json(report), rule_version, report["completed_at"], id),
            )
        return report

    def set_training_permission(self, id: str, allowed: bool, updated_at: str) -> dict:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT collection_json FROM analyses WHERE id=?", (id,)).fetchone()
            if row is None:
                raise FileNotFoundError("Analysis record does not exist")
            collection = json.loads(row[0]) if row[0] else {
                "cat_id": None, "permission_version": "photo-fgs-training-v1",
            }
            collection.update(training_allowed=allowed, permission_at=updated_at)
            db.execute("INSERT INTO training_permissions VALUES (?,?,?,?)",
                       (uuid.uuid4().hex, id, updated_at, self._json(collection)))
            db.execute("UPDATE analyses SET collection_json=?,updated_at=? WHERE id=?",
                       (self._json(collection), updated_at, id))
        return collection

    def annotation_history(self, id: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT payload_json FROM annotations WHERE analysis_id=? ORDER BY rowid", (id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def training_candidates(self) -> list[dict]:
        """Private local export source; never exposed as a public API."""
        candidates = []
        with self.connection() as db:
            rows = db.execute("SELECT * FROM analyses WHERE collection_json IS NOT NULL").fetchall()
            for row in rows:
                collection = json.loads(row["collection_json"])
                if collection.get("training_allowed") is not True:
                    continue
                first = db.execute("SELECT payload_json FROM annotations WHERE analysis_id=? ORDER BY rowid LIMIT 1", (row["id"],)).fetchone()
                if first is None:
                    continue
                candidate = training_candidate(dict(row), collection, json.loads(first[0]))
                if candidate is not None:
                    candidates.append(candidate)
        return candidates
