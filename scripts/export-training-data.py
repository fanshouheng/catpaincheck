#!/usr/bin/env python3
"""Export opted-in, first-blind user FGS annotations as weak training candidates.

Run immediately before training so withdrawn records are excluded. This does
not create verified pain labels or assign any candidate to a held-out test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.storage import Store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing exports are never overwritten")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--cloud", action="store_true", help="Private operator export from Blob; requires BLOB_READ_WRITE_TOKEN in the environment")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new version directory")
    if args.cloud:
        if not os.environ.get("BLOB_READ_WRITE_TOKEN"):
            parser.error("Set BLOB_READ_WRITE_TOKEN securely in the environment before --cloud export")
        from app.cloud_storage import BlobStore
        scratch = tempfile.TemporaryDirectory(prefix="catpain-export-")
        store = BlobStore(Path(scratch.name))
    else:
        if not (args.data_dir / "catpain.sqlite3").is_file():
            parser.error("Local database not found; no data was exported")
        store = Store(args.data_dir)
    candidates = []
    args.output.mkdir(parents=True)
    if args.copy_images:
        (args.output / "images").mkdir()
    for record in store.training_candidates():
        # Candidate enumeration may have started before a withdrawal. Recheck
        # current consent immediately before exporting metadata or photo bytes.
        current = store.get(record["analysis_id"])
        permission = json.loads(current["collection_json"]) if current and current.get("collection_json") else {}
        if permission.get("training_allowed") is not True:
            continue
        record["permission"] = {key: permission.get(key) for key in record["permission"]}
        if args.cloud:
            # Consent and the first blind annotation are checked before any
            # image download; metadata-only export reads no photo bodies.
            if args.copy_images:
                raw = store.read_image(record["analysis_id"], "image")
                if raw is None or hashlib.sha256(raw).hexdigest() != record["image_sha256"]:
                    continue
                (args.output / "images" / record["image_filename"]).write_bytes(raw)
        else:
            source = store.uploads / record["image_filename"]
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != record["image_sha256"]:
                continue
            if args.copy_images:
                shutil.copyfile(source, args.output / "images" / record["image_filename"])
        candidates.append(record)
    manifest = {
        "dataset_version": uuid.uuid4().hex,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "label_quality": "unreviewed_user_weak_labels",
        "split_policy": "unassigned; group by cat and duplicate identity before splitting",
        "permission_policy": "regenerate export immediately before training to honor current withdrawals",
        "count": len(candidates), "images_copied": args.copy_images, "source": "private_blob" if args.cloud else "local_sqlite",
    }
    (args.output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (args.output / "candidates.jsonl").open("w", encoding="utf-8") as output:
        for record in candidates:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    if args.cloud:
        scratch.cleanup()
    print(f"Exported {len(candidates)} weak-label candidates to {args.output}")
    print("No model was trained; candidates remain unassigned to training/validation/test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
