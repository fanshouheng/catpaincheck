#!/usr/bin/env python3
"""Download pinned KittyCheck research weights; never uploads photographs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.kittycheck import FILES, MODEL_VERSION, PREPROCESS_VERSION, SOURCE_COMMIT

SOURCE_REPOSITORY = "https://github.com/rmcaoile/KittyCheck"
BASE = f"https://raw.githubusercontent.com/rmcaoile/KittyCheck/{SOURCE_COMMIT}/assets/models/"


def verified(path: Path, size: int, sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == sha256


def manifest() -> dict:
    return {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "upstream_inference": SOURCE_REPOSITORY + "/blob/" + SOURCE_COMMIT + "/lib/services/ai_scoring_service.dart",
        "research_report": SOURCE_REPOSITORY + "/blob/" + SOURCE_COMMIT + "/Journal_Paper_CMSC190_RMCaoile.pdf",
        "license": "Upstream repository does not specify a LICENSE. Public download and attribution are not a license grant.",
        "upstream_report": "Undergraduate CMSC190 project; source-reported results are not site validation.",
        "runtime": "ai-edge-litert 2.3.0 CPU; models loaded from verified bytes",
        "input": "RGB float32 NHWC 1x224x224x3; regressor /255; five scorers ImageNet mean/std",
        "output": "Five independent 0/1/2 predictions from three raw logits; unavailable items are null, never zero.",
        "adapter_changes": [
            "Independent COCO cat gate is required before regression.",
            "Does not read the upstream development-only fixed training-sample landmark JSON.",
            "Invalid/missing region or scorer returns NA instead of the upstream zero fallback.",
            "Python NumPy implementation of Dart linear resizing/rotation; no claim of bit-identical Dart runtime parity.",
            "Head input preserves the full image with a red face rectangle and black letterbox.",
        ],
        "files": [{"filename": name, "size": size, "sha256": digest, "url": BASE + name}
                  for name, (size, digest) in FILES.items()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify only, with no downloads.")
    args = parser.parse_args()
    directory = Path(os.environ.get("CATPAIN_MODELS_DIR", ROOT / "models")) / "kittycheck"
    if not args.check:
        directory.mkdir(parents=True, exist_ok=True)
    try:
        for name, (size, digest) in FILES.items():
            target = directory / name
            if verified(target, size, digest):
                print(f"Verified: {name}", flush=True)
                continue
            if args.check:
                print(f"Missing or invalid model: {name}", file=sys.stderr)
                return 1
            partial = target.with_name(name + ".download")
            try:
                print(f"Downloading {name} ({size / 1_000_000:.1f} MB)...", flush=True)
                request = urllib.request.Request(BASE + name, headers={"User-Agent": "CatPainCheck-Research/1.0"})
                with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as output:
                    count = 0
                    while block := response.read(1024 * 1024):
                        count += len(block)
                        if count > size:
                            raise ValueError("Download exceeds pinned file size")
                        output.write(block)
                if not verified(partial, size, digest):
                    raise ValueError("Model failed pinned SHA256/size verification")
                os.replace(partial, target)
            finally:
                partial.unlink(missing_ok=True)
        if not args.check:
            (directory / "manifest.json").write_text(json.dumps(manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("All seven KittyCheck weights verified. Upstream has no explicit LICENSE; see manifest.json.")
        return 0
    except (OSError, ValueError, urllib.error.URLError) as error:
        print("KittyCheck setup failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
