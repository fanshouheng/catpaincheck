#!/usr/bin/env python3
"""Download and verify pinned cat-localization models and their license, without photos."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATFLW_COMMIT = "8404a7a9a98ba7752a48f55dfba0f957381ee775"
CATFLW_BASE = f"https://huggingface.co/hugocornellier/cat-face-landmarks/resolve/{CATFLW_COMMIT}"


@dataclass(frozen=True)
class Artifact:
    name: str
    size: int
    sha256: str
    url: str


ARTIFACTS = (
    Artifact(
        "cat_face_landmarks_full.tflite",
        11_549_508,
        "1a8b09c4b30a8211de89979244ab90e595935c27fdba1e64b433990f8a3a5b66",
        f"{CATFLW_BASE}/cat_face_landmarks_full.tflite",
    ),
    Artifact(
        "cat_face_localizer.tflite",
        17_230_756,
        "a9621320ba8ca39b706c82649737c3cb72ef10ba37200ee91e84346600bee0c7",
        f"{CATFLW_BASE}/cat_face_localizer.tflite",
    ),
    Artifact(
        "catflw-LICENSE.txt",
        1_486,
        "20deafce52ea778f87202f986db661d0e2f1d53983500692b976c5cd49acfa0a",
        f"{CATFLW_BASE}/LICENSE",
    ),
    Artifact(
        "coco-efficientdet-lite0.tflite",
        4_563_519,
        "2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b",
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/"
        "task_library/object_detection/android/"
        "lite-model_efficientdet_lite0_detection_metadata_1.tflite",
    ),
)


def verify(path: Path, artifact: Artifact) -> bool:
    if not path.is_file() or path.stat().st_size != artifact.size:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == artifact.sha256


def download(directory: Path, artifact: Artifact) -> None:
    target = directory / artifact.name
    partial = target.with_name(target.name + ".download")
    try:
        print(
            f"Downloading {artifact.name} ({artifact.size / 1_000_000:.1f} MB)...",
            flush=True,
        )
        request = urllib.request.Request(
            artifact.url, headers={"User-Agent": "CatPainCheck-Setup/1.0"}
        )
        started = time.monotonic()
        last_progress = 0.0
        count = 0
        with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as handle:
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > artifact.size:
                    raise ValueError("Downloaded file is larger than the pinned artifact.")
                handle.write(block)
                now = time.monotonic()
                if now - last_progress >= 3:
                    print(
                        f"  {count / artifact.size * 100:5.1f}% "
                        f"({count / 1_000_000:.1f} MB)",
                        flush=True,
                    )
                    last_progress = now
        if not verify(partial, artifact):
            raise ValueError(f"SHA256/size verification failed: {artifact.name}")
        os.replace(partial, target)
        print(f"Installed and verified in {time.monotonic() - started:.1f}s: {target}")
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify only; do not download.")
    args = parser.parse_args()
    directory = Path(os.environ.get("CATPAIN_MODELS_DIR", ROOT / "models"))
    failed = False
    for artifact in ARTIFACTS:
        target = directory / artifact.name
        try:
            if verify(target, artifact):
                print(f"Verified: {target}")
                continue
            if args.check:
                print(f"Missing, incomplete, or failed SHA256 verification: {target}", file=sys.stderr)
                failed = True
                continue
            directory.mkdir(parents=True, exist_ok=True)
            download(directory, artifact)
        except (OSError, ValueError, urllib.error.URLError) as error:
            print(f"Localization setup failed for {artifact.name}: {error}", file=sys.stderr)
            failed = True
    if failed:
        return 1
    print("All three localization models and the CatFLW license are verified.")
    print("CatFLW weights: CC BY-NC 4.0; see catflw-LICENSE.txt for attribution and terms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
