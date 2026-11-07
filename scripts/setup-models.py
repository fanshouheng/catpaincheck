#!/usr/bin/env python3
"""Download and verify the pinned upstream research classifier, without photos."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "b48eeee42596e26537fe4cc15bdf7a7954d2dac6"
NAME = "cat-pr-rn18-adamW-FL.pth"
SIZE = 112_376_644
SHA256 = "4ace5922b92e8c89c14b4d275cd41d4b1b9884de14bf8c14bac41e32c8c73581"
URL = f"https://media.githubusercontent.com/media/tongkasuyo/cat-pain-recog/{COMMIT}/{NAME}"


def verify(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == SHA256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify only; do not download.")
    args = parser.parse_args()
    directory = Path(os.environ.get("CATPAIN_MODELS_DIR", ROOT / "models"))
    target = directory / NAME
    if verify(target):
        print(f"Model verified: {target}")
        return 0
    if args.check:
        print("Model is missing, incomplete, or failed SHA256 verification.", file=sys.stderr)
        return 1
    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".download")
    try:
        print(f"Downloading pinned model ({SIZE / 1_000_000:.1f} MB)...", flush=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "CatPainCheck-Setup/1.0"})
        started = time.monotonic()
        last_progress = 0.0
        count = 0
        with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as handle:
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > SIZE:
                    raise ValueError("Downloaded file is larger than the pinned model.")
                handle.write(block)
                now = time.monotonic()
                if now - last_progress >= 3:
                    print(f"  {count / SIZE * 100:5.1f}% ({count / 1_000_000:.1f} MB)", flush=True)
                    last_progress = now
        if not verify(partial):
            raise ValueError("Model SHA256/size verification failed; no model was installed.")
        os.replace(partial, target)
        print(f"Model installed and verified in {time.monotonic() - started:.1f}s: {target}")
        print("Source: https://github.com/tongkasuyo/cat-pain-recog")
        print("Upstream has no explicit project license; no diagnostic accuracy is implied.")
        return 0
    except (OSError, ValueError, urllib.error.URLError) as error:
        partial.unlink(missing_ok=True)
        print(f"Model setup failed: {error}", file=sys.stderr)
        print("FGS manual scoring can still work while the AI model is unavailable.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
