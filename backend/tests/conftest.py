from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import create_app


@pytest.fixture
def photo() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (640, 480), "#ccbea6").save(stream, format="PNG")
    return stream.getvalue()


def prediction(label="pain", score=0.8):
    # Test fixture only: this deliberately does not assert diagnostic accuracy.
    return {
        "status": "ok", "label": label, "raw_score": score, "threshold": 0.4,
        "model_version": "test-fixture", "note": "仅用于接口测试的固定结果", "bbox": [10, 10, 200, 200],
    }


@pytest.fixture
def client(tmp_path):
    def fake_analysis(source, crop):
        Image.open(source).crop((10, 10, 200, 200)).save(crop)
        return prediction()

    application = create_app(
        data_dir=tmp_path,
        inference_fn=fake_analysis,
        model_status_fn=lambda: {"ready": True, "detail": "Test fixture", "version": "test-fixture"},
    )
    with TestClient(application) as test_client:
        yield test_client
