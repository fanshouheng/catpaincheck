"""Vercel entry: durable private Blob storage, bounded scratch and CPU ONNX."""
from __future__ import annotations

import os
import tempfile

os.environ["CATPAIN_RUNTIME"] = "vercel"

from .dual_inference import analyze_image, get_model_status
from .cloud_storage import BlobStore
from .main import create_app

_scratch = tempfile.TemporaryDirectory(prefix="catpain-")

app = create_app(
    data_dir=_scratch.name,
    inference_fn=analyze_image,
    model_status_fn=get_model_status,
    store_factory=BlobStore,
    deployment="cloud",
    max_upload=4 * 1024 * 1024,
    synchronous_inference=True,
)
