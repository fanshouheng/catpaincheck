from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .schemas import AiResult, AnalysisMeta, CollectionInfo, ImageSize, Localization, Report, ScoreSubmission, TrainingPermission
from .scoring import DUAL_RULE_VERSION, RULE_VERSION, calculate_fgs, combine_results
from .storage import Store


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD = 12 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_SIDE = 2560
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
logger = logging.getLogger("catpain")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unavailable(reason: str, status: str = "unavailable") -> AiResult:
    return AiResult(status=status, note=reason, model_version="unavailable")


def local_model_status() -> dict:
    try:
        from .dual_inference import get_model_status

        return get_model_status()
    except Exception:
        logger.exception("Model status lookup failed")
        return {"ready": False, "detail": "模型尚未准备就绪；仍可完成 FGS 人工评分。", "version": "unavailable"}


def local_analyze(image_path: Path, crop_path: Path) -> dict:
    try:
        from .dual_inference import analyze_image

        return analyze_image(image_path, crop_path)
    except ImportError:
        return unavailable("本地模型依赖尚未安装，未生成 AI 判断。可先完成 FGS 人工评分。").model_dump()


def sanitize_image(raw: bytes, destination: Path, max_bytes: int | None = None) -> tuple[int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as check:
                if check.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(415, "请上传 JPG、PNG 或 WebP 格式的照片。")
                if check.width * check.height > MAX_PIXELS:
                    raise HTTPException(422, "照片分辨率过高，请缩小到 2500 万像素以内后重试。")
                if check.width < 64 or check.height < 64:
                    raise HTTPException(422, "照片太小，请上传至少 64×64 像素的清晰照片。")
                if getattr(check, "is_animated", False):
                    raise HTTPException(422, "请上传静态照片；第一版暂不处理动图。")
                check.verify()
            with Image.open(io.BytesIO(raw)) as source:
                oriented = ImageOps.exif_transpose(source)
                rgba = oriented.convert("RGBA")
                clean = Image.new("RGB", rgba.size, (255, 255, 255))
                clean.paste(rgba, mask=rgba.getchannel("A"))
                clean.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
                clean.save(destination, format="JPEG", quality=94, optimize=True)
                if max_bytes is not None:
                    for quality in (85, 75, 65, 55, 45):
                        if destination.stat().st_size <= max_bytes:
                            break
                        clean.save(destination, format="JPEG", quality=quality, optimize=True)
                    if destination.stat().st_size > max_bytes:
                        raise HTTPException(422, "照片处理后仍过大，请缩小后重新上传。")
                return clean.width, clean.height
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(422, "照片像素数量过大，请缩小后重新上传。") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise HTTPException(422, "无法读取这张照片，文件可能已损坏，请换一张重试。") from None


def create_app(
    data_dir: Path | str | None = None,
    inference_fn: Callable[[Path, Path], dict] | None = None,
    model_status_fn: Callable[[], dict] | None = None,
    store_factory: Callable = Store,
    deployment: str = "local",
    max_upload: int = MAX_UPLOAD,
    synchronous_inference: bool = False,
) -> FastAPI:
    application = FastAPI(title="猫咪疼痛观察站", version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)
    store = store_factory(Path(data_dir or os.environ.get("CATPAIN_DATA_DIR", PROJECT_ROOT / "data")))
    store.recover_pending(
        unavailable("服务在分析期间重启，本次没有 AI 判断；可保留人工评分或重新上传照片。").model_dump(),
        now(),
    )
    inference_lock = threading.Lock()
    analyze = inference_fn or local_analyze
    model_status = model_status_fn or local_model_status
    application.state.store = store
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.middleware("http")
    async def response_headers(request: Request, call_next):
        length = request.headers.get("content-length")
        if request.url.path == "/api/analyses" and length:
            try:
                too_large = int(length) > max_upload + 256 * 1024
            except ValueError:
                return JSONResponse({"detail": "上传请求格式不正确。"}, status_code=400)
            if too_large:
                return JSONResponse({"detail": f"照片不能超过 {max_upload // (1024 * 1024)} MB，请缩小后重试。"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        fields = []
        for item in error.errors():
            location = ".".join(str(part) for part in item["loc"])
            fields.append({"field": location, "type": item["type"]})
        message = (
            "请选择一张 JPG、PNG 或 WebP 照片后上传。"
            if request.url.path == "/api/analyses"
            else "请完整填写五项评分；每项只能选 0、1、2 或“看不清”，备注不超过 2000 字。"
        )
        return JSONResponse(
            {"detail": message, "fields": fields},
            status_code=422,
        )

    def lookup(id: str) -> dict:
        if not ID_PATTERN.fullmatch(id):
            raise HTTPException(404, "未找到这次观察记录，请重新上传照片。")
        row = store.get(id)
        if row is None:
            raise HTTPException(404, "未找到这次观察记录，请重新上传照片。")
        return row

    def metadata(row: dict) -> AnalysisMeta:
        localization = None
        if row["status"] == "processing":
            localization = Localization(status="processing", reason="processing", note="正在定位猫脸。")
        elif row.get("ai_json"):
            # Whitelist geometry rather than serializing any part of AiResult:
            # pain labels and classifier scores remain hidden until submission.
            payload = json.loads(row["ai_json"]).get("localization")
            if payload is not None:
                localization = Localization.model_validate(payload)
        return AnalysisMeta(
            id=row["id"], created_at=row["created_at"], cat_name=row["cat_name"],
            image_url=f'/api/analyses/{row["id"]}/image', status=row["status"],
            image=ImageSize(width=row["width"], height=row["height"]),
            message="照片已收到，可以先完成 FGS 评分。" if row["status"] == "processing" else "分析任务已结束；完成 FGS 评分后查看两份结果。",
            localization=localization,
            collection=CollectionInfo.model_validate_json(row["collection_json"]) if row.get("collection_json") else None,
        )

    def process(id: str) -> None:
        crop_path = store.crops / f"{id}.jpg"
        try:
            with inference_lock:
                prediction = AiResult.model_validate(analyze(store.uploads / f"{id}.jpg", crop_path))
            prediction.crop_url = f"/api/analyses/{id}/crop" if crop_path.is_file() else None
        except Exception:
            logger.exception("Analysis failed for record %s", id)
            prediction = unavailable("本次 AI 分析未能完成，未生成疼痛判断。可以保留人工 FGS 评分并重新尝试。", "error")
        store.set_ai(id, prediction.model_dump(mode="json"), now())

    @application.get("/api/health")
    async def health():
        result = await run_in_threadpool(model_status)
        return {"status": "ok", "service": "catpaincheck", "model": result,
                "deployment": deployment, "max_upload_bytes": max_upload}

    @application.post("/api/analyses", status_code=202, response_model=AnalysisMeta)
    async def upload(
        background_tasks: BackgroundTasks,
        image: UploadFile = File(...),
        cat_name: str = Form(default=""),
        cat_id: str = Form(default=""),
        training_allowed: bool = Form(default=False),
    ):
        normalized_cat_id = None
        if cat_id:
            try:
                normalized_cat_id = uuid.UUID(cat_id).hex
            except (ValueError, AttributeError):
                raise HTTPException(422, "猫咪标识无效，请刷新后重试。") from None
        if len(cat_name) > 80:
            raise HTTPException(422, "猫咪名字请控制在 80 个字以内。")
        raw = await image.read(max_upload + 1)
        await image.close()
        if not raw:
            raise HTTPException(422, "照片内容为空，请重新选择。")
        if len(raw) > max_upload:
            raise HTTPException(413, f"照片不能超过 {max_upload // (1024 * 1024)} MB，请缩小后重试。")
        id = uuid.uuid4().hex
        path = store.uploads / f"{id}.jpg"
        try:
            width, height = await run_in_threadpool(sanitize_image, raw, path, max_upload if deployment == "cloud" else None)
            created = now()
            collection = CollectionInfo(cat_id=normalized_cat_id, training_allowed=training_allowed,
                                        permission_at=created).model_dump(mode="json")
            await run_in_threadpool(store.insert, id, created, cat_name.strip(), width, height, collection)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        if synchronous_inference:
            try:
                await run_in_threadpool(process, id)
            finally:
                # Cloud copies are durable; local scratch must stay bounded on warm instances.
                path.unlink(missing_ok=True)
                (store.crops / f"{id}.jpg").unlink(missing_ok=True)
        else:
            background_tasks.add_task(process, id)
        return metadata(await run_in_threadpool(lookup, id))

    @application.get("/api/analyses/{id}", response_model=AnalysisMeta)
    def status(id: str):
        return metadata(lookup(id))

    @application.get("/api/analyses/{id}/image")
    def original(id: str):
        lookup(id)
        if hasattr(store, "read_image"):
            content = store.read_image(id, "image")
            if content is None:
                raise HTTPException(404, "照片文件已不存在，请重新上传。")
            return Response(content, media_type="image/jpeg")
        path = store.uploads / f"{id}.jpg"
        if not path.is_file():
            raise HTTPException(404, "照片文件已不存在，请重新上传。")
        return FileResponse(path, media_type="image/jpeg")

    @application.get("/api/analyses/{id}/crop")
    def crop(id: str):
        row = lookup(id)
        if not row["report_json"]:
            raise HTTPException(404, "暂无可查看的猫脸局部图。")
        if hasattr(store, "read_image"):
            content = store.read_image(id, "crop")
            if content is None:
                raise HTTPException(404, "暂无可查看的猫脸局部图。")
            return Response(content, media_type="image/jpeg")
        path = store.crops / f"{id}.jpg"
        if not row["report_json"] or not path.is_file():
            raise HTTPException(404, "暂无可查看的猫脸局部图。")
        return FileResponse(path, media_type="image/jpeg")

    @application.post("/api/analyses/{id}/scores", response_model=Report)
    def submit(id: str, submission: ScoreSubmission):
        row = lookup(id)
        if row["status"] == "processing":
            raise HTTPException(409, "模型仍在分析，请稍候几秒后再次提交；你的评分不会丢失。")
        if not row["ai_json"]:
            raise HTTPException(409, "分析结果尚未就绪，请重试。")
        ai = AiResult.model_validate_json(row["ai_json"])
        fgs = calculate_fgs(submission.scores)
        result = Report(
            id=id, created_at=row["created_at"], completed_at=now(), cat_name=row["cat_name"],
            image_url=f"/api/analyses/{id}/image", image=ImageSize(width=row["width"], height=row["height"]),
            notes=submission.notes.strip(), ai=ai, localization=ai.localization, machines=ai.machines,
            collection=CollectionInfo.model_validate_json(row["collection_json"]) if row.get("collection_json") else None,
            fgs=fgs, conclusion=combine_results(ai, fgs),
        )
        stored = store.set_report(id, result.model_dump(mode="json"), DUAL_RULE_VERSION if ai.machines else RULE_VERSION)
        return Report.model_validate(stored) if stored is not None else result

    @application.get("/api/analyses/{id}/report", response_model=Report)
    def report(id: str):
        row = lookup(id)
        if not row["report_json"]:
            raise HTTPException(404, "请先独立完成五项 FGS 评分，再查看综合报告。")
        result = Report.model_validate_json(row["report_json"])
        if row.get("collection_json"):
            result.collection = CollectionInfo.model_validate_json(row["collection_json"])
        return result

    @application.post("/api/analyses/{id}/training-permission", response_model=CollectionInfo)
    def training_permission(id: str, submission: TrainingPermission):
        lookup(id)
        return store.set_training_permission(id, submission.allowed, now())

    # Register API fallback before the SPA catch-all: unknown API URLs must not return HTML.
    @application.api_route("/api", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    @application.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def missing_api(path: str = ""):
        raise HTTPException(404, "这个接口不存在。")

    dist = (PROJECT_ROOT / "frontend" / "dist").resolve()

    @application.get("/{path:path}")
    def frontend(path: str):
        candidate = (dist / path).resolve()
        if dist in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        if path.startswith("assets/"):
            raise HTTPException(404, "这个静态文件不存在。")
        index = dist / "index.html"
        if index.is_file():
            return FileResponse(index, media_type="text/html")
        return JSONResponse(
            {"detail": "前端尚未构建。请在 frontend 目录运行 npm install 和 npm run build，然后重新打开。"},
            status_code=503,
        )

    return application


app = create_app() if os.environ.get("CATPAIN_RUNTIME") != "vercel" else None
