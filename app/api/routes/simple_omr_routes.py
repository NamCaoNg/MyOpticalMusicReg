import hmac
import logging
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock
from typing import Literal

from fastapi import APIRouter, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from app.core.settings import settings
from app.error_utils import build_error_detail
from app.services.simple_omr_service import process_score_image
from src.core.config import (
    OUTPUTS_DIR,
    REQUIRED_CHECKPOINT_FILES,
    REQUIRED_SKLEARN_MODEL_FILES,
    UPLOADS_DIR,
)

router = APIRouter()
logger = logging.getLogger("uvicorn.error")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=1)
JOBS_LOCK = Lock()
JOBS: dict[str, dict] = {}
JobStatus = Literal["queued", "processing", "completed", "failed"]


class OutputFile(BaseModel):
    url: str
    filename: str
    content_type: str


class SimpleOmrDebugPaths(BaseModel):
    output_dir: str
    xml_path: str
    midi_path: str


class SimpleOmrResponse(BaseModel):
    success: bool = True
    job_id: str
    status: str = "completed"
    processing_time_sec: float
    files: dict[str, OutputFile]
    xml_content: str | None = None
    debug_paths: SimpleOmrDebugPaths | None = None


class SimpleOmrJobResponse(BaseModel):
    success: bool = True
    job_id: str
    status: JobStatus
    status_url: str
    message: str


class SimpleOmrJobStatusResponse(BaseModel):
    success: bool = True
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    processing_time_sec: float | None = None
    files: dict[str, OutputFile] | None = None
    xml_content: str | None = None
    debug_paths: SimpleOmrDebugPaths | None = None
    error: dict | None = None


class SimpleOmrCapabilities(BaseModel):
    allowed_extensions: list[str]
    max_upload_size_mb: int
    process_endpoint: str
    outputs_base_url: str
    required_assets_ready: bool
    api_key_required: bool
    supports_options: list[str]
    job_endpoint: str


def _generate_job_id() -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{now}_{uuid.uuid4().hex[:6]}"


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _sanitize_filename(filename: str) -> str:
    base_name = os.path.basename(filename).strip()
    if not base_name:
        return "unknown.png"

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", base_name)
    safe_name = safe_name.lstrip(".")
    return safe_name or "unknown.png"


def _validate_saved_image(input_path: str) -> None:
    try:
        with Image.open(input_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=build_error_detail("invalid_image", "Invalid image file"),
        )


def _verify_api_key(x_api_key: str | None) -> None:
    if not settings.api_key:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=401,
            detail=build_error_detail("unauthorized", "Invalid or missing API key"),
        )


def _static_url(job_id: str, file_path: str | None) -> str:
    if not file_path:
        raise ValueError("file_path is required")
    filename = os.path.basename(file_path)
    if not filename:
        raise ValueError("filename is required")
    url = f"/outputs/{job_id}/{filename}"
    if settings.public_base_url:
        return f"{settings.public_base_url}{url}"
    return url


def _output_file(job_id: str, path: str, content_type: str) -> OutputFile:
    return OutputFile(
        url=_static_url(job_id, path),
        filename=os.path.basename(path),
        content_type=content_type,
    )


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _should_include_xml_content(include_xml_content: bool | None) -> bool:
    return (
        settings.include_xml_content_default
        if include_xml_content is None
        else include_xml_content
    )


def _cleanup_file(path: str) -> None:
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _cleanup_dir(path: str) -> None:
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError:
            pass


def _write_upload_with_limit(file: UploadFile, input_path: str) -> None:
    total_size = 0
    try:
        with open(input_path, "wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > settings.max_upload_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=build_error_detail(
                            "simple_omr_file_too_large",
                            f"File upload exceeds the {settings.max_upload_size_mb} MB limit",
                            {"max_upload_size": settings.max_upload_size_mb},
                        ),
                    )
                buffer.write(chunk)
    finally:
        file.file.close()


def _required_assets_ready() -> bool:
    required = list(REQUIRED_CHECKPOINT_FILES) + list(REQUIRED_SKLEARN_MODEL_FILES)
    return all(path.exists() for path in required)


def _validate_upload_filename(filename: str) -> str:
    safe_filename = _sanitize_filename(filename or "unknown.png")
    _, ext = os.path.splitext(safe_filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=build_error_detail(
                "unsupported_file_type",
                f"File type '{ext}' is not supported",
                {"allowed_extensions": sorted(ALLOWED_EXTENSIONS)},
            ),
        )
    return safe_filename


def _save_upload(file: UploadFile, job_id: str, filename: str) -> str:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    input_path = str(UPLOADS_DIR / f"{job_id}_{filename}")
    _write_upload_with_limit(file, input_path)
    _validate_saved_image(input_path)
    return input_path


def _build_result_payload(
    *,
    job_id: str,
    result: dict,
    elapsed: float,
    include_xml_content: bool | None) -> dict:
    xml_path = result["xml_path"]
    midi_path = result["midi_path"]
    xml_stem = os.path.splitext(os.path.basename(xml_path))[0]
    teaser_path = os.path.join(result["output_dir"], f"{xml_stem}_teaser.png")
    pitch_path = os.path.join(result["output_dir"], f"{xml_stem}_pitch.png")

    files = {
        "musicxml": _output_file(job_id, xml_path, "application/vnd.recordare.musicxml+xml"),
        "midi": _output_file(job_id, midi_path, "audio/midi"),
    }
    if os.path.exists(teaser_path):
        files["teaser_image"] = _output_file(job_id, teaser_path, "image/png")
    if os.path.exists(pitch_path):
        files["pitch_image"] = _output_file(job_id, pitch_path, "image/png")

    debug_paths = None
    if settings.include_local_paths:
        debug_paths = SimpleOmrDebugPaths(
            output_dir=result["output_dir"],
            xml_path=xml_path,
            midi_path=midi_path,
        )

    return {
        "processing_time_sec": round(elapsed, 3),
        "files": files,
        "xml_content": _read_text(xml_path)
        if _should_include_xml_content(include_xml_content)
        else None,
        "debug_paths": debug_paths,
    }


def _run_omr_job(
    *,
    job_id: str,
    input_path: str,
    output_dir: str,
    include_xml_content: bool | None,
    without_deskew: bool,
    tempo: int | None,
    instrument: str | None) -> dict:
    start = time.perf_counter()
    result = process_score_image(
        input_path=input_path,
        output_dir=output_dir,
        without_deskew=without_deskew,
        tempo=tempo,
        instrument=instrument,
    )
    elapsed = time.perf_counter() - start
    return _build_result_payload(
        job_id=job_id,
        result=result,
        elapsed=elapsed,
        include_xml_content=include_xml_content,
    )


def _set_job(job_id: str, **updates: object) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(updates)
        job["updated_at"] = _utc_now_iso()


def _process_background_job(
    *,
    job_id: str,
    input_path: str,
    output_dir: str,
    include_xml_content: bool | None,
    without_deskew: bool,
    tempo: int | None,
    instrument: str | None) -> None:
    logger.info("Background job started: job_id=%s output_dir=%s", job_id, output_dir)
    _set_job(job_id, status="processing")
    try:
        payload = _run_omr_job(
            job_id=job_id,
            input_path=input_path,
            output_dir=output_dir,
            include_xml_content=include_xml_content,
            without_deskew=without_deskew,
            tempo=tempo,
            instrument=instrument,
        )
        _set_job(job_id, status="completed", **payload)
        logger.info("Background job completed: job_id=%s", job_id)
    except Exception as exc:
        logger.exception("Background job failed: job_id=%s error=%s", job_id, exc)
        _cleanup_dir(output_dir)
        _set_job(
            job_id,
            status="failed",
            error=build_error_detail("processing_failed", str(exc)[:500]),
        )
    finally:
        if not settings.keep_uploads:
            _cleanup_file(input_path)


@router.get("/omr/capabilities", response_model=SimpleOmrCapabilities)
def get_omr_capabilities():
    outputs_base_url = "/outputs"
    if settings.public_base_url:
        outputs_base_url = f"{settings.public_base_url}{outputs_base_url}"

    return SimpleOmrCapabilities(
        allowed_extensions=sorted(ALLOWED_EXTENSIONS),
        max_upload_size_mb=settings.max_upload_size_mb,
        process_endpoint="/api/v1/omr/process",
        outputs_base_url=outputs_base_url,
        required_assets_ready=_required_assets_ready(),
        api_key_required=bool(settings.api_key),
        supports_options=[
            "include_xml_content",
            "without_deskew",
            "tempo",
            "instrument",
        ],
        job_endpoint="/api/v1/omr/jobs",
    )


@router.post("/omr/process", response_model=SimpleOmrResponse)
def process_omr_file(
    file: UploadFile = File(...),
    include_xml_content: bool | None = Query(
        default=None,
        description="Include raw MusicXML content in the JSON response.",
    ),
    without_deskew: bool = Query(
        default=False,
        description="Disable deskew/dewarp when the input image is already straight.",
    ),
    tempo: int | None = Query(
        default=None,
        ge=20,
        le=300,
        description="Optional MIDI tempo in BPM.",
    ),
    instrument: str | None = Query(
        default=None,
        description="Optional MIDI instrument name, for example piano or violin.",
    ),
    x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """
    Internal model-service API for OMR processing.

    Upload one image and receive generated MusicXML/MIDI paths plus preview URLs.
    This endpoint runs synchronously, so the caller should use a long timeout.
    """
    _verify_api_key(x_api_key)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = _validate_upload_filename(file.filename or "unknown.png")
    job_id = _generate_job_id()
    output_dir = str(OUTPUTS_DIR / job_id)
    input_path = str(UPLOADS_DIR / f"{job_id}_{filename}")
    start = time.perf_counter()
    logger.info("Started: job_id=%s filename=%s output_dir=%s", job_id, filename, output_dir)

    try:
        input_path = _save_upload(file, job_id, filename)
        logger.info("Upload saved: job_id=%s input_path=%s", job_id, input_path)
        logger.info("Image validated: job_id=%s", job_id)
        payload = _run_omr_job(
            job_id=job_id,
            input_path=input_path,
            output_dir=output_dir,
            include_xml_content=include_xml_content,
            without_deskew=without_deskew,
            tempo=tempo,
            instrument=instrument,
        )
    except HTTPException:
        logger.warning("Job rejected: job_id=%s filename=%s", job_id, filename)
        _cleanup_dir(output_dir)
        raise
    except Exception as exc:
        logger.exception("Job failed: job_id=%s filename=%s error=%s", job_id, filename, exc)
        _cleanup_file(input_path)
        _cleanup_dir(output_dir)
        raise HTTPException(
            status_code=422,
            detail=build_error_detail("processing_failed", str(exc)[:500]),
        ) from exc
    finally:
        if not settings.keep_uploads:
            _cleanup_file(str(UPLOADS_DIR / f"{job_id}_{filename}"))

    elapsed = round(time.perf_counter() - start, 3)
    logger.info(
        "Job completed: job_id=%s elapsed=%.3fs xml=%s midi=%s",
        job_id,
        elapsed,
        payload["files"]["musicxml"].url,
        payload["files"]["midi"].url,
    )

    return SimpleOmrResponse(
        job_id=job_id,
        processing_time_sec=elapsed,
        files=payload["files"],
        xml_content=payload["xml_content"],
        debug_paths=payload["debug_paths"],
    )


@router.post("/omr/jobs", response_model=SimpleOmrJobResponse, status_code=202)
def create_omr_job(
    file: UploadFile = File(...),
    include_xml_content: bool | None = Query(
        default=None,
        description="Include raw MusicXML content in the completed job response.",
    ),
    without_deskew: bool = Query(
        default=False,
        description="Disable deskew/dewarp when the input image is already straight.",
    ),
    tempo: int | None = Query(
        default=None,
        ge=20,
        le=300,
        description="Optional MIDI tempo in BPM.",
    ),
    instrument: str | None = Query(
        default=None,
        description="Optional MIDI instrument name, for example piano or violin.",
    ),
    x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """
    Queue one OMR job and return immediately.

    Use GET /api/v1/omr/jobs/{job_id} to poll status and retrieve output URLs.
    """
    _verify_api_key(x_api_key)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = _validate_upload_filename(file.filename or "unknown.png")
    job_id = _generate_job_id()
    output_dir = str(OUTPUTS_DIR / job_id)

    try:
        input_path = _save_upload(file, job_id, filename)
    except HTTPException:
        _cleanup_dir(output_dir)
        raise

    now = _utc_now_iso()
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "processing_time_sec": None,
            "files": None,
            "xml_content": None,
            "debug_paths": None,
            "error": None,
        }

    JOB_EXECUTOR.submit(
        _process_background_job,
        job_id=job_id,
        input_path=input_path,
        output_dir=output_dir,
        include_xml_content=include_xml_content,
        without_deskew=without_deskew,
        tempo=tempo,
        instrument=instrument,
    )

    return SimpleOmrJobResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/api/v1/omr/jobs/{job_id}",
        message="Job queued. Poll the status_url until status is completed or failed.",
    )


@router.get("/omr/jobs/{job_id}", response_model=SimpleOmrJobStatusResponse)
def get_omr_job(
    job_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    _verify_api_key(x_api_key)

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=build_error_detail("job_not_found", f"Job '{job_id}' was not found"),
            )
        return SimpleOmrJobStatusResponse(**job)
