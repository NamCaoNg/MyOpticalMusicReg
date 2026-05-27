import time
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.simple_omr_routes import router as simple_omr_router
from app.core.settings import settings
from app.error_handlers import register_exception_handlers
from src.core.config import (
    OUTPUTS_DIR,
    REQUIRED_CHECKPOINT_FILES,
    REQUIRED_SKLEARN_MODEL_FILES,
    ensure_runtime_dirs,
)

load_dotenv()
ensure_runtime_dirs()

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title=settings.app_name,
    description="Upload a score image and receive MusicXML/MIDI outputs.",
    version=settings.app_version,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)

register_exception_handlers(app)


@app.on_event("startup")
async def log_startup():
    if settings.warmup_models:
        logger.info("Warming up TensorFlow OMR models")
        from src.core.inference import warmup_models

        warmup_models()
    logger.info(
        "OMR model service ready: process_endpoint=/api/v1/omr/process outputs=/outputs"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.trusted_hosts != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.info("HTTP request started: %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception(
            "HTTP request failed: %s %s after %.3fs",
            request.method,
            request.url.path,
            elapsed,
        )
        raise

    elapsed = time.perf_counter() - start
    logger.info(
        "HTTP request completed: %s %s -> %s in %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


app.include_router(simple_omr_router, prefix="/api/v1", tags=["OMR Model Service"])
app.include_router(
    simple_omr_router,
    prefix="/simple",
    tags=["Legacy Simple OMR"],
    include_in_schema=False,
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/")
def root():
    return {
        "message": f"{settings.app_name} is running",
        "service_role": "model-service",
        "process_endpoint": "/api/v1/omr/process",
        "capabilities_endpoint": "/api/v1/omr/capabilities",
    }


@app.get("/health")
def health():
    required_assets = list(REQUIRED_CHECKPOINT_FILES) + list(REQUIRED_SKLEARN_MODEL_FILES)
    missing_assets = [str(path) for path in required_assets if not path.exists()]
    return {
        "status": "ok",
        "model_assets_ready": not missing_assets,
        "missing_assets": missing_assets,
    }
