import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    cors_origins: list[str]
    trusted_hosts: list[str]
    public_base_url: str
    api_key: str
    enable_docs: bool
    max_upload_size_bytes: int
    include_local_paths: bool
    include_xml_content_default: bool
    keep_uploads: bool
    warmup_models: bool

    @property
    def max_upload_size_mb(self) -> int:
        return max(1, self.max_upload_size_bytes // (1024 * 1024))


def get_settings() -> Settings:
    max_upload_mb = _env_int("MAX_UPLOAD_MB", 15)
    return Settings(
        app_name=os.getenv("APP_NAME", "OMR Model Service"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        cors_origins=_split_csv(os.getenv("CORS_ORIGINS"), ["*"]),
        trusted_hosts=_split_csv(os.getenv("TRUSTED_HOSTS"), ["*"]),
        public_base_url=os.getenv("PUBLIC_BASE_URL", "").rstrip("/"),
        api_key=os.getenv("OMR_API_KEY", ""),
        enable_docs=_env_bool("ENABLE_DOCS", True),
        max_upload_size_bytes=max_upload_mb * 1024 * 1024,
        include_local_paths=_env_bool("INCLUDE_LOCAL_PATHS", False),
        include_xml_content_default=_env_bool("INCLUDE_XML_CONTENT_DEFAULT", False),
        keep_uploads=_env_bool("KEEP_UPLOADS", False),
        warmup_models=_env_bool("OMR_WARMUP_MODELS", False),
    )


settings = get_settings()
