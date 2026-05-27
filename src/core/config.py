import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()

CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parent.parent

CHECKPOINTS_DIR = Path(os.getenv("CHECKPOINTS_DIR", str(PROJECT_ROOT / "checkpoints")))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", str(PROJECT_ROOT / "outputs")))
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(PROJECT_ROOT / "uploads")))
SKLEARN_MODELS_DIR = Path(os.getenv("SKLEARN_MODELS_DIR", str(PROJECT_ROOT / "sklearn_models")))

CVC_DIR = CHECKPOINTS_DIR / "cvc_unet"
DS2_DIR = CHECKPOINTS_DIR / "ds2_unet"

REQUIRED_CHECKPOINT_FILES: Tuple[Path, ...] = (
    CVC_DIR / "arch.json",
    CVC_DIR / "cvc_unet.weights.h5",
    DS2_DIR / "arch.json",
    DS2_DIR / "ds2_unet.weights.h5",
)

REQUIRED_SKLEARN_MODEL_FILES: Tuple[Path, ...] = (
    SKLEARN_MODELS_DIR / "accidental.model",
    SKLEARN_MODELS_DIR / "clef.model",
    SKLEARN_MODELS_DIR / "rests.model",
    SKLEARN_MODELS_DIR / "rests_above8.model",
)


def ensure_runtime_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    SKLEARN_MODELS_DIR.mkdir(parents=True, exist_ok=True)
