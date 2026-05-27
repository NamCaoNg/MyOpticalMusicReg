import json
import os
import pickle
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import tensorflow as tf
from numpy import ndarray
from PIL import Image

from src.core.config import CVC_DIR, DS2_DIR, SKLEARN_MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


_tf_model_cache: Dict[str, tf.keras.Model] = {}
_sklearn_cache: Dict[str, dict] = {}
_tf_cache_lock = Lock()
_tf_predict_lock = Lock()
_sklearn_cache_lock = Lock()
_tf_gpu_configured = False


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r. Using default %s.", name, value, default)
        return default


def _target_pixel_bounds() -> tuple[int, int]:
    min_pixels = _env_int("OMR_RESIZE_MIN_PIXELS", 3_000_000)
    max_pixels = _env_int("OMR_RESIZE_MAX_PIXELS", 3_500_000)
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        logger.warning(
            "Invalid resize pixel bounds min=%s max=%s. Using defaults.",
            min_pixels,
            max_pixels,
        )
        return 3_000_000, 3_500_000
    return min_pixels, max_pixels


def _configure_tensorflow_gpu() -> None:
    global _tf_gpu_configured
    if _tf_gpu_configured:
        return

    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            logger.info("TensorFlow GPUs available: %s", [gpu.name for gpu in gpus])
        else:
            logger.info("TensorFlow GPU not available. Using CPU.")
    except RuntimeError as exc:
        logger.warning("TensorFlow GPU memory growth could not be configured: %s", exc)
    finally:
        _tf_gpu_configured = True


def _find_weights_file(model_path: str) -> str:
    h5_files = sorted(
        file_name
        for file_name in os.listdir(model_path)
        if file_name.lower().endswith(".h5")
    )
    if not h5_files:
        raise FileNotFoundError(f"No .h5 file found in: {model_path}")
    return os.path.join(model_path, h5_files[0])


def _load_tf_model(model_path: str) -> tf.keras.Model:
    model_path = os.path.abspath(model_path)
    with _tf_cache_lock:
        if model_path in _tf_model_cache:
            return _tf_model_cache[model_path]

        arch_path = os.path.join(model_path, "arch.json")
        weights_path = _find_weights_file(model_path)

        if not os.path.exists(arch_path):
            raise FileNotFoundError(f"Missing TensorFlow model architecture: {arch_path}")

        logger.info("Loading TensorFlow model (first time): %s", model_path)
        _configure_tensorflow_gpu()

        with open(arch_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        cfg.pop("compile_config", None)
        cfg.pop("compiled", None)

        model = tf.keras.models.model_from_json(json.dumps(cfg))
        model.load_weights(weights_path)
        _tf_model_cache[model_path] = model
        return model


def resize_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    pixel_count = w * h
    min_pixels, max_pixels = _target_pixel_bounds()
    if min_pixels <= pixel_count <= max_pixels:
        return image

    lower_bound = min_pixels / pixel_count
    upper_bound = max_pixels / pixel_count
    ratio = pow((lower_bound + upper_bound) / 2, 0.5)
    target_w = round(ratio * w)
    target_h = round(ratio * h)
    logger.info("Resize image to %sx%s", target_w, target_h)
    return image.resize((target_w, target_h))


def _load_image(img_path: str) -> ndarray:
    image_pil = Image.open(img_path)
    if image_pil.format != "GIF":
        image_cv = cv2.imread(img_path)
        if image_cv is None:
            raise ValueError(f"Cannot read image: {img_path}")
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_cv)

    image_pil = image_pil.convert("RGB")
    return np.array(resize_image(image_pil))


def _iter_patch_positions(height: int, width: int, win_size: int, step_size: int) -> list[tuple[int, int]]:
    positions = []
    for y in range(0, height, step_size):
        if y + win_size > height:
            y = height - win_size
        for x in range(0, width, step_size):
            if x + win_size > width:
                x = width - win_size
            positions.append((y, x))
    return positions


def _resolve_batch_size(batch_size: int | None) -> int:
    resolved = batch_size if batch_size is not None else _env_int("OMR_TF_BATCH_SIZE", 16)
    return max(1, resolved)


def inference(
    model_path: str,
    img_path: str,
    step_size: int = 128,
    batch_size: int | None = None,
    manual_th: Optional[Any] = None,
) -> Tuple[ndarray, ndarray]:
    model = _load_tf_model(model_path)
    batch_size = _resolve_batch_size(batch_size)
    input_shape = model.input_shape
    output_shape = model.output_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    win_size = input_shape[1]
    if win_size is None:
        raise ValueError(f"TensorFlow model has dynamic window size: {input_shape}")

    image = _load_image(img_path)
    height, width = image.shape[:2]
    if height < win_size or width < win_size:
        image = np.array(Image.fromarray(image).resize((max(width, win_size), max(height, win_size))))
        height, width = image.shape[:2]

    merged_shape = image.shape[:2] + (output_shape[-1],)
    out = np.zeros(merged_shape, dtype=np.float32)
    mask = np.zeros(merged_shape, dtype=np.float32)
    positions = _iter_patch_positions(height, width, win_size, step_size)

    for idx in range(0, len(positions), batch_size):
        if idx % (batch_size * 5) == 0:
            logger.info("Inference progress: %s/%s", min(idx + batch_size, len(positions)), len(positions))

        batch_positions = positions[idx: idx + batch_size]
        batch = np.array(
            [image[y: y + win_size, x: x + win_size] for y, x in batch_positions]
        )
        with _tf_predict_lock:
            out_batch = model.predict(batch, verbose=0)

        for hop, (y, x) in zip(out_batch, batch_positions):
            out[y: y + win_size, x: x + win_size] += hop
            mask[y: y + win_size, x: x + win_size] += 1

    out /= mask

    if manual_th is None:
        class_map = np.argmax(out, axis=-1)
    else:
        assert len(manual_th) == output_shape[-1] - 1, f"{manual_th}, {output_shape[-1]}"
        class_map = np.zeros(out.shape[:2] + (len(manual_th),))
        for idx, th in enumerate(manual_th):
            class_map[..., idx] = np.where(out[..., idx + 1] > th, 1, 0)

    return class_map, out


def warmup_models() -> None:
    for model_dir in (CVC_DIR, DS2_DIR):
        model = _load_tf_model(str(model_dir))
        input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
        win_size = input_shape[1]
        if win_size is None:
            logger.warning("Skip TensorFlow warmup for dynamic input shape: %s", input_shape)
            continue
        dummy = np.zeros((1, win_size, win_size, 3), dtype=np.uint8)
        logger.info("Warming up TensorFlow model: %s", model_dir)
        with _tf_predict_lock:
            model.predict(dummy, verbose=0)


def predict(region: ndarray, model_name: str) -> str:
    if np.max(region) == 1:
        region *= 255

    with _sklearn_cache_lock:
        if model_name in _sklearn_cache:
            m_info = _sklearn_cache[model_name]
        else:
            model_path = os.path.join(str(SKLEARN_MODELS_DIR), f"{model_name}.model")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing sklearn model: {model_path}")
            logger.info("Loading sklearn model (first time): %s", model_name)
            with open(model_path, "rb") as f:
                m_info = pickle.load(f)
            _sklearn_cache[model_name] = m_info

    model = m_info["model"]
    w = m_info["w"]
    h = m_info["h"]
    region = np.array(Image.fromarray(region.astype(np.uint8)).resize((w, h)))
    pred = model.predict(region.reshape(1, -1))
    return m_info["class_map"][pred[0]]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    img_path = project_root / "data" / "myimages" / "Be_the_one01.jpg"
    model_path = project_root / "checkpoints" / "cvc_unet"

    class_map, _ = inference(str(model_path), str(img_path))

    color_map = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)
    color_map[class_map == 0] = [0, 0, 0]
    color_map[class_map == 1] = [0, 0, 255]
    color_map[class_map == 2] = [0, 255, 0]
    color_map[class_map == 3] = [255, 0, 0]
    color_map[class_map == 4] = [255, 255, 0]
    color_map[class_map == 5] = [255, 0, 255]

    out_dir = project_root / "outputs" / "inferences"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / f"{img_path.stem}.png"
    Image.fromarray(color_map).save(str(save_path))
    logger.info("Saved to: %s", save_path)
