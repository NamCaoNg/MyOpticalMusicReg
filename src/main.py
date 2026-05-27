# Run: python -m src.main "data\myimages\Someone_you_loved02.jpg"

import os
import pickle
import argparse
import uuid
import time
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Iterator, Tuple
from argparse import Namespace, ArgumentParser

from PIL import Image
from numpy import ndarray

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from src.core import layers
from src.core.config import (
    DS2_DIR,
    CVC_DIR,
    OUTPUTS_DIR,
    REQUIRED_CHECKPOINT_FILES,
    REQUIRED_SKLEARN_MODEL_FILES,
    ensure_runtime_dirs,
)
from src.core.inference import inference
from src.utils.logger import get_logger
from src.core.dewarp import estimate_coords, dewarp
from src.extraction.staffline_extraction import staff_extract
from src.extraction.notehead_extraction import notehead_extract
from src.extraction.notegroup_extraction import notegroup_extract
from src.extraction.symbol_extraction import symbol_extract
from src.extraction.rhythm_extraction import rhythm_extract
from src.build_system.build_XML import XMLBuilder
from src.build_system.draw_teaser import teaser, pitch
from src.build_system.build_midi import xml_to_midi

logger = get_logger(__name__)


def _atomic_pickle_dump(path: Path, data: dict, retries: int = 5) -> None:
    last_err: Exception | None = None
    for _ in range(retries):
        temp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(data, f)
            os.replace(temp_path, path)
            return
        except PermissionError as e:
            last_err = e
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            time.sleep(0.05)
        except OSError as e:
            last_err = e
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            break

    if last_err is not None:
        raise last_err


@contextmanager
def _cache_file_lock(
    cache_path: Path,
    timeout_sec: float = 2.0,
    poll_interval_sec: float = 0.05) -> Iterator[bool]:
    lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
    start_time = time.time()
    fd = None

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii", errors="ignore"))
            break
        except FileExistsError:
            if (time.time() - start_time) >= timeout_sec:
                yield False
                return
            time.sleep(poll_interval_sec)

    try:
        yield True
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def _new_output_folder(base_name: str, output_dir: str | None = None) -> Path:
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_id = uuid.uuid4().hex[:8]
    folder = OUTPUTS_DIR / f"{base_name}_{timestamp}_{unique_id}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def _validate_runtime_assets() -> None:
    ensure_runtime_dirs()

    missing_files = [str(p) for p in REQUIRED_CHECKPOINT_FILES if not p.exists()]
    missing_sklearn_files = [str(p) for p in REQUIRED_SKLEARN_MODEL_FILES if not p.exists()]

    all_missing = missing_files + missing_sklearn_files
    if all_missing:
        logger.error("Missing required runtime assets:")
        for f in all_missing:
            logger.error(f"  - {f}")
        joined = "\n".join(all_missing)
        raise FileNotFoundError(f"Missing required runtime assets:\n{joined}")


def clear_data() -> None:
    layers.clear_all_layers()


def generate_pred(img_path: str) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    logger.info("Extracting staffline and symbols")
    staff_symbols_map, _ = inference(str(CVC_DIR), img_path)
    staff = np.where(staff_symbols_map == 1, 1, 0)
    symbols = np.where(staff_symbols_map == 2, 1, 0)

    logger.info("Extracting layers of different symbols")
    sep, _ = inference(str(DS2_DIR), img_path, manual_th=None)
    stems_rests = np.where(sep == 1, 1, 0)
    notehead = np.where(sep == 2, 1, 0)
    clefs_keys = np.where(sep == 3, 1, 0)

    return staff, symbols, stems_rests, notehead, clefs_keys


def polish_symbols(rgb_black_th=300):
    img = layers.get_layer('original_image')
    sym_pred = layers.get_layer('symbols_pred')

    img = Image.fromarray(img).resize((sym_pred.shape[1], sym_pred.shape[0]))
    arr = np.sum(np.array(img), axis=-1)
    arr = np.where(arr < rgb_black_th, 1, 0)  # Filter background
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    arr = cv2.dilate(cv2.erode(arr.astype(np.uint8), ker), ker)  # Filter staff lines
    mix = np.where(sym_pred + arr > 1, 1, 0)
    return mix


def register_notehead_bbox(bboxes):
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('bboxes')
    for (x1, y1, x2, y2) in bboxes:
        yi, xi = np.where(symbols[y1:y2, x1:x2] > 0)
        yi += y1
        xi += x1
        layer[yi, xi] = np.array([x1, y1, x2, y2])
    return layer


def register_note_id() -> None:
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('note_id')
    notes = layers.get_layer('notes')
    for idx, note in enumerate(notes):
        x1, y1, x2, y2 = note.bbox
        yi, xi = np.where(symbols[y1:y2, x1:x2] > 0)
        yi += y1
        xi += x1
        layer[yi, xi] = idx
        notes[idx].id = idx


def extract(args: Namespace) -> str:
    img_path = Path(args.img_path)
    f_name = os.path.splitext(img_path.name)[0]
    pkl_path = img_path.parent / f"{f_name}.pkl"

    use_cache = False
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                pred = pickle.load(f)
            notehead = pred["note"]
            symbols = pred["symbols"]
            staff = pred["staff"]
            clefs_keys = pred["clefs_keys"]
            stems_rests = pred["stems_rests"]
            use_cache = True
        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            logger.warning("Invalid cache %s (%s). Recomputing predictions.", pkl_path, e)

    if not use_cache:
        staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(str(img_path))
        if args.save_cache:
            data = {
                'staff': staff,
                'note': notehead,
                'symbols': symbols,
                'stems_rests': stems_rests,
                'clefs_keys': clefs_keys
            }
            try:
                with _cache_file_lock(pkl_path) as locked:
                    if locked and not pkl_path.exists():
                        _atomic_pickle_dump(pkl_path, data)
                    elif not locked:
                        logger.warning("Skip writing cache %s because lock acquisition timed out.", pkl_path)
            except OSError as e:
                logger.warning("Skip writing cache %s due to concurrent file contention: %s", pkl_path, e)

    image_pil = Image.open(str(img_path))
    if "GIF" != image_pil.format:
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")
    else:
        gif_image = image_pil.convert('RGB')
        gif_img_arr = np.array(gif_image)
        image = gif_img_arr[:, :, ::-1].copy()

    image = cv2.resize(image, (staff.shape[1], staff.shape[0]))

    if not args.without_deskew:
        logger.info("Dewarping")
        coords_x, coords_y = estimate_coords(staff)
        staff = dewarp(staff, coords_x, coords_y)
        symbols = dewarp(symbols, coords_x, coords_y)
        stems_rests = dewarp(stems_rests, coords_x, coords_y)
        clefs_keys = dewarp(clefs_keys, coords_x, coords_y)
        notehead = dewarp(notehead, coords_x, coords_y)
        for i in range(image.shape[2]):
            image[..., i] = dewarp(image[..., i], coords_x, coords_y)

    symbols = symbols + clefs_keys + stems_rests
    symbols[symbols > 1] = 1
    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred", clefs_keys)
    layers.register_layer("notehead_pred", notehead)
    layers.register_layer("symbols_pred", symbols)
    layers.register_layer("staff_pred", staff)
    layers.register_layer("original_image", image)

    logger.info("Extracting stafflines")
    staffs, zones = staff_extract()
    layers.register_layer("staffs", staffs)
    layers.register_layer("zones", zones)

    logger.info("Extracting noteheads")
    notes = notehead_extract()
    layers.register_layer('notes', np.array(notes))

    layers.register_layer('note_id', np.zeros(symbols.shape, dtype=np.int64) - 1)
    register_note_id()

    logger.info("Grouping noteheads")
    groups, group_map = notegroup_extract()
    layers.register_layer('note_groups', np.array(groups))
    layers.register_layer('group_map', group_map)

    logger.info("Extracting symbols")
    barlines, clefs, accidentals, rests = symbol_extract()
    layers.register_layer('barlines', np.array(barlines))
    layers.register_layer('clefs', np.array(clefs))
    layers.register_layer('accidentals', np.array(accidentals))
    layers.register_layer('rests', np.array(rests))

    logger.info("Extracting rhythm types")
    rhythm_extract()

    return Path(args.img_path).stem


def get_parser() -> ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receives an image as input, and outputs MusicXML file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("img_path", help="Path to the image.", type=str)
    parser.add_argument(
        "--save-cache",
        help="Save the predictions and the next time won't need to predict again.",
        action='store_true'
    )
    parser.add_argument(
        "-d",
        "--without-deskew",
        help="Disable the deskewing step if you are sure the image has no skew.",
        action='store_true'
    )
    parser.add_argument("--tempo", help="Tempo (BPM) for MIDI output.", type=int, default=None)
    parser.add_argument("--instrument", help="Instrument for output (piano, violin...).", type=str, default=None)
    return parser


def run_pipeline(
    img_path: str,
    output_dir: str | None = None,
    save_cache: bool = False,
    without_deskew: bool = False,
    tempo: int | None = None,
    instrument: str | None = None) -> Dict[str, str]:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The given image path doesn't exists: {img_path}")

    _validate_runtime_assets()

    args = Namespace(
        img_path=img_path,
        save_cache=save_cache,
        without_deskew=without_deskew,
        tempo=tempo,
        instrument=instrument,
    )

    with layers.isolated_layer_context():
        basename = extract(args)

        out_folder = _new_output_folder(basename, output_dir=output_dir)
        logger.info(f"Output folder: {out_folder}")
        base_out = str(out_folder / basename)

        teaser_path = base_out + "_teaser.png"
        pitch_path = base_out + "_pitch.png"
        xml_path = base_out + ".xml"

        img = teaser()
        img.save(teaser_path)

        imgp = pitch()
        imgp.save(pitch_path)

        logger.info("Building MusicXML document")
        builder = XMLBuilder(title=basename.capitalize())
        builder.build()
        xml = builder.to_xml()

        with open(xml_path, "wb") as ff:
            ff.write(xml)

        midi_path = base_out + ".mid"
        logger.info("Converting to MIDI")
        xml_to_midi(
            xml_path=xml_path,
            output_path=midi_path,
            tempo=tempo,
            instrument=instrument
        )
        logger.info(f"MIDI file saved: {midi_path}")

        return {
            "basename": basename,
            "output_folder": str(out_folder),
            "teaser_path": teaser_path,
            "pitch_path": pitch_path,
            "xml_path": xml_path,
            "midi_path": midi_path,
        }


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    try:
        run_pipeline(
            img_path=args.img_path,
            save_cache=args.save_cache,
            without_deskew=args.without_deskew,
            tempo=args.tempo,
            instrument=args.instrument,
        )
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()