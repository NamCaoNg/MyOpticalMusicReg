import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from src.main import run_pipeline

IMAGES = [
    r".\data\myimages\If_I_am_with_you01.jpg",
    r".\data\myimages\Easy01.jpg",
]

SAVE_CACHE = False
WITHOUT_DESKEW = False
TEMPO = None
INSTRUMENT = None

# số worker cho test song song
MAX_WORKERS = 2

def run_one(img_path: str) -> dict:
    start = time.perf_counter()
    try:
        result = run_pipeline(
            img_path=img_path,
            save_cache=SAVE_CACHE,
            without_deskew=WITHOUT_DESKEW,
            tempo=TEMPO,
            instrument=INSTRUMENT,
        )
        elapsed = time.perf_counter() - start
        return {
            "ok": True,
            "img_path": img_path,
            "elapsed_sec": elapsed,
            "result": result,
            "error": None,
            "traceback": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "img_path": img_path,
            "elapsed_sec": elapsed,
            "result": None,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


def print_result(tag: str, item: dict) -> None:
    status = "OK" if item["ok"] else "FAIL"
    print(f"[{tag}] {status} | {item['img_path']} | {item['elapsed_sec']:.2f}s")

    if item["ok"]:
        result = item["result"]
        print(f"output_folder: {result['output_folder']}")
        print(f"xml_path     : {result['xml_path']}")
        print(f"midi_path    : {result['midi_path']}")
    else:
        print(f"error: {item['error']}")
        print(item["traceback"])


def test_sequential(images: list[str]) -> None:
    print("\n========== TEST TUẦN TỰ ==========")
    total_start = time.perf_counter()

    results = []
    for img in images:
        results.append(run_one(img))

    total_elapsed = time.perf_counter() - total_start

    for item in results:
        print_result("SEQUENTIAL", item)

    ok_count = sum(1 for x in results if x["ok"])
    print(f"\n[SEQUENTIAL] Done: {ok_count}/{len(results)} OK | total {total_elapsed:.2f}s")


def test_thread_parallel(images: list[str], max_workers: int = 2) -> None:
    print("\n========== TEST THREAD PARALLEL ==========")
    total_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, img): img for img in images}
        for fut in as_completed(futures):
            item = fut.result()
            results.append(item)
            print_result("THREAD", item)

    total_elapsed = time.perf_counter() - total_start
    ok_count = sum(1 for x in results if x["ok"])
    print(f"\n[THREAD] Done: {ok_count}/{len(results)} OK | total {total_elapsed:.2f}s")


def test_process_parallel(images: list[str], max_workers: int = 2) -> None:
    print("\n========== TEST PROCESS PARALLEL ==========")
    total_start = time.perf_counter()
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, img): img for img in images}
        for fut in as_completed(futures):
            item = fut.result()
            results.append(item)
            print_result("PROCESS", item)

    total_elapsed = time.perf_counter() - total_start
    ok_count = sum(1 for x in results if x["ok"])
    print(f"\n[PROCESS] Done: {ok_count}/{len(results)} OK | total {total_elapsed:.2f}s")


def validate_inputs(images: list[str]) -> list[str]:
    valid = []
    for img in images:
        p = Path(img)
        if p.exists():
            valid.append(str(p))
        else:
            print(f"[WARN] File not found, skip: {img}")
    return valid


if __name__ == "__main__":
    images = validate_inputs(IMAGES)

    # Test tuần tự
    test_sequential(images)

    # Test thread song song
    test_thread_parallel(images, max_workers=MAX_WORKERS)

    # Test process song song
    test_process_parallel(images, max_workers=MAX_WORKERS)