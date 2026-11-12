"""Original DDP12 crop generator (user-provided notebook, 08/11/2026), kept verbatim except for the marked
MAGFILO_SOURCES filter (restrict to a list of source file names) and MAGFILO_SEED/INPUT overrides.
Per annotation record: 1 anchor crop per instance, 4 random positives, 3 strict negatives, native 768 windows.
"""

import json
import math
import multiprocessing as mp
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

# ========= CHANGE SETTINGS HERE =========
INPUT_ROOT = Path(os.getenv("MAGFILO_INPUT_ROOT", "/kaggle/input"))
OUTPUT_DIR = Path(os.getenv("MAGFILO_OUTPUT_DIR", "/kaggle/working/magfilo_crops_768"))
SOURCES_FILE = os.getenv("MAGFILO_SOURCES", "")  # ADDED: json list of source file names to keep (train only)
CROP = 768  # Saved image/mask size, fixed.
# Source window sizes: ~70% native scale, 15% zoom in, 15% zoom out.
# Set [768] and [1] to disable resize entirely.
SOURCE_WINDOWS = [768]
SOURCE_WINDOW_WEIGHTS = [1.0]
VAL_RATIO = 0.0  # Set 0 to crop everything (NOT recommended for honest validation)
SEED = 42
ANCHOR_PER_INSTANCE = 1
RANDOM_POS_PER_RECORD = 4
NEG_PER_RECORD = 3
MAX_ATTEMPTS = 80
MIN_VISIBLE_PIXELS = 8  # Reject *whole crop* if any clipped instance has 1..7 pixels.
NEG_MIN_NONBLACK_FRAC = 0.30  # Avoid mostly-black outside-solar-disc patches.
NEG_PIXEL_THRESHOLD = 15
PNG_COMPRESSION = 1  # Fast lossless PNG. 0=faster/larger; 9=slower/smaller.
# Reserve headroom below the user's 19 GB output budget.
MAX_OUTPUT_BYTES = 17_500_000_000
N_WORKERS = int(os.getenv("MAGFILO_WORKERS", "0")) or (
    len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
)  # CHANGED: MAGFILO_WORKERS override
# ========================================

_worker_output_dir = None
_reserved_bytes = None
_reserve_lock = None
_space_exhausted = None


def initialize_worker(output_dir, reserved_bytes, reserve_lock, space_exhausted):
    global _worker_output_dir, _reserved_bytes, _reserve_lock, _space_exhausted
    _worker_output_dir = Path(output_dir)
    _reserved_bytes = reserved_bytes
    _reserve_lock = reserve_lock
    _space_exhausted = space_exhausted
    cv2.setNumThreads(1)  # Prevent CPU oversubscription (one thread per worker).


def to_mask(segmentation, height, width):
    """Support COCO polygon, uncompressed RLE, and compressed RLE."""
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            raise ValueError("Empty polygon list")
        if isinstance(segmentation[0], (int, float)):
            segmentation = [segmentation]  # Flat single polygon -> COCO list.
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation["counts"], list):
            rle = coco_mask.frPyObjects(segmentation, height, width)
        else:
            rle = dict(segmentation)
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("ascii")
    else:
        raise TypeError(f"Unexpected segmentation type: {type(segmentation)}")
    decoded = coco_mask.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return (decoded > 0).astype(np.uint8)


def pick_anchor_origin(rng, bbox, height, width, window):
    """Random placement keeping the anchor fully visible when feasible."""
    x, y, bw, bh = [float(z) for z in bbox]

    def one_axis(start, length, full):
        last_origin = full - window
        margin = rng.choice([0, 16, 32, 64, 96])
        for pad in [margin, 0]:
            low = max(0, math.ceil(start + length + pad - window))
            high = min(last_origin, math.floor(start - pad))
            if low <= high:
                return rng.randint(low, high)
        # For very large filaments, a partial anchor is allowed.
        center = start + length / 2
        jitter = rng.uniform(-window * 0.22, window * 0.22)
        return int(np.clip(round(center - window / 2 + jitter), 0, last_origin))

    return one_axis(x, bw, width), one_axis(y, bh, height)


def box_intersects_any(boxes, x0, y0, window):
    if len(boxes) == 0:
        return False
    x1, y1 = x0 + window, y0 + window
    # boxes are [x_min, y_min, x_max, y_max].
    return bool(np.any((boxes[:, 0] < x1) & (boxes[:, 2] > x0) & (boxes[:, 1] < y1) & (boxes[:, 3] > y0)))


def clipped_instances(mask_items, x0, y0, window):
    """Return all visible instances; reject crops with tiny dangling fragments."""
    instances = []
    for item in mask_items:
        bx0, by0, bx1, by1 = item["box_xyxy"]
        if bx1 <= x0 or bx0 >= x0 + window or by1 <= y0 or by0 >= y0 + window:
            continue
        region = item["mask"][y0 : y0 + window, x0 : x0 + window]
        raw_visible_area = int(np.count_nonzero(region))
        if raw_visible_area == 0:
            continue
        if window != CROP:
            region = cv2.resize(region, (CROP, CROP), interpolation=cv2.INTER_NEAREST)
        visible_area = int(np.count_nonzero(region))
        if visible_area < MIN_VISIBLE_PIXELS:
            return None  # Resample whole crop; don't turn true filament into BG.
        rle = coco_mask.encode(np.asfortranarray(region))
        instances.append(
            {
                "source_annotation_id": str(item["ann_id"]),
                "source_category_id": item["category_id"],
                "label": 1,  # The competition predicts FILAMENT, not chirality.
                "area": visible_area,
                "bbox": [round(float(z), 3) for z in coco_mask.toBbox(rle).tolist()],
                "visible_fraction": round(raw_visible_area / item["total_area"], 5),
                "segmentation": {"size": [CROP, CROP], "counts": rle["counts"].decode("ascii")},
            }
        )
    return instances


def reserve_output_bytes(nbytes):
    """Atomic quota accounting across all writers."""
    with _reserve_lock:
        if _reserved_bytes.value + nbytes > MAX_OUTPUT_BYTES:
            _space_exhausted.value = 1
            return False
        _reserved_bytes.value += nbytes
        return True


def refund_output_bytes(nbytes):
    with _reserve_lock:
        _reserved_bytes.value -= nbytes


def process_one_record(task):
    rec_index, record, annotations, image_path, all_boxes = task
    result = {"positive": 0, "negative": 0, "skipped": 0, "bytes": 0, "error": None}
    try:
        if _space_exhausted.value:
            return result
        rng = random.Random(SEED + rec_index * 104729)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        height, width = image.shape
        if height < max(SOURCE_WINDOWS) or width < max(SOURCE_WINDOWS):
            raise ValueError(f"Image too small: {image_path} {image.shape}")

        mask_items = []
        this_boxes = []
        for ann in annotations:
            full_mask = to_mask(ann["segmentation"], height, width)
            area = int(np.count_nonzero(full_mask))
            if area == 0:
                raise ValueError(f"Empty segmentation in annotation {ann['id']}")
            x, y, bw, bh = [float(v) for v in ann["bbox"]]
            bxyxy = (x, y, x + bw, y + bh)
            mask_items.append(
                {
                    "mask": full_mask,
                    "total_area": area,
                    "box_xyxy": bxyxy,
                    "bbox": ann["bbox"],
                    "ann_id": ann["id"],
                    "category_id": int(ann["category_id"]),
                }
            )
            this_boxes.append(bxyxy)
        this_boxes = np.asarray(this_boxes, dtype=np.float64).reshape(-1, 4)
        all_boxes = np.asarray(all_boxes, dtype=np.float64).reshape(-1, 4)

        shard_file = _worker_output_dir / "parts" / f"part_{os.getpid()}.jsonl"
        used_positions = set()
        crop_number = 0

        def try_save(x0, y0, window, kind, require_positive):
            nonlocal crop_number
            if _space_exhausted.value or (x0, y0, window) in used_positions:
                return False
            if require_positive and not box_intersects_any(this_boxes, x0, y0, window):
                return False
            if not require_positive and box_intersects_any(all_boxes, x0, y0, window):
                return False
            instances = clipped_instances(mask_items, x0, y0, window)
            if instances is None:  # Very small partly clipped instance; retry.
                return False
            if require_positive and not instances:
                return False
            if not require_positive and instances:
                return False

            patch = image[y0 : y0 + window, x0 : x0 + window]
            if window != CROP:
                interpolation = cv2.INTER_AREA if window > CROP else cv2.INTER_LINEAR
                patch = cv2.resize(patch, (CROP, CROP), interpolation=interpolation)
            if not require_positive:
                fraction = float(np.count_nonzero(patch > NEG_PIXEL_THRESHOLD)) / patch.size
                if fraction < NEG_MIN_NONBLACK_FRAC:
                    return False

            ok, encoded = cv2.imencode(".png", patch, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            image_bytes = encoded.tobytes()
            crop_id = f"{rec_index:05d}_{crop_number:03d}"
            output_rel = f"images/{crop_id}.png"
            metadata = {
                "crop_id": crop_id,
                "image": output_rel,
                "width": CROP,
                "height": CROP,
                "source_file": Path(record["file_name"]).name,
                "source_image_id": str(record["id"]),
                "capture_day": Path(record["file_name"]).stem[:8],
                "crop_xywh": [x0, y0, window, window],
                "resize_to": [CROP, CROP],
                "sample_type": kind,
                "instances": instances,
            }
            line = (json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            # Include 1KB slack per crop for small metadata/FS overhead.
            billed = len(image_bytes) + len(line) + 1024
            if not reserve_output_bytes(billed):
                return False
            target_path = _worker_output_dir / output_rel
            try:
                with open(target_path, "wb") as f:
                    f.write(image_bytes)
                with open(shard_file, "ab") as f:
                    f.write(line)
            except Exception:
                try:
                    target_path.unlink(missing_ok=True)
                finally:
                    refund_output_bytes(billed)
                raise

            used_positions.add((x0, y0, window))
            crop_number += 1
            result["bytes"] += len(image_bytes) + len(line)
            if require_positive:
                result["positive"] += 1
            else:
                result["negative"] += 1
            return True

        # 1) Instance-anchor crops, diversified by random placement.
        for item in mask_items:
            for _ in range(ANCHOR_PER_INSTANCE):
                success = False
                for attempt in range(MAX_ATTEMPTS):
                    window = rng.choices(SOURCE_WINDOWS, weights=SOURCE_WINDOW_WEIGHTS)[0]
                    x0, y0 = pick_anchor_origin(rng, item["bbox"], height, width, window)
                    if try_save(x0, y0, window, "anchor", True):
                        success = True
                        break
                    if _space_exhausted.value:
                        return result
                if not success:
                    result["skipped"] += 1

        # 2) Random positives, including truncated non-anchor instances.
        for _ in range(RANDOM_POS_PER_RECORD):
            success = False
            for attempt in range(MAX_ATTEMPTS):
                window = rng.choices(SOURCE_WINDOWS, weights=SOURCE_WINDOW_WEIGHTS)[0]
                x0 = rng.randint(0, width - window)
                y0 = rng.randint(0, height - window)
                if try_save(x0, y0, window, "random_positive", True):
                    success = True
                    break
                if _space_exhausted.value:
                    return result
            if not success:
                result["skipped"] += 1

        # 3) Strict negatives: no box from ANY annotator on same source image.
        for _ in range(NEG_PER_RECORD):
            success = False
            for attempt in range(MAX_ATTEMPTS * 2):
                window = rng.choices(SOURCE_WINDOWS, weights=SOURCE_WINDOW_WEIGHTS)[0]
                x0 = rng.randint(0, width - window)
                y0 = rng.randint(0, height - window)
                if try_save(x0, y0, window, "negative", False):
                    success = True
                    break
                if _space_exhausted.value:
                    return result
            if not success:
                result["skipped"] += 1
        return result
    except Exception as exc:
        result["error"] = f"Record {rec_index}, {record['file_name']}: {type(exc).__name__}: {exc}"
        return result


def build():
    candidates = sorted(INPUT_ROOT.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected 1 MAGFiLO JSON, found {len(candidates)}: {candidates}")
    json_file = candidates[0]
    train_dir = json_file.parent / "train_images"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train images folder: {train_dir}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(
            f"Output folder already exists and is non-empty: {OUTPUT_DIR}. Change OUTPUT_DIR to avoid mixing runs."
        )
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "parts").mkdir(parents=True, exist_ok=True)

    with open(json_file, "r", encoding="utf-8") as f:
        source = json.load(f)
    records = source["images"]
    if SOURCES_FILE:  # ADDED: keep only the requested sources
        keep = set(json.loads(Path(SOURCES_FILE).read_text()))
        records = [r for r in records if Path(r["file_name"]).name in keep]
        print(
            f"MAGFILO_SOURCES: kept {len(records)} records of {len(source['images'])} (sources requested: {len(keep)})"
        )
    anns_by_id = defaultdict(list)
    for ann in source["annotations"]:
        anns_by_id[str(ann["image_id"])].append(ann)

    lookup = {}
    for path in train_dir.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            lookup[path.name] = str(path)
            lookup[path.stem] = str(path)

    for rec in records:
        n = Path(rec["file_name"]).name
        if n not in lookup and Path(n).stem not in lookup:
            raise FileNotFoundError(f"Train image missing: {n}")

    if VAL_RATIO > 0:
        groups = [Path(r["file_name"]).stem[:8] for r in records]
        split = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
        train_indices, val_indices = next(split.split(records, groups=groups))
    else:
        train_indices, val_indices = np.arange(len(records)), []
    train_records = [records[int(i)] for i in train_indices]
    val_records = [records[int(i)] for i in val_indices]

    # All annotation boxes for the same physical image: conservative negatives.
    all_boxes_by_stem = defaultdict(list)
    for rec in train_records:
        stem = Path(rec["file_name"]).stem
        for ann in anns_by_id.get(str(rec["id"]), []):
            x, y, w, h = map(float, ann["bbox"])
            all_boxes_by_stem[stem].append([x, y, x + w, y + h])

    tasks = []
    for i, rec in enumerate(train_records):
        stem = Path(rec["file_name"]).stem
        anns = anns_by_id.get(str(rec["id"]), [])
        name = Path(rec["file_name"]).name
        tasks.append((i, rec, anns, lookup.get(name, lookup[stem]), all_boxes_by_stem[stem]))

    with open(OUTPUT_DIR / "val_records.json", "w", encoding="utf-8") as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_json": str(json_file),
                "crop": CROP,
                "seed": SEED,
                "source_windows": SOURCE_WINDOWS,
                "source_window_weights": SOURCE_WINDOW_WEIGHTS,
                "val_ratio": VAL_RATIO,
                "anchor_per_instance": ANCHOR_PER_INSTANCE,
                "random_pos_per_record": RANDOM_POS_PER_RECORD,
                "neg_per_record": NEG_PER_RECORD,
                "min_visible_pixels": MIN_VISIBLE_PIXELS,
                "png_compression": PNG_COMPRESSION,
                "max_output_bytes": MAX_OUTPUT_BYTES,
                "n_workers": N_WORKERS,
                "train_records": len(train_records),
                "val_records": len(val_records),
            },
            f,
            indent=2,
        )

    print("JSON:", json_file)
    print(f"Train annotation records: {len(train_records):,}; held-out val records: {len(val_records):,}")
    print(f"CPU workers: {N_WORKERS}; max output budget: {MAX_OUTPUT_BYTES / 1e9:.1f} GB")
    print("Output:", OUTPUT_DIR)

    context = mp.get_context(
        "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    )  # CHANGED: spawn fallback for Windows (CPU check)
    with context.Manager() as manager:
        bytes_value = manager.Value("q", 0)
        lock = manager.Lock()
        stop_flag = manager.Value("i", 0)
        tally = {"positive": 0, "negative": 0, "skipped": 0, "bytes": 0}
        errors = []
        with context.Pool(
            processes=N_WORKERS,
            initializer=initialize_worker,
            initargs=(str(OUTPUT_DIR), bytes_value, lock, stop_flag),
        ) as pool:
            jobs = pool.imap_unordered(process_one_record, tasks, chunksize=1)
            bar = tqdm(jobs, total=len(tasks), desc="Generating crops", unit="record", dynamic_ncols=True)
            for result in bar:
                for k in tally:
                    tally[k] += result[k]
                if result["error"]:
                    errors.append(result["error"])
                bar.set_postfix(
                    pos=tally["positive"],
                    neg=tally["negative"],
                    skipped=tally["skipped"],
                    GB=f"{tally['bytes']/1e9:.2f}",
                )
        stopped_for_space = bool(stop_flag.value)

    # Merge worker shards, then remove them to avoid redundant disk usage.
    manifest = OUTPUT_DIR / "manifest.jsonl"
    with open(manifest, "wb") as combined:
        for shard in sorted((OUTPUT_DIR / "parts").glob("part_*.jsonl")):
            with open(shard, "rb") as stream:
                shutil.copyfileobj(stream, combined, length=4 * 1024 * 1024)
            shard.unlink()
    (OUTPUT_DIR / "parts").rmdir()

    total_files_bytes = sum(p.stat().st_size for p in (OUTPUT_DIR / "images").glob("*.png"))
    total_files_bytes += manifest.stat().st_size
    total_files_bytes += (OUTPUT_DIR / "config.json").stat().st_size
    total_files_bytes += (OUTPUT_DIR / "val_records.json").stat().st_size

    print("\n========== FINAL SUMMARY ==========")
    print(f"Positive crops : {tally['positive']:,}")
    print(f"Negative crops : {tally['negative']:,}")
    print(f"Skipped attempts: {tally['skipped']:,}")
    print(f"Output size    : {total_files_bytes/1e9:.3f} GB")
    print(f"Output manifest: {manifest}")
    print(f"Held-out records: {OUTPUT_DIR / 'val_records.json'}")
    if stopped_for_space:
        print("WARNING: Output budget reached; dataset is a *partial* crop set.")
    if errors:
        print(f"\nERRORS IN {len(errors)} RECORDS (first 20):")
        for error in errors[:20]:
            print(error)
        raise RuntimeError(f"Crop generation encountered {len(errors)} record errors. Inspect before training.")
    if total_files_bytes >= 19_000_000_000:
        raise RuntimeError("Output exceeded 19 GB: inspect files.")
    print("Done.")


if __name__ == "__main__":
    build()
