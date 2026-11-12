"""Shared-backbone full-image inference and the existing public-notebook PQ rule."""

import json
import os
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from pycocotools import mask as mu
from .data import MEAN, STD
from .model import autocast

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def pixel_metrics(c):
    tp, fp, fn, tn = map(float, c)
    fg, bg = tp / max(1, tp + fp + fn), tn / max(1, tn + fp + fn)
    return dict(
        precision=tp / max(1, tp + fp),
        recall=tp / max(1, tp + fn),
        foreground_iou=fg,
        background_iou=bg,
        miou=(fg + bg) / 2,
        dice=2 * tp / max(1, 2 * tp + fp + fn),
    )


class Context:
    def __init__(self, base):
        self.base = Path(base)
        source = json.loads((self.base / 'data/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
        self.records, self.annotations = defaultdict(list), defaultdict(list)
        for rec in source['images']:
            self.records[Path(rec['file_name']).name].append(rec)
        for ann in source['annotations']:
            self.annotations[str(ann['image_id'])].append(ann)

    def masks(self, rec):
        masks = []
        for ann in self.annotations[str(rec['id'])]:
            seg = ann['segmentation']
            if isinstance(seg, str):
                seg = json.loads(seg)
            if isinstance(seg, list):
                rle = mu.merge(mu.frPyObjects(seg, rec['height'], rec['width']))
            else:
                rle = dict(seg)
                if isinstance(rle['counts'], list):
                    rle = mu.frPyObjects(rle, rec['height'], rec['width'])
                elif isinstance(rle['counts'], str):
                    rle['counts'] = rle['counts'].encode()
            if int(mu.area(rle)) > 0:
                masks.append(mu.decode(rle).astype(bool))
        return masks

    def score(self, name, prob, thresholds):
        groups = [(r['id'], self.masks(r)) for r in self.records[name]]
        rows = []
        for threshold in thresholds:
            binary = prob >= threshold
            lab, _ = ndi.label(binary, np.ones((3, 3), np.uint8))
            areas = np.bincount(lab.ravel())
            keep = np.flatnonzero(areas >= 8)
            keep = keep[keep > 0]
            for record, masks in groups:
                ga = np.array([m.sum() for m in masks], dtype=np.int64)
                inter = np.array(
                    [np.bincount(lab[m], minlength=len(areas))[keep] for m in masks], dtype=np.int64
                ).reshape(len(masks), len(keep))
                den = ga[:, None] + areas[keep][None] - inter
                iou = np.divide(inter, den, out=np.zeros(inter.shape, float), where=den > 0).astype(np.float32)
                hit = iou > 0.5  # Public notebook: all pairs, no Hungarian matching.
                stats = [
                    int(hit.sum()),
                    int((~hit.any(0)).sum()),
                    int((~hit.any(1)).sum()),
                    float(iou[hit].astype(float).sum()),
                ]
                union = np.logical_or.reduce(masks) if masks else np.zeros(prob.shape, bool)
                sig = (inter >= 8) & (inter >= 0.05 * ga[:, None])
                rows.append(
                    dict(
                        source_file=name,
                        record=record,
                        threshold=threshold,
                        stats=stats,
                        pixels=[
                            int((binary & union).sum()),
                            int((binary & ~union).sum()),
                            int((~binary & union).sum()),
                            int((~binary & ~union).sum()),
                        ],
                        included_in_public_pq=bool(masks),
                        gt=len(masks),
                        predictions=len(keep),
                        split_gt=int((sig.sum(1) >= 2).sum()),
                        merge_predictions=int((sig.sum(0) >= 2).sum()),
                    )
                )
        return rows


def summarize(rows, threshold):
    selected = [r for r in rows if r['threshold'] == threshold]
    scored = [r for r in selected if r['included_in_public_pq']]
    stats = np.sum([r['stats'] for r in scored], axis=0) if scored else np.zeros(4)
    tp, fp, fn, si = map(float, stats)
    images = defaultdict(list)
    for r in selected:
        images[r['source_file']].append(r['pixels'])
    pixels = np.sum([np.mean(v, axis=0) for v in images.values()], axis=0) if images else np.zeros(4)
    return dict(
        threshold=threshold,
        instance=dict(
            pq=si / max(1e-12, tp + 0.5 * (fp + fn)),
            precision=tp / max(1, tp + fp),
            recall=tp / max(1, tp + fn),
            tp=tp,
            fp=fp,
            fn=fn,
        ),
        pixel=pixel_metrics(pixels),
        images=len(images),
        annotator_records=len(scored),
        gt_instances=sum(r['gt'] for r in scored),
        split_gt=sum(r['split_gt'] for r in scored),
        merge_predictions=sum(r['merge_predictions'] for r in scored),
    )


@torch.inference_mode()
def infer_shared(model, image, tile_batch=2):
    """One backbone forward per tile microbatch, then ALL heads sequentially."""
    from .preprocessing import PREPROCESS_HEADS, infer_preprocessed

    if any(name in PREPROCESS_HEADS for name in model.heads):
        return infer_preprocessed(model, image, tile_batch)
    if image.shape != (2048, 2048):
        raise ValueError(f'Expected 2048x2048 source, got {image.shape}')
    device = next(model.heads.parameters()).device
    positions = [(y, x) for y in (0, 640, 1280) for x in (0, 640, 1280)]
    window = np.maximum(np.outer(np.hanning(768), np.hanning(768)), 0.05).astype(np.float32)
    numerator = {name: np.zeros((2048, 2048), np.float32) for name in model.heads}
    denominator = np.zeros((2048, 2048), np.float32)
    for start in range(0, len(positions), tile_batch):
        coords = positions[start : start + tile_batch]
        inputs = [
            np.repeat(image[y : y + 768, x : x + 768][None], 3, axis=0).astype(np.float32) / 255.0 for y, x in coords
        ]
        batch = torch.from_numpy(np.stack([(x - MEAN) / STD for x in inputs])).to(device)
        batch = batch.contiguous(memory_format=torch.channels_last)
        features = model.features(batch)
        for name, head in model.heads.items():
            with autocast(device):
                p = head(features, batch).float().sigmoid()[:, 0].cpu().numpy()
            for pp, (y, x) in zip(p, coords):
                numerator[name][y : y + 768, x : x + 768] += pp * window
        for y, x in coords:
            denominator[y : y + 768, x : x + 768] += window
    # Preserve the exact probability rounding convention of the old reports.
    return {name: (p / denominator).astype(np.float16).astype(np.float32) for name, p in numerator.items()}


@torch.inference_mode()
def evaluate(
    model, names, context, thresholds_by_head=None, selected_thresholds=None, workers=6, tile_batch=2, progress=None
):
    model.eval()
    rows = {name: [] for name in model.heads}
    thresholds_by_head = thresholds_by_head or {name: THRESHOLDS for name in model.heads}
    started, done = time.perf_counter(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def collect(block=False):
            nonlocal done
            ready, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=None if block else 0)
            for future in ready:
                name = pending.pop(future)
                rows[name].extend(future.result())
                done += 1
                if progress and (done % 8 == 0 or done == len(names) * len(model.heads)):
                    progress(done, len(names) * len(model.heads))

        for source in names:
            image = cv2.imread(str(context.base / 'data/train_images' / source), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(source)
            maps = infer_shared(model, image, tile_batch)
            for name, probability in maps.items():
                export = os.environ.get('SOLAR_EXPORT_PREDICTIONS')
                if export:
                    folder = Path(export)
                    folder.mkdir(parents=True, exist_ok=True)
                    dest = folder / (source + '.npz')
                    temp = dest.with_name(dest.name + '.tmp')
                    assert np.isfinite(probability).all() and probability.shape == (2048, 2048)
                    with temp.open('wb') as stream:
                        np.savez_compressed(stream, probability=probability.astype(np.float16))
                    os.replace(temp, dest)
                pending[pool.submit(context.score, source, probability, thresholds_by_head[name])] = name
            collect()
            while len(pending) >= max(len(model.heads), workers * 2):
                collect(block=True)
        while pending:
            collect(block=True)
    summaries = {}
    for name, rs in rows.items():
        rs.sort(key=lambda r: (r['source_file'], str(r['record']), r['threshold']))
        sweep = [summarize(rs, t) for t in thresholds_by_head[name]]
        if selected_thresholds is None:
            best = max(
                sweep, key=lambda r: (r['instance']['pq'], r['instance']['precision'], -abs(r['threshold'] - 0.5))
            )
        else:
            best = next(r for r in sweep if r['threshold'] == selected_thresholds[name])
        summaries[name] = dict(selected=best, sweep=sweep)
    return dict(seconds=time.perf_counter() - started, heads=summaries), rows
