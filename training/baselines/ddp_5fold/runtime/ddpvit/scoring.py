"""Full-image scoring: threshold .5 (fixed), 8-connected components, min area 128, two scorers per record.

legacy : all pairs IoU>.5 (host/self-evaluation semantics): TP = #pairs, FP = components hitting no GT,
         FN = GT hit by no component. Records with no GT are NOT scored (as in the public notebook).
strict : one-to-one matching per ORIGINAL annotation record (Hungarian, only IoU>.5 pairs can match,
         max-cardinality first then IoU): TP = matched pairs, FP = unmatched components, FN = unmatched GT.
         Records with no GT ARE scored (every component is FP).
Semantics copied from experiments/refiner_rcr_v1/evaluate.py (`match`, `metric`).
"""
import gzip
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from u0_multihead.ddp_evaluation import Context as BaseContext
from u0_multihead.evaluation import infer_shared

THRESHOLD = 0.5
MIN_AREA = 128


def match(iou):
    """iou: (n_gt, n_pred) float. Returns (legacy, strict) as [tp, fp, fn, sum_iou]."""
    ng, npred = iou.shape
    q = iou > .5
    legacy = [int(q.sum()), int((~q.any(0)).sum()), int((~q.any(1)).sum()), float(iou[q].astype(float).sum())]
    if ng and npred:
        benefit = q * (1 + iou / (min(ng, npred) + 1))
        a, b = linear_sum_assignment(benefit, maximize=True)
        keep = q[a, b]
        tp = int(keep.sum())
        si = float(iou[a[keep], b[keep]].astype(float).sum())
    else:
        tp, si = 0, 0.
    return legacy, [tp, npred - tp, ng - tp, si]


def metric(c):
    tp, fp, fn, si = map(float, c)
    den = tp + .5 * (fp + fn)
    return dict(pq=si / max(1e-12, den), tp=tp, fp=fp, fn=fn, sum_iou=si,
                sq=si / max(1., tp), rq=tp / max(1e-12, den),
                precision=tp / max(1., tp + fp), recall=tp / max(1., tp + fn))


def components(prob, threshold=THRESHOLD, min_area=MIN_AREA):
    """8-connected components of prob>=threshold; returns (label image, kept ids, areas)."""
    lab, _ = ndi.label(prob >= threshold, np.ones((3, 3), np.uint8))
    areas = np.bincount(lab.ravel())
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep > 0]
    return lab, keep, areas


def iou_matrix(masks, lab, keep, areas, shape):
    ga = np.array([m.sum() for m in masks], dtype=np.int64)
    inter = np.array([np.bincount(lab[m], minlength=len(areas))[keep] for m in masks],
                     dtype=np.int64).reshape(len(masks), len(keep))
    den = ga[:, None] + areas[keep][None] - inter
    return np.divide(inter, den, out=np.zeros(inter.shape, float), where=den > 0).astype(np.float32)


class StrictContext(BaseContext):
    """Same annotation parsing as the existing Context; new score(): one row per record, both scorers."""
    def score(self, name, prob, thresholds=None):
        lab, keep, areas = components(prob)
        rows = []
        for rec in self.records[name]:
            masks = self.masks(rec)
            iou = iou_matrix(masks, lab, keep, areas, prob.shape) if masks else np.zeros((0, len(keep)), np.float32)
            legacy, strict = match(iou)
            rows.append(dict(source_file=name, record=rec['id'], gt=len(masks), predictions=int(len(keep)),
                             legacy_included=bool(masks), legacy=legacy, strict=strict))
        return rows


def summarize(rows):
    """Aggregate by summing TP/FP/FN/sum_iou over all records (never a mean of per-image PQ)."""
    legacy = np.sum([r['legacy'] for r in rows if r['legacy_included']], axis=0) if rows else np.zeros(4)
    strict = np.sum([r['strict'] for r in rows], axis=0) if rows else np.zeros(4)
    return dict(sources=len({r['source_file'] for r in rows}), records=len(rows),
                gt=int(sum(r['gt'] for r in rows)),
                legacy_records=int(sum(r['legacy_included'] for r in rows)),
                strict=metric(strict), legacy=metric(legacy),
                threshold=THRESHOLD, min_area=MIN_AREA)


def assert_expected(summary, sources, records, gt):
    want = (sources, records, gt)
    got = tuple(g if w is not None else None for g, w in zip((summary['sources'], summary['records'], summary['gt']), want))
    if got != want:
        raise AssertionError(f'evaluation totals {got} != expected {(sources, records, gt)}')


def write_rows(rows, path):
    """One compact jsonl.gz per evaluation (never one file per image)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for r in sorted(rows, key=lambda r: (r['source_file'], str(r['record']))):
            f.write(json.dumps(dict(source=r['source_file'], record=r['record'], gt=r['gt'],
                                    predictions=r['predictions'], legacy_included=r['legacy_included'],
                                    strict=r['strict'], legacy=r['legacy'])) + '\n')


@torch.inference_mode()
def score_names(model, names, context, workers=2, tile_batch=1, progress=None):
    """Infer + score `names` on the local rank. Returns the list of per-record rows."""
    model.eval()
    rows, pending, done = [], {}, 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        def collect(block):
            nonlocal done
            ready, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=None if block else 0)
            for fut in ready:
                pending.pop(fut)
                rows.extend(fut.result())
                done += 1
                if progress and done % 8 == 0:
                    progress(done, len(names))
        for source in names:
            image = cv2.imread(str(context.base / 'data/train_images' / source), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(source)
            prob = infer_shared(model, image, tile_batch)['ddp']
            pending[pool.submit(context.score, source, prob)] = source
            collect(False)
            while len(pending) >= max(2, workers * 2):
                collect(True)
        while pending:
            collect(True)
    return rows
