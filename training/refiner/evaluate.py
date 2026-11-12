"""GT-free native instance postprocessing and two explicitly separate PQ scorers."""

from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import cv2
import numpy as np
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment


def encode(mask):
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=r['size'], counts=r['counts'].decode())


def native_candidate(
    mask, anchor, box, resized_wh, score, crop_id, shape=(2048, 2048), native_anchor=None, baseline=False
):
    x0, y0, x1, y1 = map(int, box)
    nw, nh = resized_wh
    m = cv2.resize(mask[:nh, :nw].astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    a = (
        native_anchor.astype(np.uint8)
        if native_anchor is not None
        else cv2.resize(anchor[:nh, :nw].astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    )
    if baseline:
        m = a.copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8, cv2.CV_32S)
    if n <= 1:
        return None
    overlap = np.bincount(lab[a > 0], minlength=n)
    overlap[0] = 0
    k = int(overlap.argmax()) if overlap.max() > 0 else 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    if stats[k, cv2.CC_STAT_AREA] < 128:
        return None
    full = np.zeros(shape, np.uint8)
    full[y0:y1, x0:x1] = lab == k
    return dict(crop_id=crop_id, score=float(score), rle=encode(full))


def merge(candidates):
    """NMS against accepted original masks, then score-ordered exclusive pixels."""
    accepted, output = [], []
    occupied = None
    for c in sorted(candidates, key=lambda x: (-x['score'], x['crop_id'])):
        rle = c['rle']
        if accepted and float(mu.iou([rle], accepted, [0] * len(accepted)).max()) > 0.5:
            continue
        m = mu.decode(rle).astype(bool)
        if occupied is None:
            occupied = np.zeros_like(m)
        m &= ~occupied
        if m.sum() < 128:
            continue
        accepted.append(encode(m))
        occupied |= m
        output.append(encode(m))
    return output


def match(iou):
    ng, npred = iou.shape
    q = iou > 0.5
    legacy = [int(q.sum()), int((~q.any(0)).sum()), int((~q.any(1)).sum()), float(iou[q].astype(float).sum())]
    if ng and npred:
        benefit = q * (1 + iou / (min(ng, npred) + 1))
        a, b = linear_sum_assignment(benefit, maximize=True)
        keep = q[a, b]
        tp = int(keep.sum())
        si = float(iou[a[keep], b[keep]].astype(float).sum())
    else:
        tp, si = 0, 0.0
    return legacy, [tp, npred - tp, ng - tp, si]


def metric(c):
    tp, fp, fn, si = map(float, c)
    den = tp + 0.5 * (fp + fn)
    return dict(
        pq=si / max(1e-12, den),
        tp=tp,
        fp=fp,
        fn=fn,
        sum_iou=si,
        precision=tp / max(1, tp + fp),
        recall=tp / max(1, tp + fn),
    )


def gt_rle(ann, rec):
    seg = ann['segmentation']
    seg = json.loads(seg) if isinstance(seg, str) else seg
    if isinstance(seg, list):
        return mu.merge(mu.frPyObjects(seg, rec['height'], rec['width']))
    r = dict(seg)
    if isinstance(r['counts'], list):
        r = mu.frPyObjects(r, rec['height'], rec['width'])
    return r


RECORDS, ANNOTATIONS = None, None


def initialize(annotation_path):
    global RECORDS, ANNOTATIONS
    cv2.setNumThreads(1)
    source = json.loads(Path(annotation_path).read_text(encoding='utf-8'))
    RECORDS, ANNOTATIONS = defaultdict(list), defaultdict(list)
    for r in source['images']:
        RECORDS[Path(r['file_name']).name].append(r)
    for a in source['annotations']:
        ANNOTATIONS[str(a['image_id'])].append(a)


def score_source(task):
    source, paths = task
    candidates = []
    for p in paths:
        candidates.extend(json.loads(Path(p).read_text(encoding='utf-8')))
    predictions = merge(candidates)
    legacy, one, rows = np.zeros(4), np.zeros(4), []
    for rec in RECORDS[source]:
        gt = [gt_rle(a, rec) for a in ANNOTATIONS[str(rec['id'])]]
        gt = [g for g in gt if int(mu.area(g)) > 0]
        iou = (
            np.asarray(mu.iou(gt, predictions, [0] * len(predictions)), np.float32).reshape(len(gt), len(predictions))
            if gt and predictions
            else np.zeros((len(gt), len(predictions)), np.float32)
        )
        old, new = match(iou)
        if gt:
            legacy += old
        one += new
        rows.append(
            dict(
                record=rec['id'],
                gt=len(gt),
                predictions=len(predictions),
                legacy_included=bool(gt),
                legacy=old,
                one_to_one=new,
            )
        )
    return dict(source=source, legacy=legacy.tolist(), one_to_one=one.tolist(), rows=rows)


def full_image(annotation_path, sources, prediction_dir, out, workers=4):
    tasks = []
    for source in sources:
        paths = sorted(str(p) for p in Path(prediction_dir).glob(Path(source).stem + '__*.json'))
        tasks.append((source, paths))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize,
        initargs=(str(annotation_path),),
        mp_context=multiprocessing.get_context('spawn'),
    ) as pool:
        rows = list(pool.map(score_source, tasks))
    result = dict(
        sources=len(sources),
        records=sum(len(r['rows']) for r in rows),
        legacy=metric(np.sum([r['legacy'] for r in rows], axis=0)),
        one_to_one=metric(np.sum([r['one_to_one'] for r in rows], axis=0)),
        rows=rows,
    )
    Path(out).write_text(json.dumps(result), encoding='utf-8')
    return {k: v for k, v in result.items() if k != 'rows'}
