"""Post-processing - from per-crop probabilities to exclusive full-image instances.

1. Gate: keep a crop when its round-3 presence >= .5.
2. Resample the round-3 probability from the 512 letterbox to the crop's native size (bilinear), threshold > .5.
   (Thresholding at native resolution instead of at 512 + nearest upsampling gained +.014 PQ in 24/24 model-folds.)
3. Keep the connected component with the largest overlap with the anchor (the instance this prompt asked for);
   fall back to the largest if none overlaps, and drop if smaller than 128 px.
4. Greedy de-duplication over all crops of the image, in descending presence order: skip a candidate whose IoU with an
   already accepted instance is > .5, otherwise remove pixels already taken and keep it if >= 128 px remain.
Instances are pixel-disjoint. Overlapping GT can still make all-pairs and strict one-to-one scores differ.
"""

import cv2
import numpy as np

from .config import PostprocessConfig, IMAGE_SIZE

SHAPE = (IMAGE_SIZE, IMAGE_SIZE)


def native_probability(c):
    """Round-3 probability resampled (bilinear) from the letterbox to the native crop box."""
    x0, y0, x1, y1 = map(int, c['box'])
    nw, nh = map(int, c['resized_wh'])
    p = c['r3'][:nh, :nw].astype(np.float32) / 255.0
    return np.clip(cv2.resize(p, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR), 0, 1)


def tight(y0, x0, m):
    ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    if not len(ys):
        return None
    return y0 + int(ys[0]), x0 + int(xs[0]), m[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]


def select_anchor_component(m, anchor, min_area):
    """The 8-connected component of mask m overlapping the anchor most (largest component if none overlaps)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8, cv2.CV_32S)
    if n <= 1:
        return None
    overlap = np.bincount(lab[anchor], minlength=n)
    overlap[0] = 0
    k = int(overlap.argmax()) if overlap.max() > 0 else 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    keep = lab == k
    return keep if keep.sum() >= min_area else None


def candidates(crops, cfg=PostprocessConfig()):
    out = []
    for c in crops:
        s = float(c['presence'][1])
        if s < cfg.presence_threshold:
            continue
        keep = select_anchor_component(native_probability(c) > cfg.mask_threshold, c['native_anchor'], cfg.min_area)
        if keep is None:
            continue
        x0, y0 = int(c['box'][0]), int(c['box'][1])
        t = tight(y0, x0, keep)
        out.append(dict(y0=t[0], x0=t[1], m=t[2], score=s, crop_id=str(c['crop_id'])))
    return out


def _inter(a, b):
    (ay, ax, am), (by, bx, bm) = a, b
    y0, x0 = max(ay, by), max(ax, bx)
    y1, x1 = min(ay + am.shape[0], by + bm.shape[0]), min(ax + am.shape[1], bx + bm.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0
    return int((am[y0 - ay : y1 - ay, x0 - ax : x1 - ax] & bm[y0 - by : y1 - by, x0 - bx : x1 - bx]).sum())


def merge_nms(cands, cfg=PostprocessConfig()):
    """In (presence desc, crop_id) order: drop a candidate with IoU > nms_iou against an accepted instance, otherwise
    remove occupied pixels and keep it if >= final_area pixels remain. Returns [(y0, x0, local mask, score)]."""
    accepted, occupied = [], np.zeros(SHAPE, bool)
    for c in sorted(cands, key=lambda x: (-x['score'], x['crop_id'])):
        y0, x0, m = c['y0'], c['x0'], c['m']
        area = int(m.sum())
        if any(
            (i := _inter((y0, x0, m), (a[0], a[1], a[2]))) and i / (area + a[4] - i) > cfg.nms_iou for a in accepted
        ):
            continue
        m = m & ~occupied[y0 : y0 + m.shape[0], x0 : x0 + m.shape[1]]
        if m.sum() < cfg.final_area:
            continue
        occupied[y0 : y0 + m.shape[0], x0 : x0 + m.shape[1]] |= m
        accepted.append((y0, x0, m, c['score'], int(m.sum())))
    return [a[:4] for a in accepted]


def postprocess(crops, cfg=PostprocessConfig()):
    """Refined crops of ONE image -> list of instances (y0, x0, bbox-local bool mask, score)."""
    return merge_nms(candidates(crops, cfg), cfg)


def to_full(instance):
    y0, x0, m = instance[:3]
    full = np.zeros(SHAPE, bool)
    full[y0 : y0 + m.shape[0], x0 : x0 + m.shape[1]] = m
    return full
