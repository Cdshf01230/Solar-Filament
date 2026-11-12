"""Stage 2 - candidate components and the LightGBM keep/drop classifier.

The fold-mean proposal map is thresholded at a deliberately low level (.05) so that almost every filament is touched
by at least one 8-connected component. Each component (area >= 8 px) is described by 32 hand-crafted features
(shape, probability statistics, raw-image darkness/texture, local context) and scored by five LightGBM classifiers
(one per fold). The default balanced policy supplies refiner crops; the optional recall policy uses 48 times the positive
class weight for candidate review. Both keep components with mean score >= .5.
"""

from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi

from .config import ComponentConfig, IMAGE_SIZE

FEATURES = [
    'log_area',
    'log_bbox_width',
    'log_bbox_height',
    'log_aspect',
    'bbox_fill',
    'center_x',
    'center_y',
    'edge_distance',
    'prob_mean',
    'prob_std',
    'prob_min',
    'prob_max',
    'prob_gt_01',
    'prob_gt_02',
    'prob_gt_05',
    'prob_gt_08',
    'raw_mean',
    'raw_std',
    'dark5_mean',
    'dark15_mean',
    'texture5_mean',
    'gradient_mean',
    'blackhat15_mean',
    'blackhat31_mean',
    'edt_mean',
    'edt_max',
    'log_elongation',
    'bbox_context_raw_mean',
    'bbox_context_raw_std',
    'bbox_context_prob_mean',
    'raw_context_contrast',
    'threshold',
]


def components(prob, cfg=ComponentConfig()):
    """8-connected components of prob >= threshold with area >= min_area.
    Returns (n_labels, label image, stats, centroids, kept ids, boxes [x0, y0, x1, y1])."""
    n, lab, stats, centers = cv2.connectedComponentsWithStats((prob >= cfg.threshold).astype(np.uint8), 8, cv2.CV_32S)
    ids = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= cfg.min_area)
    ids = ids[ids > 0]
    s = stats[ids]
    boxes = np.column_stack([s[:, 0], s[:, 1], s[:, 0] + s[:, 2], s[:, 1] + s[:, 3]]).astype(np.int32)
    return n, lab, stats, centers, ids, boxes


def appearance(raw):
    """Raw-image maps: intensity, darkness vs Gaussian background (sigma 5, 15), local texture, gradient, black-hats."""
    f = raw.astype(np.float32) / 255
    g5 = cv2.GaussianBlur(f, (0, 0), 5)
    g15 = cv2.GaussianBlur(f, (0, 0), 15)
    mean = cv2.blur(f, (5, 5))
    texture = np.sqrt(np.maximum(cv2.blur(f * f, (5, 5)) - mean * mean, 0))
    grad = cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1))
    bh = [
        cv2.morphologyEx(raw, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (v, v))).astype(
            np.float32
        )
        / 255
        for v in (15, 31)
    ]
    return [f, g5 - f, g15 - f, texture, grad, *bh]


def _rect_sums(integral, boxes):
    x0, y0, x1, y1 = boxes.T
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def _pad_boxes(boxes, pad, shape):
    h, w = shape
    b = boxes.copy()
    b[:, :2] -= pad
    b[:, 2:] += pad
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b


def feature_matrix(prob, raw, maps, threshold, comps):
    """(n_components, 32) float32 feature matrix, column order = FEATURES."""
    n, lab, stats, centers, ids, boxes = comps
    area = stats[ids, cv2.CC_STAT_AREA].astype(float)
    flat = lab.ravel()

    def means(v):
        return np.bincount(flat, weights=v.ravel(), minlength=n)[ids] / area

    pm = means(prob)
    ps = np.sqrt(np.maximum(means(prob * prob) - pm * pm, 0))
    rm = means(maps[0])
    rs = np.sqrt(np.maximum(means(maps[0] * maps[0]) - rm * rm, 0))
    pmin = np.asarray(ndi.minimum(prob, lab, index=ids))
    pmax = np.asarray(ndi.maximum(prob, lab, index=ids))
    distance = cv2.distanceTransform((lab > 0).astype(np.uint8), cv2.DIST_L2, 5)
    edtm = means(distance)
    edtx = np.asarray(ndi.maximum(distance, lab, index=ids))
    yy, xx = np.indices(prob.shape, dtype=np.float32)
    mx = means(xx)
    my = means(yy)
    vx = np.maximum(means(xx * xx) - mx * mx, 0)
    vy = np.maximum(means(yy * yy) - my * my, 0)
    cov = means(xx * yy) - mx * my
    disc = np.sqrt(np.maximum((vx - vy) ** 2 + 4 * cov * cov, 0))
    hi = (vx + vy + disc) / 2
    lo = np.maximum((vx + vy - disc) / 2, 0)
    elong = np.sqrt((hi + 0.1) / (lo + 0.1))
    context = _pad_boxes(boxes, 16, prob.shape)
    carea = np.maximum((context[:, 2] - context[:, 0]) * (context[:, 3] - context[:, 1]), 1)
    cm = _rect_sums(cv2.integral(maps[0], sdepth=cv2.CV_64F), context) / carea
    cs = np.sqrt(
        np.maximum(_rect_sums(cv2.integral(maps[0] * maps[0], sdepth=cv2.CV_64F), context) / carea - cm * cm, 0)
    )
    cp = _rect_sums(cv2.integral(prob, sdepth=cv2.CV_64F), context) / carea
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    cx = centers[ids, 0]
    cy = centers[ids, 1]
    columns = [
        np.log1p(area),
        np.log1p(w),
        np.log1p(h),
        np.log(w / np.maximum(1, h)),
        area / (w * h),
        cx / IMAGE_SIZE,
        cy / IMAGE_SIZE,
        np.minimum.reduce([boxes[:, 0], boxes[:, 1], IMAGE_SIZE - boxes[:, 2], IMAGE_SIZE - boxes[:, 3]]) / IMAGE_SIZE,
        pm,
        ps,
        pmin,
        pmax,
        *[means((prob > v).astype(np.float32)) for v in (0.1, 0.2, 0.5, 0.8)],
        rm,
        rs,
        *[means(v) for v in maps[1:]],
        edtm,
        edtx,
        np.log1p(elong),
        cm,
        cs,
        cp,
        cm - rm,
        np.full(len(ids), threshold),
    ]
    X = np.column_stack(columns).astype(np.float32)
    if X.shape != (len(ids), len(FEATURES)) or not np.isfinite(X).all():
        raise ValueError(f'invalid component features: shape {X.shape}, all finite {bool(np.isfinite(X).all())}')
    return X


class _TextBooster:
    """LightGBM text model with the predict_proba interface of the scikit-learn wrapper (identical scores, checked by
    scripts/export_lightgbm.py)."""

    def __init__(self, path):
        import lightgbm

        self.booster = lightgbm.Booster(model_file=str(path))

    def predict_proba(self, X):
        p = self.booster.predict(X)
        return np.column_stack([1 - p, p])


def load_classifiers(weights_dir, folds, fmt='text', policy='balance'):
    """fmt='text': portable LightGBM text models (default, no pickle); fmt='joblib': the original pickled
    LGBMClassifier objects (needs lightgbm 4.6 / scikit-learn 1.6)."""
    if policy not in ('balance', 'recall'):
        raise ValueError(f'Unknown classifier policy: {policy}')
    d = Path(weights_dir) / 'ml'
    if fmt == 'text':
        return {k: _TextBooster(d / f'{policy}_fold{k}.txt') for k in folds}
    if fmt != 'joblib':
        raise ValueError(f'Unknown classifier format: {fmt}')
    import joblib

    return {k: joblib.load(d / f'{policy}_fold{k}.joblib') for k in folds}


def classifier_scores(prob, raw, comps, classifiers, cfg=ComponentConfig()):
    """Mean keep-probability of the fold classifiers for every component (ascending fold order)."""
    if not len(comps[4]):
        return np.zeros(0, np.float32)
    X = feature_matrix(prob, raw, appearance(raw), cfg.threshold, comps)
    mean = np.zeros(len(X), np.float32)
    for k in sorted(classifiers):
        mean += classifiers[k].predict_proba(X)[:, 1].astype(np.float32)
    return mean / len(classifiers)
