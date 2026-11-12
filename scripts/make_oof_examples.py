"""Report figure: out-of-fold predictions against the annotation on three labelled images, chosen by rank, not by eye.

    python scripts/make_oof_examples.py --predictions reference/oof_predictions.csv \
        --annotations /data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json --images /data/train/train_images \
        --per-image docs/results/oof/oof_per_image.json --out report/fig_examples.pdf

Candidates are the images with a single annotation record and at least six filaments. They are sorted by their strict
PQ and the images at the 25th, 50th and 75th percentiles are shown, each as a window of 896 px around the centre of
its annotated filaments. Predictions are filled green when they are a strict true positive (one-to-one, IoU > 0.5)
and orange otherwise; annotated filaments are outlined in white when matched and in magenta when missed.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from evaluate_oof import gt_records, read_predictions  # noqa: E402
from filament_pq import metrics  # noqa: E402

WINDOW = 896
TP_FILL, FP_FILL = np.array([0.10, 0.75, 0.30]), np.array([1.0, 0.55, 0.0])
GT_HIT, GT_MISS = (1.0, 1.0, 1.0), (1.0, 0.0, 1.0)


def strict_pairs(gt, pred):
    """Indices (gt, prediction) of the strict one-to-one true positives, as in filament_pq.metrics.match."""
    iou = metrics.iou_matrix(gt, pred)
    if not iou.size:
        return []
    q = iou > 0.5
    a, b = linear_sum_assignment(q * (1 + iou / (min(q.shape) + 1)), maximize=True)
    return [(i, j) for i, j in zip(a, b) if q[i, j]]


def render(gray, gt, pred, y0, x0):
    window = (slice(y0, y0 + WINDOW), slice(x0, x0 + WINDOW))
    rgb = np.repeat(gray[window][..., None].astype(np.float32) / 255, 3, axis=2)
    pairs = strict_pairs(gt, pred)
    matched_gt, matched_pred = {i for i, _ in pairs}, {j for _, j in pairs}
    for j, r in enumerate(pred):
        m = mu.decode(r)[window].astype(bool)
        rgb[m] = 0.4 * rgb[m] + 0.6 * (TP_FILL if j in matched_pred else FP_FILL)
    for i, r in enumerate(gt):
        m = mu.decode(r)[window].astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cs, -1, GT_HIT if i in matched_gt else GT_MISS, 2)
    return np.clip(rgb, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--predictions', nargs='+', required=True)
    ap.add_argument('--annotations', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--per-image', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    preds = read_predictions(a.predictions)
    records = gt_records(a.annotations)
    per_image = json.loads(Path(a.per_image).read_text(encoding='utf-8'))
    cand = [s for s, v in records.items() if len(v) == 1 and len(v[0][1]) >= 6]
    cand.sort(key=lambda s: (metrics.pq(per_image[s]['strict'])['pq'], s))
    quantiles = (0.25, 0.5, 0.75)
    picks = [cand[int(round(q * (len(cand) - 1)))] for q in quantiles]

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.42))
    info = []
    for ax, q, stem in zip(axs, quantiles, picks):
        gt, pr = records[stem][0][1], preds.get(stem, [])
        gray = cv2.imread(str(Path(a.images) / f'{stem}.jpeg'), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.nonzero(mu.decode(mu.merge(gt)))
        y0 = int(np.clip(np.median(ys) - WINDOW // 2, 0, gray.shape[0] - WINDOW))
        x0 = int(np.clip(np.median(xs) - WINDOW // 2, 0, gray.shape[1] - WINDOW))
        c = per_image[stem]['strict']
        pq = metrics.pq(c)['pq']
        ax.imshow(render(gray, gt, pr, y0, x0), interpolation='lanczos')
        ax.set_title(
            f'{int(q * 100)}th percentile: PQ {pq:.2f} (TP {c[0]:.0f}, FP {c[1]:.0f}, FN {c[2]:.0f})',
            fontsize=6,
            pad=2,
        )
        ax.axis('off')
        info.append(dict(image=stem, quantile=q, pq=pq, tp=c[0], fp=c[1], fn=c[2], window=[y0, x0, WINDOW]))
    fig.tight_layout(pad=0.1, w_pad=0.3)
    out = Path(a.out)
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight', pad_inches=0.02)
    out.with_suffix('.json').write_text(json.dumps(dict(candidates=len(cand), examples=info), indent=1))
    print(json.dumps(info))


if __name__ == '__main__':
    main()
