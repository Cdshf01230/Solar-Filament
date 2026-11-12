"""Illustrate a stored owner-excluded Stage-1 probability map against one original GT record.

No model inference or filtering: the map must be the canonical mean of four non-owner heads.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_task_figure import decode
from filament_pq.components import components


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--map', required=True)
    ap.add_argument('--image', default='20220710085152Th.jpeg')
    ap.add_argument('--record', help='Required if image has multiple annotation records')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    data = Path(a.data)
    ann = json.loads((data / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    records = [r for r in ann['images'] if Path(r['file_name']).name == a.image]
    if a.record:
        records = [r for r in records if str(r['id']) == a.record]
    assert len(records) == 1, 'Choose one original annotation record with --record'
    rec = records[0]
    masks = [decode(x, rec) for x in ann['annotations'] if str(x['image_id']) == str(rec['id'])]
    masks = [m for m in masks if m.any()]
    image = cv2.imread(str(data / 'train_images' / a.image), cv2.IMREAD_GRAYSCALE)
    with np.load(a.map, allow_pickle=False) as z:
        prob = z['probability']
    assert image.shape == prob.shape == (2048, 2048)
    assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
    n, lab, stats, _, ids, _ = components(prob)
    proposal = np.isin(lab, ids)
    gt = np.logical_or.reduce(masks)
    touched = sum(bool((m & proposal).any()) for m in masks)
    hit = sum(bool((gt & (lab == k)).any()) for k in ids)
    pixel_recall = int((gt & proposal).sum()) / int(gt.sum())
    assignment = json.loads((ROOT / 'splits/fold_assignment.json').read_text())
    owner = next(f['fold'] for f in assignment['folds'] if a.image in f['train_sources'])
    # Match the existing task figure: a 720px window centred on the largest GT.
    ys, xs = np.nonzero(max(masks, key=lambda m: m.sum()))
    cy, cx = int(ys.mean()), int(xs.mean())
    x0, x1, y0, y1 = max(cx - 360, 0), min(cx + 360, 2048), max(cy - 360, 0), min(cy + 360, 2048)
    # Reuse the task figure's exact palette, layout, titles, ROI and export scale.
    rng = np.random.default_rng(3)
    cols = np.concatenate([[[0, 0, 0, 0]], np.c_[rng.uniform(0.25, 1, (len(masks), 3)), np.full(len(masks), 0.75)]])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    ax[0].imshow(image, cmap='gray')
    ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='yellow', lw=1))
    ax[0].set_title('Input: H-alpha full disk, 2048 × 2048')

    def overlay(axis, win):
        axis.imshow(image[win], cmap='gray')
        rgba = np.zeros((*proposal[win].shape, 4), np.float32)
        rgba[proposal[win]] = [1.0, 0.55, 0.10, 0.65]
        axis.imshow(rgba, interpolation='nearest')
        for j, m in enumerate(masks, 1):
            cs, _ = cv2.findContours(m[win].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cs:
                if len(c) < 2:
                    continue
                v = np.concatenate([c[:, 0, :], c[:1, 0, :]], axis=0)
                # A dark halo separates GT colours from both orange proposals and the grey image.
                axis.plot(
                    v[:, 0],
                    v[:, 1],
                    color=cols[j][:3],
                    lw=1.6,
                    path_effects=[pe.Stroke(linewidth=2.8, foreground='#202020'), pe.Normal()],
                )

    overlay(ax[1], (slice(None), slice(None)))
    ax[1].set_title(f'Stage 1: {len(ids)} proposals + {len(masks)} GT outlines')
    overlay(ax[2], (slice(y0, y1), slice(x0, x1)))
    ax[2].set_title('Zoom (720 px): proposals + GT outlines')
    for axis in ax:
        axis.axis('off')
    plt.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches='tight', facecolor='white')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    report = dict(
        image=a.image,
        record=str(rec['id']),
        gt=len(masks),
        owner_fold=owner,
        head_folds=[k for k in range(5) if k != owner],
        checkpoint_epoch=4,
        map_file=Path(a.map).name,
        map_kind='original stored mean-OOF float32',
        threshold=0.05,
        min_area=8,
        components=len(ids),
        gt_touched=touched,
        gt_pixel_coverage=pixel_recall,
        components_touching_gt=hit,
        components_without_gt_overlap=len(ids) - hit,
        zoom_xyxy=[x0, y0, x1, y1],
        note='GT touch means any pixel overlap, not IoU matching. No LightGBM or refiner.',
    )
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
