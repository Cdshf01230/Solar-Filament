"""Compare Stage-2 policies on a stored OOF map, matching the README task/Stage-1 figure style."""

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
from filament_pq import components as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--map', required=True, help='Canonical mean-OOF probability map of this image')
    ap.add_argument('--weights', default='weights')
    ap.add_argument('--image', default='20220710085152Th.jpeg')
    ap.add_argument('--record')
    ap.add_argument('--features', help='Optional original feature NPZ for exact cross-check')
    ap.add_argument('--balance-scores', help='Optional original OOF score NPZ for exact cross-check')
    ap.add_argument(
        '--zoom-box',
        type=int,
        nargs=4,
        metavar=('X0', 'Y0', 'X1', 'Y1'),
        help='Shared 720px comparison window; defaults to Stage-1 ROI',
    )
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    data, weights = Path(a.data), Path(a.weights)
    annotations = json.loads((data / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    records = [r for r in annotations['images'] if Path(r['file_name']).name == a.image]
    if a.record:
        records = [r for r in records if str(r['id']) == a.record]
    assert len(records) == 1, 'Choose one original annotation record with --record'
    rec = records[0]
    masks = [decode(x, rec) for x in annotations['annotations'] if str(x['image_id']) == str(rec['id'])]
    masks = [m for m in masks if m.any()]
    gt = np.logical_or.reduce(masks)
    image = cv2.imread(str(data / 'train_images' / a.image), cv2.IMREAD_GRAYSCALE)
    with np.load(a.map, allow_pickle=False) as z:
        prob = z['probability']
    assert image.shape == prob.shape == (2048, 2048)
    assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
    assignment = json.loads((weights / 'fold_assignment.json').read_text())
    owner = next(f['fold'] for f in assignment['folds'] if a.image in f['train_sources'])
    folds = [k for k in range(5) if k != owner]
    comps = C.components(prob)
    _, labels, _, _, ids, boxes = comps
    X = C.feature_matrix(prob, image, C.appearance(image), 0.05, comps)
    checks = {}
    if a.features:
        with np.load(a.features, allow_pickle=False) as z:
            assert np.array_equal(ids, z['ids']) and np.array_equal(X, z['X'])
            assert int(z['owner_fold']) == owner
        checks['original_features_exact'] = True
    scores = {}
    for policy in ('balance', 'recall'):
        classifiers = C.load_classifiers(weights, folds, policy=policy)
        # Match original ML OOF evaluation: float32 columns, mean of four eligible scores.
        columns = np.stack([classifiers[k].predict_proba(X)[:, 1].astype(np.float32) for k in folds], 1)
        assert columns.shape == (len(ids), 4) and np.isfinite(columns).all()
        scores[policy] = columns.mean(1)
    if a.balance_scores:
        with np.load(a.balance_scores, allow_pickle=False) as z:
            assert int(z['owner_fold']) == owner
            assert np.array_equal(scores['balance'], z['mean_oof'])
        checks['original_balance_scores_exact'] = True
    keeps = {'before': np.ones(len(ids), bool), **{k: v >= 0.5 for k, v in scores.items()}}
    proposal_masks = {k: np.isin(labels, ids[keep]) for k, keep in keeps.items()}
    metrics = {}
    for policy, keep in keeps.items():
        union = proposal_masks[policy]
        hit = sum(bool((gt & (labels == cid)).any()) for cid in ids[keep])
        metrics[policy] = dict(
            components=int(keep.sum()),
            gt_touched=sum(bool((m & union).any()) for m in masks),
            components_touching_gt=hit,
            components_without_gt_overlap=int(keep.sum()) - hit,
            gt_pixel_coverage=int((gt & union).sum()) / int(gt.sum()),
        )
    # Fixed ROI and palette are identical to the task and Stage-1 illustrations.
    ys, xs = np.nonzero(max(masks, key=lambda m: m.sum()))
    cy, cx = int(ys.mean()), int(xs.mean())
    x0, x1, y0, y1 = max(cx - 360, 0), min(cx + 360, 2048), max(cy - 360, 0), min(cy + 360, 2048)
    if a.zoom_box:
        x0, y0, x1, y1 = a.zoom_box
        assert 0 <= x0 < x1 <= 2048 and 0 <= y0 < y1 <= 2048
        assert x1 - x0 == y1 - y0 == 720
    rng = np.random.default_rng(3)
    cols = np.concatenate([[[0, 0, 0, 0]], np.c_[rng.uniform(0.25, 1, (len(masks), 3)), np.full(len(masks), 0.75)]])

    def overlay(ax, proposal, win):
        ax.imshow(image[win], cmap='gray')
        rgba = np.zeros((*proposal[win].shape, 4), np.float32)
        rgba[proposal[win]] = [1.0, 0.55, 0.10, 0.65]
        ax.imshow(rgba, interpolation='nearest')
        for j, mask in enumerate(masks, 1):
            cs, _ = cv2.findContours(mask[win].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cs:
                if len(c) < 2:
                    continue
                v = np.concatenate([c[:, 0, :], c[:1, 0, :]], axis=0)
                ax.plot(
                    v[:, 0],
                    v[:, 1],
                    color=cols[j][:3],
                    lw=1.6,
                    path_effects=[pe.Stroke(linewidth=2.8, foreground='#202020'), pe.Normal()],
                )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = {'before': 'Before Stage 2', 'balance': 'After balanced', 'recall': 'After recall-oriented'}
    for zoom in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
        win = (slice(y0, y1), slice(x0, x1)) if zoom else (slice(None), slice(None))
        for ax, policy in zip(axes, keeps):
            overlay(ax, proposal_masks[policy], win)
            if not zoom:
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='yellow', lw=1))
            title = (
                f'Zoom (720 px): {names[policy].lower()}'
                if zoom
                else f'{names[policy]}: {metrics[policy]["components"]} proposals'
            )
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        dest = out.with_name(out.stem + '_zoom' + out.suffix) if zoom else out
        fig.savefig(dest, dpi=110, bbox_inches='tight', facecolor='white')
        fig.savefig(dest.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig)
    model_info = json.loads((weights / 'ml/RECALL_MODEL_INFO.json').read_text())
    report = dict(
        image=a.image,
        record=str(rec['id']),
        gt=len(masks),
        owner_fold=owner,
        eligible_folds=folds,
        proposal_threshold=0.05,
        min_component_area=8,
        classifier_threshold=0.5,
        recall_positive_weight_multiplier=model_info['positive_weight_multiplier'],
        map_file=Path(a.map).name,
        map_kind='stored mean-OOF, epoch 4',
        zoom_xyxy=[x0, y0, x1, y1],
        metrics=metrics,
        checks=checks,
        component_ids=ids.tolist(),
        scores={k: v.tolist() for k, v in scores.items()},
        kept_component_ids={k: ids[v].tolist() for k, v in keeps.items()},
        note='GT touch is any pixel overlap within this record, not IoU matching. Masks shown before crop padding or refinement.',
    )
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(
        json.dumps(
            {k: report[k] for k in ('image', 'recall_positive_weight_multiplier', 'metrics', 'checks')}, indent=2
        )
    )


if __name__ == '__main__':
    main()
