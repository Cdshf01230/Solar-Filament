"""Show actual post-processing of the stored balanced OOF refiner example.

One row: a native round-3 mask and its selected anchor component, then the
full-image result and original-record GT. No inference, training or tuning.
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
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.make_stage3_figure import read_npz, full_patch
from scripts.make_task_figure import decode
from filament_pq import postprocess as PP, metrics
from filament_pq.config import PostprocessConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--refined', required=True, help='Original owner-fold epoch-30 / D4-TTA output directory')
    ap.add_argument('--stage3-report', default='docs/figures/stage3_example.json')
    ap.add_argument('--component', type=int, default=21)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    previous = json.loads(Path(a.stage3_report).read_text(encoding='utf-8'))
    assert previous['classifier_policy'] == 'balance'
    assert previous['owner_fold'] == previous['refiner_fold']
    name = previous['image']
    image = cv2.imread(str(Path(a.data) / 'train_images' / name), cv2.IMREAD_GRAYSCALE)
    assert image is not None and image.shape == (2048, 2048)
    src = json.loads((Path(a.data) / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text(encoding='utf-8'))
    rec = next(r for r in src['images'] if str(r['id']) == previous['record'])
    masks = [decode(x, rec) for x in src['annotations'] if str(x['image_id']) == previous['record']]
    masks = [m for m in masks if m.any()]
    assert len(masks) == previous['gt']
    expected = {r['crop_id']: r for r in previous['crops']}
    paths = sorted(Path(a.refined).glob(f'{Path(name).stem}_*.npz'))
    assert {p.stem for p in paths} == set(expected)
    refined = []
    for p in paths:
        r = read_npz(p)
        r['crop_id'] = str(r['crop_id'])
        e = expected[r['crop_id']]
        assert r['box'].tolist() == e['box_xyxy']
        assert float(r['ml_score']) == e['ml_score'] and r['presence'].tolist() == e['presence']
        assert r['r3'].shape == (512, 512) and r['r3'].dtype == np.uint8
        refined.append(r)

    cfg = PostprocessConfig()
    candidates = PP.candidates(refined, cfg)
    by_id = {c['crop_id']: c for c in candidates}
    raw_masks, selected_masks, rows = {}, {}, []
    for r in refined:
        cid = r['crop_id']
        raw = PP.native_probability(r) > cfg.mask_threshold
        raw_masks[cid] = raw
        presence_passed = float(r['presence'][1]) >= cfg.presence_threshold
        keep = PP.select_anchor_component(raw, r['native_anchor'], cfg.min_area) if presence_passed else None
        if keep is not None:
            selected_masks[cid] = keep
            c = by_id[cid]
            tight = PP.tight(int(r['box'][1]), int(r['box'][0]), keep)
            assert c['y0'] == tight[0] and c['x0'] == tight[1] and np.array_equal(c['m'], tight[2])
        else:
            assert cid not in by_id
        n, _ = cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)
        rows.append(
            dict(
                crop_id=cid,
                box_xyxy=r['box'].tolist(),
                resized_wh=r['resized_wh'].tolist(),
                presence=float(r['presence'][1]),
                presence_passed=presence_passed,
                raw_pixels=int(raw.sum()),
                raw_components=n - 1,
                selected_pixels=int(keep.sum()) if keep is not None else 0,
                component_removed_pixels=int((raw & ~keep).sum()) if keep is not None else None,
            )
        )
    assert set(selected_masks) == set(by_id)
    final = PP.postprocess(refined, cfg)
    merged = PP.merge_nms(candidates, cfg)
    assert len(merged) == len(final)
    for x, y in zip(merged, final):
        assert x[:2] == y[:2] and np.array_equal(x[2], y[2]) and x[3] == y[3]
    final_masks = [PP.to_full(x) for x in final]
    union = np.logical_or.reduce(final_masks) if final_masks else np.zeros(image.shape, bool)
    assert sum(int(m.sum()) for m in final_masks) == int(union.sum())
    encode = lambda ms: [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in ms]
    host, strict = metrics.match(metrics.iou_matrix(encode(masks), encode(final_masks)))
    scores = dict(legacy=metrics.pq(host), strict_one_to_one=metrics.pq(strict))
    for scorer in scores:
        for key in scores[scorer]:
            assert abs(scores[scorer][key] - previous['full_image_metrics'][scorer][key]) < 1e-7

    selected_id = f'{Path(name).stem}_{a.component:05d}'
    r = next(r for r in refined if r['crop_id'] == selected_id)
    assert selected_id in selected_masks
    selected_row = next(row for row in rows if row['crop_id'] == selected_id)
    x0, y0, x1, y1 = map(int, r['box'])
    side = max(x1 - x0, y1 - y0)
    wx0, wy0 = max(0, min((x0 + x1 - side) // 2, 2048 - side)), max(0, min((y0 + y1 - side) // 2, 2048 - side))
    win = (slice(wy0, wy0 + side), slice(wx0, wx0 + side))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    for ax, p, title in zip(
        axes[:2],
        (raw_masks[selected_id], selected_masks[selected_id]),
        (f'Round 3: {selected_row["raw_components"]} components', 'Keep anchor component'),
    ):
        ax.imshow(image[win], cmap='gray')
        shown = full_patch(p, r['box'], image.shape)[win]
        rgba = np.zeros((*shown.shape, 4), np.float32)
        rgba[shown] = [1.0, 0.55, 0.10, 0.65]
        ax.imshow(rgba, interpolation='nearest')
        anchor = full_patch(r['native_anchor'], r['box'], image.shape)[win]
        cs, _ = cv2.findContours(anchor.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            v = np.concatenate([c[:, 0, :], c[:1, 0, :]], axis=0)
            ax.plot(
                v[:, 0],
                v[:, 1],
                color='cyan',
                lw=1,
                path_effects=[pe.Stroke(linewidth=1.8, foreground='#202020'), pe.Normal()],
            )
        ax.set_title(title)

    # Prediction and GT palettes identify instances within each panel, not matches.
    pred_colors = np.random.default_rng(11).uniform(0.25, 1, (len(final_masks), 3))
    gt_colors = np.random.default_rng(3).uniform(0.25, 1, (len(masks), 3))
    for ax, ms, colors, title in (
        (axes[2], final_masks, pred_colors, f'Final full disk: {len(final)} instances'),
        (axes[3], masks, gt_colors, f'Ground truth: {len(masks)} instances'),
    ):
        ax.imshow(image, cmap='gray')
        rgba = np.zeros((*image.shape, 4), np.float32)
        for m, color in zip(ms, colors):
            rgba[m] = [*color, 0.75]
        ax.imshow(rgba, interpolation='nearest')
        ax.set_title(title)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches='tight', facecolor='white')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    passed_rows = [row for row in rows if row['presence_passed']]
    candidate_pixels = sum(int(c['m'].sum()) for c in candidates)
    report = dict(
        image=name,
        record=previous['record'],
        gt=len(masks),
        classifier_policy='balance',
        refiner_fold=previous['refiner_fold'],
        refiner_epoch=previous['refiner_epoch'],
        tta=previous['tta'],
        config=cfg.__dict__,
        input_prompts=len(refined),
        presence_passed=len(passed_rows),
        presence_rejected=[row['crop_id'] for row in rows if not row['presence_passed']],
        raw_components_after_presence=sum(row['raw_components'] for row in passed_rows),
        candidate_instances=len(candidates),
        final_instances=len(final),
        component_removed_pixels=sum(row['component_removed_pixels'] or 0 for row in passed_rows),
        candidate_pixels=candidate_pixels,
        final_pixels=int(union.sum()),
        greedy_removed_instances=len(candidates) - len(final),
        greedy_removed_pixels=candidate_pixels - int(union.sum()),
        metrics=scores,
        example=dict(**selected_row, display_xyxy=[wx0, wy0, wx0 + side, wy0 + side]),
        crops=rows,
        checks=dict(
            original_stage3_inputs_exact=True,
            public_candidate_masks_exact=True,
            public_postprocess_masks_exact=True,
            final_masks_pixel_disjoint=True,
            original_oof_metrics_match=True,
        ),
        note='Left two panels are the same native-pixel crop view; right two are full 2048 images. Cyan is the anchor, orange is the prediction. GT is shown separately. No NMS removal is claimed when none occurs.',
    )
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    'input_prompts',
                    'presence_passed',
                    'raw_components_after_presence',
                    'candidate_instances',
                    'final_instances',
                    'component_removed_pixels',
                    'greedy_removed_instances',
                    'greedy_removed_pixels',
                    'example',
                    'checks',
                )
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
