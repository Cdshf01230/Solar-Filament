"""Illustrate the balanced OOF refiner using stored epoch-30 / D4-TTA outputs.

No model is rerun. --crops contains native input crops; --refined contains the
matching owner-fold NPZ outputs. See the generated JSON for checks and counts.
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
from scripts.make_task_figure import decode
from filament_pq import postprocess as PP, metrics


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def full_patch(patch, box, shape):
    x0, y0, x1, y1 = map(int, box)
    assert patch.shape == (y1 - y0, x1 - x0)
    out = np.zeros(shape, bool)
    out[y0:y1, x0:x1] = patch
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--crops', required=True)
    ap.add_argument('--refined', required=True, help='Owner-fold epoch-30 / D4-TTA output directory')
    ap.add_argument('--stage2-report', default='docs/figures/stage2_example.json')
    ap.add_argument('--weights', default='weights')
    ap.add_argument('--component', type=int, default=13, help='Selected prompt for the recurrent-round example')
    ap.add_argument('--reference-scores', help='Optional original OOF score JSON for full-image regression')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    previous = json.loads(Path(a.stage2_report).read_text())
    name, record, owner = previous['image'], previous['record'], previous['owner_fold']
    assignment = json.loads((Path(a.weights) / 'fold_assignment.json').read_text())
    assert name in assignment['folds'][owner]['train_sources']
    src = json.loads((Path(a.data) / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    rec = next(r for r in src['images'] if str(r['id']) == record)
    assert Path(rec['file_name']).name == name
    masks = [decode(x, rec) for x in src['annotations'] if str(x['image_id']) == record]
    masks = [m for m in masks if m.any()]
    assert len(masks) == previous['gt']
    image = cv2.imread(str(Path(a.data) / 'train_images' / name), cv2.IMREAD_GRAYSCALE)
    assert image is not None and image.shape == (2048, 2048)
    scores = dict(zip(previous['component_ids'], previous['scores']['balance']))
    inputs, refined = {}, []
    expected = [f'{Path(name).stem}_{k:05d}' for k in previous['kept_component_ids']['balance']]
    found = sorted(p.stem for p in Path(a.refined).glob(f'{Path(name).stem}_*.npz'))
    assert found == sorted(expected), 'Require exactly the balanced crop outputs for this image'
    anchor_union = np.zeros(image.shape, bool)
    for cid in expected:
        c = read_npz(Path(a.crops) / (cid + '.npz'))
        r = read_npz(Path(a.refined) / (cid + '.npz'))
        r['crop_id'] = str(r['crop_id'])
        assert r['crop_id'] == cid and int(c['owner_fold']) == owner
        assert np.array_equal(c['crop_box'], r['box'])
        assert np.array_equal(c['component'], r['native_anchor'])
        x0, y0, x1, y1 = map(int, r['box'])
        assert np.array_equal(c['image'], image[y0:y1, x0:x1])
        k = int(cid.rsplit('_', 1)[1])
        assert float(r['ml_score']) == float(c['ml_mean_oof']) == scores[k]
        assert r['r1'].shape == r['r3'].shape == (512, 512)
        assert r['r1'].dtype == r['r3'].dtype == np.uint8
        assert np.isfinite(r['presence']).all() and ((r['presence'] >= 0) & (r['presence'] <= 1)).all()
        anchor_union |= full_patch(c['component'], r['box'], image.shape)
        inputs[cid] = c
        refined.append(r)

    instances = PP.postprocess(refined)
    output_masks = [PP.to_full(x) for x in instances]
    output_union = np.logical_or.reduce(output_masks) if output_masks else np.zeros(image.shape, bool)
    assert sum(int(m.sum()) for m in output_masks) == int(output_union.sum())
    encode = lambda ms: [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in ms]
    host, strict = metrics.match(metrics.iou_matrix(encode(masks), encode(output_masks)))
    checks = dict(
        stage2_ids_scores_boxes_anchors_exact=True, native_input_images_exact=True, pixel_disjoint_final_masks=True
    )
    if a.reference_scores:
        ref = json.loads(Path(a.reference_scores).read_text())
        setting = next(i for i, r in enumerate(ref['results']) if r['name'] == 't8_lin50')
        reference = ref['per_source'][name][setting]
        assert np.allclose(host, reference[0], rtol=0, atol=1e-6)
        assert np.allclose(strict, reference[1], rtol=0, atol=1e-6)
        assert len(instances) == reference[2]
        checks['original_oof_full_image_counts_exact'] = True

    rng = np.random.default_rng(3)
    colors = rng.uniform(0.25, 1, (len(masks), 3))

    def overlay(ax, prediction, win, gt_indices=None, fill_color=(1.0, 0.55, 0.10, 0.65)):
        ax.imshow(image[win], cmap='gray')
        rgba = np.zeros((*prediction[win].shape, 4), np.float32)
        rgba[prediction[win]] = fill_color
        ax.imshow(rgba, interpolation='nearest')
        for j in range(len(masks)) if gt_indices is None else gt_indices:
            contours, _ = cv2.findContours(masks[j][win].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if len(contour) < 2:
                    continue
                v = np.concatenate([contour[:, 0, :], contour[:1, 0, :]], axis=0)
                ax.plot(
                    v[:, 0],
                    v[:, 1],
                    color=colors[j],
                    lw=1.6,
                    path_effects=[pe.Stroke(linewidth=2.8, foreground='#202020'), pe.Normal()],
                )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(fig, dest):
        plt.tight_layout()
        fig.savefig(dest, dpi=110, bbox_inches='tight', facecolor='white')
        fig.savefig(dest.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig)

    # Same 720px window as the task / Stage-1 illustration (largest GT filament).
    ys, xs = np.nonzero(max(masks, key=lambda m: m.sum()))
    cx, cy = int(xs.mean()), int(ys.mean())
    x0, y0 = min(max(cx - 360, 0), 1328), min(max(cy - 360, 0), 1328)
    zoom = (slice(y0, y0 + 720), slice(x0, x0 + 720))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, prediction, title in zip(
        axes[:2],
        (anchor_union, output_union),
        (f'Stage 2: {len(refined)} balanced anchors', f'Refined output: {len(instances)} instances'),
    ):
        overlay(ax, prediction, (slice(None), slice(None)))
        ax.add_patch(plt.Rectangle((x0, y0), 720, 720, fill=False, ec='yellow', lw=1))
        ax.set_title(title)
    overlay(axes[2], output_union, zoom)
    axes[2].set_title('Zoom (720 px): refined masks + GT')
    for ax in axes:
        ax.axis('off')
    save(fig, out)

    # A single native prompt, before and after the recurrent rounds. Only its
    # target GT is outlined: neighbours in the context are not this prompt's target.
    selected = f'{Path(name).stem}_{a.component:05d}'
    r = next(r for r in refined if r['crop_id'] == selected)
    c = inputs[selected]
    box = tuple(map(int, r['box']))
    bx0, by0, bx1, by1 = box
    overlaps = [int((m[by0:by1, bx0:bx1] & c['component']).sum()) for m in masks]
    assert max(overlaps) > 0
    target_index = int(np.argmax(overlaps))
    target = masks[target_index]
    target_native = target[by0:by1, bx0:bx1]
    assert int(target_native.sum()) == int(target.sum()), 'Example requires an uncropped target GT'
    side = max(bx1 - bx0, by1 - by0)
    px0 = min(max((bx0 + bx1 - side) // 2, 0), 2048 - side)
    py0 = min(max((by0 + by1 - side) // 2, 0), 2048 - side)
    win = (slice(py0, py0 + side), slice(px0, px0 + side))
    selected_metrics = {}
    stages = []
    for label, key in zip(('Input anchor', 'After round 1', 'After round 3'), (None, 'r1', 'r3')):
        if key is None:
            p = c['component']
        else:
            tmp = dict(r, r3=r[key])
            # Display the thresholded raw mask, before component selection/NMS.
            p = PP.native_probability(tmp) > 0.5
        iou = int((p & target_native).sum()) / max(1, int((p | target_native).sum()))
        selected_metrics[label] = dict(target_iou=iou, pixels=int(p.sum()))
        stages.append((label, full_patch(p, box, image.shape)))
        if key is None:
            stages.append(('Ground truth', target))
    for show_gt in (True, False):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
        for ax, (label, prediction) in zip(axes, stages):
            if label == 'Ground truth':
                overlay(ax, prediction, win, [], (*colors[target_index], 0.75))
            else:
                overlay(ax, prediction, win, [target_index] if show_gt else [])
            ax.set_title(label)
            ax.set_xlim(-0.5, side - 0.5)
            ax.set_ylim(side - 0.5, -0.5)
            ax.axis('off')
        suffix = '_prompt' if show_gt else '_prompt_no_gt'
        save(fig, out.with_name(out.stem + suffix + out.suffix))

    report = dict(
        image=name,
        record=record,
        gt=len(masks),
        classifier_policy='balance',
        owner_fold=owner,
        proposal_classifier_folds=previous['eligible_folds'],
        refiner_fold=owner,
        refiner_epoch=30,
        tta=8,
        probability_storage='uint8 round(255*p), original OOF dump',
        postprocessing=dict(
            resize='linear probability to native, then > .5',
            presence_threshold=0.5,
            min_area=128,
            nms_iou=0.5,
            final_area=128,
        ),
        input_prompts=len(refined),
        presence_rejected=[r['crop_id'] for r in refined if r['presence'][1] < 0.5],
        candidates=len(PP.candidates(refined)),
        final_instances=len(instances),
        full_image_metrics=dict(legacy=metrics.pq(host), strict_one_to_one=metrics.pq(strict)),
        zoom_xyxy=[x0, y0, x0 + 720, y0 + 720],
        prompt=dict(
            crop_id=selected,
            box_xyxy=list(box),
            target_gt_index=target_index,
            presence_round1=float(r['presence'][0]),
            presence_round3=float(r['presence'][1]),
            display_xyxy=[px0, py0, px0 + side, py0 + side],
            metrics=selected_metrics,
            panel_order=[label for label, _ in stages],
            views=['GT outlines on predictions', 'separate GT panel only'],
            crop_border_drawn=False,
            selection=f'component {a.component}, illustrative example; not an average',
        ),
        crops=[
            dict(
                crop_id=r['crop_id'],
                box_xyxy=r['box'].tolist(),
                ml_score=float(r['ml_score']),
                presence=r['presence'].tolist(),
            )
            for r in refined
        ],
        checks=checks,
        note='Balanced OOF example only. Full-image panel includes released post-processing; prompt panels show raw thresholded masks. No recall-policy refiner result is implied.',
    )
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    'input_prompts',
                    'presence_rejected',
                    'final_instances',
                    'full_image_metrics',
                    'prompt',
                    'checks',
                )
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
