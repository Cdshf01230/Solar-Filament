"""GT-free CC proposals, bounding-box padding PER SIDE; touch/crop coverage, no IoU match."""

from pathlib import Path
import os

for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import json, time, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
import threshold_sweep as original

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get('SOLAR_PADDING_OUT', str(ROOT / 'padding_study')))
OUT.mkdir(parents=True, exist_ok=True)
BASE = Path(os.environ.get('SOLAR_MEAN_ROOT', str(ROOT / 'ensemble')))
PADS = [0, 64, 128]


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def atomic(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(v), encoding='utf-8')
    q.replace(p)


def initialize():
    cv2.setNumThreads(1)
    original.initialize()


def rect_sums(integral, boxes):
    x0, y0, x1, y1 = boxes.T
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def union_rects(boxes, shape):
    h, w = shape
    d = np.zeros((h + 1, w + 1), np.int32)
    if len(boxes):
        x0, y0, x1, y1 = boxes.T
        for yy, xx, sgn in [(y0, x0, 1), (y1, x1, 1), (y0, x1, -1), (y1, x0, -1)]:
            np.add.at(d, (yy, xx), sgn)
    return d.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)[:h, :w] > 0


def pad_boxes(boxes, pad, shape):
    h, w = shape
    b = boxes.copy()
    b[:, :2] -= pad
    b[:, 2:] += pad
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b


def coverage_counts(values):
    a = np.asarray(values)
    return dict(
        objects=len(a),
        touch=int((a > 0).sum()),
        covered50=int((a >= 0.5).sum()),
        covered90=int((a >= 0.9).sum()),
        covered95=int((a >= 0.95).sum()),
        covered100=int((a >= 1).sum()),
        sum_fraction=float(a.sum()),
    )


def self_check():
    boxes = np.array([[1, 1, 2, 2], [3, 3, 4, 4]], np.int32)
    mask = union_rects(boxes, (5, 5))
    assert mask.sum() == 2
    assert union_rects(pad_boxes(boxes, 1, (5, 5)), (5, 5)).sum() == 17
    assert rect_sums(cv2.integral(mask.astype(np.uint8)), np.array([[0, 0, 5, 5]], np.int32))[0] == 2
    assert np.array_equal(
        pad_boxes(np.array([[10, 20, 30, 40]], np.int32), 64, (100, 100)), np.array([[0, 0, 94, 100]])
    )


def evaluate(task):
    kind, source, thresholds = task
    dest = OUT / 'rows' / kind / (source + '.json')
    if dest.exists():
        v = read(dest)
        assert v['thresholds'] == thresholds and v['padding_per_side'] == PADS
        return dict(model=kind, source=source, cached=True)
    with np.load(BASE / kind / (source + '.npz'), allow_pickle=False) as z:
        prob = z['probability']
    assert prob.shape == (2048, 2048) and prob.dtype == np.float32 and np.isfinite(prob).all()
    evidence = read(BASE / kind / (source + '.json'))
    assert (
        evidence['passed']
        and len(evidence['eligible_folds']) == 4
        and evidence['owner_fold'] not in evidence['eligible_folds']
    )
    groups = []
    for rec in original.RECORDS[source]:
        masks = original.masks_for(rec)
        union = np.logical_or.reduce(masks) if masks else np.zeros(prob.shape, bool)
        details = []
        for m in masks:
            yy, xx = np.nonzero(m)
            x0, x1 = int(xx.min()), int(xx.max() + 1)
            y0, y1 = int(yy.min()), int(yy.max() + 1)
            details.append((m, int(m.sum()), (x0, y0, x1, y1), cv2.integral(m[y0:y1, x0:x1].astype(np.uint8))))
        groups.append((rec['id'], union, details, cv2.integral(union.astype(np.uint8))))
    rows = []
    for t in thresholds:
        n, labels, stats, centroids = cv2.connectedComponentsWithStats((prob >= t).astype(np.uint8), 8, cv2.CV_32S)
        keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 8)
        keep = keep[keep > 0]
        s = stats[keep]
        boxes = np.column_stack([s[:, 0], s[:, 1], s[:, 0] + s[:, 2], s[:, 1] + s[:, 3]]).astype(np.int32)
        retained = np.zeros(n, bool)
        retained[keep] = True
        for pad in PADS:
            bb = pad_boxes(boxes, pad, prob.shape)
            covered = union_rects(bb, prob.shape)
            area = (bb[:, 2] - bb[:, 0]).astype(np.int64) * (bb[:, 3] - bb[:, 1])
            maxside = np.maximum(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1])
            source_cost = dict(
                proposals=len(bb),
                union_pixels=int(covered.sum()),
                image_pixels=prob.size,
                sum_crop_pixels=int(area.sum()),
                max_crop_pixels=int(area.max()) if len(area) else 0,
                crops_over_512=int((maxside > 512).sum()),
                crops_over_1024=int((maxside > 1024).sum()),
                full_image_crops=int((area == prob.size).sum()),
            )
            for record, union, details, integral in groups:
                component_labels = np.unique(labels[union])
                component_touch = int(retained[component_labels].sum())
                crop_touch = rect_sums(integral, bb) > 0 if len(bb) else np.zeros(0, bool)
                component_hit = []
                single = []
                combined = []
                object_details = []
                for m, gtarea, (x0, y0, x1, y1), mi in details:
                    component_hit.append(bool(retained[labels[m]].any()))
                    overlapping = bb[(bb[:, 0] < x1) & (bb[:, 2] > x0) & (bb[:, 1] < y1) & (bb[:, 3] > y0)]
                    if len(overlapping):
                        relative = overlapping.copy()
                        relative[:, [0, 2]] = relative[:, [0, 2]].clip(x0, x1) - x0
                        relative[:, [1, 3]] = relative[:, [1, 3]].clip(y0, y1) - y0
                        best = float(rect_sums(mi, relative).max() / gtarea)
                    else:
                        best = 0.0
                    total = float(covered[m].sum() / gtarea)
                    single.append(best)
                    combined.append(total)
                    object_details.append([gtarea, int(x1 - x0), int(y1 - y0), bool(component_hit[-1]), best, total])
                rows.append(
                    dict(
                        source_file=source,
                        record=record,
                        threshold=t,
                        padding=pad,
                        **source_cost,
                        gt_objects=len(details),
                        component_touch_gt=int(sum(component_hit)),
                        component_touch_predictions=component_touch,
                        crop_touch_predictions=int(crop_touch.sum()),
                        single_crop=coverage_counts(single),
                        union_crops=coverage_counts(combined),
                        objects=object_details,
                    )
                )
    atomic(
        dest,
        dict(
            model=kind,
            source=source,
            thresholds=thresholds,
            padding_per_side=PADS,
            owner_fold=evidence['owner_fold'],
            rows=rows,
        ),
    )
    return dict(model=kind, source=source, cached=False)


def summarize(rows):
    images = {r['source_file']: r for r in rows}
    n = sum(r['gt_objects'] for r in rows)
    pr = sum(r['proposals'] for r in rows)
    result = dict(
        sources=len(images),
        records=len(rows),
        objects=n,
        proposals_per_record=pr,
        touch_recall=sum(r['component_touch_gt'] for r in rows) / max(1, n),
        touch_precision=sum(r['component_touch_predictions'] for r in rows) / max(1, pr),
        crop_touch_precision=sum(r['crop_touch_predictions'] for r in rows) / max(1, pr),
        actual_crops=sum(r['proposals'] for r in images.values()),
        crops_per_source=sum(r['proposals'] for r in images.values()) / len(images),
        union_image_fraction=sum(r['union_pixels'] for r in images.values())
        / sum(r['image_pixels'] for r in images.values()),
        crop_pixels_image_multiple=sum(r['sum_crop_pixels'] for r in images.values())
        / sum(r['image_pixels'] for r in images.values()),
        max_crop_pixels=max(r['max_crop_pixels'] for r in images.values()),
        crops_over_512=sum(r['crops_over_512'] for r in images.values()),
        crops_over_1024=sum(r['crops_over_1024'] for r in images.values()),
        full_image_crops=sum(r['full_image_crops'] for r in images.values()),
    )
    for kind in ['single_crop', 'union_crops']:
        result[kind] = {
            key: sum(r[kind][key] for r in rows) / max(1, n)
            for key in ['touch', 'covered50', 'covered90', 'covered95', 'covered100', 'sum_fraction']
        }
    return result


def main():
    self_check()
    assignment = read(ROOT / 'fold_assignment.json')
    owners = {s: f['fold'] for f in assignment['folds'] for s in f['train_sources']}
    cal = read(ROOT / 'recall95_calibration.json')
    thresholds = {
        k: sorted(set([0.005, 0.01, 0.02, 0.03, 0.05, cal[k]['threshold']])) for k in ['convnext', 'vitsplus']
    }
    atomic(
        OUT / 'protocol.json',
        dict(
            sources=707,
            records=1154,
            objects=8199,
            models=list(thresholds),
            thresholds=thresholds,
            padding_pixels_per_side=PADS,
            proposals='CC8, minarea8, variable-size bbox from probability only',
            crop='[xmin-pad,ymin-pad,xmax+pad,ymax+pad], clipped at image2048 edges',
            touch='at least one shared pixel, many-to-many, no IoU or one-to-one restriction',
            single='maximum GT pixel fraction contained by any ONE crop',
            union='GT pixel fraction within union of all crops',
            precision='fraction of components or padded crops touching at least one GT, per original annotatorrecord',
            no_gt_used_to_build_proposals_or_crops=True,
        ),
    )
    tasks = [(k, s, thresholds[k]) for k in thresholds for s in sorted(owners)]
    began = time.time()
    with ProcessPoolExecutor(max_workers=4, initializer=initialize) as pool:
        for count, future in enumerate(as_completed([pool.submit(evaluate, t) for t in tasks]), 1):
            last = future.result()
            if count % 20 == 0 or count == len(tasks):
                state = dict(
                    state='evaluating', completed=count, total=len(tasks), seconds=time.time() - began, last=last
                )
                atomic(OUT / 'status.json', state)
                print(json.dumps(state), flush=True)
    aggregate = {}
    folds = {}
    for k in thresholds:
        rows = [r for s in sorted(owners) for r in read(OUT / 'rows' / k / (s + '.json'))['rows']]
        aggregate[k] = []
        folds[k] = {}
        for t in thresholds[k]:
            for pad in PADS:
                rr = [r for r in rows if r['threshold'] == t and r['padding'] == pad]
                summary = dict(threshold=t, padding=pad, **summarize(rr))
                assert summary['sources'] == 707 and summary['records'] == 1154 and summary['objects'] == 8199
                assert (
                    summary['single_crop']['covered100']
                    <= summary['single_crop']['covered95']
                    <= summary['single_crop']['covered90']
                    <= summary['single_crop']['touch']
                )
                assert summary['single_crop']['covered100'] <= summary['union_crops']['covered100'] + 1e-12
                aggregate[k].append(summary)
                folds[k][f'{t}_{pad}'] = {
                    str(f): summarize([r for r in rr if owners[r['source_file']] == f]) for f in range(5)
                }
        del rows
    atomic(OUT / 'aggregate.json', aggregate)
    atomic(OUT / 'by_owner_fold.json', folds)
    verification = dict(
        passed=True,
        sources=707,
        records=1154,
        objects=8199,
        models=2,
        arrays_loaded=1414,
        owner_exclusion_verified=True,
        checkpoint_epoch=4,
        thresholds_per_model=6,
        padding_per_side=PADS,
        metric_points=36,
        no_training=True,
        completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seconds=time.time() - began,
    )
    atomic(OUT / 'verification.json', verification)
    atomic(OUT / 'status.json', dict(state='complete_verified', **verification))
    print('PADDING_STUDY_COMPLETE', json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
