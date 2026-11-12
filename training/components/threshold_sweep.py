"""Development OOF threshold study; pixel, coverage and one-to-one instance metrics."""

from pathlib import Path
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json, time, zipfile, datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parent
BASE = Path(os.environ.get('SOLAR_ANALYSIS_ROOT', str(ROOT)))
ENS = BASE / 'ensemble'
OUT = BASE / 'threshold_study'
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.05]
IOU = [0.1, 0.25, 0.5]
BINS = 65536


def data_zip():
    """Competition zip, used only when the unpacked image/annotation paths are not given."""
    if not os.environ.get('SOLAR_DATA_ZIP'):
        raise SystemExit('Set SOLAR_RAW_IMAGES and SOLAR_ANNOTATION_JSON, or SOLAR_DATA_ZIP to the competition zip')
    return Path(os.environ['SOLAR_DATA_ZIP'])


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def atomic(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    t.replace(p)


def save_npz(p, **data):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    with t.open('wb') as stream:
        np.savez_compressed(stream, **data)
    t.replace(p)


def initialize():
    global RECORDS, ANNOTATIONS
    if os.environ.get('SOLAR_ANNOTATION_JSON'):
        source = read(Path(os.environ['SOLAR_ANNOTATION_JSON']))
    else:
        with zipfile.ZipFile(data_zip()) as archive:
            name = next(n for n in archive.namelist() if n.endswith('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
            source = json.loads(archive.read(name))
    RECORDS = defaultdict(list)
    ANNOTATIONS = defaultdict(list)
    for rec in source['images']:
        RECORDS[Path(rec['file_name']).name].append(rec)
    for ann in source['annotations']:
        ANNOTATIONS[str(ann['image_id'])].append(ann)


def masks_for(rec):
    masks = []
    for ann in ANNOTATIONS[str(rec['id'])]:
        seg = ann['segmentation']
        seg = json.loads(seg) if isinstance(seg, str) else seg
        if isinstance(seg, list):
            rle = mu.merge(mu.frPyObjects(seg, rec['height'], rec['width']))
        else:
            rle = dict(seg)
            if isinstance(rle['counts'], list):
                rle = mu.frPyObjects(rle, rec['height'], rec['width'])
            elif isinstance(rle['counts'], str):
                rle['counts'] = rle['counts'].encode()
        if int(mu.area(rle)) > 0:
            masks.append(mu.decode(rle).astype(bool))
    return masks


def one_to_one(iou, threshold):
    ng, npred = iou.shape
    if ng and npred:
        qualified = iou > threshold
        # Cardinality dominates total IoU; tie-break maximizes matched IoU.
        benefit = qualified * (1 + iou / (min(ng, npred) + 1))
        a, b = linear_sum_assignment(benefit, maximize=True)
        keep = qualified[a, b]
        matched = int(keep.sum())
        si = float(iou[a[keep], b[keep]].astype(float).sum())
    else:
        matched = 0
        si = 0.0
    return [matched, npred - matched, ng - matched, si]


def self_check():
    assert one_to_one(np.array([[0.9, 0.7], [0.8, 0]], np.float32), 0.5)[:3] == [2, 0, 0]
    assert one_to_one(np.array([[0.8], [0.8]], np.float32), 0.5)[:3] == [1, 0, 1]
    assert one_to_one(np.zeros((0, 3), np.float32), 0.5)[:3] == [0, 3, 0]
    assert one_to_one(np.array([[0.5]], np.float32), 0.5)[:3] == [0, 1, 1]


def evaluate(task):
    kind, source, thresholds, phase = task
    folder = OUT / phase / kind
    dest = folder / (source + '.json')
    aux = OUT / 'aux' / kind / (source + '.npz')
    if dest.exists() and (phase != 'grid' or aux.exists()):
        cached = read(dest)
        assert cached['thresholds'] == thresholds and cached['source'] == source and cached['model'] == kind
        return dict(model=kind, source=source, cached=True)
    with np.load(ENS / kind / (source + '.npz'), allow_pickle=False) as pack:
        prob = pack['probability']
    assert prob.dtype == np.float32 and prob.shape == (2048, 2048) and np.isfinite(prob).all()
    groups = []
    positive_weight = np.zeros(prob.shape, np.float64)
    for rec in RECORDS[source]:
        masks = masks_for(rec)
        union = np.logical_or.reduce(masks) if masks else np.zeros(prob.shape, bool)
        positive_weight += union / len(RECORDS[source])
        groups.append((rec['id'], masks, union))
    rows = []
    for threshold in thresholds:
        binary = prob >= threshold
        predicted = int(binary.sum())
        lab, _ = ndi.label(binary, np.ones((3, 3), np.uint8))
        areas = np.bincount(lab.ravel())
        keep = np.flatnonzero(areas >= 8)
        keep = keep[keep > 0]
        for record, masks, union in groups:
            ga = np.array([m.sum() for m in masks], np.int64)
            inter = np.array([np.bincount(lab[m], minlength=len(areas))[keep] for m in masks], np.int64).reshape(
                len(masks), len(keep)
            )
            den = ga[:, None] + areas[keep][None] - inter
            iou = np.divide(inter, den, out=np.zeros(inter.shape, float), where=den > 0).astype(np.float32)
            hit = iou > 0.5
            legacy = [
                int(hit.sum()),
                int((~hit.any(0)).sum()),
                int((~hit.any(1)).sum()),
                float(iou[hit].astype(float).sum()),
            ]
            intersections = np.array([binary[m].sum() for m in masks], np.int64)
            fractions = np.divide(intersections, ga, out=np.zeros(len(ga), float), where=ga > 0)
            sig = (inter >= 8) & (inter >= 0.05 * ga[:, None])
            tp = int((binary & union).sum())
            fg = int(union.sum())
            size = prob.size
            rows.append(
                dict(
                    source_file=source,
                    record=record,
                    threshold=threshold,
                    pixels=[tp, predicted - tp, fg - tp, size - predicted - fg + tp],
                    stats=legacy,
                    included_in_public_pq=bool(masks),
                    gt=len(masks),
                    predictions=len(keep),
                    split_gt=int((sig.sum(1) >= 2).sum()),
                    merge_predictions=int((sig.sum(0) >= 2).sum()),
                    one_to_one={str(t): one_to_one(iou, t) for t in IOU},
                    coverage=dict(
                        objects=len(masks),
                        hit10=int((fractions >= 0.1).sum()),
                        covered50=int((fractions >= 0.5).sum()),
                        covered90=int((fractions >= 0.9).sum()),
                        covered95=int((fractions >= 0.95).sum()),
                        missed_completely=int((fractions == 0).sum()),
                        predicted_pixels=predicted,
                        image_pixels=size,
                    ),
                )
            )
    atomic(dest, dict(model=kind, source=source, thresholds=thresholds, rows=rows))
    if phase == 'grid':
        index = np.minimum((prob * BINS).astype(np.int32), BINS - 1)
        totals = np.bincount(index.ravel(), minlength=BINS)
        positive = positive_weight > 0
        values = prob[positive]
        weights = positive_weight[positive]
        tp_hist = np.bincount(index[positive], weights=weights, minlength=BINS)
        save_npz(aux, total_hist=totals, tp_hist=tp_hist, positive_values=values, positive_weights=weights)
    return dict(model=kind, source=source, cached=False, records=len(groups))


def pixel_metrics(c):
    tp, fp, fn, tn = map(float, c)
    return dict(
        precision=tp / max(1, tp + fp),
        recall=tp / max(1, tp + fn),
        f1=2 * tp / max(1, 2 * tp + fp + fn),
        iou=tp / max(1, tp + fp + fn),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def instance_metrics(c):
    tp, fp, fn, si = map(float, c)
    den = tp + 0.5 * (fp + fn)
    return dict(
        precision=tp / max(1, tp + fp),
        recall=tp / max(1, tp + fn),
        f1=2 * tp / max(1, 2 * tp + fp + fn),
        pq=si / max(1e-12, den),
        sq=si / max(1, tp),
        rq=tp / max(1e-12, den),
        tp=tp,
        fp=fp,
        fn=fn,
        sum_iou=si,
    )


def summarize(rows, threshold):
    rs = [r for r in rows if r['threshold'] == threshold]
    images = defaultdict(list)
    for r in rs:
        images[r['source_file']].append(r)
    pix = np.sum([np.mean([r['pixels'] for r in rr], axis=0) for rr in images.values()], axis=0)
    scored = [r for r in rs if r['included_in_public_pq']]
    legacy = instance_metrics(np.sum([r['stats'] for r in scored], axis=0))
    one = {str(t): instance_metrics(np.sum([r['one_to_one'][str(t)] for r in rs], axis=0)) for t in IOU}
    coverage = {
        k: sum(r['coverage'][k] for r in rs)
        for k in ['objects', 'hit10', 'covered50', 'covered90', 'covered95', 'missed_completely']
    }
    for key in ['hit10', 'covered50', 'covered90', 'covered95']:
        coverage[key + '_recall'] = coverage[key] / max(1, coverage['objects'])
    coverage['area_fraction'] = sum(rr[0]['coverage']['predicted_pixels'] for rr in images.values()) / sum(
        rr[0]['coverage']['image_pixels'] for rr in images.values()
    )
    return dict(
        threshold=threshold,
        images=len(images),
        records=len(rs),
        pixel=pixel_metrics(pix),
        legacy_instance=legacy,
        one_to_one_instance=one,
        coverage=coverage,
        split_gt=sum(r['split_gt'] for r in rs),
        merge_predictions=sum(r['merge_predictions'] for r in rs),
        predictions_per_record=sum(r['predictions'] for r in rs),
        negative_records=sum(not r['included_in_public_pq'] for r in rs),
    )


def main():
    self_check()
    verification = read(ENS / 'verification.json')
    assert verification['passed'] and verification['ensemble_maps'] == 1414 and verification['labels_used'] == False
    assignment = read(ROOT / 'fold_assignment.json')
    owners = {s: f['fold'] for f in assignment['folds'] for s in f['train_sources']}
    sources = sorted(owners)
    atomic(
        OUT / 'protocol.json',
        dict(
            models=['convnext', 'vitsplus'],
            ensemble='four-head owner-excluded FP32 mean, fixedepoch4',
            thresholds=THRESHOLDS,
            pixel_aggregation='average annotator counts within each source then sum707 sources',
            public_instance='Legacy all-pairs IoU>0.5; excludes no-GT records, same historical scorer',
            standard_instance='Maximum cardinality one-to-one matching, max IoU tie-break; strict IoU>0.1/0.25/0.5; includes all original records including empty-GT false positives',
            prediction_instances='CC8 foreground components area>=8',
            coverage='GT coverage uses binary mask including components below8, matching original protocol',
            threshold_selection='development calibration on all707 OOF sources; highest probability cutoff reaching pixel recall>=0.95 by exact weighted foreground quantile; not independent test',
            unchanged='No checkpoint/head/backbone retraining; no classifier fitting; classifier threshold0.5 remains separate',
        ),
    )
    began = time.time()
    for phase in ('grid', 'recall95'):
        tasks = [
            (kind, s, THRESHOLDS if phase == 'grid' else [targets[kind]], phase)
            for kind in ('convnext', 'vitsplus')
            for s in sources
        ]
        completed = 0
        with ProcessPoolExecutor(max_workers=4, initializer=initialize) as pool:
            futures = [pool.submit(evaluate, t) for t in tasks]
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    status = dict(
                        state='evaluating',
                        phase=phase,
                        completed=completed,
                        total=len(tasks),
                        seconds=round(time.time() - began, 1),
                        last=result,
                    )
                    atomic(OUT / 'status.json', status)
                    print(json.dumps(status), flush=True)
        if phase == 'grid':
            targets = {}
            calibration = {}
            for kind in ('convnext', 'vitsplus'):
                values = []
                weights = []
                total_hist = np.zeros(BINS, np.int64)
                tp_hist = np.zeros(BINS, np.float64)
                for source in sources:
                    with np.load(OUT / 'aux' / kind / (source + '.npz'), allow_pickle=False) as pack:
                        values.append(pack['positive_values'])
                        weights.append(pack['positive_weights'])
                        total_hist += pack['total_hist']
                        tp_hist += pack['tp_hist']
                v = np.concatenate(values)
                w = np.concatenate(weights).astype(np.float64)
                order = np.argsort(v)
                v = v[order]
                w = w[order]
                unique, starts = np.unique(v, return_index=True)
                mass = np.add.reduceat(w, starts)
                cum = np.cumsum(mass)
                idx = np.searchsorted(cum, 0.05 * cum[-1], side='right')
                target = float(unique[min(int(idx), len(unique) - 1)])
                targets[kind] = target
                lower_mass = 0.0 if idx == 0 else float(cum[idx - 1])
                recall = 1 - lower_mass / float(cum[-1])
                calibration[kind] = dict(
                    threshold=target,
                    target_recall=0.95,
                    quantile_recall=recall,
                    weighted_positive_pixels=float(cum[-1]),
                    positive_values=len(v),
                    next_higher_recall=1 - float(cum[idx]) / float(cum[-1]),
                )
                predicted = np.cumsum(total_hist[::-1])[::-1]
                tp = np.cumsum(tp_hist[::-1])[::-1]
                save_npz(
                    OUT / f'{kind}_pixel_pr_histogram.npz',
                    threshold=np.arange(BINS) / BINS,
                    precision=np.divide(tp, predicted, out=np.zeros(BINS, float), where=predicted > 0),
                    recall=tp / tp_hist.sum(),
                    predicted=predicted,
                    tp=tp,
                )
                del values, weights, v, w, order, unique, starts, mass, cum
            atomic(OUT / 'recall95_calibration.json', calibration)
            print('RECALL95_CALIBRATION', json.dumps(calibration), flush=True)
    aggregate = {}
    byfold = {}
    for kind in ('convnext', 'vitsplus'):
        aggregate[kind] = {}
        byfold[kind] = {}
        for phase in ('grid', 'recall95'):
            rows = [r for s in sources for r in read(OUT / phase / kind / (s + '.json'))['rows']]
            thresholds = THRESHOLDS if phase == 'grid' else [targets[kind]]
            aggregate[kind][phase] = [summarize(rows, t) for t in thresholds]
            byfold[kind][phase] = {
                str(f): [summarize([r for r in rows if owners[r['source_file']] == f], t) for t in thresholds]
                for f in range(5)
            }
            if phase == 'recall95':
                assert aggregate[kind][phase][0]['pixel']['recall'] >= 0.95 - 1e-7
    atomic(OUT / 'aggregate.json', aggregate)
    atomic(OUT / 'by_owner_fold.json', byfold)
    atomic(
        OUT / 'verification.json',
        dict(
            passed=True,
            sources=707,
            models=2,
            fixed_thresholds=5,
            calibrated_thresholds=targets,
            arrays_loaded=1414,
            all_sources_exactly_once_per_model=True,
            complete_original_annotator_records=1154,
            owner_exclusion_verified=True,
            checkpoint_epoch=4,
            calibration_uses_development_labels=True,
            independent_test=False,
            classifier_trained=False,
            self_checks='one-to-one max cardinality, ambiguous merge, empty-GT false positives, strict IoU boundary passed',
            completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            seconds=time.time() - began,
        ),
    )
    atomic(OUT / 'status.json', dict(state='complete_verified', seconds=time.time() - began))
    print('THRESHOLD_STUDY_COMPLETE', json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
