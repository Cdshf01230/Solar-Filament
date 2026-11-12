"""Fresh707-fold proposal touch classifiers; trainONE/valFOUR, balanced vs 2x positive weight."""

from pathlib import Path
import os

for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import json, time, datetime, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
import joblib
import padding_recall as padding
import threshold_sweep as scorer

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get('SOLAR_ML_OUT', str(ROOT / 'presence_ml')))
OUT.mkdir(parents=True, exist_ok=True)
MEANS = Path(os.environ.get('SOLAR_MEAN_ROOT', str(ROOT / 'ensemble')))
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


def data_zip():
    """Competition zip, used only when the unpacked image/annotation paths are not given."""
    if not os.environ.get('SOLAR_DATA_ZIP'):
        raise SystemExit('Set SOLAR_RAW_IMAGES and SOLAR_ANNOTATION_JSON, or SOLAR_DATA_ZIP to the competition zip')
    return Path(os.environ['SOLAR_DATA_ZIP'])


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def atomic(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(v, allow_nan=False), encoding='utf-8')
    q.replace(p)


def npz(p, **data):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    with q.open('wb') as f:
        np.savez_compressed(f, **data)
    q.replace(p)


def emit(**state):
    atomic(OUT / 'status.json', state)
    print(json.dumps(state), flush=True)


def initialize():
    cv2.setNumThreads(1)
    scorer.initialize()


def load_image(source):
    if os.environ.get('SOLAR_RAW_IMAGES'):
        a = cv2.imread(str(Path(os.environ['SOLAR_RAW_IMAGES']) / source), 0)
    else:
        with zipfile.ZipFile(data_zip()) as archive:
            a = cv2.imdecode(
                np.frombuffer(archive.read('MAGFiLO_1.0_Kaggle_2026/train/train_images/' + source), np.uint8), 0
            )
    assert a is not None and a.shape == (2048, 2048)
    return a


def get_components(prob, threshold):
    n, lab, stats, centers = cv2.connectedComponentsWithStats((prob >= threshold).astype(np.uint8), 8, cv2.CV_32S)
    ids = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 8)
    ids = ids[ids > 0]
    s = stats[ids]
    boxes = np.column_stack([s[:, 0], s[:, 1], s[:, 0] + s[:, 2], s[:, 1] + s[:, 3]]).astype(np.int32)
    return n, lab, stats, centers, ids, boxes


def appearance(raw):
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


def feature_matrix(prob, raw, maps, threshold, components):
    n, lab, stats, centers, ids, boxes = components
    area = stats[ids, cv2.CC_STAT_AREA].astype(float)
    flat = lab.ravel()

    def means(v):
        return np.bincount(flat, weights=v.ravel(), minlength=n)[ids] / area

    pm = means(prob)
    ps = np.sqrt(np.maximum(means(prob * prob) - pm * pm, 0))
    rm = means(maps[0])
    rs = np.sqrt(np.maximum(means(maps[0] * maps[0]) - rm * rm, 0))
    pmin = np.asarray(scorer.ndi.minimum(prob, lab, index=ids))
    pmax = np.asarray(scorer.ndi.maximum(prob, lab, index=ids))
    distance = cv2.distanceTransform((lab > 0).astype(np.uint8), cv2.DIST_L2, 5)
    edtm = means(distance)
    edtx = np.asarray(scorer.ndi.maximum(distance, lab, index=ids))
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
    context = padding.pad_boxes(boxes, 16, prob.shape)
    carea = (context[:, 2] - context[:, 0]) * (context[:, 3] - context[:, 1])
    carea = np.maximum(carea, 1)
    cm = padding.rect_sums(cv2.integral(maps[0], sdepth=cv2.CV_64F), context) / carea
    cs = np.sqrt(
        np.maximum(padding.rect_sums(cv2.integral(maps[0] * maps[0], sdepth=cv2.CV_64F), context) / carea - cm * cm, 0)
    )
    cp = padding.rect_sums(cv2.integral(prob, sdepth=cv2.CV_64F), context) / carea
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
        cx / 2048,
        cy / 2048,
        np.minimum.reduce([boxes[:, 0], boxes[:, 1], 2048 - boxes[:, 2], 2048 - boxes[:, 3]]) / 2048,
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
    assert X.shape == (len(ids), len(FEATURES)) and np.isfinite(X).all()
    return X


def extract(task):
    kind, source, points = task
    raw = None
    maps = None
    with np.load(MEANS / kind / (source + '.npz'), allow_pickle=False) as pack:
        prob = pack['probability']
    evidence = read(MEANS / kind / (source + '.json'))
    assert (
        evidence['passed']
        and evidence['checkpoint_epoch'] == 4
        and evidence['owner_fold'] not in evidence['eligible_folds']
        and len(evidence['eligible_folds']) == 4
    )
    groups = []
    for rec in scorer.RECORDS[source]:
        masks = scorer.masks_for(rec)
        u = np.logical_or.reduce(masks) if masks else np.zeros(prob.shape, bool)
        groups.append((str(rec['id']), u))
    count = 0
    for point, threshold in points.items():
        dest = OUT / 'features' / kind / point / (source + '.npz')
        if dest.exists():
            with np.load(dest, allow_pickle=False) as z:
                assert z['X'].shape[1] == len(FEATURES) and str(z['source']) == source
            continue
        components = get_components(prob, threshold)
        n, lab, stats, centers, ids, boxes = components
        if raw is None:
            raw = load_image(source)
            maps = appearance(raw)
        X = feature_matrix(prob, raw, maps, threshold, components)
        y = np.array([np.bincount(lab[u], minlength=n)[ids] > 0 for _, u in groups], bool)
        npz(
            dest,
            X=X,
            y=y,
            ids=ids,
            boxes=boxes,
            record_ids=np.array([r for r, _ in groups]),
            source=np.array(source),
            owner_fold=np.array(evidence['owner_fold']),
            threshold=np.array(threshold),
        )
        count += len(ids)
    return dict(model=kind, source=source, new_components=count)


def confusion(y, p, weight=None):
    weight = np.ones(len(y), float) if weight is None else weight
    tp = float(weight[y & p].sum())
    fp = float(weight[~y & p].sum())
    fn = float(weight[y & ~p].sum())
    tn = float(weight[~y & ~p].sum())
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=tp / max(1e-12, tp + fp),
        recall=tp / max(1e-12, tp + fn),
        f1=2 * tp / max(1e-12, 2 * tp + fp + fn),
    )


def fit_pipeline(kind, point, sources, owners):
    from lightgbm import LGBMClassifier
    from threadpoolctl import threadpool_limits

    Xs = []
    Ys = []
    owners_unique = []
    train_weights = []
    unique_indices = []
    rowowners = []
    lookup = {}
    start = 0
    for source in sources:
        with np.load(OUT / 'features' / kind / point / (source + '.npz'), allow_pickle=False) as z:
            X = z['X']
            y = z['y']
            ids = z['ids']
        length = len(X)
        lookup[source] = (start, start + length)
        Xs.append(X)
        Ys.append(y.ravel())
        owners_unique.extend([owners[source]] * length)
        unique_indices.extend(np.tile(np.arange(start, start + length), len(y)))
        rowowners.extend([owners[source]] * y.size)
        train_weights.extend([1 / len(y)] * y.size)
        start += length
    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    unique_indices = np.array(unique_indices, np.int64)
    rowowners = np.array(rowowners, np.int8)
    base = np.array(train_weights, float)
    owners_unique = np.array(owners_unique, np.int8)
    assert X.shape[1] == len(FEATURES) and np.isfinite(X).all()
    folder = OUT / 'models' / kind / point
    folder.mkdir(parents=True, exist_ok=True)
    foldmetrics = []
    evidence = []
    predictions = {arm: np.full((len(X), 5), np.nan, np.float32) for arm in ['balance', 'recall']}
    for fold in range(5):
        train = rowowners == fold
        val = ~train
        weightedpos = base[train & Y].sum()
        weightedneg = base[train & ~Y].sum()
        assert weightedpos > 0 and weightedneg > 0
        assert {s for s in sources if owners[s] == fold} == set(
            read(ROOT / 'fold_assignment.json')['folds'][fold]['train_sources']
        )
        for arm, factor in [('balance', 1.0), ('recall', 48.0)]:
            began = time.time()
            path = folder / f'{arm}_fold{fold}.joblib'
            positiveweight = factor * weightedneg / weightedpos
            weights = base[train] * np.where(Y[train], positiveweight, 1.0)
            weights *= train.sum() / weights.sum()
            model = LGBMClassifier(
                n_estimators=250,
                num_leaves=15,
                max_depth=6,
                learning_rate=0.05,
                min_child_samples=30,
                reg_lambda=3,
                n_jobs=2,
                random_state=42,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            with threadpool_limits(limits=2):
                model.fit(X[unique_indices[train]], Y[train], sample_weight=weights)
            joblib.dump(model, path)
            with threadpool_limits(limits=2):
                p = model.predict_proba(X)[:, 1].astype(np.float32)
            loaded = joblib.load(path)
            sample = np.linspace(0, len(X) - 1, min(2048, len(X)), dtype=int)
            with threadpool_limits(limits=2):
                assert np.array_equal(loaded.predict_proba(X[sample])[:, 1].astype(np.float32), p[sample])
            predictions[arm][owners_unique != fold, fold] = p[owners_unique != fold]
            valmetrics = confusion(Y[val], p[unique_indices[val]] >= 0.5)
            weightedmetrics = confusion(Y[val], p[unique_indices[val]] >= 0.5, base[val])
            item = dict(
                model=kind,
                point=point,
                arm=arm,
                trained_fold=fold,
                train_sources=int((np.array(list(owners.values())) == fold).sum()),
                validation_sources=707 - int((np.array(list(owners.values())) == fold).sum()),
                train_record_proposals=int(train.sum()),
                validation_record_proposals=int(val.sum()),
                train_weighted_positive=float(weightedpos),
                train_weighted_negative=float(weightedneg),
                positive_weight=float(positiveweight),
                recall_multiplier=factor,
                classifier_threshold=0.5,
                train_seconds=time.time() - began,
                metrics=valmetrics,
                source_record_averaged_metrics=weightedmetrics,
            )
            foldmetrics.append(item)
            evidence.append(
                dict(
                    model=kind,
                    point=point,
                    arm=arm,
                    fold=fold,
                    reloaded_predictions_exact=True,
                    trees=int(model.n_estimators_),
                    train_sources=item['train_sources'],
                    train_days_disjoint_from_heldout=True,
                )
            )
            model.booster_.save_model(str(path.with_suffix('.txt')))
            emit(state='classifier_fitted', **item)
    for arm, p in predictions.items():
        assert np.all(np.isfinite(p).sum(axis=1) == 4) and np.all(np.isnan(p[np.arange(len(X)), owners_unique]))
        mean = np.nanmean(p, axis=1)
        for source, (a, b) in lookup.items():
            npz(
                OUT / 'scores' / kind / point / arm / (source + '.npz'),
                mean_oof=mean[a:b],
                fold_scores=p[a:b],
                owner_fold=np.array(owners[source]),
            )
    atomic(folder / 'fold_metrics.json', foldmetrics)
    atomic(folder / 'verification.json', evidence)
    return foldmetrics, evidence


def evaluate_filtered(task):
    kind, source, points = task
    dest = OUT / 'coverage_rows' / kind / (source + '.json')
    if dest.exists():
        return dict(model=kind, source=source, cached=True)
    with np.load(MEANS / kind / (source + '.npz'), allow_pickle=False) as z:
        prob = z['probability']
    groups = []
    for rec in scorer.RECORDS[source]:
        masks = scorer.masks_for(rec)
        union = np.logical_or.reduce(masks) if masks else np.zeros(prob.shape, bool)
        details = []
        for m in masks:
            yy, xx = np.nonzero(m)
            x0, x1 = int(xx.min()), int(xx.max() + 1)
            y0, y1 = int(yy.min()), int(yy.max() + 1)
            details.append((m, int(m.sum()), (x0, y0, x1, y1), cv2.integral(m[y0:y1, x0:x1].astype(np.uint8))))
        groups.append((str(rec['id']), union, details, cv2.integral(union.astype(np.uint8))))
    rows = []
    comparisons = []
    for point, threshold in points.items():
        n, lab, stats, centers, ids, boxes = get_components(prob, threshold)
        with np.load(OUT / 'features' / kind / point / (source + '.npz'), allow_pickle=False) as z:
            assert np.array_equal(ids, z['ids']) and np.array_equal(boxes, z['boxes'])
            y = z['y']
            recordids = z['record_ids']
        keeps = {'unfiltered': np.ones(len(ids), bool)}
        for arm in ['balance', 'recall']:
            with np.load(OUT / 'scores' / kind / point / arm / (source + '.npz'), allow_pickle=False) as z:
                assert z['fold_scores'].shape == (len(ids), 5) and np.all(
                    np.isfinite(z['fold_scores']).sum(axis=1) == 4
                )
                keeps[arm] = z['mean_oof'] >= 0.5
        for arm, keep in keeps.items():
            kept_ids = ids[keep]
            kept_boxes = boxes[keep]
            retained = np.zeros(n, bool)
            retained[kept_ids] = True
            for pad in [0, 64, 128]:
                bb = padding.pad_boxes(kept_boxes, pad, prob.shape)
                unioncrop = padding.union_rects(bb, prob.shape)
                area = (bb[:, 2] - bb[:, 0]).astype(np.int64) * (bb[:, 3] - bb[:, 1])
                sourcecost = dict(
                    kept_crops=len(bb),
                    sum_crop_pixels=int(area.sum()),
                    union_pixels=int(unioncrop.sum()),
                    image_pixels=prob.size,
                )
                for index, (record, union, details, integral) in enumerate(groups):
                    assert str(recordids[index]) == record
                    recalculated = np.bincount(lab[union], minlength=n)[ids] > 0
                    assert np.array_equal(recalculated, y[index])
                    confusion_counts = confusion(y[index], keep)
                    crop_hits = padding.rect_sums(integral, bb) > 0 if len(bb) else np.zeros(0, bool)
                    component_gt = 0
                    single = []
                    combined = []
                    for m, gtarea, (x0, y0, x1, y1), mi in details:
                        component_gt += int(retained[lab[m]].any())
                        overlapping = bb[(bb[:, 0] < x1) & (bb[:, 2] > x0) & (bb[:, 1] < y1) & (bb[:, 3] > y0)]
                        if len(overlapping):
                            relative = overlapping.copy()
                            relative[:, [0, 2]] = relative[:, [0, 2]].clip(x0, x1) - x0
                            relative[:, [1, 3]] = relative[:, [1, 3]].clip(y0, y1) - y0
                            best = float(padding.rect_sums(mi, relative).max() / gtarea)
                        else:
                            best = 0.0
                        single.append(best)
                        combined.append(float(unioncrop[m].sum() / gtarea))
                    rows.append(
                        dict(
                            source_file=source,
                            record=record,
                            point=point,
                            threshold=threshold,
                            arm=arm,
                            padding=pad,
                            gt_objects=len(details),
                            component_tp_gt=component_gt,
                            component_fn_gt=len(details) - component_gt,
                            proposal_confusion=confusion_counts,
                            crop_tp=int(crop_hits.sum()),
                            crop_fp=int((~crop_hits).sum()),
                            single_crop=padding.coverage_counts(single),
                            union_crops=padding.coverage_counts(combined),
                            **sourcecost,
                        )
                    )
                reference = (
                    Path(os.environ.get('SOLAR_PADDING_REFERENCE', '/nonexistent')) / 'rows' / kind / (source + '.json')
                )
                if arm == 'unfiltered' and reference.exists():
                    previous = read(reference)
                    for r in [r for r in rows if r['point'] == point and r['arm'] == arm and r['padding'] == pad]:
                        v = next(
                            x
                            for x in previous['rows']
                            if x['record'] == r['record'] and x['threshold'] == threshold and x['padding'] == pad
                        )
                        assert (
                            r['component_tp_gt'] == v['component_touch_gt']
                            and r['crop_tp'] == v['crop_touch_predictions']
                        )
                        assert r['single_crop'] == v['single_crop'] and r['union_crops'] == v['union_crops']
                    comparisons.append(dict(point=point, padding=pad, passed=True))
    atomic(dest, dict(model=kind, source=source, rows=rows, padding_reference_comparisons=comparisons))
    return dict(model=kind, source=source, cached=False, reference_comparisons=len(comparisons))


def summary(rows):
    images = {r['source_file']: r for r in rows}
    total = sum(r['gt_objects'] for r in rows)
    counts = {key: sum(r['proposal_confusion'][key] for r in rows) for key in ['tp', 'fp', 'fn', 'tn']}
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    counts.update(precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn))
    out = dict(
        sources=len(images),
        records=len(rows),
        objects=total,
        proposal=counts,
        gt_touch_tp=sum(r['component_tp_gt'] for r in rows),
        gt_touch_fn=sum(r['component_fn_gt'] for r in rows),
        crop_tp=sum(r['crop_tp'] for r in rows),
        crop_fp=sum(r['crop_fp'] for r in rows),
        actual_kept_crops=sum(r['kept_crops'] for r in images.values()),
        union_image_fraction=sum(r['union_pixels'] for r in images.values())
        / sum(r['image_pixels'] for r in images.values()),
        crop_pixels_image_multiple=sum(r['sum_crop_pixels'] for r in images.values())
        / sum(r['image_pixels'] for r in images.values()),
    )
    out['gt_touch_recall'] = out['gt_touch_tp'] / max(1, total)
    out['crop_precision'] = out['crop_tp'] / max(1, out['crop_tp'] + out['crop_fp'])
    for kind in ['single_crop', 'union_crops']:
        out[kind] = {
            key: sum(r[kind][key] for r in rows)
            for key in ['touch', 'covered50', 'covered90', 'covered95', 'covered100']
        }
        out[kind]['fn_touch'] = total - out[kind]['touch']
        out[kind]['fn_covered95'] = total - out[kind]['covered95']
        out[kind]['fn_covered100'] = total - out[kind]['covered100']
    return out


def main():
    padding.self_check()
    assignment = read(ROOT / 'fold_assignment.json')
    owners = {s: f['fold'] for f in assignment['folds'] for s in f['train_sources']}
    assert len(owners) == 707
    for a in assignment['folds']:
        for b in assignment['folds']:
            if a['fold'] != b['fold']:
                assert not set(a['days']) & set(b['days'])
    cal = read(ROOT / 'recall95_calibration.json')
    points = {k: dict(proposal005=0.05, pixelrecall95=cal[k]['threshold']) for k in ['convnext', 'vitsplus']}
    atomic(
        OUT / 'protocol.json',
        dict(
            sources=707,
            fold_sources=[141, 141, 142, 141, 142],
            fold_assignment=assignment,
            source_mean_ref='oof-threshold-analysis/1',
            proposal_points=points,
            label='at least1 overlap pixel between CC8 area>=8 and GT union WITHIN ORIGINAL RECORD; no ambiguous exclusion; average record weights within source',
            classifier='LightGBM 250trees,15leaves,maxdepth6,LR.05,minchild30,L2=3,seed42; no early stopping or val tuning',
            features=FEATURES,
            feature_count=len(FEATURES),
            split='five fits per arm/pipeline; train ONE exact ownerfold, predict otherFOUR; report each separately, plus mean of exactly4 eligible classifiers/source',
            class_weight='balance:weightedNneg/weightedNpos from TRAIN ONLY; recall:48x that positive weight; negatives1, source/record weights then normalized mean1',
            threshold=0.5,
            metrics='proposal confusion separate from GT object touch counts and padded crop coverage. No strict matching.',
            padding_per_side=[0, 64, 128],
            limitation='Development pipeline. Upstream heads shared between classifier folds; not nested end-to-end CV. pixelrecall95 proposal threshold calibrated on development707; no independent-test claim.',
        ),
    )
    began = time.time()
    tasks = [(kind, s, points[kind]) for kind in points for s in sorted(owners)]
    with ProcessPoolExecutor(max_workers=4, initializer=initialize) as pool:
        for count, future in enumerate(as_completed([pool.submit(extract, t) for t in tasks]), 1):
            last = future.result()
            if count % 20 == 0 or count == len(tasks):
                emit(
                    state='extracting_features',
                    completed=count,
                    total=len(tasks),
                    last=last,
                    seconds=time.time() - began,
                )
    atomic(
        OUT / 'feature_verification.json',
        dict(
            passed=True,
            sources=707,
            pipelines=4,
            features=len(FEATURES),
            gt_excluded_from_features=True,
            mean_oof_exclusion_verified=True,
        ),
    )
    foldmetrics = []
    verification = []
    for kind in points:
        for point in points[kind]:
            metrics, evidence = fit_pipeline(kind, point, sorted(owners), owners)
            foldmetrics += metrics
            verification += evidence
    atomic(OUT / 'fold_metrics.json', foldmetrics)
    with ProcessPoolExecutor(max_workers=4, initializer=initialize) as pool:
        for count, future in enumerate(as_completed([pool.submit(evaluate_filtered, t) for t in tasks]), 1):
            last = future.result()
            if count % 20 == 0 or count == len(tasks):
                emit(
                    state='evaluating_filtered_padding',
                    completed=count,
                    total=len(tasks),
                    last=last,
                    seconds=time.time() - began,
                )
    aggregate = {}
    byfold = {}
    comparisons = 0
    for kind in points:
        payloads = [read(OUT / 'coverage_rows' / kind / (s + '.json')) for s in sorted(owners)]
        rows = [r for p in payloads for r in p['rows']]
        comparisons += sum(len(p['padding_reference_comparisons']) for p in payloads)
        aggregate[kind] = {}
        byfold[kind] = {}
        for point in points[kind]:
            aggregate[kind][point] = {}
            byfold[kind][point] = {}
            for arm in ['unfiltered', 'balance', 'recall']:
                aggregate[kind][point][arm] = {}
                byfold[kind][point][arm] = {}
                for pad in [0, 64, 128]:
                    rr = [r for r in rows if r['point'] == point and r['arm'] == arm and r['padding'] == pad]
                    s = summary(rr)
                    assert (
                        s['sources'] == 707
                        and s['records'] == 1154
                        and s['objects'] == 8199
                        and s['gt_touch_tp'] + s['gt_touch_fn'] == 8199
                    )
                    if arm != 'unfiltered':
                        b = aggregate[kind][point]['unfiltered'][str(pad)]
                        assert s['actual_kept_crops'] <= b['actual_kept_crops'] and s['gt_touch_tp'] <= b['gt_touch_tp']
                        s['additional_gt_touch_fn_due_to_filter'] = s['gt_touch_fn'] - b['gt_touch_fn']
                        s['crops_removed'] = b['actual_kept_crops'] - s['actual_kept_crops']
                    aggregate[kind][point][arm][str(pad)] = s
                    byfold[kind][point][arm][str(pad)] = {
                        str(f): summary([r for r in rr if owners[r['source_file']] == f]) for f in range(5)
                    }
        del payloads, rows
    atomic(OUT / 'aggregate.json', aggregate)
    atomic(OUT / 'by_owner_fold.json', byfold)
    final = dict(
        passed=True,
        classifiers_loaded_and_checked=len(verification),
        expected_classifiers=40,
        sources=707,
        records=1154,
        gt_objects=8199,
        exact_train1_val4_folds=True,
        eligible_classifiers_per_source=4,
        owner_exclusion_verified=True,
        classifier_threshold=0.5,
        source_features_gt_free=True,
        padding_per_side=[0, 64, 128],
        padding_reference_comparisons=comparisons,
        calibrated_on_val=False,
        neural_inference_rerun=False,
        completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seconds=time.time() - began,
    )
    assert len(verification) == 40
    atomic(OUT / 'verification.json', final)
    emit(state='complete_verified', **final)


if __name__ == '__main__':
    main()
