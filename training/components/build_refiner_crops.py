"""Local-input adaptation of the original v1 crop builder. No cloud credentials required.
Map folder: <source>.npz (probability float32) and <source>.json (owner-exclusion evidence).
"""

import json, csv, zipfile, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, argparse
import threshold_sweep as scorer

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
for name in ('maps', 'ml', 'data-zip', 'out'):
    parser.add_argument('--' + name, required=True)
args = parser.parse_args()
ARR = Path(args.ml)
OUT = Path(args.out)
KIND, POINT, ARM, THRESHOLD, ML_THRESHOLD, PAD, MIN_AREA = 'vitsplus', 'proposal005', 'balance', 0.05, 0.5, 128, 8


class LocalResponse:
    def __init__(self, path):
        self.content = Path(path).read_bytes()

    def raise_for_status(self):
        pass


class LocalSession:
    def get(self, path, timeout=None):
        return LocalResponse(path)


def components(prob):
    n, lab, stats, _ = cv2.connectedComponentsWithStats((prob >= THRESHOLD).astype(np.uint8), 8, cv2.CV_32S)
    ids = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= MIN_AREA)
    ids = ids[ids > 0]
    s = stats[ids]
    boxes = np.column_stack([s[:, 0], s[:, 1], s[:, 0] + s[:, 2], s[:, 1] + s[:, 3]]).astype(np.int32)
    return lab, stats, ids, boxes


def build_source(source, links, archive_lock, archive, session):
    with np.load(ARR / 'features' / KIND / POINT / (source + '.npz')) as z:
        ref_ids, ref_boxes, y, record_ids, fold = (
            z['ids'],
            z['boxes'],
            z['y'],
            [str(r) for r in z['record_ids']],
            int(z['owner_fold']),
        )
    with np.load(ARR / 'scores' / KIND / POINT / ARM / (source + '.npz')) as z:
        score, fold_scores, sfold = z['mean_oof'], z['fold_scores'], int(z['owner_fold'])
    assert sfold == fold and len(score) == len(ref_ids)
    evidence = json.loads(session.get(links[source + '.json'], timeout=(30, 180)).content)
    assert (
        evidence['passed']
        and evidence['checkpoint_epoch'] == 4
        and evidence['owner_fold'] == fold
        and fold not in evidence['eligible_folds']
        and len(evidence['eligible_folds']) == 4
    )
    assert np.all(np.isnan(fold_scores[:, fold])) and np.all(np.isfinite(np.delete(fold_scores, fold, 1)))
    r = session.get(links[source + '.npz'], timeout=(30, 300))
    r.raise_for_status()
    import io

    with np.load(io.BytesIO(r.content)) as z:
        prob = z['probability']
        assert prob.shape == (2048, 2048) and prob.dtype == np.float32
    lab, stats, ids, boxes = components(prob)
    assert np.array_equal(ids, ref_ids) and np.array_equal(
        boxes, ref_boxes
    ), 'component mismatch with remote presence-ML features'
    with archive_lock:
        raw = archive.read('MAGFiLO_1.0_Kaggle_2026/train/train_images/' + source)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), 0)
    assert image.shape == (2048, 2048)
    records = scorer.RECORDS[source]
    assert [str(rec['id']) for rec in records] == record_ids
    inst = []  # (record_index, annotation_index, mask)
    for ri, rec in enumerate(records):
        for ai, m in enumerate(scorer.masks_for(rec)):
            inst.append((ri, ai, m))
    binary = prob >= THRESHOLD
    keep = score >= ML_THRESHOLD
    rows = []
    comp_rows = []
    for k, cid in enumerate(ids):
        x0, y0, x1, y1 = map(int, boxes[k])
        area = int(stats[cid, cv2.CC_STAT_AREA])
        comp = dict(
            source=source,
            owner_fold=fold,
            component=int(cid),
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            area=area,
            touch_records=int(y[:, k].sum()),
            n_records=len(records),
            ml_mean_oof=float(score[k]),
            **{f'ml_fold{f}': ('' if f == fold else float(fold_scores[k, f])) for f in range(5)},
            kept=bool(keep[k]),
        )
        comp_rows.append(comp)
        if not keep[k]:
            continue
        X0, Y0, X1, Y1 = max(0, x0 - PAD), max(0, y0 - PAD), min(2048, x1 + PAD), min(2048, y1 + PAD)
        cm = lab[Y0:Y1, X0:X1] == cid
        cmask_full_area = int(cm.sum())
        assert cmask_full_area == area
        masks = []
        irec = []
        iann = []
        iover = []
        ifull = []
        iarea = []
        for ri, ai, m in inst:
            sub = m[Y0:Y1, X0:X1]
            inside = int(sub.sum())
            if inside == 0:
                continue
            masks.append(sub)
            irec.append(ri)
            iann.append(ai)
            iover.append(int((sub & cm).sum()))
            total = int(m.sum())
            iarea.append(total)
            ifull.append(inside == total)
        irec = np.array(irec, np.int16)
        iover = np.array(iover, np.int32)
        target = np.full(len(records), -1, np.int16)
        for ri in range(len(records)):
            cand = np.flatnonzero((irec == ri) & (iover > 0))
            if len(cand):
                target[ri] = cand[np.argmax(iover[cand])]
        touch_counts = [int(((irec == ri) & (iover > 0)).sum()) for ri in range(len(records))]
        assert [t > 0 for t in touch_counts] == [bool(v) for v in y[:, k]], 'touch label mismatch with presence-ML'
        crop_id = f'{Path(source).stem}_{int(cid):05d}'
        np.savez_compressed(
            OUT / 'step3_crops' / 'crops' / (crop_id + '.npz'),
            image=image[Y0:Y1, X0:X1],
            probability=prob[Y0:Y1, X0:X1].astype(np.float16),
            component=cm,
            proposal_binary=binary[Y0:Y1, X0:X1],
            instances=np.array(masks, bool).reshape(len(masks), Y1 - Y0, X1 - X0),
            instance_record=irec,
            instance_annotation_index=np.array(iann, np.int16),
            instance_overlap_component=iover,
            instance_area_native=np.array(iarea, np.int32),
            instance_full_in_crop=np.array(ifull, bool),
            record_target=target,
            record_ids=np.array(record_ids),
            crop_box=np.array([X0, Y0, X1, Y1], np.int32),
            component_box=np.array([x0, y0, x1, y1], np.int32),
            source=np.array(source),
            owner_fold=np.array(fold),
            ml_mean_oof=np.array(score[k]),
        )
        rows.append(
            dict(
                crop_id=crop_id,
                source=source,
                owner_fold=fold,
                component=int(cid),
                ml_mean_oof=float(score[k]),
                component_x0=x0,
                component_y0=y0,
                component_x1=x1,
                component_y1=y1,
                component_area=area,
                crop_x0=X0,
                crop_y0=Y0,
                crop_x1=X1,
                crop_y1=Y1,
                crop_w=X1 - X0,
                crop_h=Y1 - Y0,
                n_records=len(records),
                records_with_target=int((target >= 0).sum()),
                positive=bool((target >= 0).any()),
                max_instances_touched_per_record=max(touch_counts) if touch_counts else 0,
                instances_in_crop=len(masks),
                neighbor_instances_not_touching=int((iover == 0).sum()),
                targets_full_in_crop=int(sum(ifull[t] for t in target if t >= 0)),
                empty_records=int((target < 0).sum()),
            )
        )
    gt = []
    kept_ids = ids[keep]
    for ri, ai, m in inst:
        under = lab[m]
        gt.append(
            dict(
                source=source,
                owner_fold=fold,
                record=record_ids[ri],
                annotation_index=ai,
                area=int(m.sum()),
                touched_by_component=bool(np.isin(under, ids).any()),
                touched_by_kept_component=bool(np.isin(under, kept_ids).any()),
            )
        )
    return comp_rows, rows, gt, len(records)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    (OUT / 'step3_crops' / 'crops').mkdir(parents=True, exist_ok=True)
    scorer.initialize()
    archive = zipfile.ZipFile(Path(args.data_zip))
    lock = threading.Lock()
    sources = sorted(p.name[:-4] for p in (ARR / 'features' / KIND / POINT).glob('*.npz'))
    assert len(sources) == 707
    links = {p.name: str(p) for p in Path(args.maps).iterdir()}
    assert all(s + '.npz' in links and s + '.json' in links for s in sources), 'missing remote ensemble maps'
    local = threading.local()
    done = [0]

    def task(s):
        if not hasattr(local, 'session'):
            local.session = LocalSession()
        out = build_source(s, links, lock, archive, local.session)
        done[0] += 1
        if done[0] % 50 == 0:
            print('sources', done[0], flush=True)
        return out

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(task, sources))
    comps = [r for x in results for r in x[0]]
    crops = [r for x in results for r in x[1]]
    gt = [r for x in results for r in x[2]]
    write_csv(
        OUT / 'step1_proposal' / 'components.csv',
        [{k: v for k, v in c.items() if not k.startswith('ml_') and k != 'kept'} for c in comps],
    )
    write_csv(OUT / 'step2_ml' / 'components_scored.csv', comps)
    write_csv(OUT / 'step3_crops' / 'manifest.csv', crops)
    write_csv(OUT / 'gt_instances.csv', gt)
    print('components', len(comps), 'crops', len(crops), 'gt', len(gt), 'records', sum(x[3] for x in results))


if __name__ == '__main__':
    main()
