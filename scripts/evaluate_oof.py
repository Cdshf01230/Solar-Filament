"""Score out-of-fold predictions per annotation record and compute the organisers' rubric statistics.

    python scripts/evaluate_oof.py --predictions reference/oof_predictions.csv \
        --annotations /data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json --out docs/results/oof

Input: predictions in submission format (filament_id = <image stem>_<n>, COCO RLE counts of a 2048 x 2048 mask) for
the labelled images, and the competition annotation file. Every (image, annotation record) pair is scored separately.

Output (`--out`): `oof_metrics.json` and `oof_distributions.{pdf,png}` with
* PQ / SQ / RQ and TP / FP / FN under the strict one-to-one rule and the organisers' all-pairs rule;
* a 95% bootstrap interval over images and the per-fold scores (folds of `splits/fold_assignment.json`);
* the statistics of the organisers' self-evaluation notebook: IoU and Dice of every overlapping (GT, prediction)
  pair, the number of predictions that overlap each GT (one-to-many) and the number of GT that each prediction
  overlaps (many-to-one); "overlap" means IoU > 0, as in that notebook;
* the IoU and Dice of the strict true-positive pairs and the foreground Dice of the union masks of each record.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from filament_pq import metrics  # noqa: E402
from filament_pq.config import IMAGE_SIZE  # noqa: E402


def read_predictions(paths):
    """{image stem: [RLE, ...]} from one or more submission-format CSV files."""
    preds = defaultdict(list)
    for path in paths:
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                stem = row['filament_id'].rsplit('_', 1)[0]
                preds[stem].append(dict(size=[IMAGE_SIZE, IMAGE_SIZE], counts=row['segmentation_rle'].encode()))
    return preds


def gt_records(annotation_file):
    """{image stem: [(record id, [RLE, ...]), ...]}; one record per annotator of an image."""
    ann = json.loads(Path(annotation_file).read_text(encoding='utf-8'))
    by_record = defaultdict(list)
    for a in ann['annotations']:
        by_record[a['image_id']].append(a)
    out = defaultdict(list)
    for rec in ann['images']:
        h, w = int(rec['height']), int(rec['width'])
        rles = []
        for a in by_record[rec['id']]:
            seg = a['segmentation']
            seg = json.loads(seg) if isinstance(seg, str) else seg
            if isinstance(seg, list):
                r = mu.merge(mu.frPyObjects(seg, h, w))
            elif isinstance(seg['counts'], list):
                r = mu.frPyObjects(seg, h, w)
            else:
                r = dict(seg, counts=seg['counts'].encode() if isinstance(seg['counts'], str) else seg['counts'])
            if mu.area(r) > 0:
                rles.append(r)
        out[Path(rec['file_name']).stem].append((rec['id'], rles))
    return out


def intersection(a, b):
    return float(mu.area(mu.merge([a, b], intersect=True)))


def score_record(gt, pred, stats):
    """Adds one record's counts and pair statistics to `stats`; returns (host, strict) counts."""
    iou = metrics.iou_matrix(gt, pred)
    host, strict = metrics.match(iou)
    ga = mu.area(gt).astype(float) if gt else np.zeros(0)
    pa = mu.area(pred).astype(float) if pred else np.zeros(0)
    hit = iou > 0
    for i, j in zip(*np.nonzero(hit)):
        inter = intersection(gt[i], pred[j])
        stats['pair_iou'].append(inter / (ga[i] + pa[j] - inter))
        stats['pair_dice'].append(2 * inter / (ga[i] + pa[j]))
    stats['gt_degree'].update(hit.sum(1).tolist() if gt else [])
    stats['pred_degree'].update(hit.sum(0).tolist() if pred else [])
    # strict true positives: recompute the one-to-one assignment to get the matched pairs
    if gt and pred:
        from scipy.optimize import linear_sum_assignment

        q = iou > 0.5
        a, b = linear_sum_assignment(q * (1 + iou / (min(q.shape) + 1)), maximize=True)
        for i, j in zip(a, b):
            if q[i, j]:
                inter = intersection(gt[i], pred[j])
                stats['tp_iou'].append(inter / (ga[i] + pa[j] - inter))
                stats['tp_dice'].append(2 * inter / (ga[i] + pa[j]))
    ug = mu.merge(gt) if gt else None
    up = mu.merge(pred) if pred else None
    sg, sp = (float(mu.area(ug)) if gt else 0.0), (float(mu.area(up)) if pred else 0.0)
    if sg + sp > 0:
        inter = intersection(ug, up) if gt and pred else 0.0
        stats['union_dice'].append(2 * inter / (sg + sp))
        stats['union_counts'] += np.array([inter, sg, sp])
    return host, strict


def summary(values):
    v = np.asarray(values, float)
    return dict(
        n=int(v.size),
        mean=float(v.mean()),
        median=float(np.median(v)),
        q25=float(np.percentile(v, 25)),
        q75=float(np.percentile(v, 75)),
    )


def degree_table(counter):
    return {str(k): int(counter[k]) for k in sorted(counter)}


def plot(stats, result, path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.75), gridspec_kw=dict(width_ratios=[1, 1, 1.25, 0.85]))
    bins = np.linspace(0, 1, 26)
    for ax, key, name in ((axs[0], 'pair_iou', 'IoU'), (axs[1], 'pair_dice', 'Dice')):
        v = np.asarray(stats[key])
        w = np.full(v.size, 1 / v.size)
        ax.hist(v, bins=bins, weights=w, color='#3b6ea8', edgecolor='white', linewidth=0.4)
        ax.axvline(v.mean(), color='#b2182b', ls='--', lw=1)
        if name == 'IoU':
            ax.axvline(0.5, color='0.4', lw=0.8, ls=':')
        ax.set_title(f'{name}: mean {v.mean():.3f} (dashed)', fontsize=6.5, pad=2)
        ax.set_xlabel(f'{name}, pairs with IoU > 0', fontsize=6.5)
        ax.set_ylabel('fraction of pairs', fontsize=6.5)
    ax = axs[2]
    gd, pdg = stats['gt_degree'], stats['pred_degree']
    top = max(max(gd.values()), max(pdg.values()))
    for side, counts, colors in ((1, gd, ('#d73027', '#1a9850')), (-1, pdg, ('#fc8d59', '#4575b4'))):
        for k, c in counts.items():
            x = side * (min(k, 4) + 0.5)
            ax.bar(x, c, width=0.8, color=colors[1] if k else colors[0])
            if k >= 2:
                ax.text(x, c + 0.02 * top, str(c), ha='center', va='bottom', fontsize=5)
    ax.text(-3.4, top * 0.92, 'predictions', ha='center', fontsize=6)
    ax.text(3.2, top * 0.92, 'GT', ha='center', fontsize=6)
    ticks = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    ax.set_xticks(ticks, ['4+', '3', '2', '1', '0', '0', '1', '2', '3', '4+'], fontsize=6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_title('one-to-many and many-to-one', fontsize=6.5, pad=2)
    ax.set_xlabel('#GT per prediction | #predictions per GT', fontsize=6.5)
    ax.set_ylabel('count', fontsize=6.5)
    ax = axs[3]
    rows = [(str(k), v['pq'], result['per_fold_ci95'][k]) for k, v in result['per_fold'].items()]
    rows.append(('all', result['strict']['pq'], result['bootstrap']['ci95']))
    for x, (name, pq, (lo, hi)) in enumerate(rows):
        ax.errorbar(
            x,
            pq,
            yerr=[[pq - lo], [hi - pq]],
            fmt='o',
            ms=3,
            capsize=2,
            lw=0.9,
            color='#b2182b' if name == 'all' else '#3b6ea8',
        )
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=6)
    ax.set_xlabel('held-out fold', fontsize=6.5)
    ax.set_ylabel('strict PQ (95% CI)', fontsize=6.5)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    for a in axs:
        a.tick_params(labelsize=6)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    for ext in ('pdf', 'png'):
        fig.savefig(path.with_suffix('.' + ext), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--predictions', nargs='+', required=True)
    ap.add_argument('--annotations', required=True)
    ap.add_argument('--folds', default=str(ROOT / 'splits/fold_assignment.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--bootstrap', type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    preds = read_predictions(a.predictions)
    records = gt_records(a.annotations)
    owner = {
        Path(s).stem: f['fold'] for f in json.loads(Path(a.folds).read_text())['folds'] for s in f['train_sources']
    }
    unknown = sorted(set(preds) - set(records))
    if unknown:
        raise ValueError(f'predictions for {len(unknown)} images without annotations, e.g. {unknown[:3]}')

    stats = dict(
        pair_iou=[],
        pair_dice=[],
        tp_iou=[],
        tp_dice=[],
        union_dice=[],
        gt_degree=Counter(),
        pred_degree=Counter(),
        union_counts=np.zeros(3),
    )
    per_image = {}
    for stem in sorted(records):
        h_img, s_img = np.zeros(4), np.zeros(4)
        for _, gt in records[stem]:
            h, s = score_record(gt, preds.get(stem, []), stats)
            h_img += h
            s_img += s
        per_image[stem] = dict(
            fold=owner.get(stem), host=h_img.tolist(), strict=s_img.tolist(), predictions=len(preds.get(stem, []))
        )

    names = sorted(per_image)
    S = np.array([per_image[n]['strict'] for n in names])
    H = np.array([per_image[n]['host'] for n in names])
    fold = np.array([per_image[n]['fold'] for n in names])
    rng = np.random.default_rng(0)
    boot = [metrics.pq(S[rng.integers(0, len(S), len(S))].sum(0))['pq'] for _ in range(a.bootstrap)]
    per_fold = {int(k): metrics.pq(S[fold == k].sum(0)) for k in sorted(set(fold.tolist()))}
    per_fold_ci = {}
    for k in per_fold:
        Sk = S[fold == k]
        b = [metrics.pq(Sk[rng.integers(0, len(Sk), len(Sk))].sum(0))['pq'] for _ in range(a.bootstrap)]
        per_fold_ci[k] = np.percentile(b, [2.5, 97.5]).tolist()
    fold_pq = np.array([v['pq'] for v in per_fold.values()])
    inter, sg, sp = stats['union_counts']
    gd, pdg = stats['gt_degree'], stats['pred_degree']
    result = dict(
        images=len(names),
        records=sum(len(v) for v in records.values()),
        gt=int(sum(len(g) for v in records.values() for _, g in v)),
        predictions=int(sum(map(len, preds.values()))),
        strict=metrics.pq(S.sum(0)),
        organisers_all_pairs=metrics.pq(H.sum(0)),
        bootstrap=dict(resamples=a.bootstrap, unit='image', seed=0, ci95=np.percentile(boot, [2.5, 97.5]).tolist()),
        per_fold=per_fold,
        per_fold_ci95=per_fold_ci,
        fold_mean=float(fold_pq.mean()),
        fold_sd=float(fold_pq.std(ddof=1)),
        overlapping_pairs=dict(iou=summary(stats['pair_iou']), dice=summary(stats['pair_dice'])),
        strict_tp_pairs=dict(iou=summary(stats['tp_iou']), dice=summary(stats['tp_dice'])),
        gt_overlapping_predictions=degree_table(gd),
        predictions_overlapping_gt=degree_table(pdg),
        one_to_many_gt=int(sum(c for k, c in gd.items() if k >= 2)),
        many_to_one_predictions=int(sum(c for k, c in pdg.items() if k >= 2)),
        union_dice_per_record=summary(stats['union_dice']),
        union_dice_pooled=float(2 * inter / (sg + sp)),
        note='Per (image, annotation record). Pairs and degrees use IoU > 0 as in the organisers\' self-evaluation '
        'notebook; a prediction is counted once per record it is scored against.',
    )
    (out / 'oof_metrics.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    (out / 'oof_per_image.json').write_text(json.dumps(per_image, indent=0), encoding='utf-8')
    plot(stats, result, out / 'oof_distributions')
    s = result['strict']
    print(
        f"strict PQ {s['pq']:.6f} (TP {s['tp']:.0f} FP {s['fp']:.0f} FN {s['fn']:.0f}); "
        f"all-pairs PQ {result['organisers_all_pairs']['pq']:.6f}; 95% CI {result['bootstrap']['ci95']}"
    )


if __name__ == '__main__':
    main()
