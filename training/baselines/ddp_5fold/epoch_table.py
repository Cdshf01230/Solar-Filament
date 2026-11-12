"""Recompute the out-of-fold table of README section 2.6 from the stored validation histories.

python training/baselines/ddp_5fold/epoch_table.py

Each fold model was validated after every epoch on its own held-out fold (threshold 0.5, 8-connected components,
area >= 128 px, no test-time augmentation). TP, FP, FN and the IoU sum are pooled over the 1154 annotation records
before PQ is computed. The models are trained for 20 epochs. The epoch-20 validation of fold 3 is stored separately
(fold3_epoch20.json).
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[3] / 'docs/results/ddp_5fold'
EPOCHS = range(1, 21)
SCORERS = ('strict', 'legacy')


def load_counts():
    counts = {}
    for fold in range(5):
        for row in json.loads((RESULTS / f'history_fold{fold}.json').read_text(encoding='utf-8')):
            if row.get('val') and row['epoch'] in EPOCHS:
                for s in SCORERS:
                    counts[fold, row['epoch'], s] = tuple(row['val'][s][k] for k in ('tp', 'fp', 'fn', 'sum_iou'))
    late = json.loads((RESULTS / 'fold3_epoch20.json').read_text(encoding='utf-8'))
    for s in SCORERS:
        counts[3, 20, s] = tuple(late[s][k] for k in ('tp', 'fp', 'fn', 'sum_iou'))
    missing = [(f, e) for f in range(5) for e in EPOCHS if (f, e, 'strict') not in counts]
    if missing:
        raise ValueError(f'Missing validation for (fold, epoch): {missing}')
    return counts


def pq(c):
    tp, fp, fn, iou = c
    return iou / (tp + 0.5 * fp + 0.5 * fn)


def pool(counts, epochs, scorer, folds=range(5)):
    total = [0.0, 0.0, 0.0, 0.0]
    for f in folds:
        total = [a + b for a, b in zip(total, counts[f, epochs[f], scorer])]
    return total


def main():
    counts = load_counts()

    def best(score):  # highest PQ, ties to the earlier epoch
        return max(EPOCHS, key=lambda e: (score(e), -e))

    own = [best(lambda e, f=f: pq(counts[f, e, 'strict'])) for f in range(5)]
    lofo = [best(lambda e, f=f: pq(pool(counts, [e] * 5, 'strict', [j for j in range(5) if j != f]))) for f in range(5)]
    common = best(lambda e: pq(pool(counts, [e] * 5, 'strict')))
    rows = [
        ('Epoch 20 for every fold (submitted)', [20] * 5),
        ('Epoch chosen on the other four folds (leave-one-fold-out)', lofo),
        ('One epoch for all folds, best pooled PQ (optimistic)', [common] * 5),
        ('Best epoch on its own held-out fold (optimistic)', own),
    ]
    print('pooled strict PQ per epoch:', ' '.join(f'{e}:{pq(pool(counts, [e] * 5, "strict")):.4f}' for e in EPOCHS))
    for name, epochs in rows:
        s, g = pool(counts, epochs, 'strict'), pool(counts, epochs, 'legacy')
        print(
            f'{name}: epochs {epochs}  strict PQ {pq(s):.4f} TP/FP/FN {s[0]:.0f}/{s[1]:.0f}/{s[2]:.0f} SQ {s[3] / s[0]:.4f}'
            f'  | legacy PQ {pq(g):.4f} TP {g[0]:.0f}'
        )


if __name__ == '__main__':
    main()
