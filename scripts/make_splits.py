"""Standard 5-fold cross-validation splits derived from the fixed date-disjoint fold assignment (never re-split).

For the refiner, holdout k trains on four folds and validates on k.
Proposal heads and LightGBM classifiers instead train on fold k only; each training image
uses the four models excluding its owner fold. This script does not redefine their split.

    python scripts/make_splits.py --folds weights/fold_assignment.json --out splits
        -> splits/holdout{k}.json            {holdout, train_sources, val_sources, train_folds}
"""

import argparse, json
from pathlib import Path


def holdout_split(assignment, k):
    folds = {f['fold']: sorted(f['train_sources']) for f in assignment['folds']}
    assert sorted(folds) == [0, 1, 2, 3, 4]
    names = [s for v in folds.values() for s in v]
    assert len(names) == len(set(names)) == 707
    train = sorted(s for j, v in folds.items() if j != k for s in v)
    val = folds[k]
    assert not set(train) & set(val) and len(train) + len(val) == 707
    assert not {s[:8] for s in train} & {s[:8] for s in val}, 'fold split must stay date-disjoint'
    return dict(holdout=k, train_folds=[j for j in range(5) if j != k], train_sources=train, val_sources=val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assignment = json.loads(Path(a.folds).read_text(encoding='utf-8'))
    for k in range(5):
        s = holdout_split(assignment, k)
        (out / f'holdout{k}.json').write_text(json.dumps(s, indent=1))
        line = f'holdout{k}: train {len(s["train_sources"])} images (folds {s["train_folds"]}), val {len(s["val_sources"])}'
        print(line)


if __name__ == '__main__':
    main()
