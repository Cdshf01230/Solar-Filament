"""Export candidate crops using either released LightGBM policy (stages 1-2 only)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', required=True)
    p.add_argument('--weights', default='weights')
    p.add_argument('--out', required=True, help='New directory for this policy and image set')
    p.add_argument('--policy', choices=['balance', 'recall'], default='balance')
    p.add_argument('--device', default='cuda')
    p.add_argument(
        '--owner-fold',
        type=int,
        choices=range(5),
        help='OOF mode: select only this owner fold and exclude its head/classifier',
    )
    a = p.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error('--out must be empty; use a separate directory for each policy and image set')
    images = sorted(x for x in Path(a.images).iterdir() if x.suffix.lower() in ('.jpeg', '.jpg', '.png'))
    folds = list(range(5))
    if a.owner_fold is not None:
        assignment = json.loads((Path(a.weights) / 'fold_assignment.json').read_text())
        allowed = set(next(f['train_sources'] for f in assignment['folds'] if f['fold'] == a.owner_fold))
        images = [x for x in images if x.name in allowed]
        folds.remove(a.owner_fold)
    if not images:
        p.error('No matching images found')
    from filament_pq.pipeline import upstream

    upstream(images, a.weights, out, device=a.device, folds=folds, classifier_policy=a.policy)
    info = dict(
        policy=a.policy,
        folds=folds,
        owner_fold=a.owner_fold,
        images=len(images),
        proposal_threshold=0.05,
        ml_threshold=0.5,
        crop_pad_each_side=128,
        scope='Candidate crops only; no refiner predictions or final pipeline PQ',
    )
    (out / 'policy.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
