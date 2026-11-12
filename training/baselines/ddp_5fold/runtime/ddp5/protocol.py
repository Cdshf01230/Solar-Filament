"""Canonical split and resumable batch boundaries; independent of CUDA."""
import json
from pathlib import Path

EPOCHS = 20
# The cosine learning-rate decay (to a 5 % floor) spans LR_HORIZON x the training length, so after the 20 epochs the
# learning rate is at 29 % of its peak.
LR_HORIZON = 1.5
SOFT_SECONDS = 11 * 3600 + 30 * 60
VAL_COUNTS = [(141, 231, 1618), (141, 248, 1817), (142, 221, 1581), (141, 228, 1682), (142, 226, 1501)]
VERSION = 'ddp12-5fold-original-crops-v1'


def partition(assignment, holdout):
    assert 0 <= holdout < 5
    folds = assignment['folds']
    assert [f['fold'] for f in folds] == list(range(5))
    all_names = [s for f in folds for s in f['train_sources']]
    assert len(all_names) == len(set(all_names)) == 707
    val = sorted(folds[holdout]['train_sources'])
    train = sorted(set(all_names) - set(val))
    assert len(val) == VAL_COUNTS[holdout][0]
    assert not ({s[:8] for s in train} & {s[:8] for s in val}), 'capture-day leakage'
    return train, val


def ranges(size, batch=8):
    """Every sample exactly once, both ranks nonempty, including remainder == 1.

    For 17 samples use 8+7+2, avoiding a final singleton. Epoch length remains ceil(N/8).
    """
    assert size >= 2 and batch == 8
    bounds = list(range(0, size, batch)) + [size]
    if size % batch == 1:
        bounds[-2] -= 1
    return list(zip(bounds[:-1], bounds[1:]))


def remaining_ranges(size, offset):
    pairs = ranges(size)
    assert offset in {a for a, _ in pairs} | {size}, 'invalid resume sample cursor'
    return [(a, b) for a, b in pairs if a >= offset]


def lr_horizon_updates(total_updates, horizon=LR_HORIZON):
    """Length of the cosine learning-rate decay, in updates, for a run of `total_updates` updates."""
    return round(total_updates * horizon)


def expected_val(args):
    n, r, g = VAL_COUNTS[args['fold']]
    return dict(sources=n, records=r, gt=g)


def read_assignment(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
