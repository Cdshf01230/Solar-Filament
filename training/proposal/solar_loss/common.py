"""Loss definitions, exact global-batch sampling, CPU quota and durable state helpers."""

import os, math, json, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

LOSSES = {'A': 'BCE + Dice', 'B': 'BCE + Tversky(FP=.6,FN=.4)', 'C': '3*BCE + Dice', 'D': 'BCE + Tversky(FP=.4,FN=.6)'}


def loss_terms(logits, target, arm):
    if arm not in LOSSES:
        raise ValueError(arm)
    z, y = logits.float(), target.float()
    p = z.sigmoid()
    dims = (1, 2, 3)
    bce = F.binary_cross_entropy_with_logits(z, y)
    tp = (p * y).sum(dims)
    fp = (p * (1 - y)).sum(dims)
    fn = ((1 - p) * y).sum(dims)
    dice = 1 - ((2 * tp + 1) / (2 * tp + fp + fn + 1)).mean()
    alpha, beta = (0.6, 0.4) if arm == 'B' else (0.4, 0.6)
    tv = 1 - ((tp + 0.5) / (tp + alpha * fp + beta * fn + 0.5)).mean()
    objective = bce + dice if arm == 'A' else 3 * bce + dice if arm == 'C' else bce + tv
    return objective, dict(bce=bce, dice=dice, common_bce_dice=bce + dice)


def cpu_budget():
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    p = Path('/sys/fs/cgroup/cpu.max')
    if p.exists():
        q, period = p.read_text().split()
        if q != 'max':
            count = min(count, max(1, math.floor(int(q) / int(period))))
    else:
        q, p = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'), Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if q.exists() and p.exists() and int(q.read_text()) > 0:
            count = min(count, max(1, int(q.read_text()) // int(p.read_text())))
    return max(1, count)


class GlobalBatches:
    # No DistributedSampler padding: every original crop is consumed exactly once.
    def __init__(self, size, batch, rank, world, seed, epoch, offset=0):
        if not 0 <= rank < world or batch < world or (size % batch and size % batch < world):
            raise ValueError('Configuration would leave a rank empty in a global batch')
        if not 0 <= offset <= size or (offset < size and offset % batch):
            raise ValueError('Invalid global cursor')
        self.size, self.batch, self.rank, self.world, self.seed, self.epoch, self.offset = (
            size,
            batch,
            rank,
            world,
            seed,
            epoch,
            offset,
        )

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        for start in range(self.offset, self.size, self.batch):
            yield [(self.epoch, i) for i in order[start : start + self.batch][self.rank :: self.world]]

    def __len__(self):
        return math.ceil((self.size - self.offset) / self.batch)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    )


def restore_rng(v):
    random.setstate(v['python'])
    np.random.set_state(v['numpy'])
    torch.set_rng_state(v['torch'])
    if v['cuda'] is not None:
        torch.cuda.set_rng_state(v['cuda'])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic(value, path, tensor=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    if tensor:
        torch.save(value, tmp)
    else:
        tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) == type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b
