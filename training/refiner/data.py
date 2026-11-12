"""Deterministic record-preserving crop loader and eight-operation prompt corruption.

Combo P+A+L(+B in model), 4-fold train / 1-fold val:
- holdout: train split = every owner_fold except `holdout`; val split = owner_fold == holdout.
- P: presence prior [logit(ML mean-OOF score), log anchor area fraction, mean probability in anchor].
- A: optional train-only image augmentation (independent stateless RNG): shrink-scale jitter .8-1.0 (p.5),
  gamma .8-1.25, contrast .85-1.15, brightness +-.1, Gaussian blur sigma .3-1 (p.5), Gaussian noise sigma .01-.03 (p.5).
- L: per-record skeleton tubes (skeletonize target, dilate r=2, keep inside target) for Skeleton Recall loss."""

from pathlib import Path
import csv
import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from torch.utils.data import Dataset


def disk(r):
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def skeleton_tube(target):
    """Tubed centerline of one binary target; falls back to the target if skeletonization is empty."""
    t = target.astype(bool)
    if not t.any():
        return t
    s = cv2.dilate(skeletonize(t).astype(np.uint8), disk(2)).astype(bool) & t
    return s if s.any() else t


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def d4_transform(array, code):
    """Eight exact square symmetries; all spatial fields use the same code."""
    assert 0 <= code < 8
    result = np.rot90(array, code % 4, axes=(-2, -1))
    if code >= 4:
        result = np.flip(result, axis=-1)
    return np.ascontiguousarray(result)


def d4_code(epoch, index):
    # Independent stream: never consume/change the original M0 sampler RNG.
    return int(np.random.default_rng(np.random.SeedSequence([42, epoch, index, 0xD4])).integers(8))


def corrupt(target, valid, neighbors, rng):
    used = []
    target = target.astype(bool)
    ys, xs = np.nonzero(target)
    near = np.argwhere((ndi.distance_transform_edt(~target) <= 64) & valid)
    for attempt in range(10):
        m = target.copy()
        for _ in range(int(rng.integers(1, 4))):
            op = int(rng.integers(8))
            used.append(op)
            if op == 0:
                m = cv2.dilate(m.astype(np.uint8), disk(int(rng.integers(2, 13)))).astype(bool)
            elif op == 1:
                eroded = cv2.erode(m.astype(np.uint8), disk(int(rng.integers(1, 5)))).astype(bool)
                if eroded.any():
                    m = eroded
            elif op == 2:
                for _ in range(int(rng.integers(1, 4))):
                    j = int(rng.integers(len(xs)))
                    cv2.circle(m.view(np.uint8), (int(xs[j]), int(ys[j])), int(rng.integers(6, 31)), 0, -1)
            elif op == 3:
                coords = np.column_stack([ys, xs]).astype(float)
                centered = coords - coords.mean(0)
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                yy, xx = np.indices(m.shape)
                projection = (yy - coords[:, 0].mean()) * vh[0, 0] + (xx - coords[:, 1].mean()) * vh[0, 1]
                lo, hi = (centered @ vh[0]).min(), (centered @ vh[0]).max()
                portion = float(rng.uniform(0.1, 0.5))
                m &= (
                    projection >= lo + portion * (hi - lo)
                    if rng.random() < 0.5
                    else projection <= hi - portion * (hi - lo)
                )
            elif op == 4:
                for _ in range(int(rng.integers(1, 4))):
                    y, x = near[int(rng.integers(len(near)))]
                    cv2.ellipse(
                        m.view(np.uint8),
                        (int(x), int(y)),
                        tuple(map(int, rng.integers(4, 31, 2))),
                        float(rng.uniform(0, 180)),
                        0,
                        360,
                        1,
                        -1,
                    )
            elif op == 5:
                candidates = [n for n in neighbors if not np.array_equal(n, target)]
                if candidates:
                    m |= candidates[int(rng.integers(len(candidates)))]
            elif op == 6:
                dx, dy = map(int, rng.integers(-12, 13, 2))
                m = cv2.warpAffine(
                    m.astype(np.uint8),
                    np.float32([[1, 0, dx], [0, 1, dy]]),
                    (m.shape[1], m.shape[0]),
                    flags=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                noise = ndi.gaussian_filter(rng.normal(size=m.shape).astype(np.float32), 4)
                noise /= max(float(noise.std()), 1e-6)
                m = ndi.gaussian_filter(m.astype(np.float32), 4) + 0.2 * noise > 0.5
            m &= valid
        overlap = iou(m, target)
        if 0.3 <= overlap <= 0.9:
            break
    return m.astype(np.float32), overlap, used


class Crops(Dataset):
    def __init__(
        self, root, holdout=1, split='train', side=512, epoch=1, sampler=True, ids=None, d4=False, photometric=False
    ):
        self.root, self.side, self.epoch, self.sampler = Path(root), side, epoch, sampler
        assert not d4 or (split == 'train' and sampler), 'D4 is train-only; evaluation/smoke stay native'
        self.d4 = d4
        assert not photometric or (split == 'train' and sampler), 'Photometric/scale augmentation is train-only'
        self.photometric = photometric
        with (self.root / 'step3_crops/manifest.csv').open(encoding='utf-8') as f:
            all_rows = list(csv.DictReader(f))
        self.rows = [r for r in all_rows if (int(r['owner_fold']) != holdout) == (split == 'train')]
        if ids is not None:
            self.rows = [r for r in self.rows if r['crop_id'] in ids]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.root / 'step3_crops/crops' / (row['crop_id'] + '.npz'), allow_pickle=False) as z:
            c = {k: z[k] for k in z.files}
        h, w = c['image'].shape
        s = self.side / max(h, w)
        arng = (
            np.random.default_rng(np.random.SeedSequence([42, self.epoch, index, 0xA6])) if self.photometric else None
        )
        if arng is not None and arng.random() < 0.5:
            s *= float(arng.uniform(0.8, 1.0))
        nw, nh = max(1, round(w * s)), max(1, round(h * s))

        def resize(a, mask=False):
            out = np.zeros((self.side, self.side), np.float32)
            out[:nh, :nw] = cv2.resize(
                a.astype(np.float32),
                (nw, nh),
                interpolation=cv2.INTER_NEAREST if mask else (cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR),
            )
            return out

        valid = np.zeros((self.side, self.side), bool)
        valid[:nh, :nw] = True
        anchor = resize(c['component'], True)
        instances = [resize(m, True).astype(bool) for m in c['instances']]
        targets = np.stack([instances[int(t)] if t >= 0 else np.zeros_like(valid) for t in c['record_target']])
        inside = ndi.distance_transform_edt(anchor > 0)
        outside = ndi.distance_transform_edt(anchor == 0)
        distance = np.clip(inside - outside, -64, 64).astype(np.float32) / 64
        distance[~valid] = -1
        current, kind, overlap, ops = anchor.copy(), 'proposal', None, []
        if self.sampler:
            rng = np.random.default_rng(42 + self.epoch * 100000 + index)
            draw = rng.random()
            positive = np.flatnonzero(targets.sum((1, 2)) > 0)
            if 0.4 <= draw < 0.7 and len(positive):
                target = targets[int(rng.choice(positive))]
                current, overlap, ops = corrupt(target, valid, instances, rng)
                kind = 'gt_noise'
            elif 0.7 <= draw < 0.9 and self.epoch >= 2:
                kind = 'rollout'
            elif draw >= 0.9:
                current = np.zeros_like(anchor)
                kind = 'empty'
        image = resize(c['image']) / 255
        if arng is not None:
            vv = valid
            image = np.clip(image, 0, 1) ** float(arng.uniform(0.8, 1.25))
            m = image[vv].mean() if vv.any() else 0.0
            image = (image - m) * float(arng.uniform(0.85, 1.15)) + m + float(arng.uniform(-0.1, 0.1))
            if arng.random() < 0.5:
                image = cv2.GaussianBlur(image.astype(np.float32), (0, 0), float(arng.uniform(0.3, 1.0)))
            if arng.random() < 0.5:
                image = image + arng.normal(0, float(arng.uniform(0.01, 0.03)), image.shape)
            image = np.clip(image, 0, 1).astype(np.float32)
            image[~vv] = 0
        arrays = dict(
            image=image,
            probability=resize(c['probability']),
            anchor=anchor,
            distance=distance,
            current=current,
            valid=valid.astype(np.float32),
        )
        transform = d4_code(self.epoch, index) if self.d4 else 0
        if self.d4:
            arrays = {k: d4_transform(v, transform) for k, v in arrays.items()}
            targets = d4_transform(targets, transform)
            # Recompute from the transformed anchor, with padding still excluded.
            a = arrays['anchor'] > 0
            arrays['distance'] = (
                np.clip(ndi.distance_transform_edt(a) - ndi.distance_transform_edt(~a), -64, 64).astype(np.float32) / 64
            )
            arrays['distance'][arrays['valid'] == 0] = -1
        out = {k: torch.from_numpy(v.copy())[None] for k, v in arrays.items()}
        a = arrays['anchor'] > 0
        v = arrays['valid'] > 0
        ml = float(np.clip(c['ml_mean_oof'], 1e-4, 1 - 1e-4))
        prior = np.array(
            [
                np.log(ml / (1 - ml)),
                np.log(max(a.sum(), 1) / max(v.sum(), 1)),
                float(arrays['probability'][a].mean()) if a.any() else 0.0,
            ],
            np.float32,
        )
        skeletons = np.stack([skeleton_tube(t) for t in targets]) if len(targets) else np.zeros_like(targets, bool)
        out.update(
            prior=torch.from_numpy(prior),
            targets=torch.from_numpy(targets.copy()),
            skeletons=torch.from_numpy(skeletons),
            crop_id=row['crop_id'],
            source=row['source'],
            box=c['crop_box'].tolist(),
            resized_wh=[nw, nh],
            ml_score=float(c['ml_mean_oof']),
            native_anchor=c['component'].copy(),
            sampler=kind,
            noise_iou=overlap,
            noise_ops=ops,
        )
        if self.d4:
            out['d4_code'] = transform
        out['photometric'] = self.photometric
        return out


def collate(samples):
    keys = ('image', 'probability', 'anchor', 'distance', 'current', 'valid', 'prior')
    return {
        **{k: torch.stack([s[k] for s in samples]) for k in keys},
        'targets': [s['targets'] for s in samples],
        'skeletons': [s['skeletons'] for s in samples],
        'meta': samples,
    }


def move(batch, device):
    return {
        k: ([t.to(device) for t in v] if k in ('targets', 'skeletons') else v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
