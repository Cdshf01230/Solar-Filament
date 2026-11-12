import json
import math
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

MEAN = np.array([.485, .456, .406], np.float32)[:, None, None]
STD = np.array([.229, .224, .225], np.float32)[:, None, None]
CURSOR_PROTOCOL = 'sample-count-v2'


def cursor_from_update(epoch, update, size, batch):
    """Consumed samples in the current epoch, including its partial last batch."""
    if min(epoch, size, batch) < 1 or update < 0:
        raise ValueError('Invalid epoch/update/dataset/batch for resume cursor')
    steps = math.ceil(size/batch)
    within = update-(epoch-1)*steps
    if not 0 <= within <= steps:
        raise ValueError('Update and epoch disagree; cannot infer sample cursor safely')
    return min(within*batch, size)


def resume_cursor(checkpoint, size, batch):
    """Migrate only the known pre-v2 preprocessing dictionary-length bug.

    A legacy run already resumed with the buggy code needs a separate history
    audit: update alone cannot establish which samples it actually consumed.
    Never silently convert an ambiguous or already versioned bad cursor.
    """
    config = checkpoint['config']
    protocol = config.get('cursor_protocol')
    if protocol not in (None, CURSOR_PROTOCOL):
        raise ValueError('Unknown checkpoint cursor protocol')
    legacy_preprocess = protocol is None and config.get('preset') == 'preprocess12'
    if legacy_preprocess and config.get('resume'):
        raise ValueError('Legacy preprocessing checkpoint was already resumed; audit sample history before migration')
    expected = cursor_from_update(checkpoint['epoch'], checkpoint['update'], size, batch)
    stored = checkpoint['offset']
    if stored == expected:
        return expected, None
    within = checkpoint['update']-(checkpoint['epoch']-1)*math.ceil(size/batch)
    if legacy_preprocess and stored == within:
        return expected, dict(reason='legacy_preprocessing_mapping_keys_instead_of_samples',
            epoch=checkpoint['epoch'], update=checkpoint['update'],
            stored_offset=stored, corrected_sample_offset=expected,
            optimizer_scheduler_rng_weights='unchanged')
    raise ValueError('Checkpoint offset disagrees with consumed sample count')


class Crops(Dataset):
    """Stateless D4: same (seed, epoch, index) survives worker changes/resume."""
    def __init__(self, root, manifest, seed=42, augment=True):
        self.root, self.seed, self.augment = Path(root), seed, augment
        with Path(manifest).open() as stream:
            self.records = [json.loads(line) for line in stream]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        r = self.records[index]
        image = cv2.imread(str(self.root/r['image']), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.root/r['semantic_mask']), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise FileNotFoundError(r['crop_id'])
        if image.shape != (768, 768) or mask.shape != image.shape or mask.max() > 1:
            raise ValueError(f"Invalid image or 0/1 semantic mask: {r['crop_id']}")
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))
            k, flip = int(rng.integers(4)), bool(rng.integers(2))
            image, mask = np.rot90(image, k), np.rot90(mask, k)
            if flip:
                image, mask = image[:, ::-1], mask[:, ::-1]
        rgb = np.repeat(image[None], 3, axis=0).astype(np.float32)/255.
        return (torch.from_numpy(np.ascontiguousarray((rgb-MEAN)/STD)),
                torch.from_numpy(np.ascontiguousarray(mask[None])).float(), r['crop_id'])


class EpochSampler(Sampler):
    def __init__(self, size, seed):
        self.size, self.seed, self.epoch, self.offset = size, seed, 1, 0

    def set_cursor(self, epoch, offset=0):
        self.epoch, self.offset = epoch, offset

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed+self.epoch)
        order = torch.randperm(self.size, generator=g).tolist()
        return iter((self.epoch, i) for i in order[self.offset:])

    def __len__(self):
        return max(0, self.size-self.offset)


def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def load_split(base, protocol):
    base = Path(base)
    root = (base/'semantic_fpn_30_70' if protocol == '30_70'
            else base/'semantic_ablations_70_30/split')
    split = json.loads((root/'source_split.json').read_text())
    train, val, quick = set(split['train']), set(split['val']), set(split['quick_val'])
    audit = val-quick
    if train & val or not quick <= val or not audit:
        raise ValueError('Invalid source-image split')
    if {n[:8] for n in train} & {n[:8] for n in val}:
        raise ValueError('Train/val capture-day leakage')
    # Preserve the historical 30/70 split exactly, while recording day overlap if any.
    overlap = sorted({n[:8] for n in quick} & {n[:8] for n in audit})
    return root/'train.jsonl', dict(train=sorted(train), quick=sorted(quick), audit=sorted(audit),
                                    quick_audit_shared_days=overlap, original=split)
