"""Fixed grayscale input ablations. Labels are never transformed by morphology."""

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import MEAN, STD

PREPROCESS_HEADS = ('raw', 'erosion', 'dilation', 'opening', 'closing', 'bilateral')
PREPROCESS_PROTOCOL = dict(
    version=1,
    input='uint8 grayscale, full source before crop/tile; before normalization',
    morphology=dict(kernel='ellipse', size=3, iterations=1, border='BORDER_REPLICATE'),
    bilateral=dict(d=5, sigmaColor=25, sigmaSpace=3, border='BORDER_REPLICATE'),
    labels='unchanged historical per-annotator masks',
    augmentation='same stateless D4 after preprocessing and crop',
)
CACHE = 'u0_preprocess_cache_v1'


def transform(image, name):
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError('Expected uint8 grayscale')
    if name == 'raw':
        return image
    if name == 'bilateral':
        return cv2.bilateralFilter(image, 5, 25, 3, borderType=cv2.BORDER_REPLICATE)
    ops = {
        'erosion': cv2.MORPH_ERODE,
        'dilation': cv2.MORPH_DILATE,
        'opening': cv2.MORPH_OPEN,
        'closing': cv2.MORPH_CLOSE,
    }
    return cv2.morphologyEx(
        image,
        ops[name],
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
        borderType=cv2.BORDER_REPLICATE,
    )


class PreprocessedCrops(Dataset):
    def __init__(self, base, manifest, seed=42, augment=True, variants=PREPROCESS_HEADS):
        self.base, self.seed, self.augment = Path(base), seed, augment
        self.variants = tuple(variants)
        assert set(self.variants) <= set(PREPROCESS_HEADS)
        self.records = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
        report = json.loads((self.base / CACHE / 'verification.json').read_text())
        assert report['state'] == 'complete' and report['protocol'] == PREPROCESS_PROTOCOL

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        r = self.records[index]
        mask = cv2.imread(str(self.base / 'magfilo_crops_768' / r['semantic_mask']), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != (768, 768) or mask.max() > 1:
            raise ValueError(r['crop_id'])
        k, flip = 0, False
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))
            k, flip = int(rng.integers(4)), bool(rng.integers(2))

        def d4(a):
            a = np.rot90(a, k)
            if flip:
                a = a[:, ::-1]
            return np.ascontiguousarray(a)

        inputs = {}
        for name in self.variants:
            path = (
                self.base / 'magfilo_crops_768' / r['image']
                if name == 'raw'
                else self.base / CACHE / name / 'crops' / r['image']
            )
            image = cv2.imread(str(path), 0)
            if image is None or image.shape != (768, 768):
                raise FileNotFoundError(path)
            inputs[name] = torch.from_numpy(d4(image)[None])
        return inputs, torch.from_numpy(d4(mask)[None]).float(), r['crop_id']


def normalized_input(value, device):
    # Transfer compact uint8 once; repeat grayscale into RGB and normalize on GPU.
    value = value.to(device, non_blocking=True).float().expand(-1, 3, -1, -1) / 255.0
    mean = torch.as_tensor(MEAN, device=device)
    std = torch.as_tensor(STD, device=device)
    return ((value - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def infer_preprocessed(model, image, tile_batch=8):
    from .model import autocast

    assert set(model.heads) <= set(PREPROCESS_HEADS)
    if image.shape != (2048, 2048):
        raise ValueError(image.shape)
    device = next(model.heads.parameters()).device
    positions = [(y, x) for y in (0, 640, 1280) for x in (0, 640, 1280)]
    window = np.maximum(np.outer(np.hanning(768), np.hanning(768)), 0.05).astype(np.float32)
    denominator = np.zeros(image.shape, np.float32)
    for y, x in positions:
        denominator[y : y + 768, x : x + 768] += window
    outputs = {}
    for name, head in model.heads.items():
        processed = transform(image, name)
        numerator = np.zeros(image.shape, np.float32)
        for start in range(0, len(positions), tile_batch):
            coords = positions[start : start + tile_batch]
            patches = np.stack([processed[y : y + 768, x : x + 768][None] for y, x in coords])
            batch = normalized_input(torch.from_numpy(patches), device)
            features = model.features(batch)
            with autocast(device):
                prob = head(features, batch).float().sigmoid()[:, 0].cpu().numpy()
            for pp, (y, x) in zip(prob, coords):
                numerator[y : y + 768, x : x + 768] += pp * window
            del features, batch
        outputs[name] = (numerator / denominator).astype(np.float16).astype(np.float32)
    return outputs
