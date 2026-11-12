"""Competition submission format: one CSV row per predicted filament.

filament_id      = <image stem>_<k>, k = 1, 2, ... per image
segmentation_rle = COCO RLE counts string (pycocotools, Fortran order) of the 2048 x 2048 binary mask
"""

import csv
from pathlib import Path
import numpy as np
from pycocotools import mask as mu

from .config import IMAGE_SIZE


def rle_counts(full_mask):
    return mu.encode(np.asfortranarray(full_mask.astype(np.uint8)))['counts'].decode('utf-8')


def decode(counts):
    return mu.decode(dict(size=[IMAGE_SIZE, IMAGE_SIZE], counts=counts.encode()))


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filament_id', 'segmentation_rle'])
        w.writerows(rows)


def validate(rows, image_names):
    """Ids unique, every id prefix is a test image stem, every RLE decodes to a non-empty 2048 x 2048 mask."""
    stems = {Path(n).stem for n in image_names}
    ids = [r[0] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate filament_id')
    unknown = [i for i in ids if i.rsplit('_', 1)[0] not in stems]
    if unknown:
        raise ValueError(f'filament_id prefix is not a test image: {unknown[:3]}')
    for i, counts in rows:
        m = decode(counts)
        if m.shape != (IMAGE_SIZE, IMAGE_SIZE) or not m.any():
            raise ValueError(f'{i}: mask is empty or not {IMAGE_SIZE}x{IMAGE_SIZE}')
    return dict(rows=len(rows), images=len(stems), images_with_prediction=len({i.rsplit('_', 1)[0] for i in ids}))
