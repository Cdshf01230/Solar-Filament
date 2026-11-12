"""Stage 2b - one refiner crop per kept component: bounding box + 128 px context, clipped to the image.

A crop carries the raw image patch, the proposal probability patch (float16) and the component mask ("anchor"),
which is the prompt telling the refiner WHICH filament to segment.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .config import ComponentConfig, IMAGE_SIZE


@dataclass
class Crop:
    crop_id: str  # <image stem>_<component id:05d>
    source: str  # image file name
    image: np.ndarray  # uint8 (h, w)
    probability: np.ndarray  # float16 (h, w), fold-mean proposal map
    component: np.ndarray  # bool (h, w), the anchor prompt
    box: tuple  # (x0, y0, x1, y1) in the 2048 image
    ml_score: float  # mean LightGBM keep-probability


def make_crops(source, image, prob, comps, scores, cfg=ComponentConfig()):
    n, lab, stats, centers, ids, boxes = comps
    out = []
    for j, cid in enumerate(ids):
        if scores[j] < cfg.ml_threshold:
            continue
        x0, y0, x1, y1 = map(int, boxes[j])
        p = cfg.crop_pad
        X0, Y0, X1, Y1 = max(0, x0 - p), max(0, y0 - p), min(IMAGE_SIZE, x1 + p), min(IMAGE_SIZE, y1 + p)
        out.append(
            Crop(
                crop_id=f'{Path(source).stem}_{int(cid):05d}',
                source=source,
                image=image[Y0:Y1, X0:X1],
                probability=prob[Y0:Y1, X0:X1].astype(np.float16),
                component=lab[Y0:Y1, X0:X1] == cid,
                box=(X0, Y0, X1, Y1),
                ml_score=float(np.float32(scores[j])),
            )
        )
    return out
