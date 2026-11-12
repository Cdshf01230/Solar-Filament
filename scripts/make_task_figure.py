"""README figure: one training image (input) and its expert annotation as instance masks (expected output).

    python scripts/make_task_figure.py --data DIR [--image NAME] --out docs/figures/task_example.png
DIR = competition data folder containing train_images/ and MAGFiLO_1.0_Annotations_kaggle2026_train.json.
Without --image, the single-annotator image with the most filaments between 10 and 14 is used (a typical busy disk).
"""

import argparse, json
from collections import defaultdict
from pathlib import Path
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mu


def decode(ann, rec):
    seg = ann['segmentation']
    if isinstance(seg, str):
        seg = json.loads(seg)
    if isinstance(seg, list):
        rle = mu.merge(mu.frPyObjects(seg, rec['height'], rec['width']))
    else:
        rle = dict(seg)
        if isinstance(rle['counts'], list):
            rle = mu.frPyObjects(rle, rec['height'], rec['width'])
        elif isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode()
    return mu.decode(rle).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--image')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    D = Path(a.data)
    src = json.loads((D / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    recs, anns = defaultdict(list), defaultdict(list)
    for r in src['images']:
        recs[r['file_name']].append(r)
    for x in src['annotations']:
        anns[str(x['image_id'])].append(x)
    if a.image:
        name = a.image
    else:
        cand = sorted(
            (len(anns[str(rs[0]['id'])]), n)
            for n, rs in recs.items()
            if len(rs) == 1 and 10 <= len(anns[str(rs[0]['id'])]) <= 14
        )
        name = cand[len(cand) // 2][1]
    rec = recs[name][0]
    img = cv2.imread(str(D / 'train_images' / name), cv2.IMREAD_GRAYSCALE)
    masks = [m for m in (decode(x, rec) for x in anns[str(rec['id'])]) if m.any()]
    lab = np.zeros(img.shape, np.int32)
    for j, m in enumerate(masks, 1):
        lab[m] = j
    rng = np.random.default_rng(3)
    cols = np.concatenate([[[0, 0, 0, 0]], np.c_[rng.uniform(0.25, 1, (len(masks), 3)), np.full(len(masks), 0.75)]])
    big = max(masks, key=lambda m: m.sum())
    ys, xs = np.nonzero(big)
    cy, cx = int(ys.mean()), int(xs.mean())
    h = 360  # 720-px window on the largest filament
    y0, y1, x0, x1 = max(cy - h, 0), min(cy + h, img.shape[0]), max(cx - h, 0), min(cx + h, img.shape[1])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    ax[0].imshow(img, cmap='gray')
    ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='yellow', lw=1))
    ax[0].set_title('Input: H-alpha full disk, 2048 × 2048')
    ax[1].imshow(img, cmap='gray')
    ax[1].imshow(cols[lab])
    ax[1].set_title(f'Output: one mask per filament ({len(masks)} here)')
    ax[2].imshow(img[y0:y1, x0:x1], cmap='gray')
    ax[2].imshow(cols[lab[y0:y1, x0:x1]])
    for j, m in enumerate(masks, 1):
        cs, _ = cv2.findContours(m[y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            ax[2].plot(c[:, 0, 0], c[:, 0, 1], color=cols[j][:3], lw=0.8)
    ax[2].set_title('Zoom (720 px): mask outlines')
    for x in ax:
        x.axis('off')
    plt.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110, bbox_inches='tight')
    print('FIGURE', name, rec['id'], len(masks), 'filaments ->', a.out)


if __name__ == '__main__':
    main()
