"""Why semantic segmentation + connected components is a sound formulation for this data.

For every annotation record of the 707 training images:
  1. union all its filament masks into one binary (semantic) mask - the perfect output of a semantic model,
  2. split it back into instances with 8-connected components (area >= 8 px, the rule of our pipeline),
  3. score these instances against the record's original masks (organisers' rule and strict one-to-one).
Also measures how far apart the filaments of one record are (gap = minimum pixel distance between two masks), how many
filaments touch a neighbour (and are merged by the union), and how many filament masks are themselves in pieces.

    python scripts/semantic_oracle.py --data DIR [--out docs/figures/semantic_oracle.json] [--workers 8]
DIR = competition data folder with MAGFiLO_1.0_Annotations_kaggle2026_train.json (images are not needed).
"""

import argparse, json, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_pq.metrics import iou_matrix, match, pq  # noqa: E402


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


def rle(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return r


def one(task):
    rec, anns = task
    gts = [m for m in (decode(a, rec) for a in anns) if m.any()]
    union = np.zeros((rec['height'], rec['width']), bool)
    for m in gts:
        union |= m
    n, lab, st, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8), 8, cv2.CV_32S)
    preds = [lab == k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 8]
    host, strict = match(iou_matrix([rle(m) for m in gts], [rle(m) for m in preds]))
    pieces = [int(cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1) for m in gts]
    gaps = []
    for i, m in enumerate(gts):  # gap to the nearest other filament of the same record
        if len(gts) < 2:
            break
        others = np.zeros_like(union)
        for j, o in enumerate(gts):
            if j != i:
                others |= o
        gaps.append(float(ndi.distance_transform_edt(~others)[m].min()))
    return host, strict, pieces, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    src = json.loads((Path(a.data) / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    anns = defaultdict(list)
    for x in src['annotations']:
        anns[str(x['image_id'])].append(x)
    tasks = [(r, anns[str(r['id'])]) for r in src['images']]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, tasks, chunksize=4))
    host = pq(np.sum([r[0] for r in res], 0))
    strict = pq(np.sum([r[1] for r in res], 0))
    pieces = np.array([p for r in res for p in r[2]])
    gaps = np.array([g for r in res for g in r[3]])
    out = dict(
        images=len({r['file_name'] for r in src['images']}),
        records=len(res),
        filaments=int(len(pieces)),
        pq_host=host,
        pq_strict=strict,
        filaments_in_one_piece=float((pieces == 1).mean()),
        gap_px_to_nearest_filament=dict(
            q05=float(np.quantile(gaps, 0.05)),
            q25=float(np.quantile(gaps, 0.25)),
            median=float(np.median(gaps)),
            filaments_with_neighbour=int(len(gaps)),
        ),
        touching_a_neighbour=int((gaps <= 1.5).sum()),
        within_10px=int((gaps <= 10).sum()),
    )
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
