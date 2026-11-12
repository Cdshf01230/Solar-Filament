"""Regression test: the release pipeline must reproduce the submitted predictions.

    python tests/test_regression.py --images /path/to/test_images [--work work]

Compares the produced submission with reference/submission.csv: byte-identical on the GPU type used for the
submission (NVIDIA L4); on other GPUs it reports the per-filament IoU between the two files instead.
"""

import argparse, csv, filecmp, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from filament_pq import pipeline, submission  # noqa: E402


def read(path):
    with open(path, encoding='utf-8') as f:
        return {r['filament_id']: r['segmentation_rle'] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--work', default=str(REPO / 'work'))
    a = ap.parse_args()
    out = Path(a.work) / 'submission_regression.csv'
    images = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in ('.jpeg', '.jpg', '.png'))
    if not pipeline.check_cache(a.work, pipeline.cache_key(images, REPO / 'weights')):
        pipeline.upstream(images, REPO / 'weights', a.work)
    pipeline.refine_all(REPO / 'weights', a.work)
    pipeline.assemble(a.work, out)
    ref = REPO / 'reference' / 'submission.csv'
    if filecmp.cmp(out, ref, shallow=False):
        print('IDENTICAL to', ref)
        return
    A, B = read(out), read(ref)
    print(f'not byte-identical: {len(A)} vs {len(B)} rows, {len(set(A) & set(B))} common ids')
    ious = []
    for k in sorted(set(A) & set(B)):
        x, y = submission.decode(A[k]).astype(bool), submission.decode(B[k]).astype(bool)
        ious.append((x & y).sum() / max(1, (x | y).sum()))
    if ious:
        print('IoU of common ids: min %.4f median %.4f' % (min(ious), float(np.median(ious))))


if __name__ == '__main__':
    main()
