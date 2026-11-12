"""Produce the competition submission from a folder of 2048 x 2048 H-alpha test images.

    python scripts/predict.py --images /path/to/test_images --weights weights --work work --out submission.csv

Runs the three stages with an on-disk cache in --work (re-running resumes; a --work made from other images,
weights or settings is refused). Needs one CUDA GPU (about 16 GB is enough); a T4 takes roughly 25 min for
upstream + 70 min per fold refiner with TTA x8 on the 180 test images.
"""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_pq import pipeline, submission  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', required=True)
    ap.add_argument('--weights', default='weights')
    ap.add_argument('--work', default='work')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    images = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in ('.jpeg', '.jpg', '.png'))
    if not images:
        ap.error(f'no .jpeg/.jpg/.png images in {a.images}')
    # Reuse an existing work directory only if it was made from the same images, weights and settings.
    try:
        cached = pipeline.check_cache(a.work, pipeline.cache_key(images, a.weights))
    except RuntimeError as e:
        ap.error(str(e))
    if not cached:
        pipeline.upstream(images, a.weights, a.work, device=a.device)
    pipeline.refine_all(a.weights, a.work, device=a.device)
    rows, _ = pipeline.assemble(a.work, a.out)
    info = submission.validate(rows, [p.name for p in images])
    print('SUBMISSION', json.dumps(info))


if __name__ == '__main__':
    main()
