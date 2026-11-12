"""Use the fixed fold assignment and original annotation records; never re-split."""

import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--images', required=True)
p.add_argument('--annotations', required=True)
p.add_argument('--out', required=True)
p.add_argument('--fold', type=int, choices=range(5), required=True)
p.add_argument('--dry-run', action='store_true')
a = p.parse_args()
# The original staging routine expects this data layout. Use links instead of copying images twice.
base = Path(a.out).resolve()
source = base / 'source'
data = source / 'data'
data.mkdir(parents=True, exist_ok=True)
links = {
    'train_images': Path(a.images).resolve(),
    'MAGFiLO_1.0_Annotations_kaggle2026_train.json': Path(a.annotations).resolve(),
}
for name, target in links.items():
    dest = data / name
    if not a.dry_run and not dest.exists():
        dest.symlink_to(target, target_is_directory=target.is_dir())
cmd = [
    sys.executable,
    '-m',
    'solar_loss.stage',
    '--source',
    str(source),
    '--dest',
    str(base / f'fold{a.fold}'),
    '--fold',
    str(a.fold),
]
print(json.dumps(cmd))
if not a.dry_run:
    subprocess.run(cmd, cwd=ROOT / 'training/proposal', check=True)
