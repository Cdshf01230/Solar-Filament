"""Stage the original DDP12 protocol: train209 crops from the ORIGINAL crop generator (anchor / random positive / strict
negative, solar_loss/magfilo_crops.py), converted to 0/1 semantic masks, plus all 707 full images and annotations for scoring.

    python -m solar_loss.stage209 --source <dir holding data/> --dest /tmp/ddp_vitb_data

Writes dest/{data/..., magfilo_crops_768/{images,masks}, train.jsonl, staging_verification.json}.
Semantic mask = union of the instances of THAT annotation record visible in the crop (the generator crops per record).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from .common import atomic, cpu_budget


def stage(source, dest):
    source, dest = Path(source), Path(dest)
    runtime = Path(__file__).resolve().parents[1]
    split = json.loads((runtime / 'split209.json').read_text())
    train = sorted(split['train209'])
    names = sorted(set(train) | set(split['val498']))
    assert len(train) == 209 and len(names) == 707
    dest.mkdir(parents=True, exist_ok=True)
    files = ['data/train_images/' + s for s in names] + ['data/MAGFiLO_1.0_Annotations_kaggle2026_train.json']
    assert all((source / f).is_file() for f in files)

    def copy(name):
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.tmp')
        shutil.copyfile(source / name, temp)
        os.replace(temp, target)

    with ThreadPoolExecutor(max_workers=min(16, cpu_budget() * 4)) as pool:
        list(pool.map(copy, files))
    from u0_multihead.evaluation import Context
    context = Context(dest)
    assert sorted(context.records) == names

    cropbase = dest / 'magfilo_crops_768'
    sources_file = dest / 'train209_sources.json'
    sources_file.write_text(json.dumps(train))
    env = dict(os.environ, MAGFILO_INPUT_ROOT=str(source), MAGFILO_OUTPUT_DIR=str(cropbase), MAGFILO_SOURCES=str(sources_file),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    code = subprocess.call([sys.executable, '-u', str(Path(__file__).with_name('magfilo_crops.py'))], env=env)
    assert code == 0, f'crop generator exited with {code}'

    (cropbase / 'masks').mkdir(exist_ok=True)
    manifest = [json.loads(l) for l in (cropbase / 'manifest.jsonl').read_text().splitlines() if l.strip()]
    rows, kinds = [], {}
    for m in manifest:
        mask = np.zeros((768, 768), np.uint8)
        for inst in m['instances']:
            rle = dict(inst['segmentation'])
            rle['counts'] = rle['counts'].encode('ascii')
            mask |= coco_mask.decode(rle).astype(np.uint8)
        assert mask.max() <= 1 and (m['sample_type'] != 'negative' or not mask.any()) and (m['sample_type'] == 'negative' or mask.any())
        rel = 'masks/' + Path(m['image']).name
        assert cv2.imwrite(str(cropbase / rel), mask)
        rows.append(dict(crop_id=m['crop_id'], image=m['image'], semantic_mask=rel, source_file=m['source_file'],
                         sample_type=m['sample_type'], crop_xywh=m['crop_xywh']))
        kinds[m['sample_type']] = kinds.get(m['sample_type'], 0) + 1
    rows.sort(key=lambda r: r['crop_id'])
    (dest / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    got = {r['source_file'] for r in rows}
    assert got == set(train), sorted(set(train) - got)[:5]
    atomic(dict(passed=True, train_sources=len(train), crops=len(rows), kinds=kinds, sources=707,
                generator='crop_gen/magfilo_crops.py (original DDP12 generator)'), dest / 'staging_verification.json')
    print('SPLIT209_STAGE_VERIFIED', len(train), len(rows), kinds, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--dest', required=True)
    a = p.parse_args()
    stage(a.source, a.dest)
