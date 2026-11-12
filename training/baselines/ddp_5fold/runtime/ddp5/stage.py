"""Original DDP crop generator, restricted to the four training folds."""
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
from pycocotools import mask as mu
from solar_loss.common import atomic, cpu_budget
from ddp5.protocol import partition, read_assignment, VAL_COUNTS, VERSION


def stage(source, dest, holdout):
    source, dest = Path(source), Path(dest)
    assignment = read_assignment(Path(__file__).with_name('fold_assignment.json'))
    train, val = partition(assignment, holdout)
    dest.mkdir(parents=True, exist_ok=True)
    files = ['data/train_images/' + s for s in sorted(train + val)] + ['data/MAGFiLO_1.0_Annotations_kaggle2026_train.json']
    assert all((source / f).is_file() for f in files), 'input lacks canonical707 images/annotations'
    def copy(name):
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, p)
    with ThreadPoolExecutor(max_workers=min(8, cpu_budget() * 2)) as pool:
        list(pool.map(copy, files))
    annotations = json.loads((dest / files[-1]).read_text())
    records = {r['id']: Path(r['file_name']).name for r in annotations['images']}
    assert len(records) == 1154
    got_val = (len(val), sum(s in val for s in records.values()), sum(records[a['image_id']] in val for a in annotations['annotations']))
    assert got_val == VAL_COUNTS[holdout], (got_val, VAL_COUNTS[holdout])
    sources_file = dest / 'train_sources.json'
    atomic(train, sources_file)
    cropbase = dest / 'magfilo_crops_768'
    env = dict(os.environ, MAGFILO_INPUT_ROOT=str(source), MAGFILO_OUTPUT_DIR=str(cropbase),
               MAGFILO_SOURCES=str(sources_file), MAGFILO_WORKERS=str(min(4, cpu_budget())), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    generator = Path(__file__).resolve().parents[1] / 'solar_loss/magfilo_crops.py'
    subprocess.run([sys.executable, '-u', str(generator)], env=env, check=True)
    manifest = [json.loads(s) for s in (cropbase / 'manifest.jsonl').read_text().splitlines() if s.strip()]
    (cropbase / 'masks').mkdir(exist_ok=True)
    rows, kinds = [], {}
    for m in manifest:
        mask = np.zeros((768, 768), np.uint8)
        for inst in m['instances']:
            r = dict(inst['segmentation']); r['counts'] = r['counts'].encode('ascii')
            mask |= mu.decode(r).astype(np.uint8)
        assert mask.max() <= 1 and ((not mask.any()) == (m['sample_type'] == 'negative'))
        rel = 'masks/' + Path(m['image']).name
        assert cv2.imwrite(str(cropbase / rel), mask)
        rows.append(dict(crop_id=m['crop_id'], image=m['image'], semantic_mask=rel, source_file=m['source_file'],
                         sample_type=m['sample_type'], crop_xywh=m['crop_xywh']))
        kinds[m['sample_type']] = kinds.get(m['sample_type'], 0) + 1
    rows.sort(key=lambda r: r['crop_id'])
    assert {r['source_file'] for r in rows} == set(train)
    assert len({r['crop_id'] for r in rows}) == len(rows)
    (dest / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    atomic(assignment, dest / 'fold_assignment.json')
    report = dict(passed=True, protocol_version=VERSION, fold=holdout, sources=707, train_sources=len(train),
                  val_sources=len(val), val_records=got_val[1], val_gt=got_val[2], crops=len(rows), kinds=kinds,
                  generator='original DDP12: anchor + random positive + strict negative, native768')
    atomic(report, dest / 'staging_verification.json')
    print('CV_STAGE_VERIFIED', json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', required=True); p.add_argument('--dest', required=True)
    p.add_argument('--fold', required=True, type=int, choices=range(5)); a = p.parse_args()
    stage(a.source, a.dest, a.fold)
