r"""Five-fold DDP baseline: test ensemble or owner-only out-of-fold scoring (README section 2.6).

python training/baselines/ddp_5fold/inference.py --images <test_images> --epoch 20 \
    --backbone weights/ddp_baseline/dinov3_convnext_base --out work_ddp/submission.csv \
    --checkpoints 0=weights/ddp_baseline/fold0_epoch20.pt 1=weights/ddp_baseline/fold1_epoch20.pt \
                  2=weights/ddp_baseline/fold2_epoch20.pt 3=weights/ddp_baseline/fold3_epoch20.pt \
                  4=weights/ddp_baseline/fold4_epoch20.pt
For out-of-fold scoring on the 707 labelled images add --oof --data-root <folder containing data/>; every image is
then segmented only by the model of its own held-out fold. The five-model ensemble refuses labelled images.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from pycocotools import mask as mu

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'runtime'))
from ddpvit.model import make_model
from ddpvit.scoring import components, StrictContext, summarize, write_rows
from u0_multihead.ddp_model import DDPHead
from u0_multihead.evaluation import infer_shared
from ddp5.protocol import VERSION


def mean_maps(maps):
    assert len(maps) == 5
    out = np.zeros_like(maps[0], dtype=np.float32)
    for p in maps:
        assert p.shape == out.shape and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
        out += p.astype(np.float32)
    return out / 5


def load_head(ck, fold):
    assert ck['config']['fold'] == fold and ck['config']['protocol_version'] == VERSION
    assert ck['config']['arm'] == 'convnext'
    h = DDPHead()
    prefix = 'heads.ddp.'
    assert ck['trainable'] and all(s.startswith(prefix) for s in ck['trainable'])
    h.load_state_dict({s[len(prefix) :]: v for s, v in ck['trainable'].items()}, strict=True)
    assert all(torch.isfinite(v).all() for v in h.state_dict().values())
    return h


def instance_rows(source, probability):
    assert probability.shape == (2048, 2048) and np.isfinite(probability).all()
    lab, keep, _ = components(probability)
    for j, cid in enumerate(keep):
        mask = np.asfortranarray((lab == cid).astype(np.uint8))
        rle = mu.encode(mask)
        assert int(mu.area(rle)) >= 128
        yield [f'{Path(source).stem}_{j+1}', rle['counts'].decode('utf-8')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True)
    p.add_argument('--backbone', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--checkpoints', nargs=5, required=True)
    p.add_argument('--epoch', type=int, help='Require this epoch in every checkpoint')
    p.add_argument('--resume', action='store_true', help='Save and resume atomic test CSV snapshots every 10 images')
    p.add_argument('--oof', action='store_true')
    p.add_argument('--data-root')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.resume and a.oof:
        raise ValueError('CSV resume is supported for test ensemble only')
    weights = dict(x.split('=', 1) for x in a.checkpoints)
    assert set(weights) == set(map(str, range(5)))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    names = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in ('.jpeg', '.jpg', '.png'))
    assert names and len({p.stem for p in names}) == len(names)
    assignment = json.loads((HERE / 'runtime/ddp5/fold_assignment.json').read_text())
    owners = {s: f['fold'] for f in assignment['folds'] for s in f['train_sources']}
    if a.oof:
        assert a.data_root and {p.name for p in names} == set(owners), 'OOF requires all 707 canonical sources'
    else:
        assert not {p.stem for p in names} & {Path(s).stem for s in owners}, 'Use --oof for labeled training sources'
    context = StrictContext(Path(a.data_root)) if a.oof else None
    model = make_model(a.backbone, 'convnext', seed=42)
    heads, provenance = {}, {}
    for k in range(5):
        ck = torch.load(weights[str(k)], map_location='cpu', weights_only=False)
        if a.epoch is not None and ck.get('epoch') != a.epoch:
            raise ValueError(f'Fold {k}: checkpoint epoch does not match requested epoch {a.epoch}')
        h = load_head(ck, k)
        heads[f'f{k}'] = h.to(a.device).eval()
        provenance[k] = dict(path=weights[str(k)], epoch=ck.get('epoch', ck.get('epochs')), update=ck['update'])
        del ck
    model.heads = nn.ModuleDict(heads)
    model.to(a.device).eval()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count, ids, scored = 0, set(), []
    start = 0
    snapshot = out.with_suffix('.resume.csv')
    progress = out.with_suffix('.resume.json')
    updates = [provenance[k]['update'] for k in range(5)]
    if a.resume and progress.exists():
        saved = json.loads(progress.read_text())
        start = saved['completed_images']
        assert 0 <= start <= len(names)
        assert saved['processed_names'] == [p.name for p in names[:start]]
        assert saved['checkpoint_updates'] == updates and saved['epoch'] == a.epoch
        assert saved['csv_bytes'] == snapshot.stat().st_size
        with snapshot.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        ids = {r['filament_id'] for r in rows}
        count = len(rows)
        assert len(ids) == count == saved['rows']
        shutil.copyfile(snapshot, out)
        print('INFERENCE_RESUMED', start, len(names), count, flush=True)
    with out.open('a' if start else 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not start:
            writer.writerow(['filament_id', 'segmentation_rle'])
        with torch.inference_mode():
            for i, path in enumerate(names):
                if i < start:
                    continue
                model.heads = (
                    nn.ModuleDict({f'f{owners[path.name]}': heads[f'f{owners[path.name]}']})
                    if a.oof
                    else nn.ModuleDict(heads)
                )
                raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    raise FileNotFoundError(path)
                maps = infer_shared(model, raw, tile_batch=1)
                prob = next(iter(maps.values())) if a.oof else mean_maps([maps[f'f{k}'] for k in range(5)])
                for name, rle in instance_rows(path.name, prob):
                    assert name not in ids
                    ids.add(name)
                    writer.writerow([name, rle])
                    count += 1
                if context:
                    scored.extend(context.score(path.name, prob))
                if a.resume and ((i + 1) % 10 == 0 or i == len(names) - 1):
                    f.flush()
                    temp_csv = snapshot.with_suffix('.csv.tmp')
                    shutil.copyfile(out, temp_csv)
                    temp_csv.replace(snapshot)
                    saved = dict(
                        completed_images=i + 1,
                        processed_names=[p.name for p in names[: i + 1]],
                        rows=count,
                        csv_bytes=snapshot.stat().st_size,
                        epoch=a.epoch,
                        checkpoint_updates=updates,
                    )
                    temp_json = progress.with_suffix('.json.tmp')
                    temp_json.write_text(json.dumps(saved))
                    temp_json.replace(progress)
                    print('INFERENCE_SNAPSHOT', i + 1, len(names), count, flush=True)
                if i % 10 == 0 or i == len(names) - 1:
                    print('INFERENCE', i + 1, len(names), count, flush=True)
    report = dict(
        mode='owner-only OOF' if a.oof else 'five-model probability ensemble',
        images=len(names),
        rows=count,
        threshold=0.5,
        min_area=128,
        tta=1,
        checkpoints=provenance,
        requested_epoch=a.epoch,
    )
    if context:
        report['metrics'] = summarize(scored)
        write_rows(scored, out.with_suffix('.rows.jsonl.gz'))
    out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
