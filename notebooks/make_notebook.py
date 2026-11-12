"""Builds pipeline.ipynb (kept as a script so the notebook source is reviewable). python notebooks/make_notebook.py"""

import json
from pathlib import Path

cells = []
md = lambda s: cells.append(dict(cell_type='markdown', metadata={}, source=s.strip('\n').splitlines(True)))
code = lambda s: cells.append(
    dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.strip('\n').splitlines(True))
)

md(r"""
# Solar filament instance segmentation - end-to-end pipeline

Phạm Đức Công, Lê Anh Đức (FPT University, Hanoi). Solar Filament Segmentation Challenge 2026.

This notebook (1) loads all released weights, (2) walks through every stage on one test image with pictures,
(3) reproduces `submission.csv` for the full test set, and (4) evaluates out-of-fold on labelled training images with
the organisers' PQ rule and a strict one-to-one rule. The notebook uses balanced filtering; recall48 candidate export is available separately through `scripts/proposals.py`. No training is needed. A CUDA GPU (T4/L4, 16 GB) is assumed.

| Stage | What | Module |
|---|---|---|
| 1 | proposal map: frozen DINOv3 ViT-S+/16 + small head, mean of 5 fold heads | `filament_pq/proposal.py` |
| 2 | components at p >= .05 + LightGBM keep/drop filter, one crop per kept component | `components.py`, `crops.py` |
| 3 | prompted recurrent refiner (DINOv3 ViT-B/16, 3 rounds, presence), 5 folds x D4 TTA | `refiner/` |
| Post-processing | presence gate, bilinear native resampling, anchor component, greedy de-duplication | `postprocess.py` |
""")
code(r"""
import sys, json, time
from pathlib import Path
REPO = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(REPO))

# ---- paths: edit these two -------------------------------------------------------------------------------------
TEST_IMAGES = Path('/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images')   # 180 test .jpeg
TRAIN_ROOT  = Path('/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train')              # optional, section 4
# -----------------------------------------------------------------------------------------------------------------
WEIGHTS = REPO/'weights'; WORK = REPO/'work'
import numpy as np, torch, matplotlib.pyplot as plt
from filament_pq import pipeline, proposal, components, crops, postprocess, submission, metrics
from filament_pq.config import PipelineConfig
cfg = PipelineConfig(); device = 'cuda'
print(torch.__version__, torch.cuda.get_device_name(0))
print(pipeline.WEIGHT_LAYOUT)
""")
md(r"""
## 1. One test image, stage by stage
`Predictor` holds every model in memory (2 backbones, 5 heads, 5 classifiers, 5 refiners, about 3 GB of GPU memory).
""")
code(r"""
pred = pipeline.Predictor(WEIGHTS, cfg, device)
names = sorted(p.name for p in TEST_IMAGES.glob('*.jpeg'))
name = names[0]
image = pipeline.read_image(TEST_IMAGES/name)
t0 = time.time(); out = pred(image, name); print(f'{name}: {len(out["crops"])} crops, {len(out["instances"])} filaments, {time.time()-t0:.0f}s')
""")
code(r"""
def overlay(ax, img, instances, title):
    ax.imshow(img, cmap='gray'); rng = np.random.default_rng(0)
    lab = np.zeros(img.shape, np.int32)
    for j, x in enumerate(instances, 1): lab[postprocess.to_full(x)] = j
    colors = np.concatenate([[[0, 0, 0, 0]], np.c_[rng.uniform(.2, 1, (max(1, len(instances)), 3)), np.full(max(1, len(instances)), .55)]])
    ax.imshow(colors[lab]); ax.set_title(title); ax.axis('off')

n, lab, stats, centers, ids, boxes = out['components']
kept = {int(c.crop_id.rsplit('_', 1)[1]) for c in out['crops']}
fig, axs = plt.subplots(1, 4, figsize=(24, 6.5))
axs[0].imshow(image, cmap='gray'); axs[0].set_title('GONG H-alpha'); axs[0].axis('off')
axs[1].imshow(out['probability'], cmap='magma', vmin=0, vmax=1); axs[1].set_title('1. proposal map (mean of 5 heads)'); axs[1].axis('off')
k = np.isin(lab, list(kept)); d = (lab > 0) & ~k
rgb = np.dstack([image/255.]*3); rgb[d] = [.2, .4, 1]; rgb[k] = [1, .8, 0]
axs[2].imshow(rgb); axs[2].set_title(f'2. components p>=.05: kept {len(kept)} (yellow) / dropped {len(ids)-len(kept)} (blue)'); axs[2].axis('off')
overlay(axs[3], image, out['instances'], f'4. final instances ({len(out["instances"])})')
plt.tight_layout(); plt.show()
""")
md(r"""
### Inside the refiner: a few crops
Anchor (prompt, cyan contour), refined probability (round 3, mean of 5 folds x 8 transforms) and the presence score.
""")
code(r"""
from filament_pq.refiner.infer import prep
import cv2
show = sorted(range(len(out['crops'])), key=lambda i: -out['crops'][i].component.sum())[:4]
fig, axs = plt.subplots(2, len(show), figsize=(5*len(show), 10))
for col, i in enumerate(show):
    c, r = out['crops'][i], out['refined'][i]
    a, _, (nw, nh) = prep(c)
    axs[0, col].imshow(a['image'][:nh, :nw], cmap='gray')
    cs, _ = cv2.findContours(a['anchor'][:nh, :nw].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for q in cs: axs[0, col].plot(q[:, 0, 0], q[:, 0, 1], c='cyan', lw=1)
    axs[0, col].set_title(f'{c.crop_id}  ML {c.ml_score:.2f}'); axs[0, col].axis('off')
    axs[1, col].imshow(r['r3'][:nh, :nw]/255., cmap='viridis', vmin=0, vmax=1)
    axs[1, col].set_title(f'refined p (round 3), presence {r["presence"][1]:.2f}'); axs[1, col].axis('off')
plt.tight_layout(); plt.show()
""")
md(r"""
## 2. Full test set -> `submission.csv`
Stage-wise with an on-disk cache (`work/`), in the same global crop order and batch size as the submitted run, so the
output is reproducible. On an L4: about 15 min (stages 1-2) + 5 x 13.5 min (stage 3), 84 min in total.

`work/` stores a fingerprint of the image names, settings and model files; a directory made from other inputs is
refused, so use a new one for a new run. On an L4 the submitted CSV was reproduced byte for byte (9 and 12 Nov 2026);
other GPU types can flip a few pixels at threshold boundaries.
""")
code(r"""
del pred; torch.cuda.empty_cache()
images = sorted(TEST_IMAGES.glob('*.jpeg'))
if not pipeline.check_cache(WORK, pipeline.cache_key(images, WEIGHTS, cfg)): pipeline.upstream(images, WEIGHTS, WORK, cfg, device)
pipeline.refine_all(WEIGHTS, WORK, cfg, device)
rows, instances = pipeline.assemble(WORK, REPO/'submission.csv', cfg)
print(submission.validate(rows, [p.name for p in images]))
""")
code(r"""
import pandas as pd
sub = pd.read_csv(REPO/'submission.csv'); print(sub.shape); sub.head()
""")
md(r"""
## 3. Morphology check on several test images
""")
code(r"""
fig, axs = plt.subplots(2, 3, figsize=(21, 14))
for ax, nm in zip(axs.ravel(), [names[i] for i in np.linspace(0, len(names)-1, 6).astype(int)]):
    overlay(ax, pipeline.read_image(TEST_IMAGES/nm), instances[nm], f'{nm}: {len(instances[nm])} filaments')
plt.tight_layout(); plt.show()
""")
md(r"""
## 4. Out-of-fold evaluation on labelled images (optional)
For a training image of fold f, stages 1-2 use the 4 heads/classifiers of the other folds and stage 3 uses the refiner
trained without fold f - exactly the cross-validation protocol of docs/RESULTS.md. Scored per annotation record with the
organisers' all-pairs rule and with a strict one-to-one matching.

These are development OOF estimates, not fully nested end-to-end CV: upstream ensemble members are reused in downstream
training inputs. Report both scorers separately, including TP/FP/FN and upstream GT misses.

The cell below uses a stricter notion of "touch" (overlap of at least 8 px and 5% of the GT) for one-to-many and
many-to-one. The statistics of the organisers' self-evaluation notebook (every overlap, IoU > 0), the bootstrap
interval and the per-fold scores of all 707 images are computed by `scripts/evaluate_oof.py` from
`reference/oof_predictions.csv` (README §4).
""")
code(r"""
N_EVAL = 10                                   # images per fold to evaluate (707 in total = full OOF, ~1.7 h on an L4)
ann = json.loads(next(TRAIN_ROOT.glob('*.json')).read_text()) if TRAIN_ROOT.exists() else None
folds = json.loads((WEIGHTS/'fold_assignment.json').read_text())['folds']
from pycocotools import mask as mu
def gt_rles(rec):
    out = []
    for a in [a for a in ann['annotations'] if a['image_id'] == rec['id']]:
        seg = a['segmentation']; seg = json.loads(seg) if isinstance(seg, str) else seg
        r = mu.merge(mu.frPyObjects(seg, rec['height'], rec['width'])) if isinstance(seg, list) else (mu.frPyObjects(seg, rec['height'], rec['width']) if isinstance(seg['counts'], list) else dict(seg, counts=seg['counts'].encode() if isinstance(seg['counts'], str) else seg['counts']))
        if mu.area(r) > 0: out.append(r)
    return out
if ann is not None:
    host, strict, rel = np.zeros(4), np.zeros(4), dict(one_to_many=0, many_to_one=0, iou=[])
    for f in folds:
        work = WORK/f'oof_fold{f["fold"]}'; others = [k for k in range(5) if k != f['fold']]
        srcs = sorted(f['train_sources'])[:N_EVAL]
        paths = [next(TRAIN_ROOT.rglob(s)) for s in srcs]
        if not pipeline.check_cache(work, pipeline.cache_key(paths, WEIGHTS, cfg, others)): pipeline.upstream(paths, WEIGHTS, work, cfg, device, folds=others, log=None)
        pipeline.refine_all(WEIGHTS, work, cfg, device, folds=[f['fold']], log=None)
        _, inst = pipeline.assemble(work, work/'oof.csv', cfg, folds=[f['fold']])
        for s in srcs:
            preds = [mu.encode(np.asfortranarray(postprocess.to_full(x).astype(np.uint8))) for x in inst[s]]
            for rec in [r for r in ann['images'] if Path(r['file_name']).name == s]:
                g = gt_rles(rec); h, o = metrics.match(metrics.iou_matrix(g, preds))
                if g: host += h
                strict += o; r = metrics.relations(g, preds)
                rel['one_to_many'] += r['one_to_many']; rel['many_to_one'] += r['many_to_one']; rel['iou'] += r['iou']
    print('organisers rule', metrics.pq(host)); print('strict 1-1    ', metrics.pq(strict))
    print('GT split over >=2 predictions:', rel['one_to_many'], ' predictions covering >=2 GT:', rel['many_to_one'])
    plt.hist(rel['iou'], bins=50, range=(.5, 1)); plt.xlabel('IoU of matched pairs'); plt.show()
""")
nb = dict(
    cells=cells,
    metadata=dict(
        kernelspec=dict(name='python3', display_name='Python 3', language='python'), language_info=dict(name='python')
    ),
    nbformat=4,
    nbformat_minor=5,
)
Path(__file__).with_name('pipeline.ipynb').write_text(
    json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8'
)
print('written', len(cells), 'cells')
