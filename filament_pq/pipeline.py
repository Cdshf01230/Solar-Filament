"""End-to-end inference, stage by stage, with an on-disk cache so that a long run can be resumed.

    stage 1+2  upstream()  images -> fold-mean proposal maps -> components -> LightGBM -> crops      work/crops/*.npz
    stage 3    refine_all()  every crop through each fold refiner (D4 TTA x8)                         work/refined/fold{k}/*.npz
    post       assemble()  mean over fold refiners -> post-processing -> submission.csv

Every cache file is written to a temporary name and renamed, so an interrupted run never leaves a truncated file
that a resumed run would trust. upstream() also stores `cache.json` (image names, configuration, folds, policy and a
fingerprint of the model files); `check_cache()` refuses to reuse a work directory made with different inputs.

Crops are refined in one global, deterministic order (images sorted by name, components by label id) with batches of
16, which is exactly how the submitted predictions were produced. `Predictor` offers the same computation in memory
for a single image (used by the notebook for visualisation).

Out-of-fold mode (`folds` = the 4 folds that did not see an image, `refiner_fold` = its owner fold) reproduces the
cross-validation protocol on labelled training images.
"""

from dataclasses import asdict
from pathlib import Path
import csv, hashlib, io, json, os, time
import cv2
import numpy as np
import torch

from . import __version__
from .config import PipelineConfig, IMAGE_SIZE
from . import proposal as P, components as C, crops as K, postprocess as PP, submission as S
from .refiner import model as RM, infer as RI

WEIGHT_LAYOUT = """weights/
  proposal/dinov3_vits16plus/      frozen DINOv3 ViT-S+/16 (Hugging Face format)
  proposal/heads/fold{k}.pt        proposal head of fold model k
  ml/balance_fold{k}.joblib        LightGBM component classifier of fold k
  refiner/dinov3_vitb16/           DINOv3 ViT-B/16 (Hugging Face format)
  refiner/fold{k}.pt               refiner decoder + fine-tuned ViT blocks 9-12 of fold k
  fold_assignment.json             date-disjoint 5-fold split of the 707 training images"""


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f'{path}: expected a {IMAGE_SIZE}x{IMAGE_SIZE} grayscale image')
    return img


def _replace_atomically(path, write):
    """Run write(binary file) on a temporary file next to `path`, then rename it to `path`."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        write(f)
    os.replace(tmp, path)


def _savez(path, **arrays):
    _replace_atomically(path, lambda f: np.savez_compressed(f, **arrays))


def _write_text(path, text):
    _replace_atomically(path, lambda f: f.write(text.encode('utf-8')))


MODEL_DIRS = ('proposal', 'ml', 'refiner')


def _file_digest(path, chunk=4 << 20):
    """SHA-256 of a small file; of its size plus first and last 4 MB for a large one (cheap but content-based)."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with open(path, 'rb') as f:
        if size <= 4 * chunk:
            h.update(f.read())
        else:
            h.update(f.read(chunk))
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.hexdigest()


def cache_key(image_paths, weights, cfg=PipelineConfig(), folds=None, classifier_policy='balance'):
    """Everything a work directory depends on; compared by check_cache() before a cache is reused."""
    w = Path(weights)
    files = sorted(p for d in MODEL_DIRS for p in (w / d).rglob('*') if p.is_file())
    return dict(
        version=__version__,
        images=sorted(Path(p).name for p in image_paths),
        config=asdict(cfg),
        folds=sorted(cfg.folds if folds is None else folds),
        classifier_policy=classifier_policy,
        weights={p.relative_to(w).as_posix(): _file_digest(p) for p in files},
    )


def check_cache(work, key):
    """False if `work` has no finished upstream cache; True if it has one made with the same key.
    Raises if it was made with different images, weights or settings, or by a version without cache.json."""
    work = Path(work)
    if not (work / 'manifest.csv').exists():
        return False
    stored = work / 'cache.json'
    if not stored.exists() or json.loads(stored.read_text(encoding='utf-8')) != json.loads(json.dumps(key)):
        raise RuntimeError(
            f'{work} holds a cache made with different inputs (or without cache.json); ' 'use a new --work directory'
        )
    return True


def load_refiner(weights, k, device):
    fw = torch.load(Path(weights) / 'refiner' / f'fold{k}.pt', map_location='cpu', weights_only=True)
    return RM.build(Path(weights) / 'refiner' / 'dinov3_vitb16', fw, device)


def _save_crop(path, c):
    _savez(
        path,
        image=c.image,
        probability=c.probability,
        component=c.component,
        crop_box=np.array(c.box, np.int32),
        source=np.array(c.source),
        ml=np.array(c.ml_score, np.float32),
    )


def _load_crop(path):
    with np.load(path, allow_pickle=False) as z:
        return K.Crop(
            crop_id=Path(path).stem,
            source=str(z['source']),
            image=z['image'],
            probability=z['probability'],
            component=z['component'],
            box=tuple(int(v) for v in z['crop_box']),
            ml_score=float(z['ml']),
        )


def upstream(
    image_paths, weights, work, cfg=PipelineConfig(), device='cuda', folds=None, log=print, classifier_policy='balance'
):
    """Stages 1-2 for every image. folds: heads/classifiers to average (default all 5)."""
    device = torch.device(device)
    work = Path(work)
    (work / 'crops').mkdir(parents=True, exist_ok=True)
    folds = tuple(cfg.folds if folds is None else folds)
    model = (
        P.ProposalModel(Path(weights) / 'proposal' / 'dinov3_vits16plus')
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    heads = P.load_heads(weights, folds)
    clf = C.load_classifiers(weights, folds, policy=classifier_policy)
    rows, t0 = [], time.monotonic()
    for i, path in enumerate(sorted(image_paths, key=lambda p: Path(p).name)):
        name = Path(path).name
        img = read_image(path)
        prob = P.fold_mean_probability(model, heads, img, folds, cfg.proposal)
        comps = C.components(prob, cfg.components)
        scores = C.classifier_scores(prob, img, comps, clf, cfg.components)
        for c in K.make_crops(name, img, prob, comps, scores, cfg.components):
            _save_crop(work / 'crops' / (c.crop_id + '.npz'), c)
            rows.append(dict(crop_id=c.crop_id, source=name, ml=c.ml_score))
        if log and (i % 20 == 0 or i == len(image_paths) - 1):
            log(f'upstream {i+1}/{len(image_paths)} images, {len(rows)} crops, {time.monotonic()-t0:.0f}s')
    _write_text(work / 'images.json', json.dumps(sorted(Path(p).name for p in image_paths)))
    _write_text(
        work / 'cache.json', json.dumps(cache_key(image_paths, weights, cfg, folds, classifier_policy), indent=1)
    )
    buffer = io.StringIO(newline='')
    w = csv.DictWriter(buffer, fieldnames=['crop_id', 'source', 'ml'])
    w.writeheader()
    w.writerows(rows)
    _write_text(work / 'manifest.csv', buffer.getvalue())  # written last: its presence marks a finished upstream
    return rows


def manifest(work):
    with open(Path(work) / 'manifest.csv', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def refine_all(weights, work, cfg=PipelineConfig(), device='cuda', folds=None, crop_ids=None, log=print):
    """Stage 3: every crop (or crop_ids) through each fold refiner; resumable per crop."""
    device = torch.device(device)
    work = Path(work)
    ids = crop_ids if crop_ids is not None else [r['crop_id'] for r in manifest(work)]
    for k in cfg.folds if folds is None else folds:
        out = work / 'refined' / f'fold{k}'
        out.mkdir(parents=True, exist_ok=True)
        todo = [i for i in ids if not (out / (i + '.npz')).exists()]
        if not todo:
            continue
        model = load_refiner(weights, k, device)
        t0 = time.monotonic()
        for s in range(0, len(todo), 50 * cfg.refiner.batch):  # load crops in chunks of 50 batches to bound memory
            part = todo[s : s + 50 * cfg.refiner.batch]
            for r in RI.refine(model, [_load_crop(work / 'crops' / (i + '.npz')) for i in part], device, cfg.refiner):
                _savez(out / (r['crop_id'] + '.npz'), **r)
            if log:
                log(f'refiner fold {k}: {min(s+len(part), len(todo))}/{len(todo)} crops, {time.monotonic()-t0:.0f}s')
        del model
        torch.cuda.empty_cache()


def _load_refined(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d['crop_id'] = str(d['crop_id'])
    return d


def assemble(work, out_csv, cfg=PipelineConfig(), folds=None, members=None):
    """Post-processing: per image, ensemble the fold refiners for each crop, post-process, write the CSV.
    members: optional {crop_id: [fold, ...]} (out-of-fold evaluation uses only the owner fold)."""
    work = Path(work)
    folds = tuple(cfg.folds if folds is None else folds)
    by = {}
    for r in manifest(work):
        by.setdefault(r['source'], []).append(r['crop_id'])
    names = json.loads((work / 'images.json').read_text())
    rows, instances = [], {}
    for name in names:
        crops = []
        for cid in by.get(name, []):
            ks = members[cid] if members else folds
            crops.append(RI.ensemble([_load_refined(work / 'refined' / f'fold{k}' / (cid + '.npz')) for k in ks]))
        inst = PP.postprocess(crops, cfg.postprocess) if crops else []
        instances[name] = inst
        rows += [(f'{Path(name).stem}_{j+1}', S.rle_counts(PP.to_full(x))) for j, x in enumerate(inst)]
    S.write_csv(out_csv, rows)
    return rows, instances


class Predictor:
    """In-memory single-image predictor (same computation, crops batched per image)."""

    def __init__(self, weights, cfg=PipelineConfig(), device='cuda', folds=None):
        self.cfg, self.device = cfg, torch.device(device)
        self.folds = tuple(cfg.folds if folds is None else folds)
        self.proposal = (
            P.ProposalModel(Path(weights) / 'proposal' / 'dinov3_vits16plus')
            .to(self.device)
            .to(memory_format=torch.channels_last)
        )
        self.heads = P.load_heads(weights, self.folds)
        self.clf = C.load_classifiers(weights, self.folds)
        self.refiners = {k: load_refiner(weights, k, self.device) for k in self.folds}

    def __call__(self, image, name='image.jpeg'):
        """Returns dict(probability, components, crops, refined, instances)."""
        prob = P.fold_mean_probability(self.proposal, self.heads, image, self.folds, self.cfg.proposal)
        comps = C.components(prob, self.cfg.components)
        scores = C.classifier_scores(prob, image, comps, self.clf, self.cfg.components)
        crops = K.make_crops(name, image, prob, comps, scores, self.cfg.components)
        per = {k: RI.refine(m, crops, self.device, self.cfg.refiner) for k, m in self.refiners.items()}
        refined = [RI.ensemble([per[k][i] for k in self.folds]) for i in range(len(crops))]
        inst = PP.postprocess(refined, self.cfg.postprocess)
        return dict(probability=prob, components=comps, scores=scores, crops=crops, refined=refined, instances=inst)
