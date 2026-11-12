"""Generate owner-excluded proposal maps, then fit the original LightGBM recipe.
One process/GPU; map output is intentionally float32 after mean of four rounded-float16 maps.
"""

import argparse, json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument('stage', choices=['maps', 'ml'])
p.add_argument('--images', required=True)
p.add_argument('--annotations', required=True)
p.add_argument('--weights', default=str(ROOT / 'weights'))
p.add_argument('--out', required=True)
a = p.parse_args()
out = Path(a.out).resolve()
owners = {
    s: f['fold']
    for f in json.loads((ROOT / 'splits/fold_assignment.json').read_text())['folds']
    for s in f['train_sources']
}
if a.stage == 'maps':
    import cv2, numpy as np
    from filament_pq.proposal import ProposalModel, load_heads, fold_mean_probability

    w = Path(a.weights)
    model = ProposalModel(w / 'proposal/dinov3_vits16plus').to('cuda').eval()
    heads = load_heads(w, range(5))
    dest = out / 'ensemble/vitsplus'
    dest.mkdir(parents=True, exist_ok=True)
    for n, s in enumerate(sorted(owners)):
        eligible = [k for k in range(5) if k != owners[s]]
        raw = cv2.imread(str(Path(a.images) / s), 0)
        assert raw is not None
        prob = fold_mean_probability(model, heads, raw, eligible)
        np.savez_compressed(dest / (s + '.npz'), probability=prob)
        (dest / (s + '.json')).write_text(
            json.dumps(dict(passed=True, checkpoint_epoch=4, owner_fold=owners[s], eligible_folds=eligible))
        )
        print(n + 1, s, flush=True)
else:
    # Import the training feature implementation, preserving record weights and balanced labels.
    os.environ.update(
        SOLAR_RAW_IMAGES=str(Path(a.images).resolve()),
        SOLAR_ANNOTATION_JSON=str(Path(a.annotations).resolve()),
        SOLAR_ANALYSIS_ROOT=str(out),
        SOLAR_MEAN_ROOT=str(out / 'ensemble'),
        SOLAR_ML_OUT=str(out / 'ml'),
    )
    sys.path.insert(0, str(ROOT / 'training/components'))
    import train_presence_ml as ml

    ml.ROOT = out
    (out / 'fold_assignment.json').write_text((ROOT / 'splits/fold_assignment.json').read_text())
    ml.initialize()
    for n, s in enumerate(sorted(owners)):
        ml.extract(('vitsplus', s, {'proposal005': 0.05}))
        print(n + 1, s, flush=True)
    ml.fit_pipeline('vitsplus', 'proposal005', sorted(owners), owners)
