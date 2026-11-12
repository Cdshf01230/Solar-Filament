"""Write each LightGBM classifier also as a plain-text LightGBM model (version-portable, human-readable, no pickle)
and verify that the text model gives bit-identical scores to the pickled one on real component features.

    python scripts/export_lightgbm.py --weights weights --images DIR [--n 5]
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_pq import components as C, proposal as P, pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--policy', choices=['balance', 'recall'], default='balance')
    a = ap.parse_args()
    W = Path(a.weights)
    import joblib, lightgbm

    pick = {k: joblib.load(W / 'ml' / f'{a.policy}_fold{k}.joblib') for k in range(5)}
    for k, m in pick.items():
        m.booster_.save_model(str(W / 'ml' / f'{a.policy}_fold{k}.txt'))
    text = {k: lightgbm.Booster(model_file=str(W / 'ml' / f'{a.policy}_fold{k}.txt')) for k in range(5)}
    device = torch.device('cuda')
    model = P.ProposalModel(W / 'proposal' / 'dinov3_vits16plus').to(device).to(memory_format=torch.channels_last)
    heads = P.load_heads(W, range(5))
    rows = 0
    maxdiff = 0.0
    for path in sorted(Path(a.images).glob('*.jpeg'))[: a.n]:
        img = pipeline.read_image(path)
        prob = P.fold_mean_probability(model, heads, img, range(5))
        comps = C.components(prob)
        X = C.feature_matrix(prob, img, C.appearance(img), 0.05, comps)
        for k in range(5):
            p1 = pick[k].predict_proba(X)[:, 1].astype(np.float32)
            p2 = text[k].predict(X).astype(np.float32)
            maxdiff = max(maxdiff, float(np.abs(p1 - p2).max()))
            assert np.array_equal(p1, p2), (path.name, k)
        rows += len(X)
    params = pick[0].get_params()
    info = dict(
        lightgbm=lightgbm.__version__,
        components_checked=rows,
        max_abs_diff=maxdiff,
        identical=maxdiff == 0.0,
        trees=[int(pick[k].n_estimators_) for k in range(5)],
        features=C.FEATURES,
        params={
            k: params[k]
            for k in (
                'n_estimators',
                'num_leaves',
                'max_depth',
                'learning_rate',
                'min_child_samples',
                'reg_lambda',
                'random_state',
                'deterministic',
            )
        },
    )
    info['policy'] = a.policy
    filename = 'MODEL_INFO.json' if a.policy == 'balance' else 'RECALL_MODEL_INFO.json'
    (W / 'ml' / filename).write_text(json.dumps(info, indent=1))
    print('LGBM', json.dumps({k: info[k] for k in ('lightgbm', 'components_checked', 'identical', 'trees')}))


if __name__ == '__main__':
    main()
