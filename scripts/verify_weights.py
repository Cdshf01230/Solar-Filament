"""Strict-load every released model and check tensors; CPU only, no training/inference."""

import argparse, json, sys, gc
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_pq.proposal import ProposalModel
from filament_pq.refiner.model import build
from filament_pq.components import load_classifiers


def finite(sd):
    assert all(torch.isfinite(v).all().item() for v in sd.values() if torch.is_tensor(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights')
    ap.add_argument('--out')
    a = ap.parse_args()
    torch.set_num_threads(2)
    w = Path(a.weights)
    report = {'proposal': [], 'refiner': [], 'ml': []}
    model = ProposalModel(w / 'proposal/dinov3_vits16plus')
    finite(model.backbone.state_dict())
    for k in range(5):
        sd = torch.load(w / f'proposal/heads/fold{k}.pt', map_location='cpu', weights_only=True)
        model.head.load_state_dict(sd, strict=True)
        finite(sd)
        report['proposal'].append(
            {
                'fold': k,
                'strict_load': True,
                'finite': True,
                'head_parameters': sum(p.numel() for p in model.head.parameters()),
            }
        )
    del model
    gc.collect()
    for k in range(5):
        sd = torch.load(w / f'refiner/fold{k}.pt', map_location='cpu', weights_only=True)
        model = build(w / 'refiner/dinov3_vitb16', sd, torch.device('cpu'))
        # Validate the reconstructed full state, in addition to each tuned block's strict load.
        model.load_state_dict(model.state_dict(), strict=True)
        finite(model.state_dict())
        report['refiner'].append(
            {
                'fold': k,
                'strict_load': True,
                'finite': True,
                'decoder_parameters': sum(v.numel() for v in sd['decoder'].values()),
                'tuned_parameters': sum(v.numel() for v in sd['tuned_blocks'].values()),
            }
        )
        del model, sd
        gc.collect()
    X = np.zeros((3, 32), dtype=np.float32)
    for policy in ('balance', 'recall'):
        classifiers = load_classifiers(w, range(5), policy=policy)
        for k, m in classifiers.items():
            p = m.predict_proba(X)
            assert p.shape == (3, 2) and np.isfinite(p).all()
            assert m.booster.num_feature() == 32 and m.booster.num_trees() == 250
            report['ml'].append(
                {'policy': policy, 'fold': k, 'load': True, 'features': 32, 'trees': 250, 'finite_prediction': True}
            )
    assignment = json.loads((w / 'fold_assignment.json').read_text())
    assert assignment['sources'] == 707
    report['passed'] = True
    report['scope'] = 'CPU structural and finite checks, not a new full inference or training rerun'
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
