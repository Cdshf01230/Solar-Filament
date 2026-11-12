"""Convert the development checkpoints into the release layout of weights/ (run once, by the authors).

Refiner checkpoints store the full ViT-B (86 M parameters), but only blocks 9-12 were trained. The export keeps the
decoder and those 4 blocks, and asserts that every other backbone tensor is bit-identical to the public DINOv3 weights,
so loading = public DINOv3 + exported tensors reproduces the trained model exactly.

    python scripts/export_weights.py --refiner-ckpt 'ckpt/fold{k}_ep30.pt' --heads 'heads/fold{k}_last_head.pt' \
        --ml 'ml/balance_fold{k}.joblib' --vitb DIR --vits DIR --folds-json fold_assignment.json --out weights
"""

import argparse, json, re, shutil, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filament_pq.refiner import model as RM  # noqa: E402

BLOCK = re.compile(r'^features\.backbone\.(?:model\.)?layer\.(\d+)\.(.+)$')


def export_refiner(ckpt, vitb, out):
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = state['model']
    base = RM.Features(vitb).state_dict()
    base = {'features.' + k: v for k, v in base.items()}
    # align the backbone key naming of the training transformers version with the installed one
    if not set(k for k in sd if k.startswith('features.')) <= set(base):
        for old, new in (
            ('features.backbone.layer.', 'features.backbone.model.layer.'),
            ('features.backbone.model.layer.', 'features.backbone.layer.'),
        ):
            r = {(new + k[len(old) :] if k.startswith(old) else k): v for k, v in sd.items()}
            if set(k for k in r if k.startswith('features.')) <= set(base):
                sd = r
                break
    assert set(k for k in sd if k.startswith('features.')) == set(base), 'backbone key mismatch'
    decoder = {k: v for k, v in sd.items() if not k.startswith('features.')}
    tuned, frozen_checked = {}, 0
    for k, v in sd.items():
        if not k.startswith('features.'):
            continue
        m = BLOCK.match(k)
        if m and int(m.group(1)) >= 12 - RM.TUNED_BLOCKS:
            tuned[f'{m.group(1)}.{m.group(2)}'] = v.clone()
            continue
        assert torch.equal(v, base[k]), f'{k} differs from the public DINOv3 weights'
        frozen_checked += 1
    torch.save(dict(decoder=decoder, tuned_blocks=tuned), out)
    return dict(
        epoch=state.get('epoch'),
        update=state.get('update'),
        decoder_tensors=len(decoder),
        tuned_tensors=len(tuned),
        frozen_tensors_verified_identical=frozen_checked,
        decoder_params=sum(v.numel() for v in decoder.values()),
        tuned_params=sum(v.numel() for v in tuned.values()),
    )


def main():
    ap = argparse.ArgumentParser()
    for k in ('refiner-ckpt', 'heads', 'ml', 'vitb', 'vits', 'folds-json', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    for d in ('proposal/heads', 'ml', 'refiner'):
        (out / d).mkdir(parents=True, exist_ok=True)
    shutil.copytree(a.vits, out / 'proposal' / 'dinov3_vits16plus', dirs_exist_ok=True)
    shutil.copytree(a.vitb, out / 'refiner' / 'dinov3_vitb16', dirs_exist_ok=True)
    shutil.copy(a.folds_json, out / 'fold_assignment.json')
    report = {}
    for k in range(5):
        h = torch.load(a.heads.format(k=k), map_location='cpu', weights_only=False)
        assert h['epoch'] == 4
        torch.save(h['head'], out / 'proposal' / 'heads' / f'fold{k}.pt')
        shutil.copy(a.ml.format(k=k), out / 'ml' / f'balance_fold{k}.joblib')
        report[f'refiner_fold{k}'] = export_refiner(a.refiner_ckpt.format(k=k), a.vitb, out / 'refiner' / f'fold{k}.pt')
        print(k, json.dumps(report[f'refiner_fold{k}']), flush=True)
    (out / 'EXPORT_REPORT.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
