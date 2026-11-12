"""Launch the released recipe without internal development gates. Linux/CUDA required."""

import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['proposal', 'refiner'])
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--data', required=True, help='Staged proposal folder or native refiner crop package')
    p.add_argument('--backbone', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--gpus', type=int, default=2)
    p.add_argument('--annotations')
    p.add_argument('--resume')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={a.gpus}']
    if a.stage == 'proposal':
        if a.gpus != 2:
            p.error('Original proposal recipe requires two GPUs')
        cfg = dict(
            fold=a.fold,
            model_kind='vitsplus',
            loss='proposal',
            epochs=4,
            lr=3e-4,
            warmup=20,
            seed=42,
            global_batch=8,
            microbatch=2,
            base=str(Path(a.data).resolve()),
            backbone=str(Path(a.backbone).resolve()),
            out=str(out),
            resume=a.resume or '',
            session_started=time.time(),
            deadline=time.time() + 37800,
        )
        config = out / 'train_config.json'
        config.write_text(json.dumps(cfg, indent=2))
        cmd += ['-m', 'solar_loss.runner', '--config', str(config)]
        cwd = ROOT / 'training/proposal'
        env['SOLAR_MODEL_KIND'] = 'vitsplus'
    else:
        if not a.annotations:
            p.error('Refiner requires --annotations (original COCO JSON)')
        cwd = ROOT / 'training/refiner'
        cmd += [
            'train.py',
            '--crops',
            str(Path(a.data).resolve()),
            '--backbone',
            str(Path(a.backbone).resolve()),
            '--annotations',
            str(Path(a.annotations).resolve()),
            '--folds',
            str(ROOT / 'splits/fold_assignment.json'),
            '--out',
            str(out),
            '--holdout',
            str(a.fold),
            '--epochs',
            '30',
            '--microbatch',
            '1',
            '--d4',
            '--photometric',
            '--layer-decay',
            '.65',
            '--drop-path',
            '.1',
            '--arm',
            'llrd_dp',
            '--keep-eval-snapshots',
        ]
        if a.resume:
            cmd += ['--resume', str(Path(a.resume).resolve())]
    print(json.dumps({'cwd': str(cwd), 'command': cmd}, indent=2))
    if not a.dry_run:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)


if __name__ == '__main__':
    main()
