"""Isolated DDP runner derived from train.py; historical runner untouched."""

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import (
    SharedFrozenModel,
    DEFAULT_HEADS,
    EXTENDED_HEADS,
    LOSS_HEADS,
    LOSS_PROTOCOL,
    SPECS,
    train_step,
    lr_factor,
)
from .data import Crops, EpochSampler, worker_init, load_split, CURSOR_PROTOCOL, cursor_from_update, resume_cursor
from .ddp_evaluation import Context, evaluate
from .ddp_model import DDPModel as SharedFrozenModel, train_step, PROTOCOL as DDP_PROTOCOL

SPECS["ddp"] = dict(parameters=0)
from .preprocessing import PREPROCESS_HEADS, PREPROCESS_PROTOCOL, PreprocessedCrops


def atomic_save(value, path, tensor=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    if tensor:
        torch.save(value, tmp)
    else:
        tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def make_optimizers(model, lr, total, warmup):
    optim = {n: torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=0.01) for n, h in model.heads.items()}
    sched = {n: torch.optim.lr_scheduler.LambdaLR(o, lambda s: lr_factor(s, total, warmup)) for n, o in optim.items()}
    return optim, sched


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    if s['cuda']:
        torch.cuda.set_rng_state_all(s['cuda'])


def smoke(model, batch, args):
    """Transactional preflight: restore weights/RNG even after an OOM."""
    initial = copy.deepcopy({n: h.cpu().state_dict() for n, h in model.heads.items()})
    model.cuda().to(memory_format=torch.channels_last)
    rng = rng_state()
    results = []
    try:
        for micro in sorted({min(args.microbatch, args.batch), 2, 1}, reverse=True):
            if micro > args.microbatch:
                continue
            for n, h in model.heads.items():
                h.load_state_dict(initial[n])
            optim, sched = make_optimizers(model, args.lr, args.total_updates, args.warmup)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            calls = [0]

            def count_forward(*_):
                calls[0] += 1

            hook = model.backbone.register_forward_hook(count_forward)
            try:
                metrics = train_step(model, *batch[:2], optim, micro, sched)
                torch.cuda.synchronize()
                assert calls[0] == math.ceil(len(batch[1]) / micro) * (
                    len(model.heads) if isinstance(batch[0], dict) else 1
                )
                results.append(
                    dict(
                        microbatch=micro,
                        passed=True,
                        seconds=time.perf_counter() - start,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                        backbone_forward_calls=calls[0],
                        metrics=metrics,
                    )
                )
                snapshot = args.out / 'smoke_roundtrip.pt'
                torch.save(
                    dict(
                        heads={n: h.state_dict() for n, h in model.heads.items()},
                        optim={n: o.state_dict() for n, o in optim.items()},
                        sched={n: s.state_dict() for n, s in sched.items()},
                    ),
                    snapshot,
                )
                reloaded = torch.load(snapshot, map_location='cpu', weights_only=False)
                for n, h in model.heads.items():
                    for k, v in h.state_dict().items():
                        torch.testing.assert_close(v.cpu(), reloaded['heads'][n][k], rtol=0, atol=0)
                    optim[n].load_state_dict(reloaded['optim'][n])
                    sched[n].load_state_dict(reloaded['sched'][n])
                snapshot.unlink()
                results[-1]['checkpoint_roundtrip'] = 'passed'
                break
            except torch.cuda.OutOfMemoryError:
                results.append(dict(microbatch=micro, passed=False, error='CUDA OOM'))
            finally:
                hook.remove()
                for p in model.heads.parameters():
                    p.grad = None
                del optim, sched
                torch.cuda.empty_cache()
    finally:
        for n, h in model.heads.items():
            h.load_state_dict(initial[n])
        restore_rng(rng)
    return dict(
        device=torch.cuda.get_device_name(),
        attempts=results,
        parameters={n: SPECS[n]['parameters'] for n in model.heads},
        passed=bool(results and results[-1]['passed']),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=Path('/content/solar_research'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--drive', type=Path)
    p.add_argument('--protocol', choices=['30_70', '70_30'], default='30_70')
    p.add_argument('--preset', choices=['base12', 'extended40', 'loss40', 'preprocess12'], default='base12')
    p.add_argument('--heads', nargs='+', choices=list(SPECS))
    p.add_argument('--epochs', type=int)
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument(
        '--val-every-epoch-from',
        type=int,
        help='Override quick validation cadence to every epoch from this epoch onward',
    )
    p.add_argument(
        '--finish-after-epoch',
        type=int,
        help='Finish and evaluate at this epoch without changing the original LR schedule budget',
    )
    p.add_argument('--no-augment', action='store_true', help='Disable D4 for a controlled training-fit experiment')
    p.add_argument(
        '--train-eval-every',
        type=int,
        default=0,
        help='Evaluate current weights on all train images at threshold .5 every N epochs',
    )
    p.add_argument(
        '--evaluate-last',
        action='store_true',
        help='Also evaluate final training weights before loading quick-selected best weights',
    )
    p.add_argument('--enable-tf32', action='store_true', help='Explicit TF32 compute protocol for a new experiment')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--eval-workers', type=int, default=12)
    p.add_argument('--tile-batch', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument(
        '--smoke-crops',
        type=int,
        default=0,
        help='Preflight only: restrict loading to first N historical crops on a fresh server',
    )
    p.add_argument('--final-eval', choices=['both', 'audit', 'none'], default='both')
    args = p.parse_args()
    if args.heads is None:
        args.heads = list(
            PREPROCESS_HEADS
            if args.preset == 'preprocess12'
            else LOSS_HEADS
            if args.preset == 'loss40'
            else EXTENDED_HEADS
            if args.preset == 'extended40'
            else DEFAULT_HEADS
        )
    if args.epochs is None:
        args.epochs = 40 if args.preset in ('extended40', 'loss40') else 12
    finish_epoch = args.finish_after_epoch or args.epochs
    if not 1 <= finish_epoch <= args.epochs:
        p.error('--finish-after-epoch must be between 1 and --epochs')
    if args.val_every_epoch_from is not None and not 1 <= args.val_every_epoch_from <= args.epochs:
        p.error('--val-every-epoch-from must be between 1 and --epochs')
    if args.smoke_crops and (not args.preflight_only or args.smoke_crops < args.batch):
        p.error('--smoke-crops requires --preflight-only and must be >= --batch')
    if (
        len(args.heads) != len(set(args.heads))
        or min(args.batch, args.microbatch, args.epochs, args.save_every, args.eval_every) < 1
    ):
        p.error('Unique heads and positive batch/microbatch/epochs/save-every required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for real-data runner; CPU tests use smoke_test.py')
    args.out.mkdir(parents=True, exist_ok=True)
    # Kernel lock prevents accidental concurrent writers; released automatically on process exit.
    import fcntl

    lock = (args.out / 'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = args.out / 'last_training.pt'
    if checkpoint.exists() and not args.resume:
        raise FileExistsError('Existing checkpoint: pass --resume or use a new --out')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    if args.preset == 'preprocess12' or args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    manifest, split = load_split(args.base, args.protocol)
    if args.preset == 'preprocess12':
        assert len(args.heads) == 1 and set(args.heads) <= set(
            PREPROCESS_HEADS
        ), 'Preprocessing ablations run one model at a time'
        dataset = PreprocessedCrops(args.base, manifest, args.seed, variants=args.heads)
    else:
        dataset = Crops(args.base / 'magfilo_crops_768', manifest, args.seed, augment=not args.no_augment)
    if args.no_augment and args.preset == 'preprocess12':
        p.error('--no-augment is only supported for raw input')
    train_crop_count = len(dataset)
    if any(r['source_file'] not in split['train'] for r in dataset.records):
        raise ValueError('Train manifest includes non-training source')
    if args.smoke_crops:
        dataset.records = dataset.records[: args.smoke_crops]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        train_crops=train_crop_count,
        train_images=len(split['train']),
        quick_images=len(split['quick']),
        audit_images=len(split['audit']),
        augmentation='none' if args.no_augment else 'stateless D4(seed,epoch,index)',
        selection='max quick full-image public-notebook PQ; no audit tuning',
        quick_audit_shared_days=split['quick_audit_shared_days'],
        cursor_protocol=CURSOR_PROTOCOL,
    )
    if any(n in LOSS_HEADS for n in args.heads):
        config['loss_protocol'] = {n: LOSS_PROTOCOL[n] for n in args.heads}
        config['architecture'] = 'All heads identical historical U0 decoder; one frozen FP32 DINOv3 ConvNeXt Base'
    if args.preset == 'preprocess12':
        config['preprocessing'] = PREPROCESS_PROTOCOL
        config['compute_protocol'] = 'FP32 backbone with matmul/cuDNN TF32 enabled; BF16 decoder'
        config['loss_protocol'] = 'BCE + original per-crop soft Dice'
        config['architecture'] = (
            'Single historical U0 decoder and frozen FP32 DINOv3 backbone; sequential preprocessing ablation'
        )
    elif args.enable_tf32:
        config['compute_protocol'] = 'FP32 backbone with matmul/cuDNN TF32 enabled; BF16 decoder'
    model = (
        SharedFrozenModel(args.base / 'backbone', args.heads, args.seed).cuda().to(memory_format=torch.channels_last)
    )
    config['model_protocol'] = DDP_PROTOCOL
    config['architecture'] = 'Binary DDP adaptation; frozen DINOv3 ConvNeXt Base'
    config['loss_protocol'] = DDP_PROTOCOL['objective']
    SPECS['ddp']['parameters'] = sum(p.numel() for p in model.heads['ddp'].parameters())
    config['head_parameters'] = SPECS['ddp']['parameters']
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=EpochSampler(len(dataset), args.seed),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        multiprocessing_context='spawn' if args.workers else None,
        worker_init_fn=worker_init,
    )
    if args.preflight_only:
        args.total_updates = math.ceil(train_crop_count / args.batch) * args.epochs
        atomic_save(config, args.out / 'preflight_config.json')
        report = smoke(model, next(iter(loader)), args)
        if report['passed']:
            quick_names = split['quick'][:1]
            selected, rows = evaluate(
                model,
                quick_names,
                Context(args.base),
                thresholds_by_head={n: [0.5] for n in model.heads},
                workers=min(args.eval_workers, 4),
                tile_batch=args.tile_batch,
            )
            report['full_image_inference_smoke'] = selected
            report['full_image_rows'] = {n: len(rs) for n, rs in rows.items()}
        atomic_save(report, args.out / 'preflight.json')
        print(json.dumps(report, indent=2), flush=True)
        if not report['passed']:
            raise RuntimeError('All preflight microbatches failed')
        return
    total = math.ceil(len(dataset) / args.batch) * args.epochs
    optim, sched = make_optimizers(model, args.lr, total, args.warmup)
    epoch, offset, update, best, history = 1, 0, 0, {}, []
    if args.resume:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck['config'].get('model_protocol') != config['model_protocol']:
            raise ValueError('Resume DDP model/sampling protocol differs')
        for key in [
            'heads',
            'protocol',
            'batch',
            'microbatch',
            'epochs',
            'seed',
            'lr',
            'warmup',
            'train_crops',
            'eval_every',
        ]:
            if ck['config'].get(key, 1 if key == 'eval_every' else None) != config[key]:
                raise ValueError(f'Resume configuration differs: {key}')
        if 'loss_protocol' in config and ck['config'].get('loss_protocol') != config['loss_protocol']:
            raise ValueError('Resume loss protocol differs')
        if args.preset == 'preprocess12' and ck['config'].get('compute_protocol') != config['compute_protocol']:
            raise ValueError('Resume compute protocol differs')
        if ck['config'].get('preprocessing') != config.get('preprocessing'):
            raise ValueError('Resume preprocessing differs')
        for key, default in [('no_augment', False), ('train_eval_every', 0), ('evaluate_last', False)]:
            if ck['config'].get(key, default) != config[key]:
                raise ValueError(f'Resume diagnostic protocol differs: {key}')
        if ck['config'].get('enable_tf32', False) != args.enable_tf32:
            raise ValueError('Resume TF32 flag differs')
        if ck['config'].get('val_every_epoch_from') != args.val_every_epoch_from:
            authorization = args.out / 'validation_schedule_change.json'
            change = json.loads(authorization.read_text()) if authorization.exists() else {}
            if not (
                ck['config'].get('val_every_epoch_from') is None
                and change.get('user_authorized') is True
                and change.get('old_eval_every') == args.eval_every
                and change.get('every_epoch_from') == args.val_every_epoch_from
            ):
                raise ValueError('Resume validation cadence change requires explicit recorded authorization')
        for n, h in model.heads.items():
            h.load_state_dict(ck['heads'][n])
            optim[n].load_state_dict(ck['optim'][n])
            sched[n].load_state_dict(ck['sched'][n])
        epoch, offset, update, best, history = (ck[k] for k in ['epoch', 'offset', 'update', 'best', 'history'])
        offset, correction = resume_cursor(ck, len(dataset), args.batch)
        if correction:
            atomic_save(correction, args.out / 'checkpoint_cursor_correction.json')
            print(json.dumps(dict(checkpoint_cursor_correction=correction)), flush=True)
        restore_rng(ck['rng'])
        # CSV may contain updates newer than the last atomic checkpoint after a crash.
        for n in model.heads:
            path = args.out / n / 'train.csv'
            if path.exists():
                with path.open(newline='') as f:
                    reader = csv.DictReader(f)
                    fields = reader.fieldnames
                    kept = [r for r in reader if int(r['update']) <= update]
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(kept)
    atomic_save(config, args.out / 'config.json')
    deadline = time.monotonic() + float(os.environ.get('SOLAR_SOFT_SECONDS', '37800'))
    stop = [False]

    def request_stop(*_):
        stop[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def mirror():
        if args.drive:
            for src in args.out.rglob('*'):
                if src.is_file() and not src.name.endswith(('.tmp', '.lock')):
                    dst = args.drive / src.relative_to(args.out)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_name(dst.name + '.tmp')
                    shutil.copyfile(src, tmp)
                    os.replace(tmp, dst)

    def save():
        if offset != cursor_from_update(epoch, update, len(dataset), args.batch):
            raise ValueError('Refusing to save a cursor inconsistent with completed updates')
        atomic_save(
            dict(
                config=config,
                heads={n: h.state_dict() for n, h in model.heads.items()},
                optim={n: o.state_dict() for n, o in optim.items()},
                sched={n: s.state_dict() for n, s in sched.items()},
                epoch=epoch,
                offset=offset,
                update=update,
                best=best,
                history=history,
                rng=rng_state(),
            ),
            checkpoint,
            tensor=True,
        )
        # Authoritative common checkpoint contains all selected weights too.
        for n, b in best.items():
            atomic_save(
                dict(
                    head=b['weights'],
                    selection=b['selection'],
                    epoch=b['epoch'],
                    update=b['update'],
                    name=n,
                    config=config,
                ),
                args.out / n / 'best_head.pt',
                tensor=True,
            )
        atomic_save(history, args.out / 'history.json')
        mirror()

    context = Context(args.base)
    backfill_path = args.out / 'quick_backfill_epoch3.pt'
    if args.resume and args.val_every_epoch_from == 3 and backfill_path.exists():
        target = next((r for r in history if r['epoch'] == 3), None)
        if target is not None and target.get('validation') is None:
            snapshot = torch.load(backfill_path, map_location='cpu', weights_only=False)
            assert snapshot['completed_epoch'] == 3 and snapshot['update'] == math.ceil(len(dataset) / args.batch) * 3
            assert snapshot['update'] <= update
            for key in ['heads', 'protocol', 'batch', 'seed', 'train_crops', 'no_augment', 'enable_tf32']:
                assert snapshot['config'].get(key) == config.get(key), f'Backfill configuration differs: {key}'
            current_weights = {
                n: {k: v.detach().cpu().clone() for k, v in h.state_dict().items()} for n, h in model.heads.items()
            }
            current_rng = rng_state()
            try:
                for n, h in model.heads.items():
                    h.load_state_dict(snapshot['heads'][n])

                def backfill_progress(done, total):
                    atomic_save(
                        dict(
                            state='backfill_quick_validation',
                            checkpoint_epoch=3,
                            resumed_update=update,
                            completed_image_heads=done,
                            total_image_heads=total,
                        ),
                        args.out / 'status.json',
                    )

                result, _ = evaluate(
                    model,
                    split['quick'],
                    context,
                    workers=args.eval_workers,
                    tile_batch=args.tile_batch,
                    progress=backfill_progress,
                )
                for n, r in result['heads'].items():
                    if n not in best or r['selected']['instance']['pq'] > best[n]['selection']['instance']['pq']:
                        best[n] = dict(
                            weights=snapshot['heads'][n], selection=r['selected'], epoch=3, update=snapshot['update']
                        )
                target['validation'] = result
                target['validation_backfilled_from_exact_epoch_checkpoint'] = True
                atomic_save(result, args.out / 'quick_backfill_epoch3.json')
            finally:
                for n, h in model.heads.items():
                    h.load_state_dict(current_weights[n])
                    assert all(torch.equal(v.detach().cpu(), current_weights[n][k]) for k, v in h.state_dict().items())
                restore_rng(current_rng)
            atomic_save(
                dict(
                    backfilled_epoch=3,
                    checkpoint_update=snapshot['update'],
                    resumed_update=update,
                    training_weights_exactly_restored=True,
                    optimizer_scheduler='not stepped',
                    rng='restored',
                    cadence='every epoch from 3; train-fit diagnostics still every 5',
                ),
                args.out / 'validation_schedule_application.json',
            )
            save()
            del current_weights, snapshot
    if not args.resume:
        # A durable update0 checkpoint makes early runtime failures resumable too.
        save()
    while epoch <= finish_epoch:
        loader.sampler.set_cursor(epoch, offset)
        began = time.perf_counter()
        for x, y, _ in loader:
            metrics = train_step(model, x, y, optim, args.microbatch, sched)
            update += 1
            offset += len(y)
            for n, row in metrics.items():
                path = args.out / n / 'train.csv'
                path.parent.mkdir(parents=True, exist_ok=True)
                exists = path.exists()
                with path.open('a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['epoch', 'update'] + list(row))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(dict(epoch=epoch, update=update, **row))
            if update % 10 == 0:
                atomic_save(
                    dict(state='training', epoch=epoch, update=update, offset=offset, metrics=metrics),
                    args.out / 'status.json',
                )
                print(
                    json.dumps(dict(epoch=epoch, update=update, losses={n: r['loss'] for n, r in metrics.items()})),
                    flush=True,
                )
            if time.monotonic() >= deadline:
                stop[0] = True
            if update % args.save_every == 0 or stop[0]:
                save()
            if stop[0]:
                atomic_save(dict(state='stopped_resumable', epoch=epoch, update=update), args.out / 'status.json')
                mirror()
                return
        train_seconds = time.perf_counter() - began
        every_epoch_now = args.val_every_epoch_from is not None and epoch >= args.val_every_epoch_from
        if epoch % args.eval_every and epoch != finish_epoch and not every_epoch_now:
            history.append(dict(epoch=epoch, update=update, train_seconds=train_seconds, validation=None))
            epoch += 1
            offset = 0
            save()
            if stop[0]:
                return
            continue

        def progress(done, total):
            atomic_save(
                dict(
                    state='quick_validation',
                    epoch=epoch,
                    update=update,
                    completed_image_heads=done,
                    total_image_heads=total,
                ),
                args.out / 'status.json',
            )

        result, _ = evaluate(
            model, split['quick'], context, workers=args.eval_workers, tile_batch=args.tile_batch, progress=progress
        )
        for n, r in result['heads'].items():
            if n not in best or r['selected']['instance']['pq'] > best[n]['selection']['instance']['pq']:
                best[n] = dict(
                    weights={k: v.detach().cpu().clone() for k, v in model.heads[n].state_dict().items()},
                    selection=r['selected'],
                    epoch=epoch,
                    update=update,
                )
        entry = dict(epoch=epoch, update=update, train_seconds=train_seconds, validation=result)
        if args.train_eval_every and (epoch % args.train_eval_every == 0 or epoch == finish_epoch):
            if args.no_augment:
                from .crop_fit import evaluate_crop_fit

                diagnostic_rng = rng_state()

                def crop_progress(done, total):
                    atomic_save(
                        dict(
                            state='fixed_train_crop_evaluation',
                            epoch=epoch,
                            update=update,
                            completed_crops=done,
                            total_crops=total,
                        ),
                        args.out / 'status.json',
                    )

                try:
                    entry['current_train_crops'] = evaluate_crop_fit(
                        model, loader, epoch, args.microbatch, crop_progress
                    )
                finally:
                    restore_rng(diagnostic_rng)
                for n, fit in entry['current_train_crops']['heads'].items():
                    score = fit['bce'] + fit['soft_dice_loss']
                    path = args.out / n / 'best_train_fit.pt'
                    metadata = args.out / n / 'best_train_fit.json'
                    previous = json.loads(metadata.read_text()) if metadata.exists() else None
                    if previous is None or score < previous['selection']['fixed_crop_bce_plus_dice']:
                        value = dict(
                            name=n,
                            epoch=epoch,
                            update=update,
                            config=config,
                            selection=dict(
                                fixed_crop_bce_plus_dice=score,
                                crop_metrics=fit,
                                criterion='minimum fixed full-training-crop BCE+Dice; no val/audit selection',
                            ),
                        )
                        atomic_save(
                            dict(
                                value,
                                head={k: v.detach().cpu().clone() for k, v in model.heads[n].state_dict().items()},
                            ),
                            path,
                            tensor=True,
                        )
                        atomic_save(value, metadata)

            def train_progress(done, total):
                atomic_save(
                    dict(
                        state='periodic_train_evaluation',
                        epoch=epoch,
                        update=update,
                        completed_image_heads=done,
                        total_image_heads=total,
                    ),
                    args.out / 'status.json',
                )

            entry['current_train'], _ = evaluate(
                model,
                split['train'],
                context,
                thresholds_by_head={n: [0.5] for n in model.heads},
                workers=args.eval_workers,
                tile_batch=args.tile_batch,
                progress=train_progress,
            )
        history.append(entry)
        epoch += 1
        offset = 0
        save()
        if stop[0]:
            return
    if args.evaluate_last:
        last = dict(epoch=finish_epoch, update=update, threshold_protocol='fixed .5; no tuning')
        for part in ['train', 'quick', 'audit']:

            def last_progress(done, total):
                atomic_save(
                    dict(
                        state='last_' + part + '_evaluation',
                        epoch=finish_epoch,
                        update=update,
                        completed_image_heads=done,
                        total_image_heads=total,
                    ),
                    args.out / 'status.json',
                )

            last[part], rows = evaluate(
                model,
                split[part],
                context,
                thresholds_by_head={n: [0.5] for n in model.heads},
                workers=args.eval_workers,
                tile_batch=args.tile_batch,
                progress=last_progress,
            )
            for n, rs in rows.items():
                atomic_save(rs, args.out / n / ('last_' + part + '_rows.json'))
            atomic_save(last, args.out / 'last_result.json')
            mirror()
        if args.no_augment and args.train_eval_every:
            fit_result = dict(
                threshold_protocol='fixed .5; checkpoint selected only by training-crop loss', selection={}
            )
            for n, h in model.heads.items():
                fit = torch.load(args.out / n / 'best_train_fit.pt', map_location='cpu', weights_only=False)
                h.load_state_dict(fit['head'])
                fit_result['selection'][n] = {k: v for k, v in fit.items() if k not in ('head', 'config')}
                del fit
            for part in ['train', 'quick', 'audit']:

                def fit_progress(done, total):
                    atomic_save(
                        dict(
                            state='best_train_fit_' + part + '_evaluation',
                            update=update,
                            completed_image_heads=done,
                            total_image_heads=total,
                        ),
                        args.out / 'status.json',
                    )

                fit_result[part], rows = evaluate(
                    model,
                    split[part],
                    context,
                    thresholds_by_head={n: [0.5] for n in model.heads},
                    workers=args.eval_workers,
                    tile_batch=args.tile_batch,
                    progress=fit_progress,
                )
                for n, rs in rows.items():
                    atomic_save(rs, args.out / n / ('best_train_fit_' + part + '_rows.json'))
                atomic_save(fit_result, args.out / 'best_train_fit_result.json')
                mirror()
    # Each head now uses its independently selected checkpoint and threshold.
    for n, h in model.heads.items():
        h.load_state_dict(best[n]['weights'])
    final = dict(
        completed_epochs=finish_epoch,
        scheduler_budget_epochs=args.epochs,
        selection={n: {k: v for k, v in b.items() if k != 'weights'} for n, b in best.items()},
    )
    for part in ['train', 'audit'] if args.final_eval == 'both' else ['audit'] if args.final_eval == 'audit' else []:
        thresholds = {n: sorted({0.5, b['selection']['threshold']}) for n, b in best.items()}

        def final_progress(done, total):
            atomic_save(
                dict(
                    state='final_' + part + '_evaluation',
                    epoch=finish_epoch,
                    update=update,
                    completed_image_heads=done,
                    total_image_heads=total,
                ),
                args.out / 'status.json',
            )

        final_progress(0, len(split[part]) * len(model.heads))
        final[part], rows = evaluate(
            model,
            split[part],
            context,
            thresholds,
            {n: b['selection']['threshold'] for n, b in best.items()},
            args.eval_workers,
            args.tile_batch,
            progress=final_progress,
        )
        atomic_save(final[part], args.out / (part + '_result.json'))
        for n, rs in rows.items():
            atomic_save(rs, args.out / n / (part + '_rows.json'))
    atomic_save(final, args.out / 'result.json')
    lines = [
        '# Shared frozen backbone / independent heads',
        '',
        'Public-notebook PQ proxy; threshold and epoch selected only on quick-val.',
        '',
        '|Head|Parameters|Epoch|Threshold|Quick PQ|Train PQ|Audit PQ|',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for n, b in best.items():

        def pq(part):
            return f"{final[part]['heads'][n]['selected']['instance']['pq']:.6f}" if part in final else '—'

        lines.append(
            f"|{n}|{SPECS[n]['parameters']}|{b['epoch']}|{b['selection']['threshold']}|{b['selection']['instance']['pq']:.6f}|{pq('train')}|{pq('audit')}|"
        )
    (args.out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    atomic_save(dict(state='complete', update=update), args.out / 'status.json')
    mirror()


if __name__ == '__main__':
    main()
