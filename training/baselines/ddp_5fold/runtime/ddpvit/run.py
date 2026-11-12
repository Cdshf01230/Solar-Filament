"""2-GPU (Kaggle T4 x2, NCCL) trainer for the DDP-on-ViT-B experiment: GPU gate, main training, evaluation.

    torchrun --standalone --nproc_per_node=2 -m ddpvit.run --config cfg.json --mode gate
    torchrun --standalone --nproc_per_node=2 -m ddpvit.run --config cfg.json --mode main

Structure follows experiments/recall707/runtime/solar_loss/runner.py (global-batch sampler, AMP retry,
rank-gathered fullstate), with: DDP binary-diffusion objective (BCE + per-crop Dice), optional tuned
backbone blocks (llrd_dp), fixed epochs / last checkpoint, report-only validation with the strict 1-1 and
legacy scorers at threshold .5, min area 128.
"""
import argparse
import copy
import csv
import datetime
import json
import math
import os
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as Distributed
from torch.utils.data import DataLoader

from solar_loss.common import GlobalBatches, rng_state, restore_rng, seed_all, atomic, exact, cpu_budget
from u0_multihead.data import Crops, worker_init
from u0_multihead.model import semantic_loss
from ddpvit.model import DDPViTModel, HEAD_LR, build_optimizer, make_model, trainable_parameters
from ddpvit.scoring import StrictContext, score_names, summarize, write_rows, assert_expected

EXPECTED_VAL = dict(sources=566, records=923, gt=6581)       # folds 1-4 when training fold 0


def expected_val(args):
    """Fold 0 keeps the exact audited totals; other folds check sources and records (gt is only logged)."""
    if args.get('split209'):
        return dict(sources=498, records=None, gt=None)
    if args['fold'] == 0:
        return EXPECTED_VAL
    folds = json.loads((Path(__file__).resolve().parents[1] / 'expected_folds.json').read_text())
    return dict(sources=707 - len(folds['folds'][args['fold']]['train_sources']),
                records=folds['annotator_records'] - folds['folds'][args['fold']]['annotator_records'], gt=None)
GATE_IOU = 0.85
RESUME_REL_TOL = 2e-4


def marker(name, **kw):
    print(name, json.dumps(kw, sort_keys=True, default=str), flush=True)


# ---------------------------------------------------------------------------------------------------------
# one optimisation step on a global batch (microbatches, AMP with retry, rank-averaged gradients)
# ---------------------------------------------------------------------------------------------------------
def distributed_step(wrapped, x, y, opt, sched, scaler, micro, global_n):
    model, device, world = wrapped.module, next(wrapped.module.parameters()).device, dist.get_world_size()
    params = trainable_parameters(model)
    saved = rng_state()
    for attempt in range(8):
        if attempt:
            restore_rng(saved)
        opt.zero_grad(set_to_none=True)
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        for start in range(0, len(y), micro):
            xx = x[start:start + micro].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            yy = y[start:start + micro].to(device, non_blocking=True)
            last = start + micro >= len(y)
            with (nullcontext() if last else wrapped.no_sync()):
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    logits = wrapped(xx, yy)
                loss, bce, dice = semantic_loss(logits, yy)
                scaler.scale(loss * (world * len(yy) / global_n)).backward()
            totals += torch.stack([loss.detach(), bce.detach(), dice.detach()]).double() * len(yy)
        scaler.unscale_(opt)
        norm = nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=False)
        bad = torch.tensor(int(not torch.isfinite(norm)), device=device)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bad.item():
            scaler.update(new_scale=scaler.get_scale() / 2)
            continue
        lr = opt.param_groups[0]['lr']
        scaler.step(opt)
        scaler.update()
        sched.step()
        dist.all_reduce(totals)
        loss_sum, bce_sum, dice_sum = (totals / global_n).cpu().tolist()
        return dict(loss=loss_sum, bce=bce_sum, dice=dice_sum, lr=lr, grad_norm=float(norm),
                    amp_scale=scaler.get_scale(), amp_retries=attempt)
    raise FloatingPointError('8 FP16 retries failed; no sample or optimizer step was skipped')


def build_state(args, device):
    model = make_model(args['backbone'], args['arm'], seed=args['seed']).to(device).to(memory_format=torch.channels_last)
    return model


def init_process(args):
    rank, world, local = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('Select Kaggle GPU T4 x2: exactly 2 GPUs required')
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    return rank, world, device


def load_data(args, world):
    base = Path(args['base'])
    if args.get('split209'):                     # original DDP12 protocol: train209 crops (original generator) / val498
        sp = json.loads((Path(__file__).resolve().parents[1] / 'split209.json').read_text())
        train, val = set(sp['train209']), set(sp['val498'])
        assert len(train) == 209 and len(val) == 498 and not train & val and not {s[:8] for s in train} & {s[:8] for s in val}
        dataset = Crops(base / 'magfilo_crops_768', base / 'train.jsonl', args['seed'], augment=True)
        assert {r['source_file'] for r in dataset.records} == train
        return base, None, dict(crops=len(dataset)), sorted(train), sorted(val), dataset
    assignment = json.loads((base / 'fold_assignment.json').read_text())
    expected = json.loads((Path(__file__).resolve().parents[1] / 'expected_folds.json').read_text())
    assert assignment == expected and assignment['sources'] == 707
    info = assignment['folds'][args['fold']]
    train, val = set(info['train_sources']), set(info['unseen_train_sources'])
    assert not train & val and len(train | val) == 707 and not {s[:8] for s in train} & {s[:8] for s in val}
    dataset = Crops(base / 'magfilo_crops_768', base / 'train.jsonl', args['seed'], augment=True)
    assert {r['source_file'] for r in dataset.records} == train and len(dataset) == info['crops']
    return base, assignment, info, sorted(train), sorted(val), dataset


# ---------------------------------------------------------------------------------------------------------
# evaluation (report only): rank-sharded inference + scoring, rows gathered on rank 0
# ---------------------------------------------------------------------------------------------------------
def evaluate_global(model, names, context, workers, label, out, rank, world, device):
    model.eval()
    started = time.time()
    local = score_names(model, names[rank::world], context, workers=workers, tile_batch=1,
                        progress=lambda d, n: print('EVAL_PROGRESS', label, rank, d, n, flush=True) if rank == 0 else None)
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    result = None
    if rank == 0:
        rows = [r for shard in gathered for r in shard]
        assert {r['source_file'] for r in rows} == set(names)
        result = summarize(rows)
        result.update(label=label, seconds=time.time() - started)
        write_rows(rows, Path(out) / f'rows_{label}.jsonl.gz')
    shared = [result]
    dist.broadcast_object_list(shared, src=0)
    dist.barrier()
    return shared[0]


# ---------------------------------------------------------------------------------------------------------
# fullstate checkpoint (rank-gathered RNG + scalers), written atomically on rank 0
# ---------------------------------------------------------------------------------------------------------
def save_fullstate(model, opt, sched, scaler, config, epoch, offset, update, history, out, records, split):
    rank, world = dist.get_rank(), dist.get_world_size()
    local = dict(rng=rng_state(), scaler=scaler.state_dict())
    states = [None] * world if rank == 0 else None
    dist.gather_object(local, states, dst=0)
    if rank == 0:
        value = dict(version=1, config=config, model=model.trainable_state(), optimizer=opt.state_dict(),
                     scheduler=sched.state_dict(), rank_states=states, world_size=world, epoch=epoch,
                     offset=offset, update=update, history=history, train_records=records, split=split)
        atomic(value, Path(out) / 'last_training.pt', True)
        atomic(history, Path(out) / 'history.json')
    dist.barrier()


def load_fullstate(path, model, opt, sched, scaler, config, rank, records, split):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('arm', 'fold', 'epochs', 'seed', 'global_batch', 'microbatch', 'warmup', 'world_size'):
        if ck['config'][key] != config[key]:
            raise ValueError('Resume mismatch: ' + key)
    assert ck['train_records'] == records and ck['split'] == split
    model.load_trainable_state(ck['model'])
    opt.load_state_dict(ck['optimizer'])
    sched.load_state_dict(ck['scheduler'])
    scaler.load_state_dict(ck['rank_states'][rank]['scaler'])
    restore_rng(ck['rank_states'][rank]['rng'])
    return ck


# ---------------------------------------------------------------------------------------------------------
# GPU gate
# ---------------------------------------------------------------------------------------------------------
def gate(args, rank, world, device):
    out = Path(args['out'])
    gate_dir = out / 'gate'
    gate_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    marker('GPU_SMOKE_GATE', stage='start', arm=args['arm'], rank=rank, device=torch.cuda.get_device_name(device))
    base, assignment, info, train_sources, val_sources, dataset = load_data(args, world)
    plain = Crops(base / 'magfilo_crops_768', base / 'train.jsonl', args['seed'], augment=False)
    step = max(1, len(plain) // 16)
    indices = list(range(0, len(plain), step))[:16]
    assert len(indices) == 16
    xs = torch.stack([plain[i][0] for i in indices])
    ys = torch.stack([plain[i][1] for i in indices])
    report = dict(arm=args['arm'], crops=indices, checks={})

    seed_all(args['seed'] + rank)
    model = build_state(args, device)
    n_params = dict(total=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in trainable_parameters(model)))

    # (1) native 768 forward/backward finite, largest microbatch that fits on every rank
    chosen, attempts = None, []
    x8, y8 = xs[:8][rank::world].to(device), ys[:8][rank::world].to(device)
    for micro in [m for m in (4, 2, 1) if m <= args['microbatch']]:
        ok, msg = True, None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(x8[:micro].contiguous(memory_format=torch.channels_last), y8[:micro])
            loss, _, _ = semantic_loss(logits, y8[:micro])
            loss.backward()
            ok = bool(torch.isfinite(loss)) and all(torch.isfinite(p.grad).all() for p in trainable_parameters(model) if p.grad is not None)
            msg = 'nonfinite' if not ok else None
        except torch.cuda.OutOfMemoryError:
            ok, msg = False, 'OOM'
        model.zero_grad(set_to_none=True)
        flag = torch.tensor(int(ok), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        attempts.append(dict(microbatch=micro, rank_ok=ok, error=msg, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2)))
        if flag.item():
            chosen = micro
            break
    report['checks']['native_forward_backward'] = dict(passed=chosen is not None, microbatch=chosen, attempts=attempts, parameters=n_params)
    marker('GPU_SMOKE_GATE', stage='native', passed=chosen is not None, microbatch=chosen, attempts=attempts, params=n_params)
    if chosen is None:
        return finish_gate(report, gate_dir, rank)

    # (2) overfit16: 16 fixed crops, no augmentation, deterministic 3-step inference IoU > .85
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    seed_all(args['seed'] + rank)
    model = build_state(args, device)
    cap = args['gate_cap_updates']
    opt, sched = build_optimizer(model, total_updates=cap, warmup=50)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    wrapped = Distributed(model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True)

    def batch(update):
        half = (update % 2) * 8
        return xs[half:half + 8][rank::world], ys[half:half + 8][rank::world]

    def overfit_iou():
        model.eval()
        inter = torch.zeros(2, device=device, dtype=torch.float64)
        with torch.inference_mode():
            mine_x, mine_y = xs[rank::world], ys[rank::world]          # 8 of the 16 crops on each rank
            for s in range(0, len(mine_x), 2):
                xx = mine_x[s:s + 2].to(device)
                yy = mine_y[s:s + 2].to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    z = model.heads['ddp'](model.features(xx.contiguous(memory_format=torch.channels_last)), xx)
                p, g = (z.float() > 0), yy > 0
                inter += torch.stack([(p & g).sum(), (p | g).sum()]).double()
        dist.all_reduce(inter)
        model.train()
        return float(inter[0] / inter[1].clamp_min(1))

    history, reached, iou = [], None, 0.
    model.train()
    for update in range(1, cap + 1):
        xb, yb = batch(update)
        m = distributed_step(wrapped, xb, yb, opt, sched, scaler, chosen, 8)
        if update % 50 == 0 or update == cap:
            iou = overfit_iou()
            history.append(dict(update=update, iou=iou, loss=m['loss']))
            marker('GPU_SMOKE_GATE', stage='overfit16', update=update, iou=round(iou, 4), loss=round(m['loss'], 4))
            if iou > GATE_IOU:
                reached = update
                break
    report['checks']['overfit16'] = dict(passed=reached is not None, iou=iou, threshold=GATE_IOU, reached_update=reached,
                                         cap=cap, history=history)
    if reached is None:
        return finish_gate(report, gate_dir, rank)

    # (3) inference determinism: two identical runs, bitwise equal logits (checked on every rank)
    model.eval()
    xx = xs[:4][rank::world].to(device)
    with torch.inference_mode():
        a = model.heads['ddp'](model.features(xx), xx)
        b = model.heads['ddp'](model.features(xx), xx)
    same = torch.tensor(int(torch.equal(a, b)), device=device)
    dist.all_reduce(same, op=dist.ReduceOp.MIN)
    report['checks']['inference_deterministic'] = dict(passed=bool(same.item()))
    marker('GPU_SMOKE_GATE', stage='determinism', passed=bool(same.item()))

    # (4) exact resume: save fullstate, run 10 updates (A), reload everything, run the same 10 again (B)
    model.train()
    saved = dict(model=model.trainable_state(), opt=copy.deepcopy(opt.state_dict()), sched=copy.deepcopy(sched.state_dict()),
                 scaler=copy.deepcopy(scaler.state_dict()), rng=rng_state())
    torch.save(saved, gate_dir / f'fullstate_rank{rank}.pt')
    loaded_back = torch.load(gate_dir / f'fullstate_rank{rank}.pt', map_location='cpu', weights_only=False)
    exact(saved['model'], loaded_back['model'])             # the on-disk file really round-trips

    def run_ten(start_update):
        losses = []
        for k in range(10):
            xb, yb = batch(start_update + k)
            losses.append(distributed_step(wrapped, xb, yb, opt, sched, scaler, chosen, 8)['loss'])
        return losses

    first = run_ten(reached + 1)
    model.load_trainable_state(loaded_back['model'])
    opt.load_state_dict(loaded_back['opt'])
    sched.load_state_dict(loaded_back['sched'])
    scaler.load_state_dict(loaded_back['scaler'])
    restore_rng(loaded_back['rng'])
    second = run_ten(reached + 1)
    rel = max(abs(p - q) / max(abs(p), 1e-12) for p, q in zip(first, second))
    report['checks']['exact_resume'] = dict(passed=rel <= RESUME_REL_TOL, max_rel_diff=rel, tolerance=RESUME_REL_TOL,
                                            bitwise_equal=first == second, losses_a=first, losses_b=second,
                                            note='tolerance covers non-deterministic CUDA atomics (grid_sample backward)')
    marker('GPU_SMOKE_GATE', stage='resume', passed=rel <= RESUME_REL_TOL, max_rel_diff=rel, bitwise_equal=first == second)
    (gate_dir / f'fullstate_rank{rank}.pt').unlink()
    return finish_gate(report, gate_dir, rank)


def finish_gate(report, gate_dir, rank):
    report['passed'] = all(c['passed'] for c in report['checks'].values()) and len(report['checks']) == 4
    dist.barrier()
    if rank == 0:
        atomic(report, gate_dir / 'verification.json')
    marker('GPU_SMOKE_GATE', stage='result', passed=report['passed'], checks={k: v['passed'] for k, v in report['checks'].items()})
    dist.barrier()
    if not report['passed']:
        raise SystemExit(3)


# ---------------------------------------------------------------------------------------------------------
# main training
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------
# optional checkpoint selection on a fixed quick subset (off unless config has select_epochs)
# ---------------------------------------------------------------------------------------------------------
def update_best(out, model, epoch, update, quick_pq, config):
    """Keep the checkpoint with the highest quick-subset strict PQ (earliest wins ties). Returns True when updated."""
    out = Path(out)
    info = out / 'best.json'
    prev = json.loads(info.read_text()) if info.exists() else None
    if prev is not None and not quick_pq > prev['quick_pq']:
        return False
    atomic(dict(trainable=model.trainable_state(), epoch=epoch, update=update, quick_pq=quick_pq, config=config), out / 'best.pt', True)
    atomic(dict(epoch=epoch, update=update, quick_pq=quick_pq), info)
    return True


def best_info(out):
    info = Path(out) / 'best.json'
    return json.loads(info.read_text()) if info.exists() else None


def evaluate_best(model, args, out, rank, world, device, context, val_sources, workers, last_epoch):
    """After the last epoch: full-val evaluation of the selected checkpoint (the last one is already evaluated)."""
    info = [best_info(out) if rank == 0 else None]
    dist.broadcast_object_list(info, src=0)
    info = info[0]
    if info is None:
        return None
    if info['epoch'] == last_epoch:
        if rank == 0:
            marker('BEST_IS_LAST', epoch=info['epoch'], quick_pq=info['quick_pq'])
        return info
    ck = torch.load(Path(out) / 'best.pt', map_location='cpu', weights_only=False)
    assert ck['epoch'] == info['epoch'] and ck['update'] == info['update']
    model.load_trainable_state(ck['trainable'])
    del ck
    res = evaluate_global(model, val_sources, context, max(1, workers), f"best_epoch{info['epoch']}_val", out, rank, world, device)
    if rank == 0:
        assert_expected(res, **expected_val(args))
        marker('EVAL_SUMMARY', epoch=info['epoch'], split='val_best', strict=res['strict'], legacy=res['legacy'],
               sources=res['sources'], records=res['records'], gt=res['gt'], quick_pq=info['quick_pq'])
        atomic(dict(selected=info, val=res), Path(out) / 'best_selected.json')
    return info


def main_train(args, rank, world, device):
    out = Path(args['out'])
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gate_file = out / 'gate' / 'verification.json'
    if not (gate_file.is_file() and json.loads(gate_file.read_text())['passed']):
        raise RuntimeError('GPU gate has not passed for this arm: refusing to start main training')
    seed_all(args['seed'] + rank)
    base, assignment, info, train_sources, val_sources, dataset = load_data(args, world)
    steps_per_epoch = math.ceil(len(dataset) / args['global_batch'])
    total = steps_per_epoch * args['epochs']
    split = dict(train=train_sources, validation=val_sources, protocol=('train209 original crops / validate val498; report-only' if args.get('split209') else f"train fold {args['fold']} / validate the other 4 folds; report-only"))
    model = build_state(args, device)
    micro = int(json.loads(gate_file.read_text())['checks']['native_forward_backward']['microbatch'])
    args = dict(args, microbatch=micro)
    if args.get('init_from'):                       # continue training: weights only, fresh optimizer and schedule
        init = torch.load(args['init_from'], map_location='cpu', weights_only=False)
        model.load_trainable_state(init['trainable'])                   # strict key set
        marker('INIT_FROM', path=str(args['init_from']), source_update=init.get('update'), rank=rank)
        del init
    opt, sched = build_optimizer(model, total, args['warmup'], float(args.get('head_lr', HEAD_LR)))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    wrapped = Distributed(model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True)
    budget = cpu_budget()
    workers = budget // world + int(rank < budget % world)
    sampler = GlobalBatches(len(dataset), args['global_batch'], rank, world, args['seed'], 1)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                        multiprocessing_context='spawn' if workers else None, worker_init_fn=worker_init,
                        generator=torch.Generator().manual_seed(args['seed'] + 1000 + rank))
    context = StrictContext(base)
    config = dict(args, world_size=world, steps_per_epoch=steps_per_epoch, total_updates=total,
                  trainable_parameters=sum(p.numel() for p in trainable_parameters(model)))
    if rank == 0:
        atomic(config, out / 'config.json')

    epoch, offset, update, history = 1, 0, 0, []
    resume = Path(args['resume']) if args.get('resume') else out / 'last_training.pt'
    if resume.exists():
        ck = load_fullstate(resume, model, opt, sched, scaler, config, rank, dataset.records, split)
        epoch, offset, update, history = ck['epoch'], ck['offset'], ck['update'], ck['history']
        assert update == (epoch - 1) * steps_per_epoch + math.ceil(offset / args['global_batch'])
        marker('RESUMED', epoch=epoch, update=update, offset=offset)
        if rank == 0 and (out / 'train.csv').exists():
            with (out / 'train.csv').open() as f:
                reader = csv.DictReader(f)
                fields, kept = reader.fieldnames, [r for r in reader if int(r['update']) <= update]
            with (out / 'train.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(kept)
        del ck
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    marker('MAIN_START', arm=args['arm'], rank=rank, train_crops=len(dataset), steps_per_epoch=steps_per_epoch,
           total_updates=total, microbatch=micro, trainable=config['trainable_parameters'], resumed_update=update)

    while epoch <= args['epochs']:
        sampler.epoch, sampler.offset = epoch, offset
        wrapped.train()
        began = time.time()
        for x, y, _ in loader:
            t0 = time.perf_counter()
            global_n = min(args['global_batch'], len(dataset) - offset)
            metrics = distributed_step(wrapped, x, y, opt, sched, scaler, micro, global_n)
            update += 1
            offset += global_n
            metrics['step_seconds'] = time.perf_counter() - t0
            if rank == 0:
                row = dict(epoch=epoch, update=update, **metrics)
                path = out / 'train.csv'
                new = not path.exists()
                with path.open('a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)
                if update % 20 == 0:
                    marker('TRAIN_UPDATE', **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
                    atomic(dict(state='training', epoch=epoch, update=update, total_updates=total), out / 'status.json')
            interrupted = torch.tensor(int(stop[0] or time.time() >= args['deadline'] - 240), device=device)
            dist.all_reduce(interrupted, op=dist.ReduceOp.MAX)
            if update % args['save_every'] == 0 or offset == len(dataset) or interrupted.item():
                save_fullstate(model, opt, sched, scaler, config, epoch, offset, update, history, out, dataset.records, split)
            if interrupted.item():
                if rank == 0:
                    atomic(dict(state='stopped_resumable', epoch=epoch, update=update), out / 'status.json')
                marker('SOFT_STOP', epoch=epoch, update=update)
                return False
        train_seconds = time.time() - began
        entry = dict(epoch=epoch, update=update, train_seconds=train_seconds)
        select_epochs = set(args.get('select_epochs', []))
        if epoch in args['eval_epochs'] or epoch in select_epochs:
            if time.time() > args['deadline'] - 1800:
                save_fullstate(model, opt, sched, scaler, config, epoch + 1, 0, update, history, out, dataset.records, split)
                if rank == 0:
                    atomic(dict(state='stopped_resumable', phase='before_eval', epoch=epoch, update=update), out / 'status.json')
                marker('SOFT_STOP', phase='before_eval', epoch=epoch, update=update)
                return False
            if epoch in select_epochs:
                quick = evaluate_global(model, args['select_quick'], context, max(1, workers), f'epoch{epoch}_quick', out, rank, world, device)
                if rank == 0:
                    entry['quick'] = quick
                    improved = update_best(out, model, epoch, update, quick['strict']['pq'], config)
                    marker('QUICK_EVAL', epoch=epoch, update=update, sources=quick['sources'], records=quick['records'],
                           strict=quick['strict'], legacy=quick['legacy'], best_updated=improved)
                dist.barrier()
            if epoch in args['eval_epochs']:
                val = evaluate_global(model, val_sources, context, max(1, workers), f'epoch{epoch}_val', out, rank, world, device)
                if rank == 0:
                    assert_expected(val, **expected_val(args))
                    entry['val'] = val
                    marker('EVAL_SUMMARY', epoch=epoch, split='val', strict=val['strict'], legacy=val['legacy'],
                           sources=val['sources'], records=val['records'], gt=val['gt'])
        if epoch == args['epochs']:
            tr = evaluate_global(model, train_sources, context, max(1, workers), f'epoch{epoch}_train', out, rank, world, device)
            if rank == 0:
                entry['train'] = tr
                marker('EVAL_SUMMARY', epoch=epoch, split='train', strict=tr['strict'], legacy=tr['legacy'],
                       sources=tr['sources'], records=tr['records'], gt=tr['gt'])
        history.append(entry)
        epoch += 1
        offset = 0
        model.train()
        save_fullstate(model, opt, sched, scaler, config, epoch, offset, update, history, out, dataset.records, split)

    assert update == total, (update, total)
    final_verification(model, args, config, out, rank, device, history, update)
    if args.get('select_epochs'):
        evaluate_best(model, args, out, rank, world, device, context, val_sources, workers, args['epochs'])
    return True


def final_verification(model, args, config, out, rank, device, history, update):
    """Write last.pt (adapter + head + tuned blocks), reload it strictly into a FRESH model, require equal outputs."""
    model.eval()
    torch.set_grad_enabled(False)
    probe = Crops(Path(args['base']) / 'magfilo_crops_768', Path(args['base']) / 'train.jsonl', args['seed'], augment=False)[0][0][None].to(device)
    live = model.heads['ddp'](model.features(probe), probe)
    if rank == 0:
        atomic(dict(trainable=model.trainable_state(), config=config, epochs=args['epochs'], update=update), out / 'last.pt', True)
    dist.barrier()
    ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    fresh = make_model(args['backbone'], args['arm'], seed=args['seed'] + 1).to(device).eval()
    fresh.load_trainable_state(ck['trainable'])                      # strict: exact key set required
    again = fresh.heads['ddp'](fresh.features(probe), probe)
    # cudnn.benchmark may pick different (equivalent) conv algorithms for two model instances, so allow tiny float noise
    # but require the same foreground decision on >99.9% of pixels.
    ok = bool(torch.allclose(live, again, atol=1e-3, rtol=1e-3)) and float(((live > 0) == (again > 0)).float().mean()) > 0.999
    flag = torch.tensor(int(ok), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise AssertionError('last.pt strict reload does not reproduce the live model outputs')
    torch.set_grad_enabled(True)
    if rank == 0:
        atomic(dict(passed=True, update=update, epochs=args['epochs'], arm=args['arm'],
                    last_checkpoint_strict_reload_equal_outputs=True, threshold=.5, min_area=128), out / 'verification.json')
        atomic(dict(state='complete', update=update, arm=args['arm']), out / 'status.json')
        marker('COMPLETE_VERIFIED', arm=args['arm'], update=update, final=history[-1])
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=('gate', 'main'), required=True)
    cli = parser.parse_args()
    args = json.loads(Path(cli.config).read_text())
    rank, world, device = init_process(args)
    try:
        if cli.mode == 'gate':
            gate(args, rank, world, device)
        else:
            finished = main_train(args, rank, world, device)
            if not finished:
                raise SystemExit(0)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
