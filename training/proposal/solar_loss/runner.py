"""Frozen DINOv3 ViT-S+ recall proposal: train one fold, unseen four."""

import argparse, csv, json, math, os, signal, time, datetime
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as Distributed
import torch.distributed as dist
from .common import LOSSES, loss_terms, cpu_budget, GlobalBatches, rng_state, restore_rng, seed_all, atomic, exact
from u0_multihead.data import Crops, worker_init
from u0_multihead.model import lr_factor
from proposal import Context, summarize, selection_key, THRESHOLDS, LOSSES, loss_terms, TrainModel as ProposalModel
from u0_multihead.evaluation import evaluate as local_evaluate

TrainModel = ProposalModel


def distributed_step(wrapped, x, y, optimizer, scheduler, scaler, micro, arm, global_n):
    """DDP averages rank gradients: world*micro/global_n preserves the global crop mean."""
    model = wrapped.module
    device = next(model.heads.parameters()).device
    world = dist.get_world_size()
    saved_rng = rng_state()
    for attempt in range(8):
        if attempt:
            restore_rng(saved_rng)
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(4, device=device, dtype=torch.float64)
        for start in range(0, len(y), micro):
            xx = x[start : start + micro].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            yy = y[start : start + micro].to(device, non_blocking=True)
            last = start + micro >= len(y)
            sync = nullcontext() if last else wrapped.no_sync()
            with sync:
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    logits = wrapped(xx, yy)
                loss, terms = loss_terms(logits, yy, arm)
                factor = world * len(yy) / global_n
                scaler.scale(loss * factor).backward()
            totals += torch.stack(
                [loss.detach(), terms['bce'].detach(), terms['dice'].detach(), terms['common_bce_dice'].detach()]
            ).double() * len(yy)
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(model.heads.parameters(), 1.0, error_if_nonfinite=False)
        bad = torch.tensor(int(not torch.isfinite(norm)), device=device)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if bad.item():
            if not scaler.is_enabled():
                raise FloatingPointError('Nonfinite CPU gradients')
            scaler.update(new_scale=scaler.get_scale() / 2)
            continue  # Same samples/cursor, same RNG and no LR step after AMP overflow.
        lr = optimizer.param_groups[0]['lr']
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        dist.all_reduce(totals)
        return dict(zip(['objective', 'bce', 'dice', 'common_bce_dice'], (totals / global_n).cpu().tolist())), dict(
            lr=lr, grad_norm=float(norm), amp_scale=scaler.get_scale(), amp_retries=attempt
        )
    raise FloatingPointError('8 FP16 retries failed; no samples or optimizer steps were skipped')


def combine_rows(rows, thresholds, selected=None):
    rows.sort(key=lambda r: (r['source_file'], str(r['record']), r['threshold']))
    sweep = [summarize(rows, t) for t in thresholds]
    chosen = max(sweep, key=selection_key) if selected is None else next(r for r in sweep if r['threshold'] == selected)
    images = defaultdict(list)
    for r in rows:
        if r['threshold'] == thresholds[0]:
            images[r['source_file']].append(r['inference_loss'])
    loss = {
        k: float(np.mean([np.mean([v[k] for v in rs]) for rs in images.values()]))
        for k in ['bce', 'soft_dice_loss', 'loss']
    }
    return dict(selected=chosen, sweep=sweep, inference_loss=loss)


def evaluate_global(model, names, context, workers, thresholds=THRESHOLDS, selected=None, out=None, label='validation'):
    rank, world = dist.get_rank(), dist.get_world_size()
    model.eval()
    started = time.time()
    local = []
    stopped = False
    # Bounded shards allow both ranks to stop together during validation.
    for start in range(0, len(names), world * 2):
        flag = torch.tensor(
            int(time.time() >= float(os.environ['SOLAR_DEADLINE']) - 180), device=next(model.heads.parameters()).device
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            stopped = True
            break
        chunk = names[start : start + world * 2][rank::world]
        if chunk:
            _, rs = local_evaluate(
                model,
                chunk,
                context,
                thresholds_by_head={'ddp': list(thresholds)},
                workers=max(1, workers),
                tile_batch=1,
            )
            local.extend(rs['ddp'])
        if out:
            atomic(
                dict(state=label, rank=rank, completed=min(start + world * 2, len(names)), total=len(names)),
                out / f'rank{rank}_status.json',
            )
        if rank == 0:
            print('EVAL_PROGRESS', label, min(start + world * 2, len(names)), len(names), flush=True)
    if stopped:
        return None, None
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    result, rows = None, None
    if rank == 0:
        rows = [r for shard in gathered for r in shard]
        assert {r['source_file'] for r in rows} == set(names)
        for t in thresholds:
            rr = [r for r in rows if r['threshold'] == t]
            assert len({(r['source_file'], str(r['record'])) for r in rr}) == len(rr)
        result = combine_rows(rows, list(thresholds), selected)
        result['seconds'] = time.time() - started
    shared = [result]
    dist.broadcast_object_list(shared, src=0)
    return shared[0], rows


def checkpoint(model, opt, sched, scaler, config, epoch, offset, update, best, history, out, records, split):
    rank, world = dist.get_rank(), dist.get_world_size()
    local = dict(rng=rng_state(), scaler=scaler.state_dict())
    states = [None] * world if rank == 0 else None
    dist.gather_object(local, states, dst=0)
    if rank == 0:
        assert update == (epoch - 1) * math.ceil(len(records) / config['global_batch']) + math.ceil(
            offset / config['global_batch']
        )
        value = dict(
            version=1,
            config=config,
            head={k: v.detach().cpu().clone() for k, v in model.heads['ddp'].state_dict().items()},
            optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),
            rank_states=states,
            world_size=world,
            epoch=epoch,
            offset=offset,
            update=update,
            best=best,
            history=history,
            train_records=records,
            split=split,
        )
        atomic(value, out / 'last_training.pt', True)
        atomic(history, out / 'history.json')
        if best:
            atomic(
                dict(head=best['weights'], epoch=best['epoch'], selection=best['selection'], config=config),
                out / 'best_head.pt',
                True,
            )
    dist.barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    cli = p.parse_args()
    args = json.loads(Path(cli.config).read_text())
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError('Select Kaggle GPU T4 x2; exactly2 GPUs required')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
    try:
        run(args, cli, rank, world, device)
    finally:
        dist.destroy_process_group()


def run(args, cli, rank, world, device):
    seed_all(args['seed'] + rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    base, out = Path(args['base']), Path(args['out'])
    out.mkdir(parents=True, exist_ok=True)
    os.environ['SOLAR_DEADLINE'] = str(args['deadline'])
    if args['loss'] not in LOSSES:
        raise ValueError('Unknown loss')
    if args['global_batch'] != 8:
        raise ValueError('This protocol fixes global batch8')
    assignment = json.loads((base / 'fold_assignment.json').read_text())
    expected = json.loads((Path(__file__).resolve().parents[1] / 'expected_folds.json').read_text())
    assert assignment == expected and assignment['sources'] == 707
    assert 0 <= args['fold'] < 5
    fold_info = assignment['folds'][args['fold']]
    allowed = set(fold_info['train_sources'])
    unseen = set(fold_info['unseen_train_sources'])
    assert not allowed & unseen and len(allowed | unseen) == 707
    assert not {s[:8] for s in allowed} & {s[:8] for s in unseen}
    split = dict(
        train=sorted(allowed),
        quick=sorted(unseen),
        validation=sorted(unseen),
        protocol='707 train-one/val-four; no quick/audit selection',
    )
    dataset = Crops(base / 'magfilo_crops_768', base / 'train.jsonl', args['seed'], augment=False)
    assert {r['source_file'] for r in dataset.records} == allowed
    assert len(dataset) == fold_info['crops'] and len(dataset) % 8 != 1
    args['steps_per_epoch'] = math.ceil(len(dataset) / 8)
    if rank == 0:
        atomic(assignment, out / 'fold_assignment.json')

    verified = json.loads((base / 'staging_verification.json').read_text())
    assert verified['passed']
    model = (
        TrainModel(args['backbone'], variant='refine', seed=args['seed'])
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    assert sum(p.numel() for p in model.heads.parameters()) < 1000000
    budget = cpu_budget()
    workers = budget // world + int(rank < budget % world)
    sampler = GlobalBatches(len(dataset), 8, rank, world, args['seed'], 1)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        multiprocessing_context='spawn' if workers else None,
        worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(args['seed'] + 1000 + rank),
    )
    hardware = dict(
        rank=rank,
        device=torch.cuda.get_device_name(device),
        capability=torch.cuda.get_device_capability(device),
        memory_gib=torch.cuda.get_device_properties(device).total_memory / 2**30,
        cpu_budget=budget,
        data_workers=workers,
        eval_workers=max(1, workers),
        ram_available_gib=os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 2**30,
        precision='FP32 backbone/FP16 head/FP32 loss+grid_sample',
    )
    atomic(hardware, out / f'hardware_rank{rank}.json')
    context = Context(base)
    micro = args['microbatch']
    wrapped = Distributed(model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True)
    config = dict(
        args,
        cpu_budget=budget,
        world_size=world,
        actual_microbatch=micro,
        model=args['model_kind'] + ' frozen DINOv3; lightweight64 fusion + RGB detail',
        selection='fixed epoch4; fixed proposal threshold0.05; heldout-four metrics report-only, no selection',
        recall_target=0.95,
        precision='FP32 backbone + FP16 head + GradScaler; TF32 off',
        cursor='global consumed samples, no sampler padding',
        loss_definition=LOSSES[args['loss']],
    )
    atomic(config, out / f'config_rank{rank}.json')
    opt = torch.optim.AdamW(model.heads.parameters(), lr=args['lr'], weight_decay=0.01)
    total = args['steps_per_epoch'] * args['epochs']
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: lr_factor(n, total, args['warmup']))
    scaler = torch.amp.GradScaler('cuda')
    epoch, offset, update, best, history = 1, 0, 0, None, []
    resume = Path(args['resume']) if args.get('resume') else out / 'last_training.pt'
    if resume.exists():
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        for k in [
            'fold',
            'loss',
            'epochs',
            'lr',
            'warmup',
            'seed',
            'global_batch',
            'actual_microbatch',
            'world_size',
            'model',
            'precision',
        ]:
            if ck['config'][k] != config[k]:
                raise ValueError('Resume mismatch ' + k)
        assert ck['world_size'] == world and ck['train_records'] == dataset.records and ck['split'] == split
        model.heads['ddp'].load_state_dict(ck['head'])
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['rank_states'][rank]['scaler'])
        epoch, offset, update, best, history = (ck[k] for k in ['epoch', 'offset', 'update', 'best', 'history'])
        restore_rng(ck['rank_states'][rank]['rng'])
        assert update == (epoch - 1) * args['steps_per_epoch'] + math.ceil(offset / 8)
        if rank == 0:
            csvpath = resume.parent / 'train.csv'
            if csvpath.exists():
                with csvpath.open() as f:
                    reader = csv.DictReader(f)
                    fields = reader.fieldnames
                    kept = [r for r in reader if int(r['update']) <= update]
                with (out / 'train.csv').open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    w.writerows(kept)
        del ck
    elif args.get('resume'):
        raise FileNotFoundError(resume)
    if rank == 0:
        atomic(config, out / 'config.json')
    stop = [False]

    def request_stop(*_):
        stop[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save():
        checkpoint(model, opt, sched, scaler, config, epoch, offset, update, best, history, out, dataset.records, split)

    # Materialize history/export in a NEW output directory even when the resumed
    # checkpoint has already finished all epochs and only final reports remain.
    save()
    if False:  # No expensive untrained baseline sweep.
        initial, _ = evaluate_global(model, split['quick'], context, max(1, workers), out=out, label='initial_quick')
        if rank == 0:
            atomic(initial, out / 'initial_quick.json')
    while epoch <= args['epochs']:
        sampler.epoch, sampler.offset = epoch, offset
        began = time.time()
        wrapped.train()
        ready_for_batch = time.perf_counter()
        for x, y, _ in loader:
            step_started = time.perf_counter()
            data_wait = step_started - ready_for_batch
            global_n = min(8, len(dataset) - offset)
            metrics, timing = distributed_step(wrapped, x, y, opt, sched, scaler, micro, args['loss'], global_n)
            timing['step_seconds'] = time.perf_counter() - step_started
            timing['crops_per_second'] = global_n / timing['step_seconds']
            timing['data_wait_seconds'] = data_wait
            update += 1
            offset += global_n
            if rank == 0:
                row = dict(epoch=epoch, update=update, **metrics, **timing)
                path = out / 'train.csv'
                exists = path.exists()
                with path.open('a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if not exists:
                        w.writeheader()
                    w.writerow(row)
                if update % 10 == 0:
                    atomic(
                        dict(
                            state='training',
                            epoch=epoch,
                            update=update,
                            total_updates=total,
                            offset=offset,
                            metrics=row,
                        ),
                        out / 'status.json',
                    )
                    print(json.dumps(row), flush=True)
            interrupted = torch.tensor(int(stop[0] or time.time() >= args['deadline'] - 180), device=device)
            dist.all_reduce(interrupted, op=dist.ReduceOp.MAX)
            if update % 20 == 0 or offset == len(dataset) or interrupted.item():
                save()
            if interrupted.item():
                if rank == 0:
                    atomic(dict(state='stopped_resumable', epoch=epoch, update=update), out / 'status.json')
                return
            ready_for_batch = time.perf_counter()
        train_seconds = time.time() - began
        if epoch == args['epochs']:
            os.environ['SOLAR_EXPORT_PREDICTIONS'] = str(out / 'heldout_predictions')
        result, rows = evaluate_global(
            model,
            split['validation'],
            context,
            max(1, workers),
            thresholds=(0.05,),
            selected=0.05,
            out=out,
            label=f'epoch{epoch}_val4_report',
        )
        os.environ.pop('SOLAR_EXPORT_PREDICTIONS', None)
        if result is None:
            save()
            if rank == 0:
                atomic(
                    dict(state='stopped_resumable', phase='quick_eval', epoch=epoch, update=update), out / 'status.json'
                )
            return
        if rank == 0:
            print('EPOCH_RESULT', epoch, json.dumps(result['selected']), flush=True)
            if True:  # Fixed final epoch; no checkpoint selection using heldout labels.
                best = dict(
                    epoch=epoch,
                    update=update,
                    selection=result['selected'],
                    weights={k: v.detach().cpu().clone() for k, v in model.heads['ddp'].state_dict().items()},
                )
            history.append(dict(epoch=epoch, update=update, train_seconds=train_seconds, validation=result))
            atomic(rows, out / 'val4_rows' / f'epoch{epoch}.json')
        epoch += 1
        offset = 0
        save()
    # Last epoch already evaluated all heldout sources and saved GT-free probabilities.
    val4 = result
    if rank == 0:
        atomic(val4, out / 'val_four_folds.json')
        atomic(rows, out / 'val_four_folds_rows.json')
        exported_names = sorted(p.name[:-4] for p in (out / 'heldout_predictions').glob('*.npz'))
        assert exported_names == sorted(unseen)
        atomic(
            dict(
                passed=True,
                trained_fold=args['fold'],
                model_kind=args['model_kind'],
                sources=exported_names,
                sources707=707,
                train_sources=sorted(allowed),
                excluded_owner_fold=True,
                threshold=0.05,
                probability_dtype='float16',
                checkpoint_epoch=4,
            ),
            out / 'prediction_verification.json',
        )
        print('VAL_FOUR_FOLDS', json.dumps(val4), flush=True)
    # Final fullstate was saved before loading the selected best head for evaluation.
    if rank == 0:
        ck = torch.load(out / 'last_training.pt', map_location='cpu', weights_only=False)
        exported = torch.load(out / 'best_head.pt', map_location='cpu', weights_only=False)
        exact(ck['best']['weights'], exported['head'])
        assert ck['update'] == total and ck['epoch'] == 5 and len(ck['history']) == 4 and len(ck['rank_states']) == 2
        atomic(dict(head=ck['head'], epoch=4, config=config), out / 'last_head.pt', True)
        atomic(
            dict(
                passed=True,
                readable_full_training_state=True,
                selected_export_matches_common=True,
                rank_rng_and_scalers=2,
                epochs=4,
                updates=update,
                fold=args['fold'],
            ),
            out / 'verification.json',
        )
        summary = dict(
            state='complete',
            sources707=707,
            model_kind=args['model_kind'],
            fold=args['fold'],
            epochs=4,
            updates=update,
            train_crops=len(dataset),
            train_sources=len(fold_info['train_sources']),
            best_epoch=exported['epoch'],
            selection=exported['selection'],
            protocol=config,
        )
        summary['val_four_folds'] = val4
        atomic(summary, out / 'fold_summary.json')
        atomic(dict(state='complete', update=update), out / 'status.json')
        (out / 'report.md').write_text(
            '# Independent proposal fold '
            + str(args['fold'])
            + '\n\n'
            + json.dumps(summary, indent=2)
            + '\n\nTrain on ONE fifth, not standard 4/5 CV. Four epochs fixed; final epoch4, fixed proposal threshold0.05. ALL707 partition, no quick/audit selection. For train-source ensemble, exclude the head trained on that source/day. Every source uses FOUR heads excluding its owner fold. Classifier threshold0.5 fixed, only after verified OOF ensemble. No audit tuning.\n'
        )
        print('COMPLETE_FOLD', json.dumps(summary), flush=True)
    dist.barrier()


if __name__ == '__main__':
    main()
