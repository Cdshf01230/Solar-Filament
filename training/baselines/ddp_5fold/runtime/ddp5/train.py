"""Small CV adapter around the verified DDP12 two-rank implementation.

Fullstate at a finished epoch retains a pending-validation cursor until validation is committed.
All epoch heads/best/history are carried with the fullstate by the resume notebook.
"""
import argparse
import csv
import json
import math
import os
import signal
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from ddpvit import run as old
from ddpvit.model import build_optimizer, trainable_parameters
from ddpvit.scoring import StrictContext, assert_expected
from solar_loss.common import atomic, seed_all
from u0_multihead.data import Crops, worker_init
from ddp5.protocol import partition, read_assignment, ranges, remaining_ranges, expected_val, lr_horizon_updates, EPOCHS, LR_HORIZON, VERSION


def load_data(args, world):
    base = Path(args['base'])
    a = read_assignment(base / 'fold_assignment.json')
    assert a == read_assignment(Path(__file__).with_name('fold_assignment.json'))
    train, val = partition(a, args['fold'])
    ds = Crops(base / 'magfilo_crops_768', base / 'train.jsonl', args['seed'], augment=True)
    assert {r['source_file'] for r in ds.records} == set(train)
    info = json.loads((base / 'staging_verification.json').read_text())
    assert info['passed'] and info['fold'] == args['fold'] and info['protocol_version'] == VERSION and info['crops'] == len(ds)
    return base, a, info, train, val, ds


class ExactBatches:
    def __init__(self, size, rank, world, seed, epoch, offset):
        assert world == 2
        self.size, self.rank, self.world, self.seed, self.epoch = size, rank, world, seed, epoch
        self.pairs = remaining_ranges(size, offset)
    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        for a, b in self.pairs:
            yield [(self.epoch, i) for i in order[a:b][self.rank::self.world]]
    def __len__(self): return len(self.pairs)


def save(args, model, opt, sched, scaler, config, epoch, offset, update, history, records, split):
    old.save_fullstate(model, opt, sched, scaler, config, epoch, offset, update, history, args['out'], records, split)


def main_train(args, rank, world, device):
    out = Path(args['out']); out.mkdir(parents=True, exist_ok=True)
    gate = json.loads((out / 'gate/verification.json').read_text())
    assert gate['passed']
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    seed_all(args['seed'] + rank)
    base, assignment, info, train, val, ds = load_data(args, world)
    size = len(ds); steps = len(ranges(size)); total = steps * args['epochs']
    split = dict(train=train, validation=val, protocol=VERSION)
    model = old.build_state(args, device)
    assert sum(p.numel() for p in trainable_parameters(model)) == 11564106
    micro = gate['checks']['native_forward_backward']['microbatch']
    args = dict(args, microbatch=micro)
    horizon = lr_horizon_updates(total, args.get('lr_horizon', LR_HORIZON))
    opt, sched = build_optimizer(model, horizon, args['warmup'])
    scaler = torch.amp.GradScaler('cuda')
    wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True)
    config = dict(args, world_size=world, steps_per_epoch=steps, total_updates=total, lr_horizon_updates=horizon,
                  protocol_version=VERSION, train_crops=size, trainable_parameters=11564106)
    epoch, offset, update, history = 1, 0, 0, []
    resume = Path(args.get('resume') or out / 'last_training.pt')
    if resume.exists():
        ck = old.load_fullstate(resume, model, opt, sched, scaler, config, rank, ds.records, split)
        for key in ('protocol_version', 'steps_per_epoch', 'total_updates', 'train_crops'):
            assert ck['config'][key] == config[key], key
        epoch, offset, update, history = ck['epoch'], ck['offset'], ck['update'], ck['history']
        assert 1 <= epoch <= args['epochs'] + 1
        assert update == (epoch - 1) * steps + sum(b <= offset for _, b in ranges(size))
        old.marker('RESUMED', fold=args['fold'], epoch=epoch, offset=offset, update=update)
        del ck
    if rank == 0: atomic(config, out / 'config.json')
    context = StrictContext(base)
    workers = min(2, old.cpu_budget() // world)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    if rank == 0 and (out / 'train.csv').exists():
        with (out / 'train.csv').open() as f:
            reader = csv.DictReader(f); fields = reader.fieldnames; rows = [r for r in reader if int(r['update']) <= update]
        with (out / 'train.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
    old.marker('MAIN_START', fold=args['fold'], rank=rank, train_sources=len(train), val_sources=len(val),
               train_crops=size, total_updates=total, resumed_update=update, epochs=args['epochs'], trainable=11564106)

    def stopping(reserve=180):
        flag = torch.tensor(int(stop[0] or time.time() >= args['deadline'] - reserve), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def finish_session(phase):
        save(args, model, opt, sched, scaler, config, epoch, offset, update, history, ds.records, split)
        if rank == 0:
            atomic(dict(state='stopped_resumable', fold=args['fold'], epoch=epoch, offset=offset, update=update,
                        phase=phase, total_updates=total), out / 'status.json')
        old.marker('SOFT_STOP', fold=args['fold'], epoch=epoch, offset=offset, update=update, phase=phase)

    while epoch <= args['epochs']:
        pairs = remaining_ranges(size, offset)
        sampler = ExactBatches(size, rank, world, args['seed'], epoch, offset)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=workers, pin_memory=True, worker_init_fn=worker_init,
                            multiprocessing_context='spawn' if workers else None,
                            generator=torch.Generator().manual_seed(args['seed'] + 1000 + rank))
        model.train(); began = time.time()
        for (a, b), (x, y, _) in zip(pairs, loader):
            t0 = time.perf_counter()
            metrics = old.distributed_step(wrapped, x, y, opt, sched, scaler, micro, b - a)
            offset = b; update += 1
            if rank == 0:
                row = dict(epoch=epoch, update=update, **metrics, step_seconds=time.perf_counter() - t0)
                path = out / 'train.csv'; exists = path.exists()
                with path.open('a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if not exists: w.writeheader()
                    w.writerow(row)
                if update % 20 == 0:
                    old.marker('TRAIN_UPDATE', fold=args['fold'], **row)
                    atomic(dict(state='training', fold=args['fold'], epoch=epoch, update=update, total_updates=total), out / 'status.json')
            if update % args['save_every'] == 0:
                save(args, model, opt, sched, scaler, config, epoch, offset, update, history, ds.records, split)
            if stopping():
                finish_session('training_or_pending_validation'); return False
        assert offset == size
        # Pending-validation fullstate makes deadline interruption/restart safe.
        save(args, model, opt, sched, scaler, config, epoch, offset, update, history, ds.records, split)
        if rank == 0:
            head = dict(trainable=model.trainable_state(), epoch=epoch, update=update, config=config)
            atomic(head, out / f'epoch_{epoch:02d}.pt', True); atomic(head, out / 'last.pt', True)
        dist.barrier()
        if stopping(reserve=900):
            finish_session('before_validation'); return False
        result = old.evaluate_global(model, val, context, max(1, workers), f'epoch{epoch:02d}_val', out, rank, world, device)
        assert_expected(result, **expected_val(args))
        if rank == 0:
            best_path = out / 'best.json'
            best = json.loads(best_path.read_text()) if best_path.exists() else None
            if best is None or result['strict']['pq'] > best['pq']:
                atomic(dict(trainable=model.trainable_state(), epoch=epoch, update=update, config=config), out / 'best.pt', True)
                atomic(dict(epoch=epoch, update=update, pq=result['strict']['pq'], legacy_pq=result['legacy']['pq'],
                            metric='full holdout strict one-to-one PQ; earliest tie'), best_path)
            old.marker('EVAL_SUMMARY', fold=args['fold'], epoch=epoch, strict=result['strict'], legacy=result['legacy'],
                       sources=result['sources'], records=result['records'], gt=result['gt'])
        history.append(dict(epoch=epoch, update=update, train_seconds_this_session=time.time() - began, val=result))
        epoch += 1; offset = 0
        save(args, model, opt, sched, scaler, config, epoch, offset, update, history, ds.records, split)
        if stopping():
            finish_session('between_epochs'); return False
    assert update == total and len(history) == args['epochs']
    old.final_verification(model, args, config, out, rank, device, history, update)
    if rank == 0:
        atomic(dict(state='complete_verified', fold=args['fold'], epoch=args['epochs'], update=update,
                    best=json.loads((out / 'best.json').read_text())), out / 'status.json')
        old.marker('DDP5_COMPLETE_VERIFIED', fold=args['fold'], epoch=args['epochs'], update=update)
    return True


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--mode', choices=['gate', 'main'], required=True)
    a = p.parse_args(); args = json.loads(Path(a.config).read_text())
    assert args['epochs'] == EPOCHS and args['arm'] == 'convnext' and args['protocol_version'] == VERSION
    old.RESUME_REL_TOL = args.get('resume_rel_tol', 2e-4)
    assert 0 < old.RESUME_REL_TOL <= 1e-3
    old.load_data = load_data; old.expected_val = expected_val
    rank, world, device = old.init_process(args)
    try:
        if a.mode == 'gate': old.gate(args, rank, world, device)
        else: main_train(args, rank, world, device)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
