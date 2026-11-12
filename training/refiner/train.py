"""PALB generalization arms (06/11): PALB code + --epochs / --ema / --lp-epochs / --layer-decay / --drop-path (defaults = PALB).
RCR combo P+A+L+B on D4: train 4 folds, val the holdout fold; fixed EPOCHS, final checkpoint, val report-only.
Run via torchrun. Internal development gates are not distributed."""

import os
import shutil

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse
import json
import math
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from data import Crops, collate, move, worker_init
import model as model_module
from model import Decoder, PartialFeatures, TUNE_BLOCKS, BACKBONE_LR
from engine import bare, backward_micro, infer, crop_counts, rng_state, restore_rng
from evaluate import native_candidate, full_image


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def save_torch(path, obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def setup():
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    return rank, world, device


def all_states(world):
    s = rng_state()
    if world == 1:
        return [s]
    states = [None] * world
    dist.all_gather_object(states, s)
    return states


def checkpoint(path, model, optimizer, scheduler, scaler, epoch, offset, update, world, config, ema=None):
    states = all_states(world)
    if int(os.environ.get('RANK', 0)) == 0:
        save_torch(
            path,
            dict(
                model=bare(model).state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                rank_rng=states,
                epoch=epoch,
                offset=offset,
                update=update,
                config=config,
                ema=ema.state_dict() if ema else None,
            ),
        )


class EMA:
    """Exponential moving average of every trainable parameter (decoder + tuned ViT blocks), with the usual warmup
    decay min(d, (1+t)/(10+t)). swapped() temporarily loads the averages for evaluation / the final EMA checkpoint."""

    def __init__(self, model, decay):
        self.decay = decay
        self.names = [n for n, p in bare(model).named_parameters() if p.requires_grad]
        params = dict(bare(model).named_parameters())
        self.shadow = {n: params[n].detach().clone().float() for n in self.names}

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        params = dict(bare(model).named_parameters())
        for n in self.names:
            self.shadow[n].mul_(d).add_(params[n].detach().float(), alpha=1 - d)

    def state_dict(self):
        return dict(decay=self.decay, shadow={n: t.cpu() for n, t in self.shadow.items()})

    def load_state_dict(self, s):
        assert s['decay'] == self.decay and set(s['shadow']) == set(self.names)
        for n in self.names:
            self.shadow[n].copy_(s['shadow'][n])

    @torch.no_grad()
    def swap(self, model):
        params = dict(bare(model).named_parameters())
        for n in self.names:
            backup = params[n].detach().clone()
            params[n].copy_(self.shadow[n].to(params[n].dtype))
            self.shadow[n].copy_(backup.float())


def param_groups(model):
    """Decoder keeps D4's LR3e-4; tuned backbone blocks use their own layer-decayed LR. Same weight decay."""
    m = bare(model)
    decoder = [p for n, p in m.named_parameters() if not n.startswith('features.')]
    groups = [dict(params=decoder, lr=3e-4, weight_decay=0.01, name='decoder')] + m.features.param_groups(0.01)
    covered = {id(p) for g in groups for p in g['params']}
    assert covered == {id(p) for p in m.parameters() if p.requires_grad}
    return groups


EPOCHS = 60  # overridden by --epochs
EVAL_EPOCHS = {1} | set(range(5, EPOCHS + 1, 5))


def lr_factor(step, total):
    if step < 200:
        return (step + 1) / 200
    progress = min(1.0, (step - 200) / max(1, total - 200))
    return 1 / 30 + (1 - 1 / 30) * 0.5 * (1 + math.cos(math.pi * progress))


def train_step(model, features, cpu_batches, optimizer, scaler, device, world):
    start_rng = rng_state()
    count = sum(len(b['targets']) for b in cpu_batches)
    for amp in (True, False):
        restore_rng(start_rng)
        optimizer.zero_grad(set_to_none=True)
        terms = []
        try:
            for j, cpu in enumerate(cpu_batches):
                b = move(cpu, device)
                # DDP averages ranks; account for tail padding using explicit sample weights.
                divisor = count / len(b['targets'])
                if hasattr(model, 'no_sync') and j < len(cpu_batches) - 1:
                    with model.no_sync():
                        terms.append(backward_micro(model, features, b, scaler, divisor, amp))
                else:
                    terms.append(backward_micro(model, features, b, scaler, divisor, amp))
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(bare(model).parameters(), 1.0)
            flag = torch.tensor(float(torch.isfinite(norm)), device=device)
            if world > 1:
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if not flag.item():
                scaler.update()
                raise FloatingPointError('Non-finite gradients')
            scaler.step(optimizer)
            scaler.update()
            return dict(
                loss=float(np.mean([t['loss'] for t in terms])),
                parts=np.mean([t['terms'] for t in terms], axis=0).tolist(),
                fp32_retry=not amp,
            )
        except FloatingPointError:
            if not amp:
                raise
            # A failed forward has not reached unscale; reset GradScaler per-optimizer state.
            scale = scaler.get_scale()
            scaler = reset_scaler_in_place(scaler, scale)
            print('FP32_BATCH_RETRY', flush=True)


def reset_scaler_in_place(scaler, scale):
    if scaler.is_enabled():
        fresh = torch.amp.GradScaler('cuda', init_scale=max(1.0, scale / 2))
        scaler.__dict__.clear()
        scaler.__dict__.update(fresh.__dict__)
    return scaler


def summarize_crop(rows):
    from scipy.stats import rankdata

    out = {}
    for t in range(5):
        merged = defaultdict(list)
        for row in rows:
            for k, v in row[t]['counts'].items():
                merged[k].extend(v)
        labels, scores = np.array(merged['labels']), np.array(merged['scores'])
        pos, neg = labels == 1, labels == 0
        auc = (
            float((rankdata(scores)[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))
            if pos.any() and neg.any()
            else None
        )

        def mean(k):
            return float(np.mean(merged[k])) if merged[k] else None

        add, rem = np.array(merged['added_fraction']), np.array(merged['removed_fraction'])
        out[str(t + 1)] = dict(
            iou=mean('iou'),
            iou_gt_half=float(np.mean(np.array(merged['iou']) > 0.5)) if merged['iou'] else None,
            baseline_iou=mean('baseline_iou'),
            presence_auroc=auc,
            negative_empty=mean('negative_empty'),
            positive_empty_wrong=mean('positive_empty'),
            empty_record_empty=mean('empty_record_empty'),
            added_quantiles=np.quantile(add, [0.05, 0.5, 0.95]).tolist(),
            removed_quantiles=np.quantile(rem, [0.05, 0.5, 0.95]).tolist(),
            changed_gt_half=float(np.mean(add + rem > 0.5)),
            loss=float(np.mean([r[t]['loss'] for r in rows])),
            loss_terms=np.mean([r[t]['terms'] for r in rows], axis=0).tolist(),
        )
    out['delta_iou_1_to_2'] = out['2']['iou'] - out['1']['iou'] if out['1']['iou'] is not None else None
    out['delta_iou_2_to_3'] = out['3']['iou'] - out['2']['iou'] if out['2']['iou'] is not None else None
    return out


def evaluate(model, features, args, split, epoch, rank, world, device):
    model.eval()
    dataset = Crops(args.crops, args.holdout, split, sampler=False)
    indices = list(range(rank, len(dataset), world))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.microbatch,
        num_workers=4 // world,
        collate_fn=collate,
        worker_init_fn=worker_init,
        pin_memory=True,
        multiprocessing_context='spawn',
    )
    folder = Path(args.out) / f'eval_epoch{epoch:02d}' / split
    folder.mkdir(parents=True, exist_ok=True)
    rows, bysource = [], defaultdict(lambda: defaultdict(list))
    # v2 (06/11): candidate building (resize + CC + full-image RLE) runs in a process pool and the per-sample GPU reads are
    # batched; crop_counts uses the CPU copy. Same values and the same candidate order as the serial loop (check_eval_fast.py).
    pool = ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context('spawn'))
    for j, cpu in enumerate(loader):
        b = move(cpu, device)
        try:
            outputs = infer(model, features, b)
        except FloatingPointError:
            outputs = infer(model, features, b, amp=False)
        cpu_out = [dict(mask=o['mask'].cpu(), presence=o['presence'].cpu()) for o in outputs]
        rows.append(
            [
                dict(counts=crop_counts(co, cpu), loss=float(o['loss']), terms=o['terms'].cpu().tolist())
                for co, o in zip(cpu_out, outputs)
            ]
        )
        masks = {t: (cpu_out[t - 1]['mask'][:, 0] > 0.5).numpy() for t in (1, 3, 5)}
        scores = {t: [float(x) for x in cpu_out[t - 1]['presence'][:, 0]] for t in (1, 3, 5)}
        pending = []
        for i, meta in enumerate(cpu['meta']):
            a = cpu['anchor'][i, 0].numpy()
            for t in (1, 3, 5):
                score = scores[t][i]
                if score >= 0.5:  # below .5 the serial loop passed an empty mask, which never yields a candidate
                    pending.append(
                        (
                            f'R{t}',
                            meta['source'],
                            pool.submit(
                                native_candidate,
                                masks[t][i],
                                None,
                                meta['box'],
                                meta['resized_wh'],
                                score,
                                meta['crop_id'],
                                native_anchor=meta['native_anchor'],
                            ),
                        )
                    )
            pending.append(
                (
                    'baseline',
                    meta['source'],
                    pool.submit(
                        native_candidate,
                        a > 0,
                        None,
                        meta['box'],
                        meta['resized_wh'],
                        meta['ml_score'],
                        meta['crop_id'],
                        native_anchor=meta['native_anchor'],
                        baseline=True,
                    ),
                )
            )
        for kind, source, future in pending:
            candidate = future.result()
            if candidate:
                bysource[kind][source].append(candidate)
        if j % 100 == 0:
            print(
                'EVAL_PROGRESS',
                json.dumps(
                    dict(epoch=epoch, split=split, rank=rank, crops=min((j + 1) * args.microbatch, len(indices)))
                ),
                flush=True,
            )
    pool.shutdown()
    for kind in ('R1', 'R3', 'R5', 'baseline'):
        dest = folder / kind
        dest.mkdir(exist_ok=True)
        for source, candidates in bysource[kind].items():
            atomic(dest / (Path(source).stem + f'__rank{rank}.json'), candidates)
    atomic(folder / f'crop_rank{rank}.json', rows)
    if world > 1:
        dist.barrier()
    result = None
    if rank == 0:
        assignment = json.loads(Path(args.folds).read_text(encoding='utf-8'))
        sources = [
            s
            for f in assignment['folds']
            if (f['fold'] != args.holdout) == (split == 'train')
            for s in f['train_sources']
        ]
        allrows = [r for i in range(world) for r in json.loads((folder / f'crop_rank{i}.json').read_text())]
        assert len(allrows) == sum(
            math.ceil(len(range(i, len(dataset), world)) / args.microbatch) for i in range(world)
        )
        # Rank shards contain every crop exactly once; source list also includes zero-crop sources.
        result = dict(crop=summarize_crop(allrows), full_image={})
        for kind in ('R1', 'R3', 'R5', 'baseline'):
            result['full_image'][kind] = full_image(
                args.annotations, sources, folder / kind, folder / (kind + '_scores.json'), workers=4
            )
        atomic(folder / 'summary.json', result)
        print('EVAL_SUMMARY', json.dumps(dict(epoch=epoch, split=split, result=result)), flush=True)
    if world > 1:
        dist.barrier()
    model.train()
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('crops', 'backbone', 'annotations', 'folds', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--holdout', type=int, default=1)
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--resume')
    p.add_argument('--d4', action='store_true', help='Only ablation: joint train-only D4; evaluation unchanged')
    p.add_argument('--photometric', action='store_true', help='A: train-only photometric + shrink-scale augmentation')
    p.add_argument('--epochs', type=int, default=40, help='fixed epochs; cosine reaches its floor at the last update')
    p.add_argument(
        '--ema',
        type=float,
        default=0.0,
        help='EMA decay of trainable weights (0 = off); evaluation and last_ema.pt use EMA',
    )
    p.add_argument('--lp-epochs', type=int, default=0, help='LP-FT: tuned ViT blocks frozen for the first N epochs')
    p.add_argument(
        '--layer-decay', type=float, default=0.8, help='LR decay per tuned ViT block below the top one (PALB .8)'
    )
    p.add_argument('--drop-path', type=float, default=0.0, help='drop path rate in the tuned ViT blocks (PALB 0)')
    p.add_argument('--arm', default='control')
    p.add_argument(
        '--keep-eval-snapshots',
        action='store_true',
        help='also keep ep<NN>.pt (and ep<NN>_ema.pt) at every eval epoch, for epoch-count selection across folds (user 08/11)',
    )
    args = p.parse_args()
    rank, world, device = setup()
    global EPOCHS, EVAL_EPOCHS
    EPOCHS = args.epochs
    EVAL_EPOCHS = {1} | set(range(5, EPOCHS + 1, 5))
    model_module.BACKBONE_LAYER_DECAY = args.layer_decay  # read by PartialFeatures.param_groups
    assert 0 <= args.holdout <= 4 and 8 % (world * args.microbatch) == 0
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model, features = Decoder().to(device), PartialFeatures(args.backbone, drop_path=args.drop_path).to(device)
    model.features = features  # registered so DDP reduces and checkpoints the tuned blocks
    if world > 1:
        model = DDP(model, device_ids=[device.index], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(param_groups(model), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999))
    n_train = len(Crops(args.crops, args.holdout, 'train')) // world * world
    steps_per_epoch = math.ceil(n_train / 8)
    total = steps_per_epoch * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total))
    ema = EMA(model, args.ema) if args.ema > 0 else None
    scaler = torch.amp.GradScaler('cuda')
    epoch0, offset0, update = 1, 0, 0
    if args.resume:
        c = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert c['config']['holdout'] == args.holdout and len(c['rank_rng']) == world
        assert (
            c['config'].get('d4', False) == args.d4 and c['config'].get('photometric', False) == args.photometric
        ), 'Resume must retain the same augmentation'
        bare(model).load_state_dict(c['model'], strict=True)
        optimizer.load_state_dict(c['optimizer'])
        scheduler.load_state_dict(c['scheduler'])
        scaler.load_state_dict(c['scaler'])
        for k in ('epochs', 'ema', 'lp_epochs', 'layer_decay', 'drop_path'):
            assert c['config'][k] == getattr(args, k), 'resume must keep the arm settings'
        if ema:
            ema.load_state_dict(c['ema'])
        restore_rng(c['rank_rng'][rank])
        epoch0, offset0, update = c['epoch'], c['offset'], c['update']
    began = time.monotonic()
    history = []
    hp = Path(args.out) / 'history.json'
    if hp.exists():
        history = json.loads(hp.read_text())
    config = dict(
        vars(args),
        epochs=EPOCHS,
        batch=8,
        rounds=3,
        checkpoint=f'fixed epoch{EPOCHS}; val report-only',
        world=world,
        n_train_per_epoch=n_train,
        steps_per_epoch=steps_per_epoch,
        total_updates=total,
        eval_epochs=sorted(EVAL_EPOCHS),
        tune_blocks=TUNE_BLOCKS,
        backbone_lr=BACKBONE_LR,
        backbone_layer_decay=args.layer_decay,
        skeleton_recall_weight=1.0,
        arm='P+A+L+B on D4, train 4 folds / val holdout fold',
    )
    if rank == 0:
        atomic(Path(args.out) / 'protocol.json', config)
    for epoch in range(epoch0, EPOCHS + 1):
        bare(model).features.set_tuning(epoch > args.lp_epochs)  # LP-FT; always on when --lp-epochs 0
        dataset = Crops(args.crops, args.holdout, 'train', epoch=epoch, d4=args.d4, photometric=args.photometric)
        # Equal crops per rank keeps per-round collectives aligned; the dropped crop rotates with the permutation.
        order = np.random.default_rng(42 + epoch).permutation(len(dataset)).tolist()[:n_train]
        losses, histogram, ops = [], [], []
        model.train()
        # Exact global batches of 8; the tail batch splits evenly over ranks.
        first_offset = offset0 if epoch == epoch0 else 0
        local_order = order[first_offset:][rank::world]
        loader = DataLoader(
            Subset(dataset, local_order),
            batch_size=8 // world,
            num_workers=4 // world,
            collate_fn=collate,
            worker_init_fn=worker_init,
            pin_memory=True,
            multiprocessing_context='spawn',
        )
        resume_rng = rng_state() if epoch == epoch0 and first_offset > 0 else None
        iterator = iter(loader)
        if resume_rng is not None:
            restore_rng(resume_rng)
        for step_index, cpu_global in enumerate(iterator):
            offset = first_offset + step_index * 8
            cpu_batches = []
            for start in range(0, len(cpu_global['meta']), args.microbatch):
                samples = cpu_global['meta'][start : start + args.microbatch]
                cpu_batches.append(collate(samples))
                for m in samples:
                    if m['noise_iou'] is not None:
                        histogram.append(m['noise_iou'])
                    ops.extend(m['noise_ops'])
            terms = train_step(model, features, cpu_batches, optimizer, scaler, device, world)
            scheduler.step()
            update += 1
            losses.append(terms)
            if ema:
                ema.update(model, update)
            if update % 10 == 0:
                print(
                    'TRAIN_UPDATE',
                    json.dumps(
                        dict(
                            rank=rank,
                            epoch=epoch,
                            update=update,
                            loss=terms['loss'],
                            lr=scheduler.get_last_lr()[0],
                            fp32_retry=terms['fp32_retry'],
                        )
                    ),
                    flush=True,
                )
            deadline = float(os.environ.get('RCR_DEADLINE_MONOTONIC', began + 11 * 3600))
            stop = torch.tensor(int(time.monotonic() > deadline), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                checkpoint(
                    Path(args.out) / 'fullstate.pt',
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    offset + 8,
                    update,
                    world,
                    config,
                    ema,
                )
                if rank == 0:
                    atomic(
                        Path(args.out) / 'status.json',
                        dict(state='soft_stopped', epoch=epoch, offset=offset + 8, update=update),
                    )
                if world > 1:
                    dist.destroy_process_group()
                return
        checkpoint(
            Path(args.out) / 'fullstate.pt',
            model,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
            0,
            update,
            world,
            config,
            ema,
        )
        if rank == 0:
            save_torch(
                Path(args.out) / 'last.pt',
                dict(model=bare(model).state_dict(), epoch=epoch, update=update, config=config),
            )
        if ema:
            ema.swap(model)
            if rank == 0:
                save_torch(
                    Path(args.out) / 'last_ema.pt',
                    dict(model=bare(model).state_dict(), epoch=epoch, update=update, config=config),
                )
            ema.swap(model)
        if args.keep_eval_snapshots and epoch in EVAL_EPOCHS and rank == 0:
            for src in ['last.pt'] + (['last_ema.pt'] if ema else []):
                dst = Path(args.out) / src.replace('last', 'ep%02d' % epoch)
                shutil.copyfile(Path(args.out) / src, dst)
        local = dict(losses=losses, noise_iou=histogram, noise_ops=ops)
        shards = [local]
        if world > 1:
            shards = [None] * world
            dist.all_gather_object(shards, local)
        record = dict(epoch=epoch, update=update, training=shards)
        if epoch in EVAL_EPOCHS:
            if ema:
                ema.swap(model)  # report the EMA weights (the arm's final checkpoint)
            record['train_fit'] = evaluate(model, features, args, 'train', epoch, rank, world, device)
            record['val'] = evaluate(model, features, args, 'val', epoch, rank, world, device)
            if args.keep_eval_snapshots and rank == 0:
                # best.pt = eval epoch with the highest own-fold val R3 1-1 PQ (user 08/11). Own-fold best is optimistic:
                # report last.pt per fold; pick the submission epoch count across folds.
                pq = record['val']['full_image']['R3']['one_to_one']['pq']
                bp = Path(args.out) / 'best.json'
                old = json.loads(bp.read_text()) if bp.exists() else None
                if old is None or pq > old['pq']:
                    shutil.copyfile(Path(args.out) / ('last_ema.pt' if ema else 'last.pt'), Path(args.out) / 'best.pt')
                    atomic(
                        bp,
                        dict(
                            epoch=epoch, update=update, pq=pq, metric='val full_image R3 one_to_one pq', ema=bool(ema)
                        ),
                    )
            if ema:
                ema.swap(model)
                if epoch == EPOCHS:
                    record['val_raw'] = evaluate(
                        model, features, args, 'val', 1000 + epoch, rank, world, device
                    )  # raw weights, last epoch only
        if rank == 0:
            history.append(record)
            atomic(hp, history)
            atomic(
                Path(args.out) / 'status.json',
                dict(state='running', epoch=epoch, update=update, elapsed=time.monotonic() - began),
            )
    if rank == 0:
        assert update == total
        saved = torch.load(Path(args.out) / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['epoch'] == EPOCHS and saved['update'] == total
        check = Decoder()
        check.features = PartialFeatures(args.backbone)
        check.load_state_dict(saved['model'], strict=True)
        assert all(torch.isfinite(t).all() for t in check.state_dict().values())
        if ema:
            saved = torch.load(Path(args.out) / 'last_ema.pt', map_location='cpu', weights_only=False)
            check.load_state_dict(saved['model'], strict=True)
            assert all(torch.isfinite(t).all() for t in check.state_dict().values())
        assignment = json.loads(Path(args.folds).read_text(encoding='utf-8'))
        counts = {
            split: sum(
                len(f['train_sources'])
                for f in assignment['folds']
                if (f['fold'] != args.holdout) == (split == 'train')
            )
            for split in ('train', 'val')
        }
        atomic(
            Path(args.out) / 'verification.json',
            dict(
                passed=True,
                epoch=EPOCHS,
                update=update,
                final_checkpoint_loaded=True,
                holdout=args.holdout,
                train_sources=counts['train'],
                val_sources=counts['val'],
            ),
        )
        atomic(
            Path(args.out) / 'status.json',
            dict(state='complete_verified', epoch=EPOCHS, update=update, elapsed=time.monotonic() - began),
        )
        print('RCR_COMPLETE_VERIFIED', flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
