"""Fixed-weight scores on every actual training crop, distinct from full-image scores."""

import time
import torch
from .model import autocast, semantic_loss


@torch.no_grad()
def evaluate_crop_fit(model, loader, epoch, microbatch, progress=None):
    if loader.dataset.augment:
        raise ValueError('Training-fit diagnostic requires unaugmented crops')
    model.eval()
    loader.sampler.set_cursor(epoch, 0)
    device = next(model.heads.parameters()).device
    totals = {
        n: dict(
            crops=0,
            bce=0.0,
            soft_dice_loss=0.0,
            hard_dice_sum=0.0,
            counts=torch.zeros(4, dtype=torch.int64, device=device),
        )
        for n in model.heads
    }
    start = time.perf_counter()
    done = 0
    for images, targets, _ in loader:
        for i in range(0, len(targets), microbatch):
            x = images[i : i + microbatch].to(device).contiguous(memory_format=torch.channels_last)
            y = targets[i : i + microbatch].to(device)
            features = model.features(x)
            for name, head in model.heads.items():
                with autocast(device):
                    logits = head(features, x)
                _, bce, dice = semantic_loss(logits, y)
                pred, gt = logits.float().sigmoid() >= 0.5, y > 0
                tp = (pred & gt).sum((1, 2, 3))
                fp = (pred & ~gt).sum()
                fn = (~pred & gt).sum()
                tn = (~pred & ~gt).sum()
                row = totals[name]
                row['crops'] += len(y)
                row['bce'] += float(bce) * len(y)
                row['soft_dice_loss'] += float(dice) * len(y)
                row['hard_dice_sum'] += float(((2 * tp + 1) / (pred.sum((1, 2, 3)) + gt.sum((1, 2, 3)) + 1)).sum())
                row['counts'] += torch.stack([tp.sum(), fp, fn, tn])
        done += len(targets)
        if progress and (done % 80 == 0 or done == len(loader.dataset)):
            progress(done, len(loader.dataset))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    output = {}
    for name, row in totals.items():
        assert row['crops'] == len(loader.dataset)
        tp, fp, fn, tn = row.pop('counts').tolist()
        n = row['crops']
        output[name] = dict(
            crops=n,
            bce=row['bce'] / n,
            soft_dice_loss=row['soft_dice_loss'] / n,
            mean_crop_hard_dice=row['hard_dice_sum'] / n,
            foreground_dice=2 * tp / max(1, 2 * tp + fp + fn),
            foreground_iou=tp / max(1, tp + fp + fn),
            precision=tp / max(1, tp + fp),
            recall=tp / max(1, tp + fn),
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
        )
    return dict(
        seconds=time.perf_counter() - start,
        heads=output,
        scope='All actual training crop targets at fixed .5, overlapping pixel visits count repeatedly; no D4; weights fixed',
    )
