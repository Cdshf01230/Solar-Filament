"""Training/evaluation primitives shared by the GPU smoke gate and the real run."""

import random
import numpy as np
import torch
from model import loss_terms


def bare(model):
    return model.module if hasattr(model, 'module') else model


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


def prepare_current(model, f16, batch, amp):
    current = batch['current'].clone()
    sel = [i for i, m in enumerate(batch['meta']) if m['sampler'] == 'rollout']
    if sel:
        with torch.no_grad(), torch.autocast(f16.device.type, dtype=torch.float16, enabled=amp and f16.is_cuda):
            z, _ = bare(model)(
                f16[sel].detach(),
                *[batch[k][sel] for k in ('image', 'probability', 'anchor', 'distance')],
                batch['anchor'][sel],
                batch['valid'][sel],
                1,
            )
            current[sel] = z.sigmoid() * batch['valid'][sel]
    return current


def backward_micro(model, features, batch, scaler, divisor=1, amp=True, rounds=3):
    """Backward one round at a time; only neck graph is retained across rounds."""
    with torch.autocast(batch['image'].device.type, dtype=torch.float16, enabled=amp and batch['image'].is_cuda):
        backbone = features(batch['image'], amp=amp)
        f16 = bare(model).encode(backbone)
        current = prepare_current(model, f16, batch, amp)
    parts, losses = [], []
    for t in range(1, rounds + 1):
        with torch.autocast(batch['image'].device.type, dtype=torch.float16, enabled=amp and batch['image'].is_cuda):
            z, presence = model(
                f16,
                *[batch[k] for k in ('image', 'probability', 'anchor', 'distance')],
                current,
                batch['valid'],
                t,
                batch['prior'],
            )
        loss, terms = loss_terms(z, presence, batch['targets'], batch['valid'], batch['skeletons'])
        finite = torch.isfinite(loss).float().detach()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError('Non-finite loss')
        next_current = z.sigmoid().detach() * batch['valid']
        scaler.scale(loss / (divisor * rounds)).backward(retain_graph=t < rounds)
        losses.append(float(loss.detach()))
        parts.append(terms.cpu().tolist())
        current = next_current
    return dict(loss=float(np.mean(losses)), terms=parts)


@torch.no_grad()
def infer(model, features, batch, rounds=5, amp=True):
    with torch.autocast(batch['image'].device.type, dtype=torch.float16, enabled=amp and batch['image'].is_cuda):
        f16 = bare(model).encode(features(batch['image'], amp=amp))
        current = batch['anchor']
        outputs = []
        for t in range(1, rounds + 1):
            z, presence = bare(model)(
                f16,
                *[batch[k] for k in ('image', 'probability', 'anchor', 'distance')],
                current,
                batch['valid'],
                t,
                batch['prior'],
            )
            loss, terms = loss_terms(z, presence, batch['targets'], batch['valid'], batch['skeletons'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite evaluation loss')
            outputs.append(dict(mask=z.sigmoid(), presence=presence.sigmoid(), terms=terms, loss=loss))
            current = z.sigmoid().detach() * batch['valid']
    return outputs


def crop_counts(output, batch):
    values, baseline, positive_labels, presence_scores, neg_empty, pos_empty, adds, removes = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    empty_records = []
    mask = (output['mask'] > 0.5) & batch['valid'].bool()
    probabilities = output['presence'].flatten()
    for i, targets in enumerate(batch['targets']):
        m = mask[i, 0]
        a = batch['anchor'][i, 0] > 0
        v = batch['valid'][i, 0].bool()
        positive = targets[:, v].sum(1) > 0
        s = float(probabilities[i])
        kept = s >= 0.5 and bool(m.any())
        for target in targets[positive]:
            y = target.bool()
            values.append(float((m & y).sum() / ((m | y).sum().clamp_min(1))))
            baseline.append(float((a & y).sum() / ((a | y).sum().clamp_min(1))))
        positive_labels.extend(positive.int().cpu().tolist())
        presence_scores.extend([s] * len(targets))
        empty_records.extend([not kept] * int((~positive).sum()))
        if positive.any():
            pos_empty.append(not kept)
        else:
            neg_empty.append(not kept)
        area = a.sum().clamp_min(1)
        adds.append(float((m & ~a).sum() / area))
        removes.append(float((a & ~m).sum() / area))
    return dict(
        iou=values,
        baseline_iou=baseline,
        labels=positive_labels,
        scores=presence_scores,
        negative_empty=neg_empty,
        positive_empty=pos_empty,
        empty_record_empty=empty_records,
        added_fraction=adds,
        removed_fraction=removes,
    )
