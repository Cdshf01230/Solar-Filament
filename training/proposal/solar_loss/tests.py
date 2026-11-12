"""CPU formula/sampler/real-model and two-rank Gloo regression tests."""

import os, copy, sys
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as Distributed
from .common import GlobalBatches, exact, restore_rng
from proposal import loss_terms


def unit():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    z = torch.randn(3, 1, 9, 7, requires_grad=True)
    y = (torch.rand_like(z) > 0.8).float()
    y[0] = 0
    a, t = loss_terms(z, y, 'A')
    c, _ = loss_terms(z, y, 'C')
    assert torch.allclose(c - a, 2 * t['bce'])
    for arm in ['A', 'B', 'C', 'D']:
        loss, terms = loss_terms(z, y, arm)
        g = torch.autograd.grad(loss, z, retain_graph=True)[0]
        assert torch.isfinite(g).all() and g.abs().sum() > 0
        p = z.sigmoid()
        tp = (p * y).sum((1, 2, 3))
        fp = (p * (1 - y)).sum((1, 2, 3))
        fn = ((1 - p) * y).sum((1, 2, 3))
        alpha, beta = (0.6, 0.4) if arm == 'B' else (0.4, 0.6)
        if arm in ['B', 'D']:
            assert torch.allclose(loss, terms['bce'] + 1 - ((tp + 0.5) / (tp + alpha * fp + beta * fn + 0.5)).mean())
    ranks = [list(GlobalBatches(4915, 8, r, 2, 42, 3)) for r in range(2)]
    assert len(ranks[0]) == len(ranks[1]) == 615
    assert len(ranks[0][-1]) == 2 and len(ranks[1][-1]) == 1
    allids = [v[1] for rank in ranks for batch in rank for v in batch]
    assert len(allids) == len(set(allids)) == 4915
    resumed = list(GlobalBatches(4915, 8, 0, 2, 42, 3, 80))
    assert resumed == ranks[0][10:]
    from u0_multihead.ddp_mechanism_model import RefinedHead

    h = RefinedHead()
    features = [torch.randn(1, c, s, s) for c, s in [(128, 8), (256, 4), (512, 2), (1024, 1)]]
    image = torch.randn(1, 3, 32, 32)
    target = (torch.rand(1, 1, 32, 32) > 0.8).float()
    assert sum(p.numel() for p in h.parameters()) == 11588011
    for arm in ['A', 'B', 'C', 'D']:
        h.zero_grad(set_to_none=True)
        logits = h.training_logits(features, image, target)
        loss, _ = loss_terms(logits, target, arm)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters()), arm
    from .runner import combine_rows

    rows = []
    for source, stats, pixels, loss in [
        ('a', [1, 0, 0, 0.8], [5, 2, 1, 10], 0.2),
        ('b', [0, 3, 1, 0], [1, 7, 3, 20], 0.6),
    ]:
        rows.append(
            dict(
                source_file=source,
                record=source,
                threshold=0.5,
                stats=stats,
                pixels=pixels,
                included_in_public_pq=True,
                gt=1,
                predictions=1,
                split_gt=0,
                merge_predictions=0,
                inference_loss=dict(bce=loss, soft_dice_loss=0.1, loss=loss + 0.1),
            )
        )
    m = combine_rows(rows, [0.5])
    assert abs(m['selected']['instance']['pq'] - 0.8 / 3) < 1e-8
    assert abs(m['inference_loss']['loss'] - 0.5) < 1e-8
    print(
        'PASS formulas/negative crops/all-head gradients; no sampler padding/tail/resume; pooled metrics, not mean PQ',
        flush=True,
    )


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({'ddp': nn.Conv2d(1, 1, 1)})
        self.noisy = False
        with torch.no_grad():
            self.heads['ddp'].weight.fill_(0.2)
            self.heads['ddp'].bias.fill_(0.1)

    def forward(self, x, y):
        return self.heads['ddp'](x + (torch.randn_like(x) * 0.01 if self.noisy else 0))


def distributed_test():
    from .runner import distributed_step, checkpoint

    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    torch.manual_seed(42 + rank)
    model = Toy()
    wrapped = Distributed(model, broadcast_buffers=False)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    x = torch.tensor([0.2, 0.6, 0.9]).reshape(3, 1, 1, 1)
    y = torch.tensor([0.0, 1.0, 0.0]).reshape_as(x)
    before = copy.deepcopy(model.state_dict())
    metrics, _ = distributed_step(wrapped, x[rank::2], y[rank::2], opt, sched, scaler, 1, 'proposal', 3)
    ref = Toy()
    ref.load_state_dict(before)
    ro = torch.optim.AdamW(ref.parameters(), lr=0.001)
    loss, _ = loss_terms(ref(x, y), y, 'proposal')
    loss.backward()
    nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
    ro.step()
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, ref.state_dict()[k], rtol=1e-6, atol=1e-7)
    assert abs(metrics['objective'] - float(loss.detach())) < 1e-6
    model.noisy = True
    out = Path(os.environ['TEST_OUTPUT'])
    checkpoint(
        model,
        opt,
        sched,
        scaler,
        dict(global_batch=3),
        1,
        3,
        1,
        None,
        [],
        out,
        [{'i': i} for i in range(3)],
        {'train': ['a']},
    )

    def reload():
        c = torch.load(out / 'last_training.pt', weights_only=False)
        model.heads['ddp'].load_state_dict(c['head'])
        opt.load_state_dict(c['optimizer'])
        sched.load_state_dict(c['scheduler'])
        scaler.load_state_dict(c['rank_states'][rank]['scaler'])
        restore_rng(c['rank_states'][rank]['rng'])

    reload()
    m1, _ = distributed_step(wrapped, x[rank::2], y[rank::2], opt, sched, scaler, 1, 'proposal', 3)
    expected = copy.deepcopy(model.state_dict())
    op = copy.deepcopy(opt.state_dict())
    reload()
    m2, _ = distributed_step(wrapped, x[rank::2], y[rank::2], opt, sched, scaler, 1, 'proposal', 3)
    exact(m1, m2)
    exact(expected, model.state_dict())
    exact(op, opt.state_dict())
    if rank == 0:
        print(
            'PASS real two-rank Gloo unequal-tail gradient weighting vs global AdamW; serialized optimizer/scheduler/per-rank RNG/scaler stochastic exact CPU resume',
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == '__main__':
    if '--distributed' in sys.argv:
        distributed_test()
    else:
        unit()
