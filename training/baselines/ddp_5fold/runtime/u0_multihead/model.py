from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from .preprocessing import PREPROCESS_HEADS, PREPROCESS_PROTOCOL, normalized_input

CHANNELS = (128, 256, 512, 1024)
SPECS = {
    'u0': dict(width=128, half=32, native=16, extra='none', parameters=1792481),
    'wide192': dict(width=192, half=32, native=16, extra='none', parameters=3778145),
    'wide256': dict(width=256, half=32, native=16, extra='none', parameters=6501089),
    'detail': dict(width=128, half=64, native=32, extra='none', parameters=1926849),
    'wide192_detail': dict(width=192, half=64, native=32, extra='none', parameters=3930945),
    'deeper': dict(width=128, half=32, native=16, extra='deeper', parameters=2678753),
    'context': dict(width=128, half=32, native=16, extra='context', parameters=2285025),
    'fpn128': dict(kind='fpn', width=128, parameters=1575041),
    'nested128': dict(kind='nested', width=128, half=32, native=16, parameters=3710945),
    'capacity512': dict(width=512, half=128, native=64, extra='none', parameters=25694081),
    'capacity1024': dict(width=1024, half=256, native=128, extra='none', parameters=100785921),
}
DEFAULT_HEADS = ('u0', 'wide192', 'deeper', 'context')
EXTENDED_HEADS = ('u0', 'wide192', 'wide256', 'detail', 'deeper', 'context', 'fpn128', 'nested128')
LOSS_HEADS = ('L0', 'L1', 'L2', 'L3', 'L4')
LOSS_PROTOCOL = {
    'L0': 'BCE + per-crop soft Dice',
    'L1': '3*BCE + per-crop soft Dice',
    'L2': 'mean((1-p_t)**2 * BCE_pixel), no alpha weighting + per-crop soft Dice',
    'L3': 'BCE + per-crop Tversky(alpha_FP=.6,beta_FN=.4,smooth=.5)',
    'L4': 'BCE + per-crop Tversky(alpha_FP=.4,beta_FN=.6,smooth=.5)',
}
for _name in LOSS_HEADS:
    SPECS[_name] = dict(SPECS['u0'], loss=LOSS_PROTOCOL[_name])
for _name in PREPROCESS_HEADS:
    SPECS[_name] = dict(SPECS['u0'], preprocessing=_name)


def block(cin, cout, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
                         nn.GroupNorm(8, cout), nn.GELU())


def resize(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ImageStem(nn.Module):
    def __init__(self, native, half):
        super().__init__()
        self.native = block(3, native)
        self.half_resolution = nn.Sequential(nn.Conv2d(native, half, 3, stride=2, padding=1, bias=False),
                                             nn.GroupNorm(8, half), nn.GELU())

    def forward(self, image):
        native = self.native(image)
        return native, self.half_resolution(native)


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(block(width, width), block(width, width))

    def forward(self, x):
        return x + self.layers(x)


class Context16(nn.Module):
    """Three dilation branches at stride16, followed by a 1x1 fusion."""
    def __init__(self, width):
        super().__init__()
        self.branches = nn.ModuleList([block(width, width, d) for d in (1, 2, 4)])
        self.fuse = nn.Sequential(nn.Conv2d(3*width, width, 1, bias=False),
                                  nn.GroupNorm(8, width), nn.GELU())

    def forward(self, x):
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class UNetHead(nn.Module):
    def __init__(self, name):
        super().__init__()
        s = SPECS[name]
        w, h, n = s['width'], s['half'], s['native']
        self.stem = ImageStem(n, h)
        self.lateral = nn.ModuleList([nn.Conv2d(c, w, 1) for c in CHANNELS])
        self.top = block(w, w)
        self.decode = nn.ModuleList([nn.Sequential(block(2*w, w), block(w, w)) for _ in range(3)])
        self.half_refine = nn.Sequential(block(w+h, h), block(h, h))
        self.native = nn.Sequential(block(h+n, n), block(n, n))
        self.classifier = nn.Conv2d(n, 1, 1)
        self.extra = nn.ModuleList([Residual(w) if s['extra'] == 'deeper' else nn.Identity()
                                    for _ in range(3)])
        self.context = Context16(w) if s['extra'] == 'context' else nn.Identity()
        assert sum(p.numel() for p in self.parameters()) == s['parameters']

    def forward(self, features, image):
        native, half = self.stem(image)
        levels = [layer(x) for layer, x in zip(self.lateral, features)]
        x = self.top(levels[-1])
        for i, (decode, extra, skip) in enumerate(zip(self.decode, self.extra, reversed(levels[:-1]))):
            x = extra(decode(torch.cat([resize(x, skip.shape[-2:]), skip], dim=1)))
            if i == 0:
                x = self.context(x)
        x = self.half_refine(torch.cat([resize(x, half.shape[-2:]), half], dim=1))
        x = self.native(torch.cat([resize(x, image.shape[-2:]), native], dim=1))
        return self.classifier(x)


class FPNHead(nn.Module):
    """Historical FPN128: top-down addition then concatenate four pyramid levels."""
    def __init__(self):
        super().__init__()
        w = 128
        self.lateral = nn.ModuleList([nn.Conv2d(c, w, 1) for c in CHANNELS])
        self.smooth = nn.ModuleList([block(w, w) for _ in CHANNELS])
        self.fuse = nn.Sequential(block(4*w, w), block(w, w))
        self.classifier = nn.Conv2d(w, 1, 1)
        assert sum(p.numel() for p in self.parameters()) == SPECS['fpn128']['parameters']

    def forward(self, features, image):
        pyramid = [layer(x) for layer, x in zip(self.lateral, features)]
        for i in (2, 1, 0):
            pyramid[i] = pyramid[i] + resize(pyramid[i+1], pyramid[i].shape[-2:])
        pyramid = [layer(x) for layer, x in zip(self.smooth, pyramid)]
        merged = torch.cat([resize(x, pyramid[0].shape[-2:]) if i else x
                            for i, x in enumerate(pyramid)], dim=1)
        return resize(self.classifier(self.fuse(merged)), image.shape[-2:])


class NestedHead(nn.Module):
    """UNet++-inspired dense nested skips over DINO stages; no deep supervision.

    Nodes X[i,j] concatenate X[i,0..j-1] and upsampled X[i+1,j-1].
    Retains U0's independent image stem and stride2/1 refinement.
    """
    def __init__(self):
        super().__init__()
        w = 128
        self.stem = ImageStem(16, 32)
        self.lateral = nn.ModuleList([nn.Conv2d(c, w, 1) for c in CHANNELS])
        self.top = block(w, w)
        self.nodes = nn.ModuleDict({f'{i}_{j}': nn.Sequential(block((j+1)*w, w), block(w, w))
                                    for j in (1, 2, 3) for i in range(4-j)})
        self.half_refine = nn.Sequential(block(w+32, 32), block(32, 32))
        self.native = nn.Sequential(block(32+16, 16), block(16, 16))
        self.classifier = nn.Conv2d(16, 1, 1)
        assert sum(p.numel() for p in self.parameters()) == SPECS['nested128']['parameters']

    def forward(self, features, image):
        native, half = self.stem(image)
        nodes = {(i, 0): layer(x) for i, (layer, x) in enumerate(zip(self.lateral, features))}
        nodes[3, 0] = self.top(nodes[3, 0])
        for j in (1, 2, 3):
            for i in range(4-j):
                values = [nodes[i, k] for k in range(j)]
                values.append(resize(nodes[i+1, j-1], nodes[i, 0].shape[-2:]))
                nodes[i, j] = self.nodes[f'{i}_{j}'](torch.cat(values, dim=1))
        x = self.half_refine(torch.cat([resize(nodes[0, 3], half.shape[-2:]), half], dim=1))
        return self.classifier(self.native(torch.cat([resize(x, image.shape[-2:]), native], dim=1)))


def build_head(name):
    if name in LOSS_HEADS or name in PREPROCESS_HEADS: return UNetHead('u0')
    if name == 'fpn128': return FPNHead()
    if name == 'nested128': return NestedHead()
    return UNetHead(name)


class SharedFrozenModel(nn.Module):
    def __init__(self, backbone_path, head_names=DEFAULT_HEADS, seed=42, backbone=None):
        super().__init__()
        if backbone is None:
            from transformers import DINOv3ConvNextBackbone
            backbone = DINOv3ConvNextBackbone.from_pretrained(
                str(backbone_path), local_files_only=True,
                out_features=['stage1', 'stage2', 'stage3', 'stage4'])
            assert tuple(backbone.channels) == CHANNELS
        self.backbone = backbone.requires_grad_(False).eval()
        heads = {}
        for name in head_names:
            # Initialization is independent of head order; no shared trainable stem/weights.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                heads[name] = build_head(name)
        self.heads = nn.ModuleDict(heads)
        seen = set()
        for head in self.heads.values():
            ids = {id(p) for p in head.parameters()}
            assert not seen & ids
            seen |= ids

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, x):
        # no_grad, NOT inference_mode: heads must save these tensors for backward.
        with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
            return [f.detach() for f in self.backbone(pixel_values=x.float()).feature_maps]


def autocast(device):
    return torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()


def semantic_loss(logits, target):
    """Identical to existing U0: BCE plus mean per-crop soft Dice, including negatives."""
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    intersection = (p*target).sum((1, 2, 3))
    dice = 1 - ((2*intersection+1)/(p.sum((1, 2, 3))+target.sum((1, 2, 3))+1)).mean()
    return bce+dice, bce, dice


def ablation_loss(logits, target, name):
    """Keep BCE/Dice monitoring identical; only change optimization objective."""
    baseline, bce, dice = semantic_loss(logits, target)
    if name not in LOSS_HEADS or name == 'L0': return baseline, bce, dice
    if name == 'L1': return 3*bce+dice, bce, dice
    logits, target = logits.float(), target.float()
    p = logits.sigmoid()
    if name == 'L2':
        pt = torch.where(target > .5, p, 1-p)
        focal = ((1-pt).square()*F.binary_cross_entropy_with_logits(logits, target, reduction='none')).mean()
        return focal+dice, bce, dice
    alpha, beta = (.6, .4) if name == 'L3' else (.4, .6)
    tp = (p*target).sum((1, 2, 3))
    fp = (p*(1-target)).sum((1, 2, 3))
    fn = ((1-p)*target).sum((1, 2, 3))
    # smooth=.5 makes alpha=beta=.5 algebraically identical to existing Dice.
    tversky = 1-((tp+.5)/(tp+alpha*fp+beta*fn+.5)).mean()
    return bce+tversky, bce, dice


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step+1)/max(1, warmup)
    fraction = min(1., max(0., (step-warmup)/max(1, total-1-warmup)))
    return .05 + .95*.5*(1+__import__('math').cos(__import__('math').pi*fraction))


def train_step(model, x_cpu, y_cpu, optimizers, microbatch, schedulers=None):
    """One effective-batch update per head; features computed once per microbatch."""
    model.train()
    device = next(model.heads.parameters()).device
    separate_inputs = isinstance(x_cpu, dict)
    if separate_inputs:assert set(x_cpu)==set(model.heads) and set(model.heads)<=set(PREPROCESS_HEADS)
    n = len(y_cpu)
    totals = {name: dict(loss=0., bce=0., dice_loss=0., counts=torch.zeros(4, device=device,
                                                                    dtype=torch.int64))
              for name in model.heads}
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    for start in range(0, n, microbatch):
        y = y_cpu[start:start+microbatch].to(device, non_blocking=True)
        if not separate_inputs:
            x = x_cpu[start:start+microbatch].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            features = model.features(x)
        for name, head in model.heads.items():
            if separate_inputs:
                x = normalized_input(x_cpu[name][start:start+microbatch], device)
                features = model.features(x)
            with autocast(device):
                logits = head(features, x)
                loss, bce, dice = ablation_loss(logits, y, name)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss: {name}')
            fraction = len(x)/n
            (loss*fraction).backward()
            with torch.no_grad():
                pred, gt = logits.detach().float().sigmoid() >= .5, y > 0
                totals[name]['counts'] += torch.stack([(pred & gt).sum(), (pred & ~gt).sum(),
                                                       (~pred & gt).sum(), (~pred & ~gt).sum()])
            for key, value in [('loss', loss), ('bce', bce), ('dice_loss', dice)]:
                totals[name][key] += float(value.detach())*fraction
            del logits, loss, bce, dice
            if separate_inputs:del features,x
        if not separate_inputs:del features,x
        del y
    # Validate ALL gradients before stepping ANY head.
    assert all(p.grad is None for p in model.backbone.parameters())
    for name, head in model.heads.items():
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        totals[name]['grad_norm'] = float(norm)
        totals[name]['lr'] = optimizers[name].param_groups[0]['lr']
    for name in model.heads:
        optimizers[name].step()
        if schedulers is not None:
            schedulers[name].step()
        totals[name]['counts'] = totals[name]['counts'].cpu().tolist()
    return totals
