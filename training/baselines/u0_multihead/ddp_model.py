"""Binary adaptation of JiYuanFeng/DDP, without MMCV compiled operators.

Single-level deformable attention uses PyTorch grid_sample (align_corners=False).
Architecture follows official ConvNeXt-B config; backbone, objective, schedule and
deterministic inference noise are explicit adaptations. DDP means diffusion here.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import CHANNELS, SharedFrozenModel, autocast, semantic_loss

PROTOCOL = dict(
    version='ddp-binary-v1',
    reference='https://github.com/JiYuanFeng/DDP',
    width=256,
    layers=6,
    attention_heads=8,
    points=4,
    ffn=1024,
    time_embedding='learned sinusoidal 16 -> 1024 -> 1024; input cosine logSNR',
    bit_scale=0.01,
    training_t=[0.0, 0.999],
    sampling_steps=3,
    time_difference=1,
    sampling='DDIM discrete predicted-class embedding; official shifted next-time rule',
    inference_noise='CPU generator seed42, one latent shared across tiles; independent of batch and GT',
    objective='BCE(logit_fg-logit_bg) + original mean per-crop Dice smooth1; no auxiliary head',
    backbone='frozen FP32 DINOv3 ConvNeXt Base, TF32 enabled',
    attention_backend='torch grid_sample FP32; projections/FFN BF16 autocast',
    neck='FPN256 GN32 no activation, bilinear merge4levels stride4 -> Conv1x1 GN32',
    initialization='seed42; Xavier transformer matrices; radial attention offsets; other PyTorch defaults',
)


def log_snr(t):
    return -torch.log((torch.cos((t + 0.0002) / (1 + 0.00025) * math.pi * 0.5).pow(-2) - 1).clamp(min=1e-5))


def alpha_sigma(t):
    s = log_snr(t)
    return s.sigmoid().sqrt(), (-s).sigmoid().sqrt()


def sample_values(value, locations, weights):
    """value B,heads,dim,H,W; locations B,N,heads,points,xy in [0,1]."""
    b, heads, dim, h, w = value.shape
    n, points = locations.shape[1], locations.shape[3]
    grid = locations.permute(0, 2, 1, 3, 4).reshape(b * heads, n, points, 2)
    # CUDA grid_sample does not reliably implement BF16; accumulate in FP32.
    with torch.autocast(value.device.type, enabled=False):
        sampled = F.grid_sample(
            value.reshape(b * heads, dim, h, w).float(), grid.float() * 2 - 1, align_corners=False, padding_mode='zeros'
        )
        sampled = sampled.reshape(b, heads, dim, n, points).permute(0, 3, 1, 4, 2)
        return (sampled * weights.float()[..., None]).sum(3).reshape(b, n, heads * dim)


class DeformAttention(nn.Module):
    def __init__(self, width=256, heads=8, points=4):
        super().__init__()
        self.heads, self.points, self.width = heads, points, width
        self.offset = nn.Linear(width, heads * points * 2)
        self.weight = nn.Linear(width, heads * points)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.reset_attention()

    def reset_attention(self):
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)
        angles = torch.arange(self.heads) * 2 * math.pi / self.heads
        radial = torch.stack([angles.cos(), angles.sin()], -1)
        radial = radial / radial.abs().amax(-1, keepdim=True)
        radial = radial[:, None, :] * torch.arange(1, self.points + 1)[None, :, None]
        with torch.no_grad():
            self.offset.bias.copy_(radial.flatten())
        for layer in (self.value, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, pos, h, w):
        b, n, c = x.shape
        q = x + pos
        offset = self.offset(q).reshape(b, n, self.heads, self.points, 2)
        weight = self.weight(q).reshape(b, n, self.heads, self.points).softmax(-1)
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
        ref = torch.stack([(xx + 0.5) / w, (yy + 0.5) / h], -1).reshape(1, n, 1, 1, 2)
        locations = ref + offset.float() / x.new_tensor([w, h], dtype=torch.float32)
        value = self.value(x).reshape(b, h, w, self.heads, c // self.heads).permute(0, 3, 4, 1, 2)
        return self.output(sample_values(value, locations, weight))


def position(h, w, device, width=256):
    num = width // 2
    dim = 10000 ** (2 * torch.div(torch.arange(num, device=device), 2, rounding_mode='floor') / num)
    y = (torch.arange(h, device=device) + 0.5) / (h + 1e-6) * 2 * math.pi
    x = (torch.arange(w, device=device) + 0.5) / (w + 1e-6) * 2 * math.pi
    y, x = y[:, None] / dim, x[:, None] / dim
    y = torch.stack([y[:, 0::2].sin(), y[:, 1::2].cos()], -1).flatten(1)
    x = torch.stack([x[:, 0::2].sin(), x[:, 1::2].cos()], -1).flatten(1)
    return torch.cat([y[:, None].expand(h, w, num), x[None].expand(h, w, num)], -1).reshape(1, h * w, width)


class TimeLayer(nn.Module):
    def __init__(self, width=256, heads=8, time_dim=1024):
        super().__init__()
        self.attention = DeformAttention(width, heads)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * width))
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.xavier_uniform_(p)
        self.attention.reset_attention()

    def forward(self, x, pos, t, h, w):
        x = self.norm1(x + self.attention(x, pos, h, w))
        x = self.norm2(x + self.ffn(x))
        scale, shift = self.time(t).chunk(2, -1)
        return x * (1 + scale[:, None]) + shift[:, None]


def conv_norm(cin, cout, kernel):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False), nn.GroupNorm(32, cout))


class DDPHead(nn.Module):
    def __init__(self, width=256, layers=6, heads=8, time_dim=1024):
        super().__init__()
        assert width % 32 == 0 and width % heads == 0
        self.lateral = nn.ModuleList([conv_norm(c, width, 1) for c in CHANNELS])
        self.smooth = nn.ModuleList([conv_norm(width, width, 3) for _ in CHANNELS])
        self.merge = conv_norm(4 * width, width, 1)
        self.embedding = nn.Embedding(3, width)
        self.transform = nn.Conv2d(2 * width, width, 1)
        self.frequencies = nn.Parameter(torch.randn(8))
        self.time = nn.Sequential(nn.Linear(17, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.layers = nn.ModuleList([TimeLayer(width, heads, time_dim) for _ in range(layers)])
        self.classifier = nn.Conv2d(width, 2, 1)

    def image_features(self, features):
        levels = [layer(f) for layer, f in zip(self.lateral, features)]
        for i in (2, 1, 0):
            levels[i] = levels[i] + F.interpolate(levels[i + 1], size=levels[i].shape[-2:], mode='nearest')
        levels = [layer(f) for layer, f in zip(self.smooth, levels)]
        return self.merge(
            torch.cat(
                [F.interpolate(f, size=levels[0].shape[-2:], mode='bilinear', align_corners=False) for f in levels], 1
            )
        )

    def clean(self, labels):
        return (self.embedding(labels).sigmoid() * 2 - 1).permute(0, 3, 1, 2) * 0.01

    def decode(self, image_features, latent, t):
        x = self.transform(torch.cat([image_features, latent], 1))
        b, c, h, w = x.shape
        s = log_snr(t.float())[:, None]
        phase = s * self.frequencies[None] * 2 * math.pi
        time = self.time(torch.cat([s, phase.sin(), phase.cos()], -1))
        pos = position(h, w, x.device, c)
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, pos, time, h, w)
        return self.classifier(x.transpose(1, 2).reshape(b, c, h, w))

    def training_logits(self, features, image, target):
        fused = self.image_features(features)
        labels = F.interpolate(target.float(), size=fused.shape[-2:], mode='nearest')[:, 0].long()
        clean = self.clean(labels)
        t = torch.rand(len(image), device=image.device) * 0.999
        a, s = alpha_sigma(t)
        latent = a[:, None, None, None] * clean + s[:, None, None, None] * torch.randn_like(clean)
        logits = self.decode(fused, latent, t)
        return F.interpolate(
            logits[:, 1:2] - logits[:, :1], size=image.shape[-2:], mode='bilinear', align_corners=False
        )

    def forward(self, features, image):
        fused = self.image_features(features)
        b, c, h, w = fused.shape
        # Local CPU generator does not consume global training RNG, even in validation.
        noise = torch.randn(1, c, h, w, generator=torch.Generator().manual_seed(42))
        latent = noise.to(image.device).expand(b, -1, -1, -1)
        for step in range(3):
            t = torch.full((b,), 1 - step / 3, device=image.device)
            tn = torch.full((b,), max(1 - (step + 2) / 3, 0), device=image.device)
            logits = self.decode(fused, latent, t)
            clean = self.clean(logits.argmax(1))
            a, s = alpha_sigma(t)
            an, sn = alpha_sigma(tn)
            eps = (latent - a[:, None, None, None] * clean) / s[:, None, None, None].clamp(min=1e-8)
            latent = an[:, None, None, None] * clean + sn[:, None, None, None] * eps
        return F.interpolate(
            logits[:, 1:2] - logits[:, :1], size=image.shape[-2:], mode='bilinear', align_corners=False
        )


class DDPModel(SharedFrozenModel):
    experiment_protocol = PROTOCOL

    def __init__(self, backbone_path, head_names=('ddp',), seed=42, backbone=None):
        assert list(head_names) == ['ddp']
        super().__init__(backbone_path, [], seed, backbone)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.heads = nn.ModuleDict({'ddp': DDPHead()})


def train_step(model, x_cpu, y_cpu, optimizers, microbatch, schedulers=None):
    model.train()
    device = next(model.heads.parameters()).device
    optimizer = optimizers['ddp']
    optimizer.zero_grad(set_to_none=True)
    totals = dict(loss=0.0, bce=0.0, dice_loss=0.0, counts=torch.zeros(4, device=device, dtype=torch.int64))
    n = len(y_cpu)
    for start in range(0, n, microbatch):
        x = x_cpu[start : start + microbatch].to(device).contiguous(memory_format=torch.channels_last)
        y = y_cpu[start : start + microbatch].to(device)
        features = model.features(x)
        with autocast(device):
            logits = model.heads['ddp'].training_logits(features, x, y)
            loss, bce, dice = semantic_loss(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('DDP loss not finite')
        (loss * (len(y) / n)).backward()
        with torch.no_grad():
            p, g = logits.detach().sigmoid() >= 0.5, y > 0
            totals['counts'] += torch.stack([(p & g).sum(), (p & ~g).sum(), (~p & g).sum(), (~p & ~g).sum()])
        for k, v in [('loss', loss), ('bce', bce), ('dice_loss', dice)]:
            totals[k] += float(v.detach()) * len(y) / n
    assert all(p.grad is None for p in model.backbone.parameters())
    totals['grad_norm'] = float(nn.utils.clip_grad_norm_(model.heads.parameters(), 1.0, error_if_nonfinite=True))
    totals['lr'] = optimizer.param_groups[0]['lr']
    optimizer.step()
    if schedulers is not None:
        schedulers['ddp'].step()
    totals['counts'] = totals['counts'].cpu().tolist()
    return dict(ddp=totals)
