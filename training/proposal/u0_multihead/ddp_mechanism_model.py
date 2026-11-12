"""Controlled decoder mechanisms; historical DDP implementation is untouched."""

import copy
import torch
from torch import nn
from torch.nn import functional as F
from .model import SharedFrozenModel
from .ddp_model import DDPHead, PROTOCOL, position

VARIANTS = ('ddp', 'direct', 'refine')


class DirectLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.attention, self.norm1 = layer.attention, layer.norm1
        self.ffn, self.norm2 = layer.ffn, layer.norm2

    def forward(self, x, pos, h, w):
        x = self.norm1(x + self.attention(x, pos, h, w))
        return self.norm2(x + self.ffn(x))


class DirectHead(DDPHead):
    """Same FPN and six spatial layers, without latent, noise or time affine."""

    def __init__(self):
        super().__init__()
        old = self.transform
        self.transform = nn.Conv2d(256, 256, 1)
        with torch.no_grad():
            self.transform.weight.copy_(old.weight[:, :256])
            self.transform.bias.copy_(old.bias)
        self.layers = nn.ModuleList([DirectLayer(layer) for layer in self.layers])
        del self.embedding, self.frequencies, self.time

    def forward(self, features, image):
        x = self.transform(self.image_features(features))
        b, c, h, w = x.shape
        pos = position(h, w, x.device)
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, pos, h, w)
        logits = self.classifier(x.transpose(1, 2).reshape(b, c, h, w))
        return F.interpolate(logits[:, 1:2] - logits[:, :1], image.shape[-2:], mode='bilinear', align_corners=False)

    def training_logits(self, features, image, target):
        # No GT reaches the predictor; target is used only by the outer loss.
        return self(features, image)


def detail_block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU()
    )


class RefinedHead(DDPHead):
    """Keep diffusion unchanged; add a small half-resolution residual logit head."""

    def __init__(self):
        super().__init__()
        self.raw_detail = detail_block(3, 32, stride=2)
        self.low_detail = nn.Sequential(nn.Conv2d(128, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.GELU())
        self.refinement = nn.Sequential(detail_block(65, 32), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)

    def refine(self, coarse, features, image):
        raw = self.raw_detail(image)
        size = raw.shape[-2:]
        low = F.interpolate(self.low_detail(features[0]), size, mode='bilinear', align_corners=False)
        coarse_half = F.interpolate(coarse, size, mode='bilinear', align_corners=False)
        residual = self.refinement(torch.cat([raw, low, coarse_half], 1))
        return coarse + F.interpolate(residual, image.shape[-2:], mode='bilinear', align_corners=False)

    def training_logits(self, features, image, target):
        return self.refine(super().training_logits(features, image, target), features, image)

    def forward(self, features, image):
        # Refinement runs once after the unchanged three-step DDIM sampler.
        return self.refine(super().forward(features, image), features, image)


def protocol(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    p = copy.deepcopy(PROTOCOL)
    p.update(version='ddp-mechanism-comparison-v1', variant=variant)
    if variant == 'direct':
        p.update(
            sampling_steps=1,
            sampling='deterministic direct prediction',
            time_embedding='removed',
            bit_scale=None,
            training_t=None,
            inference_noise='none',
            mechanism='FPN256 + same six spatial attention/FFN layers; no diffusion/time affine',
        )
    if variant == 'refine':
        p['refinement'] = (
            'raw stride2/32 + stage1 projection32 + coarse logit; 65->32->1 zero-init residual; once after DDIM'
        )
    p['initialization_comparison'] = (
        'same seed42 and exact common FPN/spatial weights; refine starts identical baseline logits; direct transform takes image-channel slice'
    )
    return p


class MechanismModel(SharedFrozenModel):
    def __init__(self, backbone_path, head_names=('ddp',), seed=42, backbone=None, variant='ddp'):
        assert list(head_names) == ['ddp']
        super().__init__(backbone_path, [], seed, backbone)
        self.experiment_protocol = protocol(variant)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            head = {'ddp': DDPHead, 'direct': DirectHead, 'refine': RefinedHead}[variant]()
        self.heads = nn.ModuleDict({'ddp': head})
