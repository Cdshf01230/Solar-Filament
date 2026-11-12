"""Stage 1 - proposal map.

A frozen DINOv3 ViT-S+/16 provides features from blocks 3, 6, 9 and 12. A small FPN-like head (`ProposalHead`,
426,993 parameters) fuses them with a half-resolution image stem and predicts a per-pixel filament probability.
The 2048 x 2048 image is processed as 9 overlapping 768 x 768 tiles blended with a floored Hann window.

Five heads were trained, each trained on its own fold only. At test time the five probability maps are averaged
(each map rounded to float16 first, accumulated in fold order in float32, divided by the number of heads).
"""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import ProposalConfig, IMAGE_SIZE

CHANNELS = (384, 384, 384, 384)


def block(cin, cout, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False), nn.GroupNorm(8, cout), nn.GELU()
    )


def resize(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ProposalHead(nn.Module):
    """Top-down fusion of four ViT feature maps (stride 16) plus a raw-image branch at half resolution."""

    def __init__(self):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in CHANNELS])
        self.smooth = nn.ModuleList([block(64, 64) for _ in CHANNELS])
        self.fuse = nn.Sequential(block(256, 64), block(64, 32))
        self.raw = block(3, 16)
        self.detail = nn.Sequential(block(48, 32), nn.Conv2d(32, 1, 1))

    def forward(self, features, image):
        xs = [m(x) for m, x in zip(self.lateral, features)]
        for i in (2, 1, 0):
            xs[i] = xs[i] + resize(xs[i + 1], xs[i].shape[-2:])
        xs = [m(x) for m, x in zip(self.smooth, xs)]
        z = self.fuse(torch.cat([resize(x, xs[0].shape[-2:]) for x in xs], 1))
        raw = self.raw(F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False))
        z = self.detail(torch.cat([resize(z, raw.shape[-2:]), raw], 1))
        return resize(z, image.shape[-2:])


class ProposalModel(nn.Module):
    """Frozen DINOv3 ViT-S+/16 backbone + one swappable proposal head."""

    def __init__(self, backbone_dir):
        super().__init__()
        from transformers import DINOv3ViTModel

        backbone = DINOv3ViTModel.from_pretrained(str(backbone_dir), local_files_only=True, attn_implementation='sdpa')
        self.backbone = backbone.requires_grad_(False).eval()
        c = backbone.config
        assert c.hidden_size == 384 and c.patch_size == 16 and c.use_gated_mlp and c.num_hidden_layers == 12
        self.head = ProposalHead()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, x):
        with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
            out = self.backbone(pixel_values=x.float(), output_hidden_states=True)
            n = (x.shape[-2] // 16) * (x.shape[-1] // 16)
            # the last n tokens are the spatial patches (CLS and register tokens are dropped)
            return [
                self.backbone.norm(out.hidden_states[i])[:, -n:]
                .transpose(1, 2)
                .reshape(x.shape[0], 384, x.shape[-2] // 16, x.shape[-1] // 16)
                .contiguous()
                for i in (3, 6, 9, 12)
            ]


def load_heads(weights_dir, folds):
    """State dicts of the fold heads (`weights/proposal/heads/fold{k}.pt`)."""
    heads = {}
    for k in folds:
        state = torch.load(
            Path(weights_dir) / 'proposal' / 'heads' / f'fold{k}.pt', map_location='cpu', weights_only=True
        )
        heads[k] = state
    return heads


def _autocast(device):
    from contextlib import nullcontext

    return torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()


@torch.inference_mode()
def tiled_probability(model, image, cfg=ProposalConfig()):
    """Probability map of one 2048 x 2048 uint8 image for the head currently loaded in `model`."""
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f'expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {image.shape}')
    device = next(model.head.parameters()).device
    positions = [(y, x) for y in cfg.tile_starts for x in cfg.tile_starts]
    t = cfg.tile
    window = np.maximum(np.outer(np.hanning(t), np.hanning(t)), cfg.window_floor).astype(np.float32)
    mean = np.array(cfg.mean, np.float32)[:, None, None]
    std = np.array(cfg.std, np.float32)[:, None, None]
    numerator = np.zeros(image.shape, np.float32)
    denominator = np.zeros(image.shape, np.float32)
    for start in range(0, len(positions), cfg.tile_batch):
        coords = positions[start : start + cfg.tile_batch]
        inputs = [np.repeat(image[y : y + t, x : x + t][None], 3, axis=0).astype(np.float32) / 255.0 for y, x in coords]
        batch = (
            torch.from_numpy(np.stack([(v - mean) / std for v in inputs]))
            .to(device)
            .contiguous(memory_format=torch.channels_last)
        )
        features = model.features(batch)
        with _autocast(device):
            p = model.head(features, batch).float().sigmoid()[:, 0].cpu().numpy()
        for q, (y, x) in zip(p, coords):
            numerator[y : y + t, x : x + t] += q * window
        for y, x in coords:
            denominator[y : y + t, x : x + t] += window
    # float16 rounding kept on purpose: it is the convention every reported number was produced with
    return (numerator / denominator).astype(np.float16).astype(np.float32)


@torch.inference_mode()
def fold_mean_probability(model, heads, image, folds, cfg=ProposalConfig()):
    """Mean of the fold heads' maps (float16-rounded maps, ascending fold order, float32 accumulation)."""
    mean = np.zeros(image.shape, np.float32)
    for k in sorted(folds):
        model.head.load_state_dict(heads[k], strict=True)
        model.eval()
        mean += tiled_probability(model, image, cfg).astype(np.float16).astype(np.float32)
    return mean / len(folds)
