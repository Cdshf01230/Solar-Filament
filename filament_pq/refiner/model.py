"""Stage 3 model - prompt-conditioned recurrent instance refiner ("RCR").

Inputs per crop, letterboxed to 512 x 512: raw image, proposal probability, anchor (component) mask, signed distance to
the anchor, the current mask estimate, and a validity mask for the letterbox padding.

* Backbone: DINOv3 ViT-B/16; blocks 1-8, embeddings and final norm frozen, blocks 9-12 fine-tuned with layer-wise LR
  decay ("llrd"). Hidden states of blocks 3/6/9/12 are summed through 1x1 projections into one stride-16 map.
* Decoder: a 5-stage convolutional stem on the 5 input channels (strides 1..16) is added to the ViT map, 4 global
  self-attention blocks (256-d, 8 heads, sine position code, round embedding) mix information along the filament,
  and a U-Net style upsampling path with the stem skips returns a full-resolution mask logit.
* Recurrence: the predicted mask of round t is fed back as the "current" channel of round t+1 (3 rounds).
* Presence: logit(ML score) + a small residual MLP over pooled tokens (anchor-weighted and global) and anchor
  statistics; it answers "is there a filament for this anchor at all?".
"""

import torch
from torch import nn
from torch.nn import functional as F


def conv(a, b, stride=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1), nn.GroupNorm(8, b), nn.GELU())


def pair(a, b, stride=1):
    return nn.Sequential(conv(a, b, stride), conv(b, b))


def position(h, w, device, dtype):
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    omega = 1 / (10000 ** (torch.arange(64, device=device).float() / 64))
    x, y = xx.flatten()[:, None] * omega, yy.flatten()[:, None] * omega
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], 1).to(dtype)[None]


class GlobalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(256), nn.LayerNorm(256)
        self.qkv = nn.Linear(256, 768)
        self.proj = nn.Linear(256, 256)
        self.mlp = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, x, valid):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, 8, 32).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        x = x + self.proj(y.transpose(1, 2).reshape(b, n, 256))
        return x + self.mlp(self.norm2(x))


class Decoder(nn.Module):
    def __init__(self, r_max=3):
        super().__init__()
        self.r_max = r_max
        self.neck = nn.ModuleList([nn.Sequential(nn.Conv2d(768, 256, 1), nn.GroupNorm(8, 256)) for _ in range(4)])
        self.stems = nn.ModuleList([pair(5, 32), pair(32, 64, 2), pair(64, 96, 2), pair(96, 128, 2), pair(128, 256, 2)])
        self.round_embed = nn.Embedding(r_max + 1, 256)
        self.blocks = nn.ModuleList([GlobalBlock() for _ in range(4)])
        self.ups = nn.ModuleList([pair(256 + 128, 128), pair(128 + 96, 96), pair(96 + 64, 64), pair(64 + 32, 32)])
        self.mask = nn.Conv2d(32, 1, 1)
        self.presence = nn.Sequential(nn.Linear(512 + 2, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1))

    def encode(self, features):
        return sum(layer(f) for layer, f in zip(self.neck, features))

    def forward(self, f16, image, probability, anchor, distance, current, valid, round_index, prior):
        x = torch.cat([image, probability, anchor, distance, current], 1)
        skips = []
        for layer in self.stems:
            x = layer(x)
            skips.append(x)
        b, _, h, w = x.shape
        emb = self.round_embed.weight[min(round_index, self.r_max)][None, :, None, None]
        tokens = (x + f16 + emb).flatten(2).transpose(1, 2)
        tokens = tokens + position(h, w, x.device, x.dtype)
        v = F.adaptive_max_pool2d(valid.float(), (h, w)).flatten(1).bool()
        if not v.any(1).all():
            raise ValueError('Every crop must contain valid pixels')
        for block in self.blocks:
            tokens = block(tokens, v)
        a = F.interpolate(anchor.float(), (h, w), mode='nearest').flatten(1) * v
        vf = v.to(tokens.dtype)
        mean = (tokens * vf[..., None]).sum(1) / vf.sum(1).clamp_min(1)[:, None]
        anchored = (tokens * a[..., None]).sum(1) / a.sum(1).clamp_min(1)[:, None]
        anchored = torch.where(a.sum(1)[:, None] > 0, anchored, mean)
        prior = prior.to(tokens.dtype)
        presence = prior[:, :1] + self.presence(torch.cat([anchored.detach(), mean.detach(), prior[:, 1:]], 1))
        x = tokens.transpose(1, 2).reshape(b, 256, h, w)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(torch.cat([F.interpolate(x, skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))
        return self.mask(x), presence


TUNED_BLOCKS = 4  # blocks 9..12 of the 12 ViT-B blocks carry fine-tuned weights


class Features(nn.Module):
    """DINOv3 ViT-B/16 feature extractor (inference): hidden states 3/6/9/12, final norm, patch tokens only."""

    def __init__(self, backbone_dir):
        super().__init__()
        from transformers import DINOv3ViTModel

        self.backbone = DINOv3ViTModel.from_pretrained(
            str(backbone_dir), local_files_only=True, attn_implementation='sdpa'
        )
        c = self.backbone.config
        assert c.hidden_size == 768 and c.patch_size == 16 and c.num_hidden_layers == 12
        self.backbone.requires_grad_(False).eval()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def blocks(self):
        return next(m for m in self.backbone.modules() if isinstance(m, nn.ModuleList) and len(m) == 12)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, image, amp=True):
        h, w = image.shape[-2:]
        with torch.autocast(image.device.type, dtype=torch.float16, enabled=amp and image.is_cuda):
            out = self.backbone(
                pixel_values=(image.repeat(1, 3, 1, 1) - self.mean) / self.std, output_hidden_states=True
            )
            n = h // 16 * (w // 16)
            return [
                self.backbone.norm(out.hidden_states[i])[:, -n:]
                .transpose(1, 2)
                .reshape(image.shape[0], 768, h // 16, w // 16)
                .contiguous()
                for i in (3, 6, 9, 12)
            ]


def build(backbone_dir, fold_weights, device):
    """Decoder + Features with the fold's decoder weights and its 4 fine-tuned ViT blocks.

    fold_weights: dict(decoder=state_dict, tuned_blocks={'<block index>.<param name>': tensor}) as written by
    scripts/export_weights.py. The other ViT weights come unchanged from the public DINOv3 checkpoint
    (export verified that they were frozen, i.e. bit-identical, during training)."""
    model = Decoder()
    model.features = Features(backbone_dir)
    missing = model.load_state_dict(fold_weights['decoder'], strict=False)
    assert not missing.unexpected_keys and all(k.startswith('features.') for k in missing.missing_keys), missing
    stack = model.features.blocks()
    loaded = 0
    for i in range(12 - TUNED_BLOCKS, 12):
        sd = {k.split('.', 1)[1]: v for k, v in fold_weights['tuned_blocks'].items() if int(k.split('.', 1)[0]) == i}
        stack[i].load_state_dict(sd, strict=True)
        loaded += len(sd)
    assert loaded == len(fold_weights['tuned_blocks'])
    return model.to(device).eval()
