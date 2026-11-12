"""RCR v1 combo P+A+L+B: absolute masks, detached recurrent prompts, DINOv3 ViT-B last 4 blocks fine-tuned (B),
ML-anchored presence (P), Skeleton Recall loss (L). Augmentation A lives in data.py."""

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
        nn.init.constant_(self.mask.bias, -2.0)
        # P: presence = logit(ML score) + residual MLP on DETACHED pooled tokens + anchor stats; dropout .3; zero-init residual.
        self.presence = nn.Sequential(nn.Linear(512 + 2, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1))
        nn.init.zeros_(self.presence[-1].weight)
        nn.init.zeros_(self.presence[-1].bias)

    def encode(self, features):
        return sum(layer(f) for layer, f in zip(self.neck, features))

    def forward(self, f16, image, probability, anchor, distance, current, valid, round_index, prior=None):
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
        if prior is None:
            prior = torch.zeros(b, 3, device=tokens.device, dtype=tokens.dtype)
        prior = prior.to(tokens.dtype)
        presence = prior[:, :1] + self.presence(torch.cat([anchored.detach(), mean.detach(), prior[:, 1:]], 1))
        x = tokens.transpose(1, 2).reshape(b, 256, h, w)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(torch.cat([F.interpolate(x, skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))
        return self.mask(x), presence


TUNE_BLOCKS = 4
BACKBONE_LR, BACKBONE_LAYER_DECAY = 2e-5, 0.8


class PartialFeatures(nn.Module):
    """Arm B: DINOv3 ViT-B with only the last TUNE_BLOCKS transformer blocks trainable.
    Embeddings, earlier blocks and the final norm stay frozen; the module stays in eval mode."""

    def __init__(self, path, tune=TUNE_BLOCKS, drop_path=0.0):
        super().__init__()
        from transformers import DINOv3ViTModel

        # drop_path > 0 builds DropPath (no parameters) in every block; only the tuned blocks ever run in train mode.
        extra = dict(drop_path_rate=drop_path) if drop_path > 0 else {}
        self.backbone = DINOv3ViTModel.from_pretrained(
            str(path), local_files_only=True, attn_implementation='sdpa', **extra
        )
        self.drop_path = drop_path
        c = self.backbone.config
        assert c.hidden_size == 768 and c.patch_size == 16 and c.num_hidden_layers == 12
        self.backbone.requires_grad_(False).eval()
        stacks = [m for m in self.backbone.modules() if isinstance(m, nn.ModuleList) and len(m) == 12]
        assert len(stacks) == 1
        self.tune = tune
        for block in stacks[0][12 - tune :]:
            block.requires_grad_(True)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def blocks(self):
        return next(m for m in self.backbone.modules() if isinstance(m, nn.ModuleList) and len(m) == 12)

    def param_groups(self, weight_decay):
        """Top block at BACKBONE_LR, each lower tuned block x BACKBONE_LAYER_DECAY."""
        stack = self.blocks()
        return [
            dict(
                params=list(stack[11 - k].parameters()),
                lr=BACKBONE_LR * BACKBONE_LAYER_DECAY**k,
                weight_decay=weight_decay,
                name=f'backbone_block{12-k}',
            )
            for k in range(self.tune)
        ]

    def set_tuning(self, on):
        """LP-FT: freeze (on=False) or release the tuned blocks; frozen blocks get no grad, so AdamW skips them."""
        for block in self.blocks()[12 - self.tune :]:
            block.requires_grad_(on)

    def train(self, mode=True):
        super().train(False)
        if mode and self.drop_path > 0:
            for block in self.blocks()[12 - self.tune :]:
                block.train(True)  # drop path only where weights train
        return self

    def forward(self, image, amp=True):
        # Frozen prefix records no graph (no input/parameter requires grad); tuned blocks do.
        h, w = image.shape[-2:]
        with torch.autocast(image.device.type, dtype=torch.float16, enabled=amp and image.is_cuda):
            result = self.backbone(
                pixel_values=(image.repeat(1, 3, 1, 1) - self.mean) / self.std, output_hidden_states=True
            )
            n = h // 16 * (w // 16)
            return [
                self.backbone.norm(result.hidden_states[i])[:, -n:]
                .transpose(1, 2)
                .reshape(image.shape[0], 768, h // 16, w // 16)
                .contiguous()
                for i in (3, 6, 9, 12)
            ]


def lovasz_hinge(logits, labels):
    signs = 2 * labels - 1
    errors, order = (1 - logits * signs).sort(descending=True)
    fg = labels[order]
    total = fg.sum()
    intersection = total - fg.cumsum(0)
    union = total + (1 - fg).cumsum(0)
    grad = 1 - intersection / union.clamp_min(1)
    grad = torch.cat([grad[:1], grad[1:] - grad[:-1]])
    return torch.dot(F.relu(errors), grad)


SKELETON_RECALL_WEIGHT = 1.0


def loss_terms(logits, presence, targets, valid, skeletons):
    """Crop-balanced loss; targets/skeletons are lists of original-record stacks (R,H,W).

    L adds Skeleton Recall on positive records: 1 - sum(p*S)/sum(S) with S the tubed GT centerline."""
    losses, parts = [], []
    for i, records in enumerate(targets):
        v = valid[i, 0].bool()
        z, y = logits[i, 0].float()[v], records.float()[:, v]
        s = skeletons[i].float()[:, v]
        positive = y.sum(1) > 0
        zero = z.sum() * 0
        mask_loss = (
            torch.stack([F.binary_cross_entropy_with_logits(z, r) + lovasz_hinge(z, r) for r in y[positive]]).mean()
            if positive.any()
            else zero
        )
        pres_loss = F.binary_cross_entropy_with_logits(presence[i].float().expand(len(y)), positive.float())
        empty_loss = F.binary_cross_entropy_with_logits(z, torch.zeros_like(z)) if not positive.any() else zero
        p = torch.sigmoid(z)
        skel_loss = (
            torch.stack([1 - (p * k).sum() / k.sum().clamp_min(1) for k in s[positive]]).mean()
            if positive.any()
            else zero
        )
        losses.append(mask_loss + 0.5 * pres_loss + 0.5 * empty_loss + SKELETON_RECALL_WEIGHT * skel_loss)
        parts.append(torch.stack([mask_loss.detach(), pres_loss.detach(), empty_loss.detach(), skel_loss.detach()]))
    return torch.stack(losses).mean(), torch.stack(parts).mean(0)
