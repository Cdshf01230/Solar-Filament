"""Small supervised proposal head and coverage metrics; no diffusion or GT input."""

import torch
from torch import nn
from torch.nn import functional as F
from u0_multihead.model import block, resize

CHANNELS = (384, 384, 384, 384)
from u0_multihead.ddp_evaluation import Context as BaseContext
from u0_multihead.evaluation import summarize as segmentation_summary

THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class ProposalHead(nn.Module):
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


class TrainModel(nn.Module):
    def __init__(self, backbone_path, variant=None, seed=42, backbone=None):
        super().__init__()
        if backbone is None:
            from transformers import DINOv3ViTModel

            backbone = DINOv3ViTModel.from_pretrained(
                str(backbone_path), local_files_only=True, attn_implementation='sdpa'
            )
        self.backbone = backbone.requires_grad_(False).eval()
        assert backbone.config.hidden_size == 384 and backbone.config.patch_size == 16
        assert backbone.config.use_gated_mlp and backbone.config.num_hidden_layers == 12
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.heads = nn.ModuleDict({'ddp': ProposalHead()})
        assert not any(p.requires_grad for p in self.backbone.parameters())

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, x):
        with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
            result = self.backbone(pixel_values=x.float(), output_hidden_states=True)
            n = (x.shape[-2] // 16) * (x.shape[-1] // 16)
            # Last n tokens are spatial patches; CLS/registers are excluded.
            return [
                self.backbone.norm(result.hidden_states[i])[:, -n:]
                .transpose(1, 2)
                .reshape(x.shape[0], 384, x.shape[-2] // 16, x.shape[-1] // 16)
                .contiguous()
                .detach()
                for i in (3, 6, 9, 12)
            ]

    def forward(self, image, target=None):
        return self.heads['ddp'](self.features(image), image)


LOSSES = {'proposal': '4-positive-weight BCE + Tversky(FP=.2,FN=.8,smooth=.5)'}


def loss_terms(logits, target, arm):
    assert arm == 'proposal'
    z, y = logits.float(), target.float()
    p = z.sigmoid()
    d = (1, 2, 3)
    bce = F.binary_cross_entropy_with_logits(z, y)
    wbce = F.binary_cross_entropy_with_logits(z, y, pos_weight=z.new_tensor(4.0))
    tp = (p * y).sum(d)
    fp = (p * (1 - y)).sum(d)
    fn = ((1 - p) * y).sum(d)
    dice = 1 - ((2 * tp + 1) / (2 * tp + fp + fn + 1)).mean()
    tv = 1 - ((tp + 0.5) / (tp + 0.2 * fp + 0.8 * fn + 0.5)).mean()
    return wbce + tv, dict(bce=bce, dice=dice, common_bce_dice=bce + dice)


def coverage_counts(prob, masks, threshold):
    binary = prob >= threshold
    fractions = [float(binary[m].mean()) for m in masks]
    return dict(
        objects=len(masks),
        hit10=sum(v >= 0.1 for v in fractions),
        covered50=sum(v >= 0.5 for v in fractions),
        covered90=sum(v >= 0.9 for v in fractions),
        covered95=sum(v >= 0.95 for v in fractions),
        missed_completely=sum(v == 0 for v in fractions),
        fractions=fractions,
        predicted_pixels=int(binary.sum()),
        image_pixels=int(binary.size),
    )


class Context(BaseContext):
    def score(self, name, prob, thresholds):
        rows = super().score(name, prob, thresholds)
        masks = {r['id']: self.masks(r) for r in self.records[name]}
        for r in rows:
            r['coverage'] = coverage_counts(prob, masks[r['record']], r['threshold'])
        return rows


def summarize(rows, threshold):
    result = segmentation_summary(rows, threshold)
    rs = [r for r in rows if r['threshold'] == threshold]
    n = sum(r['coverage']['objects'] for r in rs)
    totals = {
        k: sum(r['coverage'][k] for r in rs)
        for k in ['objects', 'hit10', 'covered50', 'covered90', 'covered95', 'missed_completely']
    }
    per_image = {r['source_file']: r['coverage'] for r in rs}
    predicted = sum(v['predicted_pixels'] for v in per_image.values())
    size = sum(v['image_pixels'] for v in per_image.values())
    result['proposal'] = dict(
        **totals,
        hit10_recall=totals['hit10'] / max(1, n),
        coverage90_recall=totals['covered90'] / max(1, n),
        coverage95_recall=totals['covered95'] / max(1, n),
        area_fraction=predicted / max(1, size),
        feasible=predicted / max(1, size) <= 0.05 and result['pixel']['precision'] >= 0.1,
    )
    return result


def selection_key(row):
    m = row['proposal']
    # A mask covering everything is never eligible for a successful proposal claim.
    if not m['feasible']:
        return (0, row['pixel']['precision'], -m['area_fraction'], 0.0)
    return (1, m['coverage90_recall'], row['pixel']['recall'], -m['area_fraction'])


def epoch_key(row):
    return selection_key(row['selected'])
