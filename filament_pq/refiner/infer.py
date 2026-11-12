"""Stage 3 inference - letterbox preparation, 3 recurrent rounds, D4 test-time augmentation.

`prep` reproduces exactly the evaluation inputs used during training (same resize rules, same distance transform,
same presence prior), which was verified crop-by-crop against the training data loader.
Outputs are stored per crop as uint8 probabilities round(255 p) on the 512 letterbox, as during development.
"""

import numpy as np
import torch
import cv2
from scipy import ndimage as ndi

from ..config import RefinerConfig

SPATIAL = ('image', 'probability', 'anchor', 'distance', 'current', 'valid')


def prep(crop, side=512):
    """Letterbox arrays (top-left aligned, zero padded), presence prior and the resized (w, h)."""
    h, w = crop.image.shape
    s = side / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))

    def resize(a, mask=False):
        out = np.zeros((side, side), np.float32)
        out[:nh, :nw] = cv2.resize(
            a.astype(np.float32),
            (nw, nh),
            interpolation=cv2.INTER_NEAREST if mask else (cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR),
        )
        return out

    valid = np.zeros((side, side), bool)
    valid[:nh, :nw] = True
    anchor = resize(crop.component, True)
    inside = ndi.distance_transform_edt(anchor > 0)
    outside = ndi.distance_transform_edt(anchor == 0)
    distance = np.clip(inside - outside, -64, 64).astype(np.float32) / 64
    distance[~valid] = -1
    arrays = dict(
        image=resize(crop.image) / 255,
        probability=resize(crop.probability),
        anchor=anchor,
        distance=distance,
        current=anchor.copy(),
        valid=valid.astype(np.float32),
    )
    a = anchor > 0
    ml = float(np.clip(crop.ml_score, 1e-4, 1 - 1e-4))
    # prior = [logit(ML score), log anchor area fraction, mean proposal probability inside the anchor]
    prior = np.array(
        [
            np.log(ml / (1 - ml)),
            np.log(max(a.sum(), 1) / max(valid.sum(), 1)),
            float(arrays['probability'][a].mean()) if a.any() else 0.0,
        ],
        np.float32,
    )
    return arrays, prior, [nw, nh]


def _d4(t, k, flip):
    t = torch.rot90(t, k, (-2, -1))
    return torch.flip(t, (-1,)) if flip else t


def _d4_inv(t, k, flip):
    if flip:
        t = torch.flip(t, (-1,))
    return torch.rot90(t, -k, (-2, -1))


@torch.no_grad()
def forward(model, b, rounds=3, amp=True):
    """[(mask probability, presence probability)] for rounds 1..rounds."""
    with torch.autocast(b['image'].device.type, dtype=torch.float16, enabled=amp and b['image'].is_cuda):
        f16 = model.encode(model.features(b['image'], amp=amp))
        current = b['anchor']
        out = []
        for t in range(1, rounds + 1):
            z, presence = model(
                f16, *[b[k] for k in ('image', 'probability', 'anchor', 'distance')], current, b['valid'], t, b['prior']
            )
            m = z.float().sigmoid()
            p = presence.float().sigmoid()
            if not (torch.isfinite(m).all() and torch.isfinite(p).all()):
                raise FloatingPointError
            out.append((m, p))
            current = m * b['valid']
    return out


def infer_tta(model, b, n=8, rounds=3):
    """Mean over the first n of the 8 dihedral transforms (identity first); fp32 fallback if fp16 overflows."""
    ops = [(k, f) for f in (False, True) for k in range(4)][:n]
    acc = None
    for k, f in ops:
        t = dict(b)
        for key in SPATIAL:
            t[key] = _d4(b[key], k, f)
        try:
            out = forward(model, t, rounds)
        except FloatingPointError:
            out = forward(model, t, rounds, amp=False)
        cur = [(_d4_inv(m, k, f), p) for m, p in out]
        acc = cur if acc is None else [(a[0] + c[0], a[1] + c[1]) for a, c in zip(acc, cur)]
    return [(m / len(ops), p / len(ops)) for m, p in acc]


def refine(model, crops, device, cfg=RefinerConfig(), progress=None):
    """Refine a list of crops (batched in the given order). Returns one dict per crop:
    r1, r3 (uint8 512x512), presence [p1, p3], box, resized_wh, ml_score, native_anchor, crop_id."""
    results = []
    for s in range(0, len(crops), cfg.batch):
        chunk = crops[s : s + cfg.batch]
        ps = [prep(c, cfg.side) for c in chunk]
        b = {k: torch.from_numpy(np.stack([p[0][k] for p in ps]))[:, None].to(device) for k in SPATIAL}
        b['prior'] = torch.from_numpy(np.stack([p[1] for p in ps])).to(device)
        res = infer_tta(model, b, cfg.tta, cfg.rounds)
        q = lambda t: np.round(t.cpu().numpy() * 255).astype(np.uint8)
        r1, r3 = q(res[0][0][:, 0]), q(res[-1][0][:, 0])
        p1, p3 = res[0][1][:, 0].cpu().numpy(), res[-1][1][:, 0].cpu().numpy()
        for j, c in enumerate(chunk):
            results.append(
                dict(
                    r1=r1[j],
                    r3=r3[j],
                    presence=np.array([p1[j], p3[j]], np.float32),
                    box=np.array(c.box, np.int32),
                    resized_wh=np.array(ps[j][2], np.int32),
                    ml_score=np.float32(c.ml_score),
                    native_anchor=c.component.astype(bool),
                    crop_id=c.crop_id,
                )
            )
        if progress:
            progress(min(s + cfg.batch, len(crops)), len(crops))
    return results


def ensemble(members):
    """Average the fold models' outputs of one crop: mean probability (re-quantised to uint8) and mean presence."""
    c = dict(members[0])
    c['r3'] = np.round(np.mean([m['r3'].astype(np.float32) for m in members], 0)).astype(np.uint8)
    c['r1'] = np.round(np.mean([m['r1'].astype(np.float32) for m in members], 0)).astype(np.uint8)
    c['presence'] = np.mean([m['presence'] for m in members], 0).astype(np.float32)
    return c
