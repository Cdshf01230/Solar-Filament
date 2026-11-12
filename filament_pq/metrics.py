"""Evaluation used during development (all at full-image level, per original annotator record).

* `host` : the organisers' rule - every (GT, prediction) pair with IoU > .5 is a TP; FP = prediction matching no GT;
           FN = GT matching no prediction; summed over all (annotator, image) pairs, then PQ.
* `strict`: the same but with a one-to-one maximum matching (Hungarian), so a prediction can match one GT at most.
  Pixel-disjoint predictions can still match overlapping GT differently under the two rules. Report each separately;
  `strict` is primary because it cannot be inflated by duplicate predictions.
* `relations`: one-to-many / many-to-one counts (a GT touched by >= 2 predictions, a prediction touching >= 2 GT),
  using "touch" = overlap >= 8 px and >= 5% of the GT, plus per-matched-pair IoU and Dice distributions.
"""

import numpy as np
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment


def iou_matrix(gt_rles, pred_rles):
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), np.float32)
    return np.asarray(mu.iou(gt_rles, pred_rles, [0] * len(pred_rles)), np.float32).reshape(
        len(gt_rles), len(pred_rles)
    )


def match(iou):
    """Returns host [TP, FP, FN, sumIoU] and strict [TP, FP, FN, sumIoU] for one (annotator, image)."""
    ng, npred = iou.shape
    q = iou > 0.5
    host = [int(q.sum()), int((~q.any(0)).sum()), int((~q.any(1)).sum()), float(iou[q].astype(float).sum())]
    if ng and npred:
        a, b = linear_sum_assignment(q * (1 + iou / (min(ng, npred) + 1)), maximize=True)
        keep = q[a, b]
        tp = int(keep.sum())
        si = float(iou[a[keep], b[keep]].astype(float).sum())
    else:
        tp, si = 0, 0.0
    return host, [tp, npred - tp, ng - tp, si]


def pq(counts):
    tp, fp, fn, si = map(float, counts)
    den = tp + 0.5 * (fp + fn)
    return dict(pq=si / max(den, 1e-12), sq=si / max(tp, 1), rq=tp / max(den, 1e-12), tp=tp, fp=fp, fn=fn)


def relations(gt_rles, pred_rles):
    """one_to_many: GT touched by >= 2 predictions; many_to_one: predictions touching >= 2 GT; plus matched IoU/Dice."""
    iou = iou_matrix(gt_rles, pred_rles)
    if not iou.size:
        return dict(one_to_many=0, many_to_one=0, iou=[], dice=[])
    ga = mu.area(gt_rles).astype(float)[:, None]
    pa = mu.area(pred_rles).astype(float)[None]
    inter = iou * (ga + pa) / (1 + iou)
    touch = (inter >= 8) & (inter >= 0.05 * ga)
    q = iou > 0.5
    a, b = np.nonzero(q)
    return dict(
        one_to_many=int((touch.sum(1) >= 2).sum()),
        many_to_one=int((touch.sum(0) >= 2).sum()),
        iou=iou[a, b].tolist(),
        dice=(2 * inter[a, b] / (ga[a, 0] + pa[0, b])).tolist(),
    )
