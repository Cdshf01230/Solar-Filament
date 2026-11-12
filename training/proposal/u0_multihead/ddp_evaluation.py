"""Existing PQ scorer plus full-image inference losses, averaged by source image."""

from collections import defaultdict
import numpy as np
from .evaluation import Context as OriginalContext, evaluate as original_evaluate


class Context(OriginalContext):
    def score(self, name, prob, thresholds):
        rows = super().score(name, prob, thresholds)
        losses = {}
        p = np.clip(prob, 1e-7, 1 - 1e-7)
        log_p, log_bg = np.log(p), np.log1p(-p)
        for rec in self.records[name]:
            masks = self.masks(rec)
            target = np.logical_or.reduce(masks) if masks else np.zeros(prob.shape, bool)
            bce = float(-np.where(target, log_p, log_bg).mean(dtype=np.float64))
            dice = float(
                1 - (2 * prob[target].sum(dtype=np.float64) + 1) / (prob.sum(dtype=np.float64) + target.sum() + 1)
            )
            losses[rec['id']] = dict(bce=bce, soft_dice_loss=dice, loss=bce + dice)
        for row in rows:
            row['inference_loss'] = losses[row['record']]
        return rows


def evaluate(*args, **kwargs):
    result, rows = original_evaluate(*args, **kwargs)
    for name, rs in rows.items():
        groups = defaultdict(list)
        # Loss independent of threshold: include each annotator record once.
        first = rs[0]['threshold']
        for row in rs:
            if row['threshold'] == first:
                groups[row['source_file']].append(row['inference_loss'])
        loss = {
            k: float(np.mean([np.mean([r[k] for r in records]) for records in groups.values()]))
            for k in ('bce', 'soft_dice_loss', 'loss')
        }
        loss['definition'] = (
            'full-image sampled inference; mean annotator loss within each image, then mean images; negatives included'
        )
        result['heads'][name]['inference_loss'] = loss
    return result, rows
