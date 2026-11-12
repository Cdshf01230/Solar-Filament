# Results

Protocol: development OOF on 5 date-disjoint folds of 707 labelled images / 1154 original records / 8199 GT.
Direct predictions use proposal heads/classifiers of the other four folds and the refiner trained on the other four.
Upstream ensemble members are shared across downstream training datasets: this is not fully nested end-to-end CV. Scores are computed on
full 2048 images, per original annotation record, and include ground truth lost before the refiner (an image without
any crop still counts its filaments as FN). TP/FP/FN and summed IoU are pooled over all records before PQ.

## Final pipeline (out-of-fold, 707 images, 8199 GT)

| Setting | Legacy PQ | Legacy TP / FP / FN | Strict PQ | Strict TP / FP / FN |
|---|---:|---|---:|---|
| ep30, threshold at 512 + nearest, no TTA | .422733 | 4994 / 2621 / 3205 | .422675 | 4993 / 2621 / 3206 |
| + bilinear native resampling, t = .5 | .436333 | 5066 / 2549 / 3133 | .436273 | 5065 / 2549 / 3134 |
| **+ D4 TTA x8 (submitted)** | **.440524** | **5101 / 2512 / 3098** | **.440463** | **5100 / 2512 / 3099** |

Source: stored inference evaluation `oof_scores_ep30.json` from the submitted run; settings `t1_old`, `t1_lin50`,
`t8_lin50`. Submitted SQ/RQ: legacy .682764/.645206; strict .682761/.645120. Pixel-disjoint predictions do not
ensure identical matching when original GT masks overlap.

Leave-one-fold-out post-processing selection among 11 stored-prediction settings gives .4396 legacy / .4395 strict.
For each evaluation fold, choose settings on the other four; no model is retrained inside this check. This is a check
of settings selection, not nested end-to-end evaluation or proof that selection is free from development bias.

The organisers' notebook (`get_pq_score`), run on our OOF predictions written in submission format, returns .440524,
equal to our legacy scorer.

Per fold (strict, submitted setting): .4295, .4300, .4472, .4521, .4454.

Public leaderboard (test, about half of 180 images): **.39**. For reference, the organisers stated that "any PQ score
of greater than 0.35 is of great value", and one human annotator scored against another reaches .34 on the training
images.

## Fresh out-of-fold rerun of the released code (12 Nov 2026)

The released package and weights, downloaded from the public Hugging Face copy (identical to GitHub by hash) and
installed with `pip install -r requirements.txt`, were rerun on two Colab NVIDIA L4 runtimes in out-of-fold mode
(stages 1-2 with the four non-owner heads and classifiers, the owner refiner, D4 TTA, bilinear native resampling).
Predictions: `reference/oof_predictions.csv`; scores and statistics: `python scripts/evaluate_oof.py` (output in
`docs/results/oof/`).

| Fold | Strict PQ | Strict TP / FP / FN | Legacy PQ | Legacy TP / FP / FN |
|---:|---:|---|---:|---|
| 0 | .430005 | 995 / 546 / 623 | .430005 | 995 / 546 / 623 |
| 1 | .429906 | 1127 / 602 / 690 | .429906 | 1127 / 602 / 690 |
| 2 | .446747 | 1003 / 486 / 578 | .447055 | 1004 / 486 / 577 |
| 3 | .453073 | 1055 / 450 / 627 | .453073 | 1055 / 450 / 627 |
| 4 | .444595 | 915 / 414 / 586 | .444595 | 915 / 414 / 586 |
| **Pooled** | **.440507** | 5095 / 2498 / 3104 | .440568 | 5096 / 2498 / 3103 |

Recorded submitted setting: .440463 strict / .440524 legacy. 555 of the 707 images give the same strict TP / FP / FN
and IoU sum (to .001); the rest differ because the release recomputes the stage-1 maps (development read stored maps), giving
5497 crops instead of 5493 (per fold 1123, 1167, 1061, 1120, 1026 against 1122, 1166, 1058, 1119, 1028).
Strict 95% bootstrap interval over images (2000 resamples, seed 0): [.4318, .4491]; fold mean .4409, SD .0104.
Fold 2 differs between the scorers by .0003 PQ (one TP); the pooled rows by .00006 (one TP).

Organisers' rubric statistics on the rerun, per record (pairs and degrees with IoU > 0, as in their self-evaluation
notebook): 6696 overlapping pairs, mean IoU .598, mean Dice .730; strict true positives mean IoU .683, mean Dice
.808; GT overlapped by 0 / 1 / 2 / 3 predictions: 1738 / 6242 / 203 / 16; (prediction, record) pairs overlapping
0 / 1 / 2 / 3 GT: 1071 / 6352 / 166 / 4; Dice of the union masks per record: mean .664, pooled .690.

## Training history of the refiner (5 folds pooled, threshold at 512 rule, no TTA)

These are the original training-evaluator records (strict), retained as recorded. Its epoch-30 result
(.4226; 4992/2621/3207 TP/FP/FN) differs slightly from rescoring the stored inference dump above
(.422675; 4993/2621/3206 strict). The two evaluation paths are not interchangeable; do not combine their counts.

| Epoch | Crop IoU train | Crop IoU val | PQ train | PQ val | Val TP / FP / FN |
|---|---|---|---|---|---|
| 1 | .548 | .548 | .345 | .344 | 4239 / 3428 / 3960 |
| 5 | .582 | .580 | .403 | .398 | 4788 / 2925 / 3411 |
| 10 | .588 | .584 | .420 | .411 | 4894 / 2788 / 3305 |
| 15 | .599 | .591 | .426 | .417 | 4961 / 2730 / 3238 |
| 20 | .604 | .593 | .434 | .423 | 5002 / 2630 / 3197 |
| 25 | .608 | .594 | .436 | .420 | 4972 / 2611 / 3227 |
| 30 | .611 | .595 | .440 | .423 | 4992 / 2621 / 3207 |

"Train" = each fold model on its own 566 training images without augmentation. The gap grows slowly (.017 PQ at epoch
30) and validation is flat after epoch 20; the fixed epoch 30 is kept (leave-one-fold-out epoch choice: .4203).

## Error audit of the submitted setting (OOF)

Strict scorer: 5100 TP / 2512 FP / 3099 FN. FN (3099) by cause, each GT counted once:

| Cause | FN | PQ if all fixed* |
|---|---|---|
| inside some crop but no prediction touches it | 1329 | .499 |
| &nbsp;&nbsp;- of which: the GT was the target of a crop whose mask was gated or missed it | 495 | |
| &nbsp;&nbsp;- of which: the GT only lies in crop padding (no anchor touches it) | 514 | |
| incomplete mask (precision >= .7, recall < .7) | 558 | .483 |
| lost upstream (no crop at all) | 402 | .459 |
| mask too wide (recall >= .7, precision < .7) | 332 | .465 |
| merged with a neighbouring GT | 219 | .454 |
| misaligned | 146 | .452 |
| split into >= 2 predictions | 82 | .447 |
| covered only by predictions already matched to another GT | 29 | .442 |
| blocked by the presence gate | 2 | .441 |

\* each fixed GT counted as a TP with IoU .6, and a touching unmatched prediction removed.

FP (2512): 1079 on annotation background (removing all of them: .473), 1433 overlap a labelled filament with
IoU <= .5. Background here is relative to that original record; incomplete annotations can leave real structures unmatched.

Incomplete masks: 55% of the missing pixels are inside the anchor and 43% outside the anchor but inside the crop;
77% of these GT are touched by a single anchor that covers a median 81% of them. **The refiner sees the missing part
but does not paint it.** 506 GT (6.2%) are covered by an anchor (median 99.8% coverage) but are never the target of
any crop because the same anchor contains another filament; 481 of them are FN. Both observations point to the
one-output-per-prompt design as the main remaining limit.

## Negative result: multiple instances per prompt without retraining

We tested whether the existing refiner can already produce a second instance per anchor (OOF, 707 images, both
scorers agree within .0001):

| Setting | PQ (1-1) | TP | FP |
|---|---|---|---|
| submitted (one instance per prompt) | .4405 | 5100 | 2512 |
| keep every connected component touching the anchor | .4305 | 5208 | 3098 |
| re-prompt on the unexplained part of the anchor, parent ML prior, presence >= .8, area >= 512 | .4195 | 5257 | 3616 |
| same, neutral prior .5, presence >= .6, area >= 512 | .4409 | 5143 | 2585 |

Re-prompting on the residual anchor (anchor pixels with p >= .3 that are more than 5 px from the first mask) finds up
to 164 of the 506 never-targeted filaments, so the second instance is visible to the model, but at about 14%
precision: the presence head was never trained on residual prompts. The best zero-shot variant is within noise of the
submission (+.0004), so it is not used. A trained second output slot (with its own presence) is the natural next step.

## Head comparison (README §2.5)

The comparison of FPN / U0 / DDP-mini (train209, quick74 selection, audit424 reporting) uses legacy all-pairs only;
strict was not measured. Their audit424 legacy PQ values are .2968 / .3396 / .3687. It uses a different split and is
not a controlled comparison with the 707-image development OOF pipeline.

## Standard DDP head on the five folds (README §2.6)

Same folds, records and scorers as the final pipeline; threshold .5, 8-connected components, area ≥ 128 px, no TTA.
Each fold model is trained for 20 epochs (cosine learning-rate decay ending at 29 % of the peak). Both scorers, epoch
20 of every fold:

| Fold | Images | Records | GT | Strict PQ | Strict TP / FP / FN | Legacy PQ | Legacy TP / FP / FN |
|---:|---:|---:|---:|---:|---|---:|---|
| 0 | 141 | 231 | 1618 | .391846 | 1064 / 943 / 554 | .391846 | 1064 / 943 / 554 |
| 1 | 141 | 248 | 1817 | .386599 | 1190 / 1118 / 627 | .386599 | 1190 / 1118 / 627 |
| 2 | 142 | 221 | 1581 | .386243 | 1046 / 1039 / 535 | .386508 | 1047 / 1039 / 534 |
| 3 | 141 | 228 | 1682 | .393669 | 1193 / 1226 / 489 | .393669 | 1193 / 1226 / 489 |
| 4 | 142 | 226 | 1501 | .384297 | 1064 / 1183 / 437 | .384297 | 1064 / 1183 / 437 |
| **Pooled** | 707 | 1154 | 8199 | **.388575** | 5557 / 5509 / 2642 | .388626 | 5558 / 5509 / 2641 |

Epoch selection within epochs 1–20 (pooled; strict / legacy PQ): epoch 20 for every fold, submitted, .388575 / .388626;
epoch chosen on the other four folds (19, 19, 19, 18, 18) .392391 / .392443; one epoch for all folds chosen on the
pooled score (19) .398402 / .398455; each fold's own best epoch (20, 20, 19, 19, 19) .405720 / .405775. The last two
select on the reported data. `training/baselines/ddp_5fold/epoch_table.py` recomputes these values from
`docs/results/ddp_5fold/`. Fold 2 is the only row where the scorers differ by more than .0001 PQ (.000265, one TP).

## Scope of the annotation-assistance claim

Both filter policies are released, but all refiner results use balanced crops. Recall48 candidate coverage is not
end-to-end instance recall or final PQ. No user study has measured review time, correction quality or improvement
in annotator agreement. Imperfect labels and shared upstream OOF dependencies limit the conclusions above.
