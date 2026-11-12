# Design decisions and the evidence behind them

Choices below use development results on labelled images, including historical splits. Current OOF is not fully
nested end-to-end CV because upstream models are reused across downstream fold datasets. Numbers are strict
one-to-one PQ unless marked legacy (organisers' all-pairs); rounded agreement does not imply identical TP/FP/FN. The
first head comparison uses an older split (209 training / 498 held-out images) and is marked *historical*: it guided
the direction but is not directly comparable with the 5-fold numbers.

## D1. Detect-then-refine instead of semantic segmentation + connected components

PQ counts instances. Semantic segmentation followed by connected components fails in exactly the ways the challenge
description lists: one filament broken into several components (fragmentation), or neighbouring dark islands joined
into one (over-merging). Both cost a TP and add FPs.

* *Historical* full-image semantic heads on frozen DINOv3 ConvNeXt-B + CC, 20 epochs, 209 training images, epoch and
  threshold chosen on 74 validation images, legacy PQ on the other 424 (README §2.5): FPN .297 (1.58 M parameters),
  U-Net-style head .340 (1.79 M), diffusion-style head DDP-mini .369 (2.12 M).
* The best of them, DDP, at its standard size (11.56 M parameters, README §2.6), trained for 20 epochs on the five
  folds, scores .3886 strict OOF without TTA (.3924 with its epoch chosen on the other four folds); about half of its
  predictions are false positives. On the same folds the pipeline scores .4363 without TTA, and on the public
  leaderboard the five-fold DDP ensemble scores .34 against .39.
* The prompted refiner was chosen because it is the only design with a per-instance handle (prompt, presence,
  recurrence) on fragmentation and merging. Trained on four folds it reaches .4226 5-fold OOF before the
  post-processing changes (rescoring the stored inference dumps: .422675 strict / .422733 legacy) and .4405 after.
* A prompted refiner segments **one** filament per prompt, so each prediction is one instance by construction, and a
  presence output rejects prompts judged absent. One output per prompt does not ensure that it matches exactly one GT.

## D2. Recall-first proposals with a frozen self-supervised backbone

The refiner can only segment filaments it is prompted with, so stage 1 is tuned for recall, not precision.

* Frozen DINOv3 features generalise from few images; only a 0.43 M-parameter head is trained.
* ViT-S+/16 vs ConvNeXt-B (both frozen, same head budget, same data): at 95% pixel recall the ViT map has precision
  19.3% vs 15.4%, so ViT-S+ was kept.
* Threshold .05 (fixed before validation): pixel recall .908; components touch 8138 / 8199 GT (99.3%).
* LightGBM on 32 interpretable features removes most false components cheaply (5493 of 16721 kept, recall of the
  "touches a filament" label .843 at precision .788), before the expensive refiner runs.
* For annotation review, an optional recall policy uses 48 times the positive weight: 9411 candidates, component-label
  recall .9557 / precision .5198. Balanced remains the default for all refiner/PQ results. The 48/64 arms were added
  after inspecting earlier development results; recall-policy final PQ has not been measured.

## D3. One target per annotation record, never a union of annotators

Annotators disagree strongly: scoring one expert's masks against another's on the 296 multi-annotated training
images gives PQ **.34** (SQ .63). A union of annotators would create targets no annotator drew. The refiner target for
a crop and a record is the record's filament overlapping the anchor most ("K1"); a neighbouring filament that only
falls inside the 128 px padding is not a target.

## D4. Refiner architecture

The stem concatenates five channels (image, proposal probability, anchor, signed distance, current mask); validity
is separate. Trainable parameters: 34,820,802 = 6,466,242 decoder + 28,354,560 tuned ViT-B blocks.

* **Prompt channels** (anchor, signed distance, current mask) tell the network which instance is wanted; the
  proposal probability is a strong extra cue.
* **Global attention at stride 16**: filaments are long and thin (up to several hundred pixels); attention lets the
  two ends of a filament inform each other, which a purely convolutional decoder of this size does not.
* **Recurrence (3 rounds)** with the previous mask as input: each round corrects the previous one; training
  supervises every round. Round 3 is used.
* **Presence prior anchored on the ML score**: presence = logit(ML) + learned residual. This alone was the best
  single change on the one-fold comparison (P .397 vs baseline .388).

## D5. Training choices that controlled over-fitting (one change at a time)

All arms below changed exactly one factor against their baseline, with fixed epochs and the last checkpoint.

| Change | Evidence |
|---|---|
| D4 (dihedral) augmentation | +.0385 val PQ; train-val gap .056 instead of .197 |
| ML-anchored presence (P) | one-fold val .397 vs .388 |
| photometric + scale augmentation (A) | .396 vs .388 |
| Skeleton Recall loss (L) on the filament centre-line | .391 vs .388; keeps thin parts connected |
| fine-tune ViT-B blocks 9-12 (B), with P+A+L, trained on 4 folds | val fold 1: .4079 |
| + layer-wise LR decay .65 and drop path .1 ("llrd_dp") | val fold 1: **.4146**, smallest train-val gap; adopted |
| EMA of weights (.999) | .4079, no gain; not adopted |

Epoch count: 30, fixed. With 5-fold snapshots, choosing the epoch for each fold from the other four folds
(leave-one-fold-out) gives .4203, below the fixed epoch 30 (.4226), so the fixed rule was kept. The train-val gap at
epoch 30 is .017.

## D6. Post-processing at native resolution, plus test-time augmentation

* Thresholding at 512 and upsampling the binary mask with nearest neighbour produces staircase borders on small
  crops. Resampling the **probability** bilinearly to native size and thresholding afterwards improved 24/24
  model-fold pairs (+.014 on average) and +.0136 on the final 5-fold OOF (.4227 -> .4363).
* D4 TTA (8 transforms) adds +.0042 (.4363 -> .4405).
* Things tested on the same OOF predictions and **not** adopted (no gain): mask threshold .45 (.4392) or .55 (.4406,
  within noise; the pre-registered .5 was kept), presence gates combining refiner presence with the ML score
  (.4385-.4393), a 256 px minimum area (.4405). Choosing among all settings with leave-one-fold-out post-processing selection gives
  .4395 strict / .4396 legacy. This selects post-processing using the other four folds of stored OOF predictions;
  it does not retrain the pipeline inside each split or establish an unbiased end-to-end score.

## D7. What did not work

| Idea | Result |
|---|---|
| "Carve" decoder (residual growth from the anchor, ConvGRU) | .372 vs .388; masks too fat, cannot reach filament parts outside the anchor |
| Completion training (truncated anchors as corruption) | +.0005, within epoch-to-epoch noise |
| Mask-quality presence target | precision .70 but recall .46; PQ .372 |
| Higher LR + more unfrozen blocks | fits train better (train .50), validation lower (.375) |

## D8. We do not exploit the all-pairs matching rule

Under the all-pairs rule, k copies of the same mask count as k true positives against one ground truth. On our OOF
predictions, duplicating every mask 10 times would raise the score from .44 to about .53 without any better
segmentation. The organisers asked participants not to "hack the system" (announcement of 20 Aug 2026). Our
predictions are pixel-disjoint, we optimise the strict one-to-one score, and report both rules separately. Submitted OOF legacy PQ is .440524 (5101 TP / 2512 FP / 3098 FN);
strict PQ is .440463 (5100 TP / 2512 FP / 3099 FN). Overlapping GT can produce this difference.

## D9. Known limits (from the OOF error audit, see RESULTS.md)

The largest remaining error groups are (a) incomplete masks - the refiner paints only part of a filament whose anchor
already covers most of it (558 FN), (b) filaments never prompted because the proposal component was filtered or
missed (402 lost upstream + 514 only in crop padding), and (c) one anchor covering two filaments, of which only one
can be output (481 FN). Group (c) motivates a trained multi-instance refiner (0-2 outputs per prompt); a zero-shot
probe changed PQ by only .0004 and does not demonstrate improvement.

Annotations are imperfect, so background FP means unmatched to the record, not a verified physical absence.
Annotation-assistance benefits (review time, correction quality, inter-annotator agreement) have not been tested in a
user study. Similar baseline errors do not establish that accumulating errors across stages is harmless.
