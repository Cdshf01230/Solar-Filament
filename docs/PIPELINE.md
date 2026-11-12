# Pipeline, stage by stage

Input: one 2048 x 2048 8-bit grayscale GONG H-alpha image. Output: a list of pixel-disjoint binary masks, one per
filament. All constants live in [`filament_pq/config.py`](../filament_pq/config.py).

## 0. Data and folds

* 707 labelled images (2011-2022), 1154 annotation records (an image can be annotated by 1-3 experts; 411/145/151
  images have 1/2/3 records), 8199 filament instances. Test: 180 images.
* Every record is a separate ground truth, exactly as the organisers score it. **We never merge annotators**; a
  training target always comes from one record.
* 5 folds split by observation date (141/141/142/141/142 images, seed 42), so near-duplicate images taken minutes
  apart never sit on both sides of a split. The same folds are used by every stage
  (`weights/fold_assignment.json`).
* Proposal heads and classifiers each train on one fold; labelled-image predictions average the other four.
  Refiner k trains on four folds and predicts its held-out fold k; test predictions average five models per stage.
* These are development OOF estimates, not fully nested end-to-end CV: upstream members reused to build refiner
  training inputs include models trained on its held-out fold. No held-out image is directly used to fit that refiner.

## 1. Proposal map (`filament_pq/proposal.py`)

* Backbone: **DINOv3 ViT-S+/16** (frozen, 28.7 M parameters). Hidden states of blocks 3, 6, 9, 12 after the final norm,
  patch tokens only (stride 16).
* Head (`ProposalHead`, 0.43 M parameters): 1x1 lateral projections to 64 channels, top-down addition, 3x3 smoothing,
  fusion to 32 channels, then a raw-image branch at half resolution (16 channels) restores thin structures; bilinear
  upsampling to full resolution, sigmoid.
* Tiling: 9 tiles of 768 x 768 at offsets {0, 640, 1280}^2, blended with a Hann window floored at .05.
* Ensemble: one head per fold; the test map is the mean of the 5 heads (each map rounded to float16, summed in fold
  order in float32).

Why: retain suspicious regions with high recall. A filament without component overlap may still be recovered from
another anchor's padded crop; only one outside every retained crop is unreachable. At the
low threshold .05 the OOF map has pixel recall .908 (precision .295) and its components touch 8138 of the 8199
ground-truth filaments.

## 2. Components and LightGBM filter (`filament_pq/components.py`, `crops.py`)

* Components: 8-connected regions of probability >= .05 with at least 8 pixels (OOF: 16721 components).
* 32 features per component: log area / bbox size / aspect / fill, position and distance to the image border,
  probability statistics (mean, std, min, max, fractions above .1/.2/.5/.8), raw-image intensity, darkness relative to
  Gaussian backgrounds (sigma 5 and 15), local texture, gradient, black-hat responses (15 and 31 px), distance-transform
  thickness, elongation from second moments, and contrast with a 16 px context box.
* Classifier: LightGBM (250 trees, 15 leaves, depth 6, class-balanced weights), one per fold, mean score; keep >= .5
  (fixed). Balanced is the default: 5493 of 16721 components kept; anchors touch 7282 GT, padded crops intersect 7794.
* Optional recall48 policy: same features/threshold, positive training weight multiplied by 48. It keeps 9411
  components, component-label recall 95.57% / precision 51.98%; padded crops intersect 8155/8199 GT and fully
  contain 7949. Use `scripts/proposals.py --policy recall` for candidate export. It was selected on development
  results, including extra 48/64 trials after inspection; refiner PQ on recall crops has not been evaluated.
* Crop: component bounding box + 128 px on every side, clipped to the image. Median long side 377 px, 90.5% <= 512.

The component that produced a crop is kept as the **anchor**: the prompt telling the refiner which filament to
segment.

## 3. Prompted recurrent refiner (`filament_pq/refiner/`)

Inputs (letterboxed to 512 x 512, aspect preserved, top-left aligned): image, proposal probability, anchor mask,
signed distance to the anchor (clipped to +-64 px), current mask estimate (initialised with the anchor), validity
mask of the letterbox. The first five arrays are concatenated into the stem; validity is supplied separately for
attention, pooling and recurrence. There is also a 3-number presence prior [logit(ML score), log anchor area fraction, mean probability in
the anchor].

* Backbone: **DINOv3 ViT-B/16**; embeddings, blocks 1-8 and final norm frozen; blocks 9-12 fine-tuned.
* Trainable total: **34,820,802** = decoder **6,466,242** + tuned backbone blocks **28,354,560**.
* Decoder: a 5-level convolutional stem over the 5 input channels gives skips at strides
  1, 2, 4, 8, 16; at stride 16 it is added to the projected ViT features, a learned round embedding and a sine
  position code; 4 global self-attention blocks (256-d, 8 heads, padding masked) let distant parts of a long filament
  exchange information; a U-Net path with the stem skips returns a full-resolution mask logit.
* Recurrence: round t+1 receives the sigmoid mask of round t as its "current" channel. Three rounds; round 3 is used.
* Presence: logit(ML score) + a residual MLP (zero-initialised) over the anchor-pooled and globally pooled tokens
  (gradient-detached) and the anchor statistics. It decides whether this anchor corresponds to a filament at all, which
  is how the pipeline removes false components that survived stage 2.
* Ensemble at test time: 5 fold refiners x 8 dihedral transforms (D4), probabilities and presences averaged.
  Each model quantizes probability after TTA as `round(255*p)` uint8; the fold mean is rounded to uint8 again.
  OOF uses only the owner-fold refiner, with D4 TTA.

## 4. Post-processing (`filament_pq/postprocess.py`)

1. Drop crops with presence < .5.
2. Remove letterbox padding and resample the round-3 probability to the crop's native size with bilinear interpolation, threshold > .5.
3. Keep the 8-connected component that overlaps the anchor most; if none overlaps, keep the largest. Drop if < 128 px.
4. Greedy de-duplication over the image in descending presence (crop ID breaks ties): a candidate whose IoU with an accepted instance
   exceeds .5 is discarded; otherwise pixels already taken are removed and the rest is kept if >= 128 px.

The result is a set of pixel-disjoint instances. Overlapping GT masks can still let one prediction match multiple
GT under all-pairs scoring. Report both scorers separately: submitted OOF strict PQ .440463 (5100/2512/3099
TP/FP/FN), legacy .440524 (5101/2512/3098), on 707 images / 1154 records / 8199 GT.

## 5. Submission (`filament_pq/submission.py`)

`filament_id = <image stem>_<k>` and `segmentation_rle` = COCO RLE counts (pycocotools, Fortran order) of the
2048 x 2048 mask. `validate()` checks unique ids, valid image stems and that every RLE decodes to a non-empty mask.
