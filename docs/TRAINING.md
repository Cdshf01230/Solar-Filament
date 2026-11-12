# Training recipes

Inference needs none of this: all weights are in `weights/`. `training/` contains original model/data/optimisation
code with portable wrappers that adapt paths and omit internal development gates. These wrappers differ from the
original cloud launchers; a complete fresh training run through the portable route has not been verified.

## Folds

`splits/fold_assignment.json`: 5 folds by observation date, 141/141/142/141/142 images, seed 42.
Upstream models are reused across downstream training datasets; the resulting OOF scores are development
estimates, not fully nested end-to-end validation (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

## Stage 1 - proposal heads (`training/proposal/`)

* One head per fold, **trained on that fold only (about 141 images) and validated on the other four**; at test time
  all five are averaged, for a training image the four heads that did not see it.
* Training data: 9 tiles of 768 x 768 per image and per annotation record (binary target of that record only).
* Frozen DINOv3 ViT-S+/16 in fp32, head in fp16 autocast; AdamW, LR 3e-4, weight decay .01, global batch 8, 4 epochs,
  no augmentation, final epoch kept (no early stopping).
* Loss: BCE with positive weight 4 + Tversky (alpha .2 on FP, beta .8 on FN), i.e. recall-oriented.

## Stage 2 - LightGBM component classifiers (`training/components/`)

* Components of the OOF mean map at .05, 32 features, label = "touches a filament of this record".
* One classifier per fold, trained on that fold's images, class-balanced weights; 250 trees, 15 leaves, depth 6,
  LR .05, min child samples 30, L2 3, deterministic.
* Both `balance` and `recall` model sets are included. With per-record base weight 1 / number of records of the source,
  the positive multiplier is weighted training negatives / positives for balance, and 48 times that ratio for recall;
  negative multiplier 1, final weights normalised to mean 1. Both use score threshold .5. The ML training command
  below already fits both sets; no additional neural training is required. The refiner and reported final PQ use balance.
* Crops for the refiner are built from the OOF maps and OOF scores, so the refiner is trained on the same kind of
  imperfect inputs it receives at test time.

## Stage 3 - refiner (`training/refiner/`)

* **34,820,802 trainable parameters**: decoder 6,466,242 + tuned ViT-B blocks 9-12 28,354,560.
* Five concatenated stem channels: image, probability, anchor, signed distance, current mask. Letterbox validity
  is a separate mask used for attention, pooling and recurrence.

* One refiner per held-out fold, trained on the crops of the other four folds (about 4400 crops / 566 images).
* Targets per crop and per annotation record: K1 = the record's filament overlapping the anchor most (or empty).
* Prompt ("current" channel) sampling per crop and epoch: 40% the anchor itself, 30% a corrupted version of a target
  (dilate, erode, holes, truncation along the main axis, added blobs, merged neighbour, shift, noisy contour; IoU with
  the target kept in [.3, .9]), 20% the model's own prediction (rollout, from epoch 2), 10% empty.
* Augmentation: D4 (one of 8 dihedral transforms per sample), photometric (gamma .8-1.25, contrast .85-1.15,
  brightness +-.1, blur, noise) and shrink-scale .8-1.0.
* Loss per record and per round (3 rounds, averaged): BCE + Lovasz hinge on positive records, .5 x presence BCE,
  .5 x "empty" BCE on negative crops, 1.0 x Skeleton Recall (1 - soft recall of the record's centre-line tube, r = 2).
* Optimiser: AdamW, decoder LR 3e-4, top ViT block 2e-5 with layer decay .65 per block below, drop path .1 in the
  tuned blocks, weight decay .01, batch 8, 30 epochs, cosine schedule to 1/30 at the last update, mixed precision.
  The last checkpoint (epoch 30) is used; validation is report-only.
## Historical semantic baselines

FPN, U0 and DDP-mini code and exact parameter counts are in [training/baselines](../training/baselines/README.md).
Its `run_all.sh` / `capacity_queue.sh` require the original staged cloud environment and are historical launchers,
not standalone clone-and-run entrypoints. Their train209/quick74/audit424 protocol differs from the pipeline folds.

## Hardware

Kaggle 2 x T4 (folds 0, 2, 3, 4) and Colab L4 (fold 1); inference and evaluation on Colab L4.

## Commands

Run from the repository root on Linux with CUDA. Install `requirements-training.txt`.
Use the original Kaggle annotation JSON and images; the data are not included.
The examples train fold 0; repeat for k = 0, 1, 2, 3, 4, keeping separate output folders.

```bash
python scripts/prepare_proposal.py --images /data/train_images --annotations /data/MAGFiLO_1.0_Annotations_kaggle2026_train.json --out /scratch/proposal_data --fold 0
python scripts/train.py proposal --fold 0 --data /scratch/proposal_data/fold0 --backbone weights/proposal/dinov3_vits16plus --out /scratch/proposal_fold0 --gpus 2
```

The original proposal runner requires two GPUs. It writes `last_head.pt` (epoch 4);
`best_head.pt` is a diagnostic artifact and is not used. The supplied manifests keep targets separate by annotation record.
No fold is generated or shuffled again. If replacing the released heads, export the `head` field of each `last_head.pt`
to `weights/proposal/heads/fold{k}.pt` before generating maps.

```bash
python scripts/build_oof.py maps --images /data/train_images --annotations /data/MAGFiLO_1.0_Annotations_kaggle2026_train.json --out /scratch/upstream
python scripts/build_oof.py ml --images /data/train_images --annotations /data/MAGFiLO_1.0_Annotations_kaggle2026_train.json --out /scratch/upstream
python training/components/build_refiner_crops.py --maps /scratch/upstream/ensemble/vitsplus --ml /scratch/upstream/ml --data-zip /data/filament-segmentation-2026.zip --out /scratch/refiner_crops
python scripts/train.py refiner --fold 0 --data /scratch/refiner_crops --backbone weights/refiner/dinov3_vitb16 --annotations /data/MAGFiLO_1.0_Annotations_kaggle2026_train.json --out /scratch/refiner_fold0 --gpus 2
```

The ML wrapper uses the original feature extraction and record-balanced training weights. It fits both released
policies. The default end-to-end pipeline uses `balance_fold{k}`; candidate export can select `recall_fold{k}` with
`scripts/proposals.py --policy recall`. Export either model set as text using `scripts/export_lightgbm.py --policy
balance` or `--policy recall` (also supply `--weights` and `--images` as in the script usage).
The native crop builder preserves K1 per original record and writes `gt_instances.csv` for upstream misses.
Fold 1 originally used one L4; set `--gpus 1` for its refiner run. Other folds used two T4 GPUs.
Global batch remains 8, with microbatch 1 and accumulation for the refiner. Refiner warmup is 200 updates;
proposal warmup is 20 updates. Export refiner weights with `scripts/export_weights.py` (see its usage header),
and text LightGBM with `scripts/export_lightgbm.py`. Training checkpoints can be resumed via `--resume` on the launcher;
keep the same GPU count and configuration. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for verified scope.
