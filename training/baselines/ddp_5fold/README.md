# Standard DDP head, five folds (README §2.6)

The original DDP head (11,564,106 trainable parameters) on the frozen DINOv3 ConvNeXt-B, trained once per held-out
fold of [`splits/fold_assignment.json`](../../../splits/fold_assignment.json) on the other four folds for **20
epochs**; the released checkpoints and the submitted CSV use the epoch-20 model of every fold.

| Path | What it is |
|---|---|
| `runtime/` | The training code of the five models: crop staging, two-GPU trainer, model and scoring. It performs the same updates as the training runs (data order, learning rate at every update, loss); the notebook launcher is not included. |
| `runtime/ddp5/stage.py` | Copies the 707 images and annotations, runs the original crop generator (`runtime/solar_loss/magfilo_crops.py`) on the four training folds and writes the semantic target of each annotation record. |
| `runtime/ddp5/train.py` | Two-GPU trainer (`torch.distributed`), with a GPU gate, epoch-wise validation on the held-out fold, epoch snapshots and a resumable full state (model, optimiser, scheduler, scaler, RNG and data cursor). |
| `inference.py` | Five-model test ensemble (probabilities averaged before the threshold) or owner-only out-of-fold scoring. |
| `epoch_table.py` | Recomputes the out-of-fold table of README §2.6 from [`docs/results/ddp_5fold/`](../../../docs/results/ddp_5fold). |

Weights: [`weights/ddp_baseline/`](../../../weights/ddp_baseline) holds the backbone used for the submission
(`dinov3_convnext_base/`) and `fold{k}_epoch20.pt`, the epoch-20 model of each fold; SHA-256 sums are in
`weights/ddp_baseline/PROVENANCE.json`. The submitted CSV is
[`reference/ddp_5fold_epoch20_submission.csv`](../../../reference/ddp_5fold_epoch20_submission.csv).

Check (12 Nov 2026): `inference.py` with the released weights, run on an RTX 3050 for 5 of the 180 test images,
loaded all five heads strictly and produced the same 51 instances as the submitted CSV (made on an A100); 46 rows are
byte-identical, and the other five (in three images) change the union of an image's masks by at most 12 pixels, at
threshold boundaries.
The full test set and the out-of-fold scores were not recomputed with the released files.

## Recipe

20 epochs (30,620–31,600 updates depending on the fold). AdamW with a peak learning rate of 3e-4: 100 warm-up updates,
then a cosine decay towards a 5 % floor whose length is 1.5 × the training length, so the learning rate ends at 29 % of
its peak. Global batch 8 on two GPUs (micro-batch 2), D4 augmentation, BCE + per-crop Dice, seed 42. Native
768 px crops per annotation record: one centred on each filament, four random crops containing filaments and three
crops with no filament (12,248–12,637 crops per fold). Validation after every epoch on the whole held-out fold:
nine 768 px tiles blended with a Hann window, threshold 0.5, 8-connected components, area ≥ 128 px, no test-time
augmentation, scored per annotation record.

## Commands

Inference on test images, from the repository root:

```bash
python training/baselines/ddp_5fold/inference.py --images /data/test_images --epoch 20 --backbone weights/ddp_baseline/dinov3_convnext_base --out work_ddp/submission.csv --checkpoints 0=weights/ddp_baseline/fold0_epoch20.pt 1=weights/ddp_baseline/fold1_epoch20.pt 2=weights/ddp_baseline/fold2_epoch20.pt 3=weights/ddp_baseline/fold3_epoch20.pt 4=weights/ddp_baseline/fold4_epoch20.pt
```

Training one held-out fold *k* (two GPUs). `SOURCE` contains `data/train_images/` and
`data/MAGFiLO_1.0_Annotations_kaggle2026_train.json`; the backbone is `weights/ddp_baseline/dinov3_convnext_base`.

```bash
cd training/baselines/ddp_5fold/runtime
python -m ddp5.stage --source $SOURCE --dest /tmp/ddp_fold$K --fold $K
python -m torch.distributed.run --standalone --nproc_per_node=2 -m ddp5.train --config config.json --mode gate
python -m torch.distributed.run --standalone --nproc_per_node=2 -m ddp5.train --config config.json --mode main
```

`config.json` holds `arm: "convnext"`, `fold`, `epochs: 20`, `lr_horizon: 1.5` (optional, the default), `seed: 42`, `global_batch: 8`, `microbatch: 2`,
`warmup: 100`, `base` (the staging folder), `backbone`, `out`, `resume` (empty, or the `last_training.pt` of an
interrupted run), `deadline` (Unix time of the soft stop), `save_every: 200`, `gate_cap_updates: 800`,
`protocol_version: "ddp12-5fold-original-crops-v1"` and `resume_rel_tol` (2e-4; 1e-3 for folds 0 and 3, whose first gate
failed only the resume check, by a relative loss drift just above 2e-4). Each run was split into sessions of at
most 11.5 h and resumed from the full state; the main mode refuses to start unless the gate has passed.
