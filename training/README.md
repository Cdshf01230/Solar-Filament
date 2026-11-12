# Training code (reference)

This contains the optimisation and data code used on Kaggle and Colab. Portable launchers adapt paths; internal development gates are omitted.
It is not needed for inference and its paths refer to our development storage; see [docs/TRAINING.md](../docs/TRAINING.md)
for the recipes and hyper-parameters.

| Folder | Stage | Entry points |
|---|---|---|
| `baselines/` | README §2 baselines (FPN, U-Net, DDP-mini) on the frozen ConvNeXt-B, 209 / 498 split | `stage/stage209.py`, `u0_multihead/train.py`, `u0_multihead/ddp_mini_engine.py`, see its README |
| `baselines/ddp_5fold/` | README §2.6: standard DDP head trained on the five folds for 20 epochs | `runtime/ddp5/stage.py`, `runtime/ddp5/train.py`, `inference.py`, see its README |
| `proposal/` | stage 1 proposal heads of the submitted system (frozen DINOv3 ViT-S+/16) | `proposal_vitsplus.py` (head, loss), `u0_multihead/` (data, tiled inference, evaluation), `solar_loss/` (training loop) |
| `components/` | stage 2 component features + LightGBM | `train_presence_ml.py` (features, per-fold classifiers), `padding_recall.py`, `threshold_sweep.py` (record-level scoring helpers) |
| `refiner/` | stage 3 prompted recurrent refiner | `train.py` (DDP training, fixed epochs, last checkpoint), `data.py` (crop loader, prompt corruption, augmentation), `model.py`, `engine.py`, `evaluate.py` |

Crops for the refiner were built from the out-of-fold proposal maps and classifier scores with exactly the rule of
`filament_pq/crops.py` (component bbox + 128 px, clipped), with per-record K1 targets as described in docs/TRAINING.md.

Splits: `splits/fold_assignment.json` (fixed date-disjoint folds) and `splits/holdout{k}.json` (train = 4 folds != k), produced by
`scripts/make_splits.py`. Only the refiner uses holdout{k}.json. Proposal and LightGBM model k train on fold k itself; OOF averages the four models excluding the owner. Baselines use the separate historical split209.json.
