# Weights in this repository

The default inference pipeline uses the balanced filter; recall-oriented filter weights are an additional option for
candidate export. Nothing is downloaded at run time. Total about 1.1 GB (Git LFS), plus about 0.6 GB for the
separate DDP baseline in `ddp_baseline/`.

| File | Model | Parameters | Trained on | Notes |
|---|---|---|---|---|
| `proposal/dinov3_vits16plus/` | DINOv3 ViT-S+/16 (`facebook/dinov3-vits16plus-pretrain-lvd1689m`) | 28.7 M | public pre-training (LVD-1689M) | frozen, unchanged; DINOv3 License |
| `proposal/heads/fold{k}.pt` | `ProposalHead` state dict | 0.43 M each | images of fold k only (about 141), 4 epochs, last epoch | k = 0..4 |
| `ml/balance_fold{k}.txt` | LightGBM booster, text format (used by default) | 250 trees, 15 leaves | components of fold k's images, OOF proposal maps | identical scores to the pickle (`MODEL_INFO.json`) |
| `ml/balance_fold{k}.joblib` | same model as pickled `LGBMClassifier` | | | needs lightgbm 4.6.0, scikit-learn 1.6.1 |
| `ml/recall_fold{k}.txt` | recall-oriented LightGBM booster | 250 trees, 15 leaves | components of fold k's images, same OOF proposal maps | positive class weight 48 times the balanced recipe; threshold .5; k = 0..4 |
| `ml/recall_fold{k}.joblib` | same recall-oriented model as pickled `LGBMClassifier` | | | text/joblib score equality on all 16,721 original component features: `ml/POLICIES_VERIFICATION.json` |
| `refiner/dinov3_vitb16/` | DINOv3 ViT-B/16 | 85.7 M | public pre-training | blocks 1-8, embeddings, final norm used as is |
| `refiner/fold{k}.pt` | refiner decoder (6.47 M) + fine-tuned ViT-B blocks 9-12 (28.35 M) | 34,820,802 trainable each | crops of the 4 folds other than k (about 566 images), 30 epochs, last checkpoint | k = 0..4 |
| `fold_assignment.json` | the date-disjoint 5-fold split of the 707 training images | | | seed 42 |
| `ddp_baseline/dinov3_convnext_base/` | DINOv3 ConvNeXt-B | 87.6 M | public pre-training | baseline of README §2.6 only, frozen; DINOv3 License |
| `ddp_baseline/fold{k}_epoch20.pt` | full-size DDP head: `trainable` (state dict, prefix `heads.ddp.`), `epoch`, `update`, `config` | 11,564,106 each | crops of the 4 folds other than k, 20 epochs | baseline only, not used by the final pipeline; SHA-256 in `ddp_baseline/PROVENANCE.json`; code in `training/baselines/ddp_5fold/` |

`refiner/fold{k}.pt` = `dict(decoder=<state dict>, tuned_blocks={'<block>.<param>': tensor})`. It was produced by
`scripts/export_weights.py`, which asserted that all 145 other backbone tensors of every training checkpoint are
bit-identical to the public DINOv3 ViT-B/16 weights.

The refiner stem concatenates five channels (image, probability, anchor, distance, current mask); letterbox validity
is separate. Exact trainable total: 6,466,242 decoder + 28,354,560 tuned backbone = 34,820,802 per model.
There are 20 learned models across both policies, plus two shared pretrained backbones; text/joblib are duplicate formats.

Validation scope: the original 15 balanced-pipeline models passed CPU load/finite checks on 10 Nov; five additional
recall48 classifiers passed full-feature text/joblib score equality (LightGBM 4.7.0 check; 4.6.0 training). On 12 Nov
2026 a fresh end-to-end out-of-fold run of all 15 balanced-pipeline models, downloaded from the HF copy, gave strict
PQ .440507 against the recorded .440463 (README §4). No retraining has been repeated. The refiner is trained/evaluated on balanced crops
only; recall48 final PQ is unmeasured. OOF scores are development estimates with shared upstream dependencies.

The HF copy was synchronized with GitHub on 12 Nov 2026 and holds every file listed above, including
`ml/recall_fold{k}` and `ddp_baseline/`.

Seeds and determinism: proposal heads seed 42; LightGBM `random_state=42, deterministic=True, force_col_wise=True`;
refiner prompt sampling and augmentation use stateless generators seeded by (42, epoch, sample index). The historical 9 Nov L4 inference regression reproduced the submission byte for byte (`tests/test_regression.py`).
