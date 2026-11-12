# Reproducibility evidence

The released pipeline uses proposal heads and component classifiers trained on **one fold each**.
The refiner trains on the other **four folds** for each holdout. The original 707-image assignment is fixed.
`configs/pipeline.json` records the training and inference parameters without changing the algorithm.

| Check | Evidence | Limit |
|---|---|---|
| End-to-end inference | On 9 Nov and 12 Nov 2026 the release predictor on NVIDIA L4 reproduced the submitted 180-image CSV byte for byte (SHA-256 `c8a9a8c446ea…`). The 12 Nov run used the public HF copy and `pip install -r requirements.txt` on Colab (84 min) | Same GPU type as the submission; other GPUs can flip boundary pixels |
| End-to-end out-of-fold | On 12 Nov 2026 the same environment reran all 707 labelled images out of fold: strict PQ .440507 vs .440463 recorded; 555 images give the same counts and IoU sum (to .001) | Stage-1 maps are recomputed, while development read stored maps: 5497 crops instead of 5493 |
| Line endings | Since 12 Nov 2026 `*.csv` is stored unchanged (`.gitattributes`: `-text`). Before, git stored the reference CSVs with LF, so on Linux the CRLF output of `predict.py` compared unequal although its content was identical | Fixed in the repository; older clones need the new `reference/*.csv` |
| Recall48 filter | On 10 Nov, all 5 new classifiers reloaded in text/joblib formats; scores matched on all 16,721 original feature vectors (`weights/ml/POLICIES_VERIFICATION.json`) | Verified with LightGBM 4.7.0 against models trained with pinned 4.6.0; no recall-policy refiner evaluation |
| Recovered weights | On 10 Nov all 5 proposal heads, 5 refiners and 5 LightGBM text models loaded successfully; neural tensors finite | CPU structural checks, not an accuracy evaluation |
| Refiner export | Original export checked 145 frozen tensors per fold equal to the pretrained backbone; exported decoder + 4 tuned blocks | `weights/EXPORT_REPORT.json` records the original check |
| Submission reference | Recovered archive CSV equals `reference/submission.csv` byte for byte | Checks the saved output, not a rerun |
| Training scripts | Original optimisation/data code with portable launchers, canonical manifests and local-input crop builder | Full training from scratch has not been rerun to confirm identical learned weights |

There are 20 learned models across both policies: 5 proposal heads, 10 LightGBM classifiers (5 per policy),
5 refiners, plus two shared pretrained backbones. Text/joblib are two formats of the same classifier. The original
15-model structural check and the additional recall48 score check have different scope; neither reruns training.

The HF copy was synchronized with GitHub on 12 Nov 2026 (including `recall_fold{k}` and `weights/ddp_baseline/`):
every file was compared by hash (SHA-256 for Git LFS files, the git blob hash for the others). No upload is performed by verification scripts.

Inference environment: Python 3.13, CUDA 13.0, NVIDIA L4; pinned packages in `requirements.txt`.
Weight validation on 10 Nov used CPU, torch 2.11.0+cu128 and transformers 4.57.6.
Different hardware, library versions, batch layout or mixed-precision kernels can change boundary pixels.
Seeds alone do not establish bitwise training reproducibility. The training launcher omits internal development gates;
this does not change the loss, optimiser, schedule, data split or model architecture.

Run from the repository root:

```bash
python scripts/verify_weights.py --weights weights
python tests/test_regression.py --images /data/test_images --work /scratch/regression
```

The regression command reports equality and, if different, row counts and common-ID IoUs. Inspect its output;
a successful process exit by itself does not assert byte equality. Use a new work directory for a new inference run; a work directory made from other images, settings or model files is refused.
No competition images or annotations are redistributed.

OOF scores are development cross-validation estimates, not fully nested end-to-end validation: the upstream ensemble
members are shared across downstream fold datasets. Report the strict 1-to-1 scorer per original annotation record
separately from legacy all-pairs, including ground truth lost upstream.
