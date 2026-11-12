# Baselines (README §2): FPN, U-Net (U0), DDP-mini

All three use the frozen DINOv3 ConvNeXt-B backbone and the 209 / 498 capture-day split of
[`splits/split209.json`](../../splits/split209.json) (`quick74`, scored every 5 epochs, chooses the epoch and threshold; `audit424` reports).

| Path | What it is |
|---|---|
| `stage/stage209.py` | Stages the data: copies the 707 images and annotations, runs the original crop generator (`stage/magfilo_crops.py`: per annotation record, one crop centred on each filament, four random positive crops, three crops with no filament) on the 209 training images, and writes the 0/1 semantic masks and `train.jsonl`. |
| `u0_multihead/train.py` | Single-GPU runner for the FPN (`fpn128`) and U-Net (`u0`) heads. Several heads train side by side on one shared frozen backbone, each with its own optimiser. |
| `u0_multihead/ddp_engine.py` | Single-GPU runner for the DDP head (`ddp_model.py`). |
| `u0_multihead/ddp_mini_engine.py` | Same runner for DDP-mini (`ddp_mini.py`: width 128, 3 layers). |
| `ddp_5fold/` | Separate from the 209 / 498 runs: the standard DDP head trained on the five date-disjoint folds (README §2.6); see its own README. |
| `run_all.sh` | Historical staging/launch script; see capacity_queue.sh for the updated sequence. |

The runners expect `--base` laid out as in `run_all.sh`: `data/`, `backbone/`, `magfilo_crops_768/`, and
`semantic_fpn_30_70/{source_split.json, train.jsonl}` (the name is historical). Each run writes `config.json`,
`history.json` (`quick74` scores of the evaluated epochs) and `result.json` (chosen and last epoch on `audit424`).

## Capacity comparison (10 Nov 2026)

Order on A100: U0 + FPN first, then DDP-mini. All 20 epochs, quick74 selects
the epoch (5/10/15/20) and threshold (0.3–0.7); audit424 reports, and the last epoch is also scored at threshold 0.5.
Earlier U0/FPN evaluations at epochs 1-4 are retained as history and excluded from selection.

FPN: 1,575,041 parameters; U0: 1,792,481; DDP-mini: 2,117,146. Results: README §2.5.
DDP-mini uses width128, 3 layers, 4 attention heads, FFN512 and time512.
The diffusion objective, noise schedule and 3 inference steps are unchanged.
The original DDP head (11,564,106) retains its defaults as a historical reference.

`run_all.sh` stages the data in its original Colab environment; `capacity_queue.sh` runs the reported sequence
and evaluates every five epochs. They retain `/content` paths and require the original `stage_rt` / module context
and staged inputs; they are historical launchers, not standalone portable scripts for a fresh clone.
Reported capacity-run scores use legacy all-pairs; strict one-to-one was not measured. These results use one
seed/split and do not establish a generalisation advantage across splits.
