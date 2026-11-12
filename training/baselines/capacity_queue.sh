#!/bin/bash
# Reported 20-epoch baselines (README §2.5), run after run_all.sh has staged the data.
# On 10 Nov 2026 U0/FPN ran first (resumed after epoch 5 when evaluation moved to every 5 epochs), then DDP-mini.
set -euo pipefail
B=${B:-/content/solar_research}
cd "$B"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 SOLAR_SOFT_SECONDS=37800
COMMON="--epochs 20 --batch 8 --eval-every 5 --train-eval-every 5 --evaluate-last --enable-tf32 --save-every 200 --final-eval both --microbatch 4"
resume () { [ -f "$1/last_training.pt" ] && echo --resume || true; }

if [ ! -f "$B/u0fpn20/train/result.json" ]; then
  python -u -m u0_multihead.train --heads u0 fpn128 $COMMON --workers 5 --eval-workers 6 --tile-batch 1 \
    --out "$B/u0fpn20/train" $(resume "$B/u0fpn20/train") >> "$B/u0fpn20_train.log" 2>&1
fi
test -f "$B/u0fpn20/train/result.json"

if [ ! -f "$B/ddpmini20/train/result.json" ]; then
  python -u -m u0_multihead.ddp_mini_engine --heads ddp $COMMON --workers 4 --eval-workers 4 --tile-batch 1 \
    --out "$B/ddpmini20/train" $(resume "$B/ddpmini20/train") >> "$B/ddpmini20_train.log" 2>&1
fi
test -f "$B/ddpmini20/train/result.json"
echo CAPACITY_ALL_DONE
