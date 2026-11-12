#!/bin/bash
# Baseline 20-epoch re-runs on one Colab A100 (DECISIONS 2026-11-10). Split 209 train / 498 val (quick74 picks the
# epoch, audit424 reports), crops from the original DDP generator, archived single-GPU runner (u0_multihead).
set -euo pipefail
B=/content/solar_research; IN=/content/inputs/solar_loss_inputs; ST=/content/stage_rt
mkdir -p $B; cd $B
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
if [ ! -f $B/staging_verification.json ]; then
  (cd $ST && PYTHONPATH=$ST python -u -m solar_loss.stage209 --source $IN --dest $B) > $B/stage.log 2>&1
fi
python - <<'PY'
import json, shutil
from pathlib import Path
B=Path('/content/solar_research'); sp=json.loads(Path('/content/stage_rt/split209.json').read_text())
d=B/'semantic_fpn_30_70'; d.mkdir(exist_ok=True)
(d/'source_split.json').write_text(json.dumps(dict(train=sp['train209'], val=sp['val498'], quick_val=sp['quick74'],
    note='209/498 capture-day split (seed 42); quick74 selects the epoch, audit424 = val498 - quick74 reports')))
shutil.copyfile(B/'train.jsonl', d/'train.jsonl')
if not (B/'backbone'/'config.json').exists(): shutil.copytree('/content/inputs/solar_loss_inputs/backbone', B/'backbone', dirs_exist_ok=True)
print('SPLIT_READY', len(sp['train209']), len(sp['val498']), len(sp['quick74']), flush=True)
PY
COMMON="--epochs 20 --batch 8 --workers 5 --eval-workers 6 --tile-batch 1 --eval-every 1 --train-eval-every 5 --enable-tf32 --save-every 200 --final-eval both"
run_job () {  # name module heads...
  local name=$1 mod=$2; shift 2
  local out=$B/$name
  if [ ! -f $out/preflight/preflight.json ]; then python -u -m u0_multihead.$mod --heads "$@" $COMMON --microbatch 4 --out $out/preflight --preflight-only > $B/${name}_preflight.log 2>&1; fi
  local micro; micro=$(python -c "import json;print(json.load(open('$out/preflight/preflight.json'))['attempts'][-1]['microbatch'])")
  local extra=""; [ -f $out/train/last_training.pt ] && extra="--resume"
  python -u -m u0_multihead.$mod --heads "$@" $COMMON --microbatch $micro --out $out/train $extra > $B/${name}_train.log 2>&1
  echo "JOB_DONE $name" >> $B/jobs.log
}
run_job u0fpn20 train u0 fpn128 &
run_job ddp20 ddp_engine ddp &
wait
echo ALL_DONE >> $B/jobs.log
