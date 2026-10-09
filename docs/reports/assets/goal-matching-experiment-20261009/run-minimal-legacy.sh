#!/usr/bin/env bash
set -eu
cd /data/zzx/inter-nav-goal-matching-smoke
source /data20t/embodied/share/inter-nav/env.sh
R=$(cat /data/zzx/inter-nav-experiment-current.txt)
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GO2_STEP_TIMING=1
export INTER_NAV_EXPERIMENT_ROOT="$PWD/$R/minimal-online-legacy"
"$ISAAC_PYTHON" "$R/supervise.py" --log "$R/minimal-online-legacy.log" --timeout 240 -- \
  "$ISAAC_PYTHON" -u "$R/diagnostic-navigation.py" \
  --scene grscene --profile "$R/minimal-profile.json" \
  --gpu 0 --perception-gpu 5 --qwen-device cuda:7 --headless --no-qwen \
  --detection-mode open_vocab --semantic-classifier qwen-vl \
  --grounding-dino-url http://127.0.0.1:12187/gdino \
  --mobile-sam-url http://127.0.0.1:12188/mobile_sam \
  --qwen-vl-url http://127.0.0.1:12186/classify --qwen-vl-max-candidates 4 \
  --target 'a large appliance used to keep food cold' --goal-matching legacy \
  --max-steps 1200 --mapping-warmup-steps 80 --record-every 80 --record-raw-rgb \
  --record-dir "$R/minimal-online-legacy"
