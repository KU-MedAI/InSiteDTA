#!/bin/bash
# 1b_distill_feat_match: EMA distillation with bottleneck_p2l feature matching, 1-seed screening
# Usage: bash run_train.sh [GPU_ID]
# GPU_ID를 인자로 받음, 생략 시 기본값 0

GPU=${1:-0}
SEEDS=(312)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for SEED in "${SEEDS[@]}"; do
    echo "===== Training seed=${SEED} on GPU ${GPU} ====="
    python "$SCRIPT_DIR/train.py" \
        --device "$GPU" \
        --seed "$SEED"
    echo "===== Done seed=${SEED} ====="
    echo ""
done
