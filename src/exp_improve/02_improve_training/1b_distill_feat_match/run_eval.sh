#!/bin/bash
# 1b_distill_feat_match: ckpts 디렉토리 내 모든 .pt 파일 평가
# Usage: bash run_eval.sh [GPU_ID]
# GPU_ID를 인자로 받음, 생략 시 기본값 0

GPU=${1:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/../../ckpts/02_improve_training/1b_distill_feat_match"

for CKPT in "$CKPT_DIR"/*.pt; do
    [ -f "$CKPT" ] || continue
    BASENAME="$(basename "$CKPT" .pt)"
    RESULT_FILE="$CKPT_DIR/${BASENAME}_results.json"

    echo "===== Evaluating ${BASENAME} on GPU ${GPU} ====="
    python "$SCRIPT_DIR/evaluate.py" \
        --ckpt "$CKPT" \
        --result_file "$RESULT_FILE" \
        --device "$GPU"
    echo "===== Done ${BASENAME} ====="
    echo ""
done
