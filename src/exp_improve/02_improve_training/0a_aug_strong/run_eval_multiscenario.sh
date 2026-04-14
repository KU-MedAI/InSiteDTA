#!/bin/bash
# 0a_aug_strong: 5개 coreset scenario에 대해 모든 ckpt 평가
# Usage: bash run_eval_multiscenario.sh [GPU_ID]
# GPU_ID를 인자로 받음, 생략 시 기본값 0

GPU=${1:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../../../.."
CKPT_DIR="$SCRIPT_DIR/../../ckpts/02_improve_training/0a_aug_strong"

SCENARIOS=("crystal" "redocked" "p2rank" "alphafold" "boltz2")

# ckpt 목록 수집
CKPTS=()
for CKPT in "$CKPT_DIR"/*.pt; do
    [ -f "$CKPT" ] || continue
    CKPTS+=("$CKPT")
done

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "No checkpoint files found in $CKPT_DIR"
    exit 1
fi

echo "Found ${#CKPTS[@]} checkpoint(s):"
for CKPT in "${CKPTS[@]}"; do
    echo "  $(basename "$CKPT")"
done
echo ""

for SCENARIO in "${SCENARIOS[@]}"; do
    echo "===== Scenario: ${SCENARIO} ====="
    python "$REPO_ROOT/05-reproduce.py" \
        --scenario "$SCENARIO" \
        --ckpt "${CKPTS[@]}" \
        --device "$GPU"
    echo "===== Done: ${SCENARIO} ====="
    echo ""
done
