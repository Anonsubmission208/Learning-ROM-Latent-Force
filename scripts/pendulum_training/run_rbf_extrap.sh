#!/usr/bin/env bash
set -euo pipefail

TRAIN_SCRIPT="rbf.py"

CONFIG_DIR="../../configs/pendulum/rbf/extrap"

SEEDS=(6)

declare -a SUCCESS_LIST=()
declare -a FAIL_LIST=()

for CONFIG_PATH in "$CONFIG_DIR"/*.yaml; do
    CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
    echo "=== Running config: $CONFIG_NAME ==="

    for SEED in "${SEEDS[@]}"; do
        echo "  → Seed $SEED"

        if python "$TRAIN_SCRIPT" --config "$CONFIG_PATH" --seed "$SEED"; then
            SUCCESS_LIST+=("$CONFIG_NAME:seed=$SEED")
        else
            echo "    Seed $SEED failed — skipping"
            FAIL_LIST+=("$CONFIG_NAME:seed=$SEED")
            continue
        fi
    done
done

echo
echo "====================== SUMMARY ======================"
echo

echo "Successful runs:"
if [ "${#SUCCESS_LIST[@]}" -eq 0 ]; then
    echo "  (none)"
else
    for ITEM in "${SUCCESS_LIST[@]}"; do
        echo "  $ITEM"
    done
fi

echo
echo "Failed runs:"
if [ "${#FAIL_LIST[@]}" -eq 0 ]; then
    echo "  (none)"
else
    for ITEM in "${FAIL_LIST[@]}"; do
        echo "  $ITEM"
    done
fi

echo
echo "======================================================"