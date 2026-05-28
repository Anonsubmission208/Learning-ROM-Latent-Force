#!/usr/bin/env bash
set -euo pipefail

TRAIN_SCRIPT="physics.py"

CONFIG_DIR="../../configs/physics"

SEEDS=(1 3 5 9)

for CONFIG_PATH in "$CONFIG_DIR"/*.yaml; do
    CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)  
    echo "=== Running config: $CONFIG_NAME ==="

    for SEED in "${SEEDS[@]}"; do
        echo "  → Seed $SEED"
        python "$TRAIN_SCRIPT" \
            --config "$CONFIG_PATH" \
            --seed "$SEED"
    done
done

