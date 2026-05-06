#!/usr/bin/env bash
# Run the full aLoRA answerability experiment pipeline.
#
# Usage (from aLoRA/ project root):
#   bash run_experiments.sh
#
# Smoke test (fast sanity check, 10 samples, 2 epochs):
#   SMOKE=1 bash run_experiments.sh

set -euo pipefail

# ── Activate environment ───────────────────────────────────────────────────────
source .venv/bin/activate

SMOKE=${SMOKE:-0}
SMOKE_FLAGS=""
MODEL_DIR="smollm2-135m"
if [ "$SMOKE" = "1" ]; then
    SMOKE_FLAGS="--max_samples 10 --num_epochs 2"
    echo "=== SMOKE TEST MODE (10 samples, 2 epochs) ==="
fi

# ── Step 1: Preprocess ─────────────────────────────────────────────────────────
echo ""
echo "=== Step 1: Preprocessing ==="
if [ ! -f data/train.jsonl ]; then
    python src/preprocess.py --input reference.jsonl --output_dir data
else
    echo "Preprocessed data already exists, skipping."
fi

# ── Step 2 + 3: Train and evaluate each variant ────────────────────────────────
for TYPE in baseline shorter descriptive generic; do
    echo ""
    echo "=== Training: ${TYPE} ==="
    python src/train.py \
        --invoc_type "${TYPE}" \
        --output_dir "adapters/${MODEL_DIR}/${TYPE}" \
        --train_data data/train.jsonl \
        --val_data   data/val.jsonl \
        ${SMOKE_FLAGS}

    echo ""
    echo "=== Evaluating: ${TYPE} ==="
    python src/evaluate.py \
        --adapter_path "adapters/${MODEL_DIR}/${TYPE}" \
        --test_data    data/test.jsonl \
        --output       "results/${MODEL_DIR}/${TYPE}_results.json"
done

# ── Step 4: Comparison table ───────────────────────────────────────────────────
echo ""
echo "=== Final Comparison ==="
python src/evaluate.py --compare "results/${MODEL_DIR}/"
