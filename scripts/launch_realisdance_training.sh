#!/usr/bin/env bash
# Launch LTX Union NFT RealisDance Training on RD-H200-2
#
# Prerequisites:
#   1. Clone Flow-Factory repo to RD-H200-2
#   2. Install: pip install -e . (from Flow-Factory root)
#   3. Ensure ltx_core and ltx_trainer are importable
#   4. Download models (see MODEL PATHS section below)
#   5. Run prepare_realisdance.py first (see DATASET PREP section)
#
# Usage:
#   bash scripts/launch_realisdance_training.sh
set -euo pipefail

# ─── MODEL PATHS (update these to actual locations on RD-H200-2) ───
# Download from: https://huggingface.co/Lightricks/LTX-2.3
export LTX_MODEL_PATH="${LTX_MODEL_PATH:-/path/to/ltx-2.3-22b-checkpoint.safetensors}"
# Download from: https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control
export UNION_LORA_PATH="${UNION_LORA_PATH:-/path/to/IC-LoRA-Union-Control.safetensors}"
# Download from: https://huggingface.co/google/gemma-3-12b-it (or compatible)
export GEMMA_PATH="${GEMMA_PATH:-/path/to/gemma-model}"

# ─── DATASET ───
DATASET_ROOT="/home/ubuntu/RealisDance-Val"
GVHMR_ROOT="/data/GVHMR"
TRAIN_JSONL="${DATASET_ROOT}/train.jsonl"
GT_JOINTS="${DATASET_ROOT}/gt_joints.pt"

# ─── DATASET PREP (run once) ───
if [ ! -f "${TRAIN_JSONL}" ]; then
    echo "==> Preparing dataset: generating train.jsonl and gt_joints.pt..."
    python scripts/prepare_realisdance.py \
        --dataset-root "${DATASET_ROOT}" \
        --gvhmr-root "${GVHMR_ROOT}" \
        --output-jsonl "${TRAIN_JSONL}" \
        --output-joints "${GT_JOINTS}"
    echo "==> Dataset prepared."
else
    echo "==> train.jsonl already exists, skipping preparation."
fi

# ─── VALIDATE PATHS ───
echo "==> Validating paths..."
missing=0
for path in "${LTX_MODEL_PATH}" "${UNION_LORA_PATH}" "${GEMMA_PATH}" "${TRAIN_JSONL}"; do
    if [ ! -e "${path}" ]; then
        echo "ERROR: ${path} not found"
        missing=1
    fi
done
if [ "${missing}" -eq 1 ]; then
    echo "Fix missing paths above and re-run."
    exit 1
fi
echo "==> All paths valid."

# ─── UPDATE CONFIG WITH ACTUAL PATHS ───
CONFIG="configs/ltx_union_nft_realisdance.yaml"
echo "==> Updating config with model paths..."
# Use python to do safe YAML update (preserves structure)
python3 -c "
import yaml

with open('${CONFIG}') as f:
    config = yaml.safe_load(f)

config['model']['model_name_or_path'] = '${LTX_MODEL_PATH}'
config['model']['union_lora_path'] = '${UNION_LORA_PATH}'
config['model']['gemma_path'] = '${GEMMA_PATH}'

with open('${CONFIG}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print('Config updated.')
"

# ─── LAUNCH TRAINING ───
echo "==> Launching NFT training (8 GPUs, bf16)..."
ff-train "${CONFIG}"
