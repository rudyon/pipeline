#!/bin/bash

export OMP_NUM_THREADS=1

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv pip install -r requirements.txt
# activate venv
source .venv/bin/activate

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ ! "$(ls -A data_cache 2>/dev/null)" ]; then
    # --- Primary dataset: FineWeb-Edu (educational, high quality) ---
    python get_data.py HuggingFaceFW/fineweb-edu \
        -c text -C sample-10BT \
        --cache data_cache \
        --max-docs 700000      # ~70% of the mix

    # --- Secondary dataset: DCLM-Baseline (diverse web text) ---
    # Downloaded into the SAME data_cache so tokenize_data.py picks it up.
    # We pull roughly 25% worth (250K docs).
    python get_data.py mlfoundations/dclm-baseline-1.0 \
        -c text -C default \
        --cache data_cache \
        --max-docs 250000

    # Train a shared tokenizer over the full mixed raw data
    python train_tokenizer.py --cache data_cache -c text --vocab-size 32768

    # Tokenize everything into shards
    python tokenize_data.py --cache data_cache -c text --tokenizer tokenizer.json
fi

# Run with torchrun
# --standalone: single-node multi-gpu
# --nproc_per_node: uses all detected GPUs
#
# Steps: Chinchilla-optimal for depth=12 (497M total params) is ~18,965 steps.
# We overtrain to ~1.4x Chinchilla (~26,500 steps) which is cheap and improves
# quality meaningfully for small models without much extra wall-clock time.
torchrun --standalone --nproc_per_node=$NUM_GPUS train.py --steps 26500 \
    --batch 524288 \
    --micro 16 \
    --vocab-size 32768 \
    --tokenizer tokenizer.json \
    --wandb $1