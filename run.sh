#!/bin/bash
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

# Run with torchrun
# --standalone: single-node multi-gpu
# --nproc_per_node: uses all detected GPUs
torchrun --standalone --nproc_per_node=$NUM_GPUS train.py 19073 \
    --batch 524288 \
    --micro 16 \
    --wandb $1