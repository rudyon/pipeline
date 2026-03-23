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

if [ ! "$(ls -A data_cache 2>/dev/null)" ]; then
    python get_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT --cache data_cache
    python train_tokenizer.py --cache data_cache -c text --vocab-size 32768
    python tokenize_data.py --cache data_cache -c text --tokenizer tokenizer.json
fi

# Run with torchrun
# --standalone: single-node multi-gpu
# --nproc_per_node: uses all detected GPUs
torchrun --standalone --nproc_per_node=$NUM_GPUS train.py 19073 \
    --batch 524288 \
    --micro 16 \
    --vocab-size 32768 \
    --tokenizer tokenizer.json \
    --wandb $1