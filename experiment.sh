#!/bin/bash
# this is designed to run on Kaggle
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv pip install -r requirements.txt || pip install -r requirements.txt -q
source .venv/bin/activate 2>/dev/null || true
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
if [ ! "$(ls -A test_cache 2>/dev/null)" ]; then
    python prepare_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT -m 2 --cache test_cache
fi
torchrun --standalone --nproc_per_node=$NUM_GPUS train.py 500 \
    --depth 4 \
    --batch 32768 \
    --micro 4 \
    --cache test_cache \
    --experiment $1