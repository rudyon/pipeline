#!/bin/bash
# this is designed to run on Kaggle
pip install -r requirements.txt -q

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ ! "$(ls -A test_cache 2>/dev/null)" ]; then
    python prepare_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT -m 2 --cache test_cache
fi

torchrun --standalone --nproc_per_node=$NUM_GPUS train.py 300 \
    --depth 4 \
    --batch 32768 \
    --micro 4 \
    --cache test_cache \
    --experiment $1