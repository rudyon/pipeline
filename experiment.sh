#!/bin/bash
# this is designed to run on Kaggle
pip install -r requirements.txt -q

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ ! "$(ls -A test_cache 2>/dev/null)" ]; then
    python get_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT --cache test_cache --max-docs 200000
    python train_tokenizer.py --cache test_cache -c text --vocab-size 32768
    python tokenize_data.py --cache test_cache -c text --tokenizer tokenizer.json -m 2
fi

MINUTES=5
EXPERIMENT_ARG="--experiment"

# Get experiment name (first argument)
EXPERIMENT_NAME="$1"

torchrun --standalone --nproc_per_node=$NUM_GPUS train.py \
    --minutes $MINUTES \
    --depth 4 \
    --batch 32768 \
    --micro 4 \
    --cache test_cache \
    --vocab-size 32768 \
    --tokenizer tokenizer.json \
    $EXPERIMENT_ARG $EXPERIMENT_NAME