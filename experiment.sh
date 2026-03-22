#!/bin/bash
# this is designed to run on Kaggle
pip install -r requirements.txt -q

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ ! "$(ls -A test_cache 2>/dev/null)" ]; then
    python prepare_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT -m 2 --cache test_cache
fi

# Default to 300 steps, use 600 if -l/--long flag is present
STEPS=300
EXPERIMENT_ARG="--experiment"
if [[ "$*" == *"-l"* ]] || [[ "$*" == *"--long"* ]]; then
    STEPS=600
    EXPERIMENT_ARG="--experimentlong"
    echo "Running long experiment: 600 steps"
fi

# Get experiment name (first non-flag argument)
EXPERIMENT_NAME=""
for arg in "$@"; do
    if [[ "$arg" != "-l" ]] && [[ "$arg" != "--long" ]]; then
        EXPERIMENT_NAME="$arg"
        break
    fi
done

torchrun --standalone --nproc_per_node=$NUM_GPUS train.py $STEPS \
    --depth 4 \
    --batch 32768 \
    --micro 4 \
    --cache test_cache \
    $EXPERIMENT_ARG $EXPERIMENT_NAME