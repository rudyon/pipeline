#!/bin/bash
# this is designed to run on Kaggle
pip install -r requirements.txt -q

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Default paths
RAW_CACHE="${RAW_CACHE:-raw_data_cache}"
TOKENIZER="${TOKENIZER:-tokenizer.json}"
VOCAB_SIZE="${VOCAB_SIZE:-10000}"

echo "Raw data cache: $RAW_CACHE"
echo "Tokenizer path: $TOKENIZER"
echo "Vocab size: $VOCAB_SIZE"

# Step 1: Download raw data if needed
if [ ! -d "$RAW_CACHE/fineweb-edu" ] || [ ! "$(ls -A $RAW_CACHE/fineweb-edu 2>/dev/null)" ]; then
    echo "=== Step 1: Downloading raw test data ==="
    python get_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT -m 10 --cache $RAW_CACHE
else
    echo "=== Step 1: Raw data already exists, skipping ==="
fi

# Step 2: Train tokenizer if needed
if [ ! -f "$TOKENIZER" ]; then
    echo "=== Step 2: Training test tokenizer ==="
    python train_tokenizer.py $RAW_CACHE/fineweb-edu -v $VOCAB_SIZE -o $TOKENIZER
else
    echo "=== Step 2: Tokenizer already exists ($TOKENIZER), skipping ==="
fi

# Step 3: Tokenize data if needed
if [ ! -d "test_cache" ] || [ ! "$(ls -A test_cache 2>/dev/null)" ]; then
    echo "=== Step 3: Tokenizing test data ==="
    python tokenize_data.py $RAW_CACHE/fineweb-edu $TOKENIZER --cache test_cache -s 1000000
else
    echo "=== Step 3: Tokenized data already exists, skipping ==="
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

echo "=== Step 4: Running experiment: $EXPERIMENT_NAME ==="
torchrun --standalone --nproc_per_node=$NUM_GPUS train.py $STEPS \
    --depth 4 \
    --batch 32768 \
    --micro 4 \
    --cache test_cache \
    --tokenizer $TOKENIZER \
    $EXPERIMENT_ARG $EXPERIMENT_NAME