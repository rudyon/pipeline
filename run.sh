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

# Default paths
RAW_CACHE="${RAW_CACHE:-raw_data_cache}"
TOKENIZER="${TOKENIZER:-tokenizer.json}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
VOCAB_SIZE="${VOCAB_SIZE:-50304}"

echo "Using dataset: $DATASET"
echo "Raw data cache: $RAW_CACHE"
echo "Tokenizer path: $TOKENIZER"
echo "Vocab size: $VOCAB_SIZE"

# Step 1: Download raw data if needed
if [ ! -d "$RAW_CACHE/fineweb-edu" ] || [ ! "$(ls -A $RAW_CACHE/fineweb-edu 2>/dev/null)" ]; then
    echo "=== Step 1: Downloading raw data ==="
    python get_data.py $DATASET -c text -C sample-10BT --cache $RAW_CACHE
else
    echo "=== Step 1: Raw data already exists, skipping download ==="
fi

# Step 2: Train tokenizer if needed
if [ ! -f "$TOKENIZER" ]; then
    echo "=== Step 2: Training tokenizer ==="
    python train_tokenizer.py $RAW_CACHE/fineweb-edu -v $VOCAB_SIZE -o $TOKENIZER
else
    echo "=== Step 2: Tokenizer already exists ($TOKENIZER), skipping training ==="
fi

# Step 3: Tokenize data if needed
if [ ! -d "data_cache" ] || [ ! "$(ls -A data_cache 2>/dev/null)" ]; then
    echo "=== Step 3: Tokenizing data ==="
    python tokenize_data.py $RAW_CACHE/fineweb-edu $TOKENIZER --cache data_cache
else
    echo "=== Step 3: Tokenized data already exists, skipping ==="
fi

# Step 4: Train
echo "=== Step 4: Starting training ==="
echo "Using tokenizer: $TOKENIZER"

torchrun --standalone --nproc_per_node=$NUM_GPUS train.py 19073 \
    --batch 524288 \
    --micro 16 \
    --wandb $1 \
    --tokenizer $TOKENIZER