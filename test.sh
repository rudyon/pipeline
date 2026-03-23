#!/bin/bash

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv

# install the repo dependencies
uv pip install -r requirements.txt

# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Default paths (using small values for testing)
RAW_CACHE="${RAW_CACHE:-test_raw_cache}"
TOKENIZER="${TOKENIZER:-test_tokenizer.json}"
VOCAB_SIZE="${VOCAB_SIZE:-1000}"
SHARD_SIZE="${SHARD_SIZE:-1000000}"

echo "Raw data cache: $RAW_CACHE"
echo "Tokenizer path: $TOKENIZER"
echo "Vocab size: $VOCAB_SIZE"
echo "Shard size: $SHARD_SIZE"

# Step 1: Download test data if needed (using small shards for speed)
if [ ! -d "$RAW_CACHE/fineweb-edu" ] || [ ! "$(ls -A $RAW_CACHE/fineweb-edu 2>/dev/null)" ]; then
    echo "=== Step 1: Downloading test data ==="
    python get_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT -m 2 --cache $RAW_CACHE -s $SHARD_SIZE
else
    echo "=== Step 1: Raw data already exists, skipping ==="
fi

# Step 2: Train tokenizer if needed (small vocab for fast testing)
if [ ! -f "$TOKENIZER" ]; then
    echo "=== Step 2: Training test tokenizer ==="
    python train_tokenizer.py $RAW_CACHE/fineweb-edu -v $VOCAB_SIZE -o $TOKENIZER
else
    echo "=== Step 2: Tokenizer already exists ($TOKENIZER), skipping ==="
fi

# Step 3: Tokenize data if needed
if [ ! -d "test_cache" ] || [ ! "$(ls -A test_cache 2>/dev/null)" ]; then
    echo "=== Step 3: Tokenizing test data ==="
    python tokenize_data.py $RAW_CACHE/fineweb-edu $TOKENIZER --cache test_cache -s $SHARD_SIZE
else
    echo "=== Step 3: Tokenized data already exists, skipping ==="
fi

# Step 4: Test training (small model for speed)
echo "=== Step 4: Running test training ==="
echo "Using tokenizer: $TOKENIZER"

python train.py 50 -d 1 -b 1024 -m 1 -s 256 --cache test_cache --tokenizer $TOKENIZER

echo "=== Test complete! ==="