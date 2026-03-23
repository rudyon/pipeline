#!/bin/bash

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv pip install -r requirements.txt
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Check if we should use custom tokenizer
USE_CUSTOM_TOKENIZER=false
VOCAB_SIZE=50304
TOKENIZER_ARG=""
for arg in "$@"; do
    if [[ "$arg" == "--custom-tokenizer" ]]; then
        USE_CUSTOM_TOKENIZER=true
    fi
done

# Train custom tokenizer if requested and doesn't exist
if [ "$USE_CUSTOM_TOKENIZER" = true ] && [ ! -f "test_tokenizer.json" ]; then
    echo "Training test tokenizer..."
    python train_tokenizer.py HuggingFaceFW/fineweb-edu 10000 \
        -c text -C sample-10BT -v 32000 \
        -m 50000 --cache test_cache \
        -o test_tokenizer.json
fi

# download some test data and tokenize it if not already downloaded
# using test_cache as the test path and max shards of 2 here so it's fast for testing
if [ ! "$(ls -A test_cache 2>/dev/null)" ]; then
    if [ "$USE_CUSTOM_TOKENIZER" = true ]; then
        TOKENIZER_ARG="--tokenizer test_tokenizer.json"
        VOCAB_SIZE=32000
    fi
    python prepare_data.py HuggingFaceFW/fineweb-edu \
        -c text -C sample-10BT -m 2 --cache test_cache $TOKENIZER_ARG
fi

# do a simple test training run of 50 steps. this is meant to be able to run on my i5 10th gen cpu
python train.py 50 -d 1 -b 1024 -m 1 -s 256 --cache test_cache --vocab-size $VOCAB_SIZE