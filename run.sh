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
if [ "$USE_CUSTOM_TOKENIZER" = true ] && [ ! -f "tokenizer.json" ]; then
    echo "Training custom tokenizer..."
    python train_tokenizer.py HuggingFaceFW/fineweb-edu 50000 \
        -c text -C sample-10BT -v 50304 \
        -o tokenizer.json
fi

if [ ! "$(ls -A data_cache 2>/dev/null)" ]; then
    if [ "$USE_CUSTOM_TOKENIZER" = true ]; then
        TOKENIZER_ARG="--tokenizer tokenizer.json"
    fi
    python prepare_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT $TOKENIZER_ARG
fi

# Get wandb name (non-flag argument)
WANDB_NAME=""
for arg in "$@"; do
    if [[ "$arg" != "--custom-tokenizer" ]]; then
        WANDB_NAME="$arg"
        break
    fi
done

# Run with torchrun
# --standalone: single-node multi-gpu
# --nproc_per_node: uses all detected GPUs
torchrun --standalone --nproc_per_node=$NUM_GPUS train.py 19073 \
    --batch 524288 \
    --micro 16 \
    --vocab-size $VOCAB_SIZE \
    --wandb "$WANDB_NAME"