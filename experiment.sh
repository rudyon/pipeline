#!/bin/bash
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv pip install -r requirements.txt
source .venv/bin/activate

if [ ! "$(ls -A test_cache 2>/dev/null)" ]; then
    python prepare_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT -m 2 --cache test_cache
fi

python train.py 500 \
    --depth 4 \
    --batch 32768 \
    --micro 4 \
    --cache test_cache \
    --experiment $1