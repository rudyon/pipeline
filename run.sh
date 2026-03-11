#!/bin/bash

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv pip install -r requirements.txt
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# download data and tokenize it if not already downloaded
if [ ! "$(ls -A data_cache 2>/dev/null)" ]; then
    python prepare_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT
fi

# actually begin training
python train.py 19073 -w $1 # ~1 epoch, if data is 10B tokens and batch size is 0.5M tokens