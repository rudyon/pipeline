#!/bin/bash

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv pip install -r requirements.txt
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# download some test data and tokenize it if not already downloaded
# using test_cache as the test path and max shards of 2 here so it's fast for testing
if [ ! "$(ls -A test_cache 2>/dev/null)" ]; then
    python prepare_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT -m 2 --cache test_cache
fi

# do a simple test training run of 50 steps. this is meant to be able to run on my i5 10th gen cpu
python train.py 50 -d 1 -b 1024 -m 1 -s 256 --cache test_cache