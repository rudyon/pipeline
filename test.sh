#!/bin/bash

export OMP_NUM_THREADS=1

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
    python get_data.py HuggingFaceFW/fineweb-edu -c text -C sample-10BT --cache test_cache --max-docs 200000
    python train_tokenizer.py --cache test_cache -c text --vocab-size 32768
    python tokenize_data.py --cache test_cache -c text --tokenizer tokenizer.json -m 2
fi

# do a simple test training run of 50 steps. this is meant to be able to run on my i5 10th gen cpu
python train.py --minutes 0.5 -d 1 -b 1024 -m 1 -s 256 --cache test_cache --vocab-size 32768 --tokenizer tokenizer.json