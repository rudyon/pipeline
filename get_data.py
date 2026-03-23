from datasets import load_dataset
import json
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Download raw text data from HuggingFace and save as JSONL files")
parser.add_argument('dataset')
parser.add_argument('-c', '--column', default="text")
parser.add_argument('-C', '--config', default="default")
parser.add_argument('--cache', default="data_cache")
parser.add_argument('--max-docs', type=int, default=None)
parser.add_argument('--docs-per-file', type=int, default=100000)
args = parser.parse_args()

hf_dataset = args.dataset
column_name = args.column
hf_dataset_name = hf_dataset.split('/')[1]
local_dir = args.cache
remote_name = args.config

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

dataset = load_dataset(hf_dataset, name=remote_name, split="train", streaming=True)

file_index = 0
doc_count = 0
current_file = None
writer = None

progress = tqdm(desc="Downloading docs", unit=" docs")

for doc in dataset:
    text = doc[column_name]
    if not text:
        continue

    # open a new JSONL file when needed
    if current_file is None:
        path = os.path.join(DATA_CACHE_DIR, f"{hf_dataset_name}_raw_{file_index:06d}.jsonl")
        current_file = open(path, "w", encoding="utf-8")
        file_count_in_current = 0

    current_file.write(json.dumps({column_name: text}) + "\n")
    file_count_in_current += 1
    doc_count += 1
    progress.update(1)

    # rotate to next file after docs_per_file documents
    if file_count_in_current >= args.docs_per_file:
        current_file.close()
        current_file = None
        file_index += 1

    if args.max_docs is not None and doc_count >= args.max_docs:
        break

if current_file is not None:
    current_file.close()

progress.close()
print(f"Saved {doc_count} documents in {file_index + 1} file(s) to {DATA_CACHE_DIR}")

# force exit — HuggingFace streaming dataset leaves background threads
# that prevent the process from terminating after breaking out of the iterator
os._exit(0)
