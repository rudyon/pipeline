"""
Download raw text data from HuggingFace datasets.
Saves as text shards (no tokenization) for tokenizer training.
"""

from datasets import load_dataset
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Download raw text data from HuggingFace")
parser.add_argument(
    "dataset", help="HuggingFace dataset name (e.g., HuggingFaceFW/fineweb)"
)
parser.add_argument("-c", "--column", default="text", help="Text column name")
parser.add_argument("-C", "--config", default="default", help="Dataset config")
parser.add_argument(
    "-s", "--shard-size", type=int, default=10000000, help="Characters per shard"
)
parser.add_argument(
    "-m", "--max-shards", type=int, default=None, help="Max shards to download"
)
parser.add_argument("--cache", default="raw_data_cache", help="Output directory")
args = parser.parse_args()

# Setup paths
hf_dataset = args.dataset
hf_dataset_name = (
    hf_dataset.split("/")[1] if "/" in hf_dataset else hf_dataset.replace("/", "_")
)
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), args.cache, hf_dataset_name)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

print(f"Downloading {hf_dataset}...")
dataset = load_dataset(hf_dataset, name=args.config, split="train", streaming=True)

shard_index = 0
current_shard = []
current_size = 0

for doc in tqdm(dataset, desc="Downloading documents"):
    text = doc.get(args.column, "")
    if not text:
        continue

    current_shard.append(text)
    current_size += len(text)

    # Write shard when it reaches target size
    if current_size >= args.shard_size:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{split}_{shard_index:06d}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n\n<|endoftext|>\n\n".join(current_shard))

        if shard_index == 0:
            print(f"Saved validation shard: {filename} ({current_size:,} chars)")

        shard_index += 1
        current_shard = []
        current_size = 0

        if args.max_shards and shard_index >= args.max_shards:
            break

# Write final partial shard if any
if current_shard:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"{split}_{shard_index:06d}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n\n<|endoftext|>\n\n".join(current_shard))
    shard_index += 1

print(f"Downloaded {shard_index} shards to {DATA_CACHE_DIR}")
