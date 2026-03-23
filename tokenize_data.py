"""
Tokenize raw text data using a trained tokenizer.
Converts .txt files to .npy shards for training.
"""

from tokenizers import Tokenizer
import numpy as np
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Tokenize raw text using trained tokenizer"
)
parser.add_argument("data_dir", help="Directory containing raw .txt files")
parser.add_argument("tokenizer", help="Path to tokenizer.json file")
parser.add_argument(
    "--cache", default="data_cache", help="Output directory for tokenized data"
)
parser.add_argument(
    "-s", "--shard-size", type=int, default=100000000, help="Tokens per shard"
)
args = parser.parse_args()

DATA_DIR = os.path.abspath(args.data_dir)
TOKENIZER_PATH = os.path.abspath(args.tokenizer)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), args.cache)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer
print(f"Loading tokenizer from {TOKENIZER_PATH}...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocab_size:,}")

# Get EOT token ID
eot_token = "<|endoftext|>"
eot_id = tokenizer.token_to_id(eot_token)
if eot_id is None:
    print(f"Warning: EOT token '{eot_token}' not found, using 0")
    eot_id = 0

print(f"EOT token ID: {eot_id}")


def get_text_files():
    """Get all .txt files from data directory."""
    files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".txt"):
            split = "val" if "val_" in f else "train"
            files.append((os.path.join(DATA_DIR, f), split))
    # Sort by split and filename
    files.sort(key=lambda x: (x[1], x[0]))
    return files


# Get all files
files = get_text_files()
if not files:
    raise ValueError(f"No .txt files found in {DATA_DIR}")

print(f"Found {len(files)} text files")

# Process files and create shards
train_tokens = []
val_tokens = []
train_count = 0
val_count = 0

print("Tokenizing files...")
for filepath, split in tqdm(files, desc="Processing"):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize
    encoding = tokenizer.encode(text)
    tokens = [eot_id] + encoding.ids
    tokens_array = np.array(tokens, dtype=np.uint32)

    if split == "val":
        val_tokens.append(tokens_array)
        val_count += 1
    else:
        train_tokens.append(tokens_array)
        train_count += 1

print(f"Tokenized {train_count} train files, {val_count} val files")


def write_shards(tokens_list, split_name, shard_size):
    """Write tokens to shard files."""
    if not tokens_list:
        return 0

    # Concatenate all tokens
    all_tokens = np.concatenate(tokens_list)
    total_tokens = len(all_tokens)

    print(f"{split_name}: {total_tokens:,} total tokens")

    # Split into shards
    shard_index = 0
    for start in range(0, total_tokens, shard_size):
        end = min(start + shard_size, total_tokens)
        shard_tokens = all_tokens[start:end]

        filename = os.path.join(OUTPUT_DIR, f"{split_name}_{shard_index:06d}.npy")
        np.save(filename, shard_tokens)
        print(f"  Saved {filename} ({len(shard_tokens):,} tokens)")

        shard_index += 1

    return shard_index


# Write shards
train_shards = write_shards(train_tokens, "train", args.shard_size)
val_shards = write_shards(val_tokens, "val", args.shard_size)

print(f"\n=== Tokenization Complete ===")
print(f"Train shards: {train_shards}")
print(f"Val shards: {val_shards}")
print(f"Output directory: {OUTPUT_DIR}")

# Verify tokens are in valid range
all_tokens_check = np.concatenate(train_tokens + val_tokens)
max_token = np.max(all_tokens_check)
min_token = np.min(all_tokens_check)
print(f"Token range: [{min_token}, {max_token}]")
print(f"Vocab size: {vocab_size}")

if max_token >= vocab_size:
    print(f"WARNING: Tokens exceed vocabulary size!")
else:
    print(f"✓ All tokens within vocabulary range")
