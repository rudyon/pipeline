from tokenizers import Tokenizer
import numpy as np
import os
import json
import glob
from tqdm import tqdm
import argparse

from util import write_datafile

parser = argparse.ArgumentParser(description="Tokenize raw JSONL data into .npy shards")
parser.add_argument('--cache', default="data_cache")
parser.add_argument('-c', '--column', default="text")
parser.add_argument('-s', '--shard-size', type=int, default=100000000)
parser.add_argument('-m', '--max-shards', type=int, default=None)
parser.add_argument('--tokenizer', default="tokenizer.json", help="path to tokenizer.json")
args = parser.parse_args()

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), args.cache)
column_name = args.column
shard_size = args.shard_size

# discover the dataset name from the raw JSONL files
raw_files = sorted(glob.glob(os.path.join(DATA_CACHE_DIR, "*_raw_*.jsonl")))
assert len(raw_files) > 0, f"No raw JSONL files found in {DATA_CACHE_DIR}. Run get_data.py first."

# extract dataset name
basename = os.path.basename(raw_files[0])
dataset_name = basename.rsplit("_raw_", 1)[0]

# load HF tokenizer
enc = Tokenizer.from_file(args.tokenizer)
eot_token_str = chr(60) + "|endoftext|" + chr(62)
eot = enc.token_to_id(eot_token_str)
assert eot is not None, f"EOT token {eot_token_str} not found in tokenizer"
vocab_size = enc.get_vocab_size()
print(f"Loaded tokenizer from {args.tokenizer} (vocab size: {vocab_size})")

# determine dtype based on vocab size
if vocab_size < 2**16:
    token_dtype = np.uint16
else:
    token_dtype = np.uint32

def tokenize_doc(text):
    if not text:
        return np.array([], dtype=token_dtype)
    encoded = enc.encode(text)
    tokens = [eot] + encoded.ids
    tokens_np = np.array(tokens, dtype=token_dtype)
    return tokens_np

shard_index = 0
all_tokens_np = np.empty((shard_size,), dtype=token_dtype)
token_count = 0
progress_bar = None
done = False

for path in raw_files:
    if done:
        break
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            text = doc[column_name]
            tokens = tokenize_doc(text)
            if len(tokens) == 0:
                continue

            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"{dataset_name}_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                if progress_bar is not None:
                    progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                if args.max_shards is not None and shard_index >= args.max_shards:
                    done = True
                    break
                leftover = tokens[remainder:]
                all_tokens_np = np.empty((shard_size,), dtype=token_dtype)
                all_tokens_np[0:len(leftover)] = leftover
                token_count = len(leftover)

if not done and token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"{dataset_name}_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])
