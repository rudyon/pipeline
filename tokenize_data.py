import tiktoken
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
parser.add_argument('--tokenizer', default="gpt2", help="tiktoken encoding name (default: gpt2)")
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

enc = tiktoken.get_encoding(args.tokenizer)
eot = enc._special_tokens[chr(60) + "|endoftext|" + chr(62)]

def tokenize_doc(text):
    if not text:
        return np.array([], dtype=np.uint16)
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    return tokens_np.astype(np.uint16)

shard_index = 0
all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
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
                all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
                all_tokens_np[0:len(leftover)] = leftover
                token_count = len(leftover)

if not done and token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"{dataset_name}_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])
