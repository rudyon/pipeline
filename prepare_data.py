from datasets import load_dataset
import tiktoken
import numpy as np
import os
import multiprocess as mp
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('-c', '--column', default="text")
parser.add_argument('-C', '--config', default="default")
parser.add_argument('-s', '--shard-size', type=int, default=100000000)
parser.add_argument('-m', '--max-shards', type=int, default=None)
parser.add_argument('-ca' '--cache', default="data_cache")
args = parser.parse_args()

hf_dataset = args.dataset
column_name = args.column
hf_dataset_name = hf_dataset.split('/')[1]
local_dir = args.cache
remote_name = args.config
shard_size = args.shard_size

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

dataset = load_dataset(hf_dataset, name=remote_name, split="train", streaming=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']
def tokenize(doc):
    text = doc[column_name]
    if not text:
        return np.array([], dtype=np.uint16)
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, dataset, chunksize=16):
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{hf_dataset_name}_{split}_{shard_index:06d}")
            remainder = shard_size - token_count
            if progress_bar is not None:
                progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            if args.max_shards is not None and shard_index >= args.max_shards:
                pool.terminate()
                break
            leftover = tokens[remainder:]
            all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
            all_tokens_np[0:len(leftover)] = leftover
            token_count = len(leftover)
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{hf_dataset_name}_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
        