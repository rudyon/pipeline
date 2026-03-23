import torch
import os
import numpy as np
import random

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def fmt_elapsed(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.2f}h"
    else:
        return f"{seconds/86400:.2f}d"

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root="data_cache"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        assert split in {'train', 'val'}
        self.data_root = data_root
        shards = os.listdir(data_root)
        shards = [os.path.join(data_root, s) for s in shards if split in s]
        shards.sort() # Ensure consistent order across all GPUs
        self.shards = shards
        
        assert len(shards) > 0, f"no shards found for split {split}"
        if process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # Each process starts at a unique offset
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        
        # Advance by the total tokens processed by the whole GPU fleet
        self.current_position += B * T * self.num_processes
        
        # If the next jump goes out of bounds, move to the next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y