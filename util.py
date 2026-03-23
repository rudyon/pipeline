import torch
import os
import numpy as np
import random
import math
from torch.distributed import init_process_group


def fmt_elapsed(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f}h"
    else:
        return f"{seconds / 86400:.2f}d"


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(
        self, B, T, process_rank, num_processes, split, data_root="data_cache"
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {"train", "val"}
        self.data_root = data_root
        shards = os.listdir(data_root)
        shards = [os.path.join(data_root, s) for s in shards if split in s]
        shards.sort()  # Ensure consistent order across all GPUs
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
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
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


# DDP setup shared across training scripts
def setup_ddp():
    """Initialize distributed data parallel. Returns tuple of DDP parameters."""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA required for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device


def get_lr(step, warmup_steps, max_steps, stable_ratio, max_lr, min_lr):
    """Cosine learning rate schedule with warmup and stable phase."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step < max_steps * stable_ratio:
        return max_lr
    decay_steps = max_steps - int(max_steps * stable_ratio)
    it_in_decay = step - int(max_steps * stable_ratio)
    coeff = 0.5 * (1.0 + math.cos(math.pi * it_in_decay / decay_steps))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(path, model, optimizer, step, val_loss):
    """Save model checkpoint including optimizer states."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": [opt.state_dict() for opt in optimizer],
        "step": step,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, device, master_process=True):
    """Load checkpoint and restore model/optimizer states.

    Returns: Tuple of (step, val_loss) from checkpoint.
    """
    if master_process:
        print(f"Resuming from checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    checkpoint_optimizers = checkpoint["optimizer"]
    if isinstance(checkpoint_optimizers, list) and len(checkpoint_optimizers) == len(
        optimizer
    ):
        for opt, state in zip(optimizer, checkpoint_optimizers):
            opt.load_state_dict(state)
    else:
        if master_process:
            print(
                "Warning: Optimizer checkpoint format mismatch. Starting optimizers from scratch."
            )
    return checkpoint["step"], checkpoint["val_loss"]
