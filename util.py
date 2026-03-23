import torch
import os
import numpy as np
import random


def fmt_elapsed(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f}h"
    else:
        return f"{seconds / 86400:.2f}d"


def load_tokenizer(tokenizer_path_or_name):
    """Load a tokenizer from a file or use tiktoken.

    Args:
        tokenizer_path_or_name: Path to tokenizer.json or "gpt2" for tiktoken

    Returns:
        tuple: (tokenizer, vocab_size)
    """
    if tokenizer_path_or_name == "gpt2":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        # tiktoken doesn't have explicit vocab_size property, but we know it's 50257
        vocab_size = 50257
        return enc, vocab_size
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path_or_name)
        vocab_size = tokenizer.get_vocab_size()
        return tokenizer, vocab_size


def encode_text(tokenizer, text, eot_token=None):
    """Encode text using tokenizer.

    Args:
        tokenizer: Tokenizer object (tiktoken or HF)
        text: Text to encode
        eot_token: EOT token to prepend (if None, no EOT added)

    Returns:
        list: Token IDs
    """
    # Check if it's tiktoken
    if hasattr(tokenizer, "encode_ordinary"):
        tokens = tokenizer.encode_ordinary(text)
        if eot_token is not None:
            tokens = [eot_token] + tokens
        return tokens
    else:
        # HF tokenizer
        encoding = tokenizer.encode(text)
        tokens = encoding.ids
        if eot_token is not None:
            eot_id = tokenizer.token_to_id("<|endoftext|>")
            if eot_id is not None:
                tokens = [eot_id] + tokens
        return tokens


def decode_tokens(tokenizer, tokens):
    """Decode tokens using tokenizer.

    Args:
        tokenizer: Tokenizer object (tiktoken or HF)
        tokens: List of token IDs

    Returns:
        str: Decoded text
    """
    # Check if it's tiktoken
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(tokens)
    else:
        # HF tokenizer
        return tokenizer.decode(tokens)


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
        # If data is too small, tile/repeat it to ensure enough tokens
        min_required = self.B * self.T * self.num_processes + 1
        if len(self.tokens) < min_required:
            repeats = (min_required // len(self.tokens)) + 1
            self.tokens = self.tokens.repeat(repeats)
        # Each process starts at a unique offset
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # Calculate required tokens for this batch
        required_tokens = B * T + 1

        # If we're near the end of this shard, wrap around to beginning
        if self.current_position + required_tokens > len(self.tokens):
            self.current_position = 0
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])

        buf = self.tokens[
            self.current_position : self.current_position + required_tokens
        ]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        # Advance by the total tokens processed by the whole GPU fleet
        self.current_position += B * T * self.num_processes

        return x, y
