import argparse
import os
import time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from util import setup_ddp, fmt_elapsed
import torch.distributed as dist
import json

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument(
    "steps",
    type=int,
    help="Number of merge operations (vocab_size = 256 + num_special_tokens + steps)",
)
parser.add_argument("-c", "--column", default="text")
parser.add_argument("-C", "--config", default="default")
parser.add_argument("-o", "--output", default="tokenizer.json")
parser.add_argument("--cache", default="data_cache")
parser.add_argument(
    "-v",
    "--vocab-size",
    type=int,
    default=None,
    help="Target vocab size (calculated from steps if not provided)",
)
parser.add_argument(
    "-m",
    "--max-samples",
    type=int,
    default=None,
    help="Max samples to use for training",
)
parser.add_argument(
    "-s",
    "--save-every",
    type=int,
    default=1000,
    help="Save intermediate tokenizer every N steps",
)
parser.add_argument("--wandb", default=None, help="wandb run name for logging")
args = parser.parse_args()

# Special tokens
SPECIAL_TOKENS = ["<|endoftext|>", "<pad>", "<unk>"]

ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_ddp()

if master_process and args.wandb is not None:
    import wandb

    wandb.init(project="pipeline", name=args.wandb)

# Calculate actual vocab size
# We start with 256 byte tokens, add special tokens, then perform BPE merges
base_vocab = 256
num_special = len(SPECIAL_TOKENS)
target_vocab = (
    args.vocab_size if args.vocab_size else (base_vocab + num_special + args.steps)
)
num_merges = target_vocab - base_vocab - num_special

if master_process:
    print(f"Training BPE tokenizer")
    print(f"  Target vocab size: {target_vocab}")
    print(f"  Byte tokens: {base_vocab}")
    print(f"  Special tokens: {num_special}")
    print(f"  Merge operations: {num_merges}")
    print(f"  Dataset: {args.dataset}")
    print()

# Initialize tokenizer
# BPE model with byte-level pre-tokenization
# This ensures we can encode ANY unicode text, even emojis or weird characters
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

# ByteLevel pre-tokenizer splits on whitespace but keeps track of byte offsets
# It handles unicode properly by operating on bytes
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

# Trainer with our target vocab size
trainer = BpeTrainer(
    vocab_size=target_vocab,
    special_tokens=SPECIAL_TOKENS,
    min_frequency=2,  # Only merge pairs that appear at least twice
    show_progress=master_process,
)

# Load dataset in streaming mode (same as prepare_data.py)
# We only need the master process to load and train the tokenizer
# Since tokenizer training is deterministic, all processes get same result
if master_process:
    time_start = time.time()

    dataset = load_dataset(
        args.dataset, name=args.config, split="train", streaming=True
    )

    # Iterator that yields text from dataset
    def text_iterator():
        count = 0
        for doc in dataset:
            if args.max_samples and count >= args.max_samples:
                break
            text = doc.get(args.column, "")
            if text:
                yield text
            count += 1
            if count % 10000 == 0 and master_process:
                print(f"Processed {count} documents...")

    # Train the tokenizer
    print("Starting tokenizer training...")
    tokenizer.train_from_iterator(
        text_iterator(), trainer=trainer, length=args.max_samples
    )

    # Set up post-processing to add EOT token
    # This ensures every document ends with EOT token
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))],
    )

    training_time = time.time() - time_start

    # Save the tokenizer
    tokenizer.save(args.output)
    print(f"\nTokenizer saved to {args.output}")
    print(f"Training took {fmt_elapsed(training_time)}")

    # Log vocab info
    vocab = tokenizer.get_vocab()
    print(f"Actual vocab size: {len(vocab)}")
    print(f"Sample tokens:")
    # Show some interesting tokens
    sample_tokens = list(vocab.items())[:10] + list(vocab.items())[-10:]
    for token, idx in sample_tokens:
        print(f"  '{token}': {idx}")

    # Test encoding/decoding
    test_text = "Hello world! This is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\nTest encoding:")
    print(f"  Input: '{test_text}'")
    print(f"  Tokens: {encoded.tokens}")
    print(f"  IDs: {encoded.ids}")
    print(f"  Decoded: '{decoded}'")

    if args.wandb:
        wandb.log(
            {
                "vocab_size": len(vocab),
                "training_time_sec": training_time,
                "num_merges": num_merges,
            }
        )
        wandb.finish()

if ddp:
    dist.barrier()  # Wait for master to finish
    dist.destroy_process_group()

if master_process:
    print("\nTokenizer training complete!")
    print(f"Use it with: tokenizer = Tokenizer.from_file('{args.output}')")
