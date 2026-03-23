"""
Train a BPE tokenizer on raw text data using HuggingFace tokenizers.
"""

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers.processors import TemplateProcessing
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description="Train BPE tokenizer on raw text")
parser.add_argument("data_dir", help="Directory containing .txt files")
parser.add_argument(
    "-v", "--vocab-size", type=int, default=50304, help="Target vocabulary size"
)
parser.add_argument(
    "-o", "--output", default="tokenizer.json", help="Output tokenizer file"
)
parser.add_argument(
    "-s",
    "--special-tokens",
    nargs="+",
    default=["<|endoftext|>", "<|pad|>", "<|unk|>"],
    help="Special tokens to include",
)
args = parser.parse_args()

DATA_DIR = os.path.abspath(args.data_dir)


def get_files():
    """Get all .txt files from data directory."""
    files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".txt"):
            files.append(os.path.join(DATA_DIR, f))
    return sorted(files)


def file_iterator():
    """Iterator over all text files."""
    files = get_files()
    if not files:
        raise ValueError(f"No .txt files found in {DATA_DIR}")

    print(f"Training on {len(files)} files...")
    total_chars = 0

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            total_chars += len(text)
            yield text

    print(f"Total training data: {total_chars:,} characters")


print(f"Initializing BPE tokenizer (vocab size: {args.vocab_size})...")

# Initialize BPE model
tokenizer = Tokenizer(models.BPE())

# Use ByteLevel pre-tokenizer (like GPT-2)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Set decoder for detokenization
tokenizer.decoder = decoders.ByteLevel()

# Configure trainer
trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    special_tokens=args.special_tokens,
    min_frequency=2,
    show_progress=True,
)

print("Training tokenizer (this may take a while)...")
tokenizer.train_from_iterator(file_iterator(), trainer=trainer, length=len(get_files()))

# Set post-processor for special tokens handling
tokenizer.post_processor = TemplateProcessing(
    single="$A",
    special_tokens=[
        (t, tokenizer.token_to_id(t))
        for t in args.special_tokens
        if tokenizer.token_to_id(t) is not None
    ],
)

# Enable truncation and padding
tokenizer.enable_truncation(max_length=1024)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<|pad|>"), pad_token="<|pad|>")

# Save tokenizer
output_path = os.path.abspath(args.output)
tokenizer.save(output_path)

print(f"\nTokenizer saved to: {output_path}")

# Print statistics
vocab = tokenizer.get_vocab()
print(f"\n=== Tokenizer Statistics ===")
print(f"Vocabulary size: {len(vocab):,}")
print(f"Special tokens: {args.special_tokens}")

# Test compression on sample
test_files = get_files()[:3]
total_chars = 0
total_tokens = 0
for filepath in test_files:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()[:100000]  # Sample first 100k chars
        total_chars += len(text)
        encoding = tokenizer.encode(text)
        total_tokens += len(encoding.ids)

if total_tokens > 0:
    compression = total_chars / total_tokens
    print(f"Compression ratio: {compression:.2f} chars/token")
    print(f"Sample tokenization: {total_tokens:,} tokens from {total_chars:,} chars")

print("\n=== Tokenizer configuration ===")
config = {
    "vocab_size": len(vocab),
    "special_tokens": args.special_tokens,
    "model_type": "BPE",
    "padding_token": "<|pad|>",
    "eot_token": "<|endoftext|>",
}
print(json.dumps(config, indent=2))
