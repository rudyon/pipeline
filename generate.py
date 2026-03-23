from model import LLM, LLMConfig
from util import load_tokenizer
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("-d", "--depth", type=int, default=12)
parser.add_argument(
    "-t",
    "--tokenizer",
    default=None,
    help='Tokenizer: "gpt2" or path to tokenizer.json (auto-detected if not specified)',
)
parser.add_argument(
    "-p", "--prompt", default="The model architecture is", help="Prompt text"
)
parser.add_argument(
    "-n", "--tokens", type=int, default=50, help="Number of tokens to generate"
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Auto-detect tokenizer from checkpoint directory if not specified
if args.tokenizer is None:
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    if not checkpoint_dir:
        checkpoint_dir = "."

    # Look for tokenizer.json in checkpoint directory or parent directory
    possible_paths = [
        os.path.join(checkpoint_dir, "tokenizer.json"),
        os.path.join(checkpoint_dir, "test_tokenizer.json"),
        os.path.join(checkpoint_dir, "..", "tokenizer.json"),
        os.path.join(checkpoint_dir, "..", "test_tokenizer.json"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            args.tokenizer = os.path.abspath(path)
            break

    if args.tokenizer is None:
        print("Error: Could not auto-detect tokenizer.")
        print("Please specify with --tokenizer /path/to/tokenizer.json")
        print("Or use --tokenizer gpt2 for GPT-2 tokenizer")
        exit(1)

# Load tokenizer and get vocab_size
tokenizer, vocab_size = load_tokenizer(args.tokenizer)
print(f"Using tokenizer: {args.tokenizer}")
print(f"Vocab size: {vocab_size}")

checkpoint = torch.load(args.checkpoint, map_location=device)
config = LLMConfig(depth=args.depth, vocab_size=vocab_size)
model = LLM(config)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print(model.generate(args.prompt, max_new_tokens=args.tokens, tokenizer=tokenizer))
