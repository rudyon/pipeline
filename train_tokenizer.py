from tokenizers import ByteLevelBPETokenizer
import os
import json
import glob
import argparse

parser = argparse.ArgumentParser(description="Train a BPE tokenizer on raw JSONL data")
parser.add_argument('--cache', default="data_cache")
parser.add_argument('-c', '--column', default="text")
parser.add_argument('--vocab-size', type=int, default=32768)
parser.add_argument('--min-frequency', type=int, default=2)
args = parser.parse_args()

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), args.cache)

# discover raw JSONL files
raw_files = sorted(glob.glob(os.path.join(DATA_CACHE_DIR, "*_raw_*.jsonl")))
assert len(raw_files) > 0, f"No raw JSONL files found in {DATA_CACHE_DIR}. Run get_data.py first."

print(f"Found {len(raw_files)} raw file(s) in {DATA_CACHE_DIR}")

# iterator that yields text from JSONL files
def text_iterator():
    for path in raw_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    text = doc.get(args.column, "")
                    if text:
                        yield text

tokenizer = ByteLevelBPETokenizer()

EOT = chr(60) + "|endoftext|" + chr(62)
PAD = chr(60) + "|padding|" + chr(62)

tokenizer.train_from_iterator(
    text_iterator(),
    vocab_size=args.vocab_size,
    min_frequency=args.min_frequency,
    special_tokens=[EOT, PAD],
)

# save in current working directory (like model checkpoints)
output_path = "tokenizer.json"
tokenizer.save(output_path)
print(f"Tokenizer saved to {output_path} (vocab size: {tokenizer.get_vocab_size()})" )
