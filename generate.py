from model import LLM, LLMConfig
from tokenizers import Tokenizer
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('-d', '--depth', type=int, default=12)
parser.add_argument('-v', '--vocab-size', type=int, default=32768)
parser.add_argument('-t', '--tokenizer', default="tokenizer.json", help="path to tokenizer.json")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

enc = Tokenizer.from_file(args.tokenizer)

checkpoint = torch.load(args.checkpoint, map_location=device)
padded_vocab_size = ((args.vocab_size + 63) // 64) * 64
config = LLMConfig(depth=args.depth, vocab_size=padded_vocab_size)
model = LLM(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print(model.generate("The model architecture is", max_new_tokens=50, enc=enc))