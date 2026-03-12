from model import GPT, GPTConfig
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('-d', '--depth', type=int, default=12)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(args.checkpoint, map_location=device)
model = GPT(GPTConfig(depth=args.depth, vocab_size=50304))
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print(model.generate("Hello, how are you?", max_new_tokens=100))