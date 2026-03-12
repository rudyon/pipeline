from model import GPT, GPTConfig
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('-d', '--depth', type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = "cuda" if device.startswith("cuda") else "cpu"
autocast_device = device if device_type == "cuda" else "cpu"
use_autocast = device_type == "cuda"

checkpoint = torch.load(args.checkpoint, map_location=device)
model = GPT(GPTConfig(depth=args.depth, vocab_size=50304))
model = model.load_state_dict(checkpoint['model'])

model.eval()
print(model.generate("Hello, how are you?", max_new_tokens=20))