import torch
from datasets import load_dataset
import tiktoken
from torch.nn import functional as F

print("loading hellaswag...")
ds = load_dataset("Rowan/hellaswag", split="validation")
enc = tiktoken.get_encoding("gpt2")

def get_hellaswag_acc(model, device, limit=200):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, example in enumerate(ds):
            if limit is not None and idx >= limit:
                break
            ctx = example["ctx"]
            endings = example["endings"]
            label = int(example["label"])
            losses = []
            for ending in endings:
                text = ctx + " " + ending
                tokens = enc.encode(text)
                ctx_tokens = enc.encode(ctx + " ")
                ctx_len = len(ctx_tokens)
                if len(tokens) > 1024:
                    tokens = tokens[:1024]
                x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
                y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
                with torch.autocast(device_type=device if device != 'mps' else 'cpu', dtype=torch.bfloat16):
                    logits, _ = model(x)
                shift_logits = logits[0, ctx_len-1:, :]
                shift_labels = y[0, ctx_len-1:]
                if shift_labels.size(0) == 0:
                    loss = float('inf')
                else:
                    loss = F.cross_entropy(shift_logits, shift_labels).item()
                losses.append(loss)
            pred = losses.index(min(losses))
            if pred == label:
                correct += 1
            total += 1
    return correct / total