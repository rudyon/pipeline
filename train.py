import torch
from model import *
from util import *
import time
import math
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('steps', type=int)
parser.add_argument('-d', '--depth', type=int, default=12)
parser.add_argument('-b', '--batch', type=int, default=524288)
parser.add_argument('-m', '--micro', type=int, default=16)
parser.add_argument('-s', '--sequence', type=int, default=1024)
parser.add_argument('-w', '--wandb', default=None)
parser.add_argument('-c', '--cache', default="data_cache")
parser.add_argument('-r', '--resume', default=None)
args = parser.parse_args()

if args.wandb is not None:
    wandb.init(project="build-nanogpt", name=args.wandb)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device: {device}")

autocast_device = device if device in ('cuda', 'mps') else 'cpu'
use_autocast = device != 'cpu'

device_type = "cuda" if device.startswith("cuda") else "cpu"

total_batch_size = args.batch # in number of tokens
B = args.micro # micro batch size
T = args.sequence # sequence length
assert total_batch_size % (B * T) == 0, "make sure total batch size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size {total_batch_size}")
print(f"=> calculated gradient accumulation steps {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, split="train", data_root=args.cache)
val_loader = DataLoaderLite(B=B, T=T, split="val", data_root=args.cache)

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(depth=args.depth, vocab_size=50304)) # nice numbers make GPU go faster
model.to(device)

if device != 'cpu':
    model = torch.compile(model)

max_lr = 0.0006
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = args.steps
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=0.0006, device=device)

start_step = 0
if args.resume is not None:
    print(f"resuming from checkpoint {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_step = checkpoint['step'] + 1
time_start = time.time()
for step in range(start_step, max_steps):
    last_step = (step == max_steps - 1)
    t0 = time.time()
    if step != 0 and (step % 250 == 0 or last_step):
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        print(f"validation loss {val_loss_accum.item():.4f}")
        if step > 0 and (step % 5000 == 0 or last_step):
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': model.config,
                'step': step,
                'val_loss': val_loss_accum.item()
            }
            torch.save(checkpoint, f"model_{step:05d}.pt")
            print(f"checkpoint saved to model_{step:05d}.pt")
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if use_autocast:
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    time_elapsed = fmt_elapsed(t1 - time_start)
    dt = (t1 - t0)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    if args.wandb is not None:
        wandb.log({
            "loss": loss_accum.item(),
        }, step=step)
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | elapsed {time_elapsed} | dt {dt*1000:.2f}ms | tok/sec {tokens_per_sec:.2f}")