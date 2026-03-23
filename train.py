import torch
import os
import numpy as np
import time
import math
import argparse
import wandb
from torch.distributed import destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import LLM, LLMConfig
from util import (
    DataLoaderLite,
    fmt_elapsed,
    setup_ddp,
    get_lr,
    save_checkpoint,
    load_checkpoint,
)
from hellaswag import get_hellaswag_acc
import json

parser = argparse.ArgumentParser()
parser.add_argument("steps", type=int)
parser.add_argument("-d", "--depth", type=int, default=12)
parser.add_argument("-b", "--batch", type=int, default=524288)
parser.add_argument("-m", "--micro", type=int, default=16)
parser.add_argument("-s", "--sequence", type=int, default=1024)
parser.add_argument("-w", "--wandb", default=None)
parser.add_argument("-c", "--cache", default="data_cache")
parser.add_argument("-r", "--resume", default=None)
parser.add_argument("-e", "--experiment", default=None)
parser.add_argument("-el", "--experimentlong", default=None)
parser.add_argument(
    "--vocab-size",
    type=int,
    default=50304,
    help="Vocabulary size (must match tokenizer)",
)
args = parser.parse_args()

if args.experimentlong is not None and args.experiment is not None:
    print("can't mix long and short experiment arguments. sorry")
    exit(1)

# manual seed for experimentation only!
if args.experiment is not None or args.experimentlong is not None:
    torch.manual_seed(42)

# Setup DDP
ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_ddp()

if master_process and args.wandb is not None:
    wandb.init(project="pipeline", name=args.wandb)

device_type = "cuda" if device.startswith("cuda") else "cpu"
autocast_device = device if device_type == "cuda" else "cpu"
use_autocast = device_type == "cuda"

# Batch calculations
total_batch_size = args.batch
B, T = args.micro, args.sequence
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total batch size: {total_batch_size}")
    print(f"=> grad accum steps: {grad_accum_steps}")

# Data Loaders
train_loader = DataLoaderLite(
    B=B,
    T=T,
    split="train",
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    data_root=args.cache,
)
val_loader = DataLoaderLite(
    B=B,
    T=T,
    split="val",
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    data_root=args.cache,
)

torch.set_float32_matmul_precision("high")

model = LLM(LLMConfig(depth=args.depth, vocab_size=args.vocab_size))
model.to(device)
if device_type == "cuda":
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=0.0006, device=device
)

# Learning rate schedule
max_lr, min_lr = 0.001, 0.00006
warmup_steps, max_steps = 715, args.steps
stable_ratio = 0.8


start_step = 0
if args.resume:
    start_step, _ = load_checkpoint(
        args.resume, raw_model, optimizer, device, master_process
    )
    start_step += 1

time_start = time.time()
best_val_loss = float("inf")

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    # Validation and Evaluation
    if step != 0 and (step % 250 == 0 or last_step):
        model.eval()
        val_loader.reset()
        val_loss_accum = torch.zeros(1, device=device)
        val_loss_steps = 20
        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                val_loss_accum += loss / val_loss_steps

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            hw_acc = get_hellaswag_acc(
                raw_model, device, limit=1000 if last_step else 200
            )
            print(
                f"step {step} | val loss {val_loss_accum.item():.4f} | hellaswag {hw_acc * 100:.2f}%"
            )
            if args.wandb:
                wandb.log(
                    {"val loss": val_loss_accum.item(), "hellaswag": hw_acc * 100},
                    step=step,
                )

            save_checkpoint(
                f"model_{step:05d}.pt",
                raw_model,
                optimizer,
                step,
                val_loss_accum.item(),
            )
            if val_loss_accum.item() < best_val_loss:
                best_val_loss = val_loss_accum.item()
                save_checkpoint(
                    "model_best.pt", raw_model, optimizer, step, val_loss_accum.item()
                )

    # Training loop
    model.train()
    for opt in optimizer:
        opt.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=use_autocast
        ):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step, warmup_steps, max_steps, stable_ratio, max_lr, min_lr)
    for param_group in optimizer[0].param_groups:
        param_group["lr"] = lr * 10
    for param_group in optimizer[1].param_groups:
        param_group["lr"] = lr

    for opt in optimizer:
        opt.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    if master_process:
        dt = time.time() - t0
        tokens_per_sec = total_batch_size / dt
        print(
            f"step {step:4d} | loss: {loss_accum.item():.6f} | dt {dt * 1000:.2f}ms | tok/sec {tokens_per_sec:.2f} | elapsed {fmt_elapsed((time.time() - time_start))}"
        )
        if args.wandb:
            wandb.log({"train loss": loss_accum.item(), "lr": lr}, step=step)

if master_process:
    print(f"training took {fmt_elapsed((time.time() - time_start))}")
    if args.experiment:
        result = {
            "id": len(open("experiments.jsonl").readlines())
            if os.path.exists("experiments.jsonl")
            else 0,
            "name": args.experiment,
            "val_loss": val_loss_accum.item(),
            "kept": None,  # filled in manually later
        }
        with open("experiments.jsonl", "a") as f:
            f.write("\n" + json.dumps(result))
        print(
            f"logged experiment '{args.experiment}' with val loss {val_loss_accum.item():.4f}"
        )
    elif args.experimentlong:
        result = {
            "id": len(open("experiments_long.jsonl").readlines())
            if os.path.exists("experiments_long.jsonl")
            else 0,
            "name": args.experimentlong,
            "val_loss": val_loss_accum.item(),
            "kept": None,  # filled in manually later
        }
        with open("experiments_long.jsonl", "a") as f:
            f.write("\n" + json.dumps(result))
        print(
            f"logged experimentlong '{args.experimentlong}' with val loss {val_loss_accum.item():.4f}"
        )


if ddp:
    destroy_process_group()
