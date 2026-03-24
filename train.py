import torch
import os
import numpy as np
import time
import math
import argparse
import wandb
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import LLM, LLMConfig
from util import DataLoaderLite, fmt_elapsed
from hellaswag import get_hellaswag_acc
from tokenizers import Tokenizer
import json

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=0, help="Fixed number of steps to train (overrides --minutes if > 0)")
parser.add_argument("--minutes", type=float, default=5.0, help="Training time budget in minutes")
parser.add_argument("-d", "--depth", type=int, default=12)
parser.add_argument("-b", "--batch", type=int, default=524288)
parser.add_argument("-m", "--micro", type=int, default=16)
parser.add_argument("-s", "--sequence", type=int, default=1024)
parser.add_argument("-w", "--wandb", default=None)
parser.add_argument("-c", "--cache", default="data_cache")
parser.add_argument("-r", "--resume", default=None)
parser.add_argument("-e", "--experiment", default=None)
parser.add_argument("-v", "--vocab-size", type=int, default=32768)
parser.add_argument("-t", "--tokenizer", default="tokenizer.json", help="path to tokenizer.json")
args = parser.parse_args()

# load tokenizer for hellaswag eval
tokenizer = Tokenizer.from_file(args.tokenizer)

# manual seed for experimentation only!
if args.experiment is not None:
    torch.manual_seed(42)

# Setup DDP
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA required for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

# round vocab_size up to nearest multiple of 64 for GPU efficiency
padded_vocab_size = ((args.vocab_size + 63) // 64) * 64
model = LLM(LLMConfig(depth=args.depth, vocab_size=padded_vocab_size))
model.to(device)
if device_type == "cuda":
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=0.0006, device=device
)

# Budget setup
use_steps_budget = args.steps > 0
if use_steps_budget:
    max_steps = args.steps
    estimated_total_steps = max_steps
else:
    max_minutes = args.minutes
    target_end_time = time.time() + (max_minutes * 60)
    # Learning rate schedule (approximate steps based on time)
    # We assume ~250 steps per 5 minutes as a rough baseline for the LR schedule
    estimated_total_steps = int((max_minutes / 5.0) * 250)
    if estimated_total_steps < 100:
        estimated_total_steps = 100

max_lr, min_lr = 0.001, 0.00006
warmup_steps = int(estimated_total_steps * 0.1) # 10% warmup
stable_ratio = 0.8


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it < estimated_total_steps * stable_ratio:
        return max_lr
    decay_steps = estimated_total_steps - (estimated_total_steps * stable_ratio)
    it_in_decay = it - (estimated_total_steps * stable_ratio)
    coeff = 0.5 * (1.0 + math.cos(math.pi * it_in_decay / decay_steps))
    return min_lr + coeff * (max_lr - min_lr)


start_step = 0
if args.resume:
    if master_process:
        print(f"Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    raw_model.load_state_dict(checkpoint["model"])
    checkpoint_optimizers = checkpoint["optimizer"]
    if isinstance(checkpoint_optimizers, list) and len(checkpoint_optimizers) == len(
        optimizer
    ):
        for opt, state in zip(optimizer, checkpoint_optimizers):
            opt.load_state_dict(state)
    else:
        if master_process:
            print(
                "Warning: Optimizer checkpoint format mismatch. Starting optimizers from scratch."
            )
    start_step = checkpoint["step"] + 1

# Read bytes_per_token from cache (saved by tokenize_data.py)
bytes_per_token_path = os.path.join(args.cache, "bytes_per_token.txt")
if os.path.exists(bytes_per_token_path):
    with open(bytes_per_token_path, "r") as f:
        bytes_per_token = float(f.read().strip())
else:
    print(f"Warning: {bytes_per_token_path} not found. BPB will not be calculated correctly. Using 1.0 fallback.")
    bytes_per_token = 1.0

time_start = time.time()
best_val_loss = float("inf")

step = start_step
while True:
    current_time = time.time()
    
    if use_steps_budget:
        if step >= max_steps:
            if master_process:
                print(f"Step budget of {max_steps} steps reached. Stopping training.")
            break
        last_step = step == max_steps - 1
    else:
        if current_time >= target_end_time:
            if master_process:
                print(f"Time budget of {max_minutes} minutes reached. Stopping training.")
            break
        # Force validation on the very last step we run before timeout
        time_remaining = target_end_time - current_time
        last_step = time_remaining < 5.0 # if less than 5 seconds left, assume it's the last step

    t0 = current_time

    # Validation and Evaluation
    if step != 0 and (step % 50 == 0 or last_step):
        model.eval()
        val_loader.reset()
        val_loss_accum = torch.zeros(1, device=device)
        val_loss_steps = 20

        # Accumulators for probe diagnostics
        all_probe_scores = []   # list of (B*T,) tensors
        all_top1_correct = []   # list of (B*T,) bool tensors
        all_token_nll = []      # list of (B*T,) per-token NLL tensors
        probe_aux_accum = 0.0

        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss, probe_scores, probe_aux = raw_model(x, y, return_probe_scores=True)

                val_loss_accum += loss / val_loss_steps
                probe_aux_accum += probe_aux.item() / val_loss_steps

                # Per-token NLL (no reduction)
                token_nll = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction="none",
                )  # (B*T,)

                top1 = logits.argmax(dim=-1).view(-1)   # (B*T,)
                correct = (top1 == y.view(-1)).float()  # (B*T,)

                all_probe_scores.append(probe_scores.view(-1).float().cpu())
                all_top1_correct.append(correct.cpu())
                all_token_nll.append(token_nll.float().cpu())

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            hw_acc = get_hellaswag_acc(
                raw_model, device, tokenizer, limit=1000 if last_step else 200
            )
            val_loss_scalar = val_loss_accum.item()
            val_bpb = val_loss_scalar / (math.log(2) * bytes_per_token)

            # ---- Probe diagnostics ----
            scores_cat = torch.cat(all_probe_scores)        # (N,)
            correct_cat = torch.cat(all_top1_correct)       # (N,)
            nll_cat = torch.cat(all_token_nll)              # (N,)

            probe_mean = scores_cat.mean().item()
            probe_variance = scores_cat.var().item()

            # Pearson correlation between probe confidence and top-1 accuracy
            s_z = scores_cat - scores_cat.mean()
            c_z = correct_cat - correct_cat.mean()
            denom = (s_z.std() * c_z.std() + 1e-8)
            probe_acc_corr = ((s_z * c_z).mean() / denom).item()

            # BPB per confidence bin (low < 0.33 ≤ mid < 0.67 ≤ high)
            log2e = 1.0 / math.log(2)
            def bin_bpb(mask):
                if mask.sum() == 0:
                    return float("nan")
                return (nll_cat[mask].mean().item() * log2e) / bytes_per_token

            low_mask  = scores_cat < 0.33
            mid_mask  = (scores_cat >= 0.33) & (scores_cat < 0.67)
            high_mask = scores_cat >= 0.67

            bpb_by_confidence = {
                "low":  bin_bpb(low_mask),
                "mid":  bin_bpb(mid_mask),
                "high": bin_bpb(high_mask),
            }
            # ---- End probe diagnostics ----

            print(
                f"step {step} | val loss {val_loss_scalar:.4f} | val bpb {val_bpb:.4f} | hellaswag {hw_acc * 100:.2f}%"
                f" | probe_mean {probe_mean:.3f} | probe_var {probe_variance:.3f} | probe_corr {probe_acc_corr:.3f}"
            )
            if args.wandb:
                wandb.log(
                    {
                        "val loss": val_loss_scalar,
                        "val bpb": val_bpb,
                        "hellaswag": hw_acc * 100,
                        "probe_mean": probe_mean,
                        "probe_variance": probe_variance,
                        "probe_aux_loss": probe_aux_accum,
                        "probe_acc_corr": probe_acc_corr,
                        "bpb_low": bpb_by_confidence["low"],
                        "bpb_mid": bpb_by_confidence["mid"],
                        "bpb_high": bpb_by_confidence["high"],
                    },
                    step=step,
                )

            # Write to log.jsonl
            log_entry = {
                "step": step,
                "val_bpb": round(val_bpb, 6),
                "probe_mean": round(probe_mean, 6),
                "probe_variance": round(probe_variance, 6),
                "probe_aux_loss": round(probe_aux_accum, 6),
                "probe_acc_corr": round(probe_acc_corr, 6),
                "bpb_by_confidence": {k: round(v, 6) if not math.isnan(v) else None
                                      for k, v in bpb_by_confidence.items()},
            }
            with open("log.jsonl", "a") as log_f:
                log_f.write(json.dumps(log_entry) + "\n")

            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": [opt.state_dict() for opt in optimizer],
                "step": step,
                "val_loss": val_loss_scalar,
                "val_bpb": val_bpb,
            }
            torch.save(checkpoint, f"model_{step:05d}.pt")
            if val_loss_accum.item() < best_val_loss:
                best_val_loss = val_loss_accum.item()
                torch.save(checkpoint, "model_best.pt")

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
    lr = get_lr(step)
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
        if use_steps_budget:
            print(
                f"step {step:4d} | loss: {loss_accum.item():.6f} | dt {dt * 1000:.2f}ms | tok/sec {tokens_per_sec:.2f} | elapsed {fmt_elapsed((time.time() - time_start))} "
            )
        else:
            print(
                f"step {step:4d} | loss: {loss_accum.item():.6f} | dt {dt * 1000:.2f}ms | tok/sec {tokens_per_sec:.2f} | elapsed {fmt_elapsed((time.time() - time_start))} | remaining {fmt_elapsed((target_end_time - time.time()))}"
            )
        if args.wandb:
            wandb.log({"train loss": loss_accum.item(), "lr": lr}, step=step)
            
    step += 1

if master_process:
    print(f"training took {fmt_elapsed((time.time() - time_start))}")
    if args.experiment:
        result = {
            "id": len(open("experiments.jsonl").readlines())
            if os.path.exists("experiments.jsonl")
            else 0,
            "name": args.experiment,
            "val_bpb": val_loss_accum.item() / (math.log(2) * bytes_per_token),
            "kept": None,  # filled in manually later
        }
        with open("experiments.jsonl", "a") as f:
            f.write("\n" + json.dumps(result))
        print(
            f"logged experiment '{args.experiment}' with val bpb {result['val_bpb']:.4f}"
        )


if ddp:
    destroy_process_group()
