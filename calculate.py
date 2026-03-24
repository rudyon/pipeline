"""
calculate.py — Training cost and feasibility estimator.

Imports LLMConfig directly from model.py so it automatically stays in sync
with any architecture changes — no manual updates needed.

Usage:
    python calculate.py

Outputs:
    results.csv          — per (gpu_node, depth) rows with all computed metrics
    results_time.png     — training time by depth per GPU node
    results_cost.png     — total training cost by depth per GPU node
    results_params.png   — total vs active parameters by depth
    results_vram.png     — free VRAM headroom during training by depth per GPU node
"""

import csv
import math
import sys
import importlib
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ---------------------------------------------------------------------------
# Dynamically import LLMConfig from model.py so this script always uses the
# current architecture — even if model.py changes tomorrow.
# ---------------------------------------------------------------------------
_model_path = os.path.join(os.path.dirname(__file__), "model.py")
_spec = importlib.util.spec_from_file_location("model", _model_path)
_model_module = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_model_module)
    LLMConfig = _model_module.LLMConfig
    print(f"[calculate] Loaded LLMConfig from {_model_path}")
except Exception as e:
    print(f"[calculate] ERROR: Could not load LLMConfig from model.py: {e}")
    sys.exit(1)

# ===========================================================================
# ✏️  CONFIGURE EVERYTHING HERE
# ===========================================================================

# --- Compute resources ---
# Each entry: name -> {tflops, vram_gb, cost_per_hour}
# tflops: peak BF16/FP16 TFLOP/s of a *single node* (all GPUs included)
# vram_gb: total VRAM across all GPUs in a single node (GB)
# cost_per_hour: USD/hour for the entire node
COMPUTE_RESOURCES = {
    "A1000 (Prime Intellect)": {
        "tflops": 312.0,
        "vram_gb": 80,
        "cost_per_hour": 1.23,
    },
    "RTX5090 (Prime Intellect)": {
        "tflops": 209.5,
        "vram_gb": 32,
        "cost_per_hour": 0.92,
    },
    "H100 (Prime Intellect)": {
        "tflops": 989.0,
        "vram_gb": 80,
        "cost_per_hour": 2.43,
    },
}

# --- Model / training configuration ---
DEPTHS = [12, 16, 18, 24]      # values of `depth` to evaluate

VOCAB_SIZE = 32768          # must match training vocab_size (padded to multiple of 64 internally)
DTYPE_BYTES = 2             # BF16 = 2 bytes per element

# Chinchilla: ~20 tokens per parameter is compute-optimal
TOKENS_PER_PARAM = 20.0

# Batch size in tokens (total_batch_size in train.py)
BATCH_SIZE_TOKENS = 524288  # 512 K tokens per step

# MFU: how efficiently the hardware actually runs (0–1).  60% is realistic for
# well-optimised transformer training on modern GPUs.
MFU = 0.60

# Training overhead multiplier: Adam/Muon optimizer states, gradients, etc.
# Rule of thumb: 16–20× model params in bytes when using BF16 mixed precision
# (model weights BF16 + FP32 copy + gradients BF16 + Adam/Muon states FP32×2)
OPTIMIZER_OVERHEAD = 18.0   # multiplier on raw model parameter count (in bytes per param)

# KV cache estimate during training forward pass: negligible relative to model
# weights at typical batch×sequence sizes, but we include a rough estimate.
# train.py uses B=16, T=1024 by default.
MICRO_BATCH_SIZE = 16
SEQ_LEN = 1024

# ===========================================================================
# Helper functions
# ===========================================================================

def padded_vocab(vocab_size: int) -> int:
    return ((vocab_size + 63) // 64) * 64


def count_parameters(cfg: LLMConfig, vocab_size: int) -> tuple[int, int]:
    """
    Returns (total_parameters, active_parameters).

    total_parameters : every weight in the model
    active_parameters: weights touched during a *single forward pass*
                       (MoE routing means only n_active_experts / n_experts
                        of the MoE weight is active per token)
    """
    vsize = padded_vocab(vocab_size)

    # Token embedding (shared with lm_head via weight tying → counted once)
    embed = vsize * cfg.n_embd

    per_layer_total = 0
    per_layer_active = 0

    # ── Attention ──────────────────────────────────────────────────────────
    q_dim  = cfg.n_embd
    kv_dim = cfg.n_kv_head * (cfg.n_embd // cfg.n_head)
    # Depthwise conv1d (kernel_size=3, groups=n_embd → n_embd params per layer)
    l_conv = cfg.n_embd * 3      # kernel_size=3
    c_attn = cfg.n_embd * (q_dim + 2 * kv_dim)
    c_proj = cfg.n_embd * cfg.n_embd
    per_layer_total  += l_conv + c_attn + c_proj
    per_layer_active += l_conv + c_attn + c_proj  # all attention weights active

    # RMSNorm ×2 per block
    per_layer_total  += 2 * cfg.n_embd
    per_layer_active += 2 * cfg.n_embd

    # ── MoE ───────────────────────────────────────────────────────────────
    # Router
    router = cfg.n_embd * cfg.n_experts
    # Each expert: SwiGLU w_v (n_embd → 2*ffn_dim) + c_proj (ffn_dim → n_embd)
    expert_params = cfg.n_embd * 2 * cfg.ffn_dim + cfg.ffn_dim * cfg.n_embd
    total_expert_params = cfg.n_experts * expert_params

    per_layer_total += router + total_expert_params
    # Only n_active_experts are used per token
    active_expert_params = cfg.n_active_experts * expert_params
    per_layer_active += router + active_expert_params  # router always runs

    total_per_all_layers  = per_layer_total  * cfg.n_layer
    active_per_all_layers = per_layer_active * cfg.n_layer

    # Final RMSNorm
    final_norm = cfg.n_embd
    total_per_all_layers  += final_norm
    active_per_all_layers += final_norm

    total_params  = embed + total_per_all_layers
    active_params = embed + active_per_all_layers  # embedding always active

    return total_params, active_params


def flops_per_token(active_params: int) -> float:
    """
    Approximate FLOPs for a *single token* forward (and backward = ×3 total).
    Standard approximation: FLOPs ≈ 6 × active_params per training step token.
    """
    return 6.0 * active_params


def required_training_steps(total_params: int, batch_size_tokens: int, tokens_per_param: float) -> int:
    """Chinchilla-optimal number of gradient steps."""
    total_tokens = total_params * tokens_per_param
    return math.ceil(total_tokens / batch_size_tokens)


def training_time_seconds(active_params: int, total_tokens: float, node_tflops: float, mfu: float) -> float:
    """
    Estimate wall-clock seconds.
    total_flops = 6 * active_params * total_tokens
    effective_tflops = node_tflops * mfu
    time = total_flops / (effective_tflops * 1e12)
    """
    total_flops = 6.0 * active_params * total_tokens
    effective_tflops = node_tflops * mfu * 1e12
    return total_flops / effective_tflops


def vram_estimate_gb(cfg: LLMConfig, total_params: int, vocab_size: int,
                     micro_batch: int, seq_len: int,
                     dtype_bytes: int, optimizer_overhead: float) -> float:
    """
    Very rough VRAM estimate (GB) for one training node.

    Includes:
      - Model weights (BF16)
      - Optimizer states (overhead multiplier on top of weight bytes)
      - Activations during forward pass (rough estimate)
    """
    # Model weights in BF16
    weights_bytes = total_params * dtype_bytes

    # Optimizer states: FP32 copy of weights + Adam/Muon second moments etc.
    # optimizer_overhead is a multiplier on the *number of params* → gives total bytes
    optimizer_bytes = total_params * optimizer_overhead  # bytes (FP32 states dominate)

    # Activations: rough heuristic — n_layer × micro_batch × seq_len × n_embd × 4 bytes (FP32)
    act_bytes = cfg.n_layer * micro_batch * seq_len * cfg.n_embd * 4

    total_bytes = weights_bytes + optimizer_bytes + act_bytes
    return total_bytes / (1024 ** 3)   # bytes → GB


def fmt_time(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f} hr"
    else:
        return f"{seconds / 86400:.2f} days"


# ===========================================================================
# Main computation
# ===========================================================================

def main():
    rows = []

    print(f"\n{'GPU Node':<20} {'Depth':>5} {'Total Params':>14} {'Active Params':>14} "
          f"{'Steps':>8} {'Time':>12} {'Cost/hr':>9} {'Total Cost':>12} {'Free VRAM (GB)':>15} {'Status':>10}")
    print("-" * 120)

    for node_name, node_info in COMPUTE_RESOURCES.items():
        node_tflops    = node_info["tflops"]
        node_vram_gb   = node_info["vram_gb"]
        cost_per_hour  = node_info["cost_per_hour"]

        for depth in DEPTHS:
            cfg = LLMConfig(depth=depth, vocab_size=padded_vocab(VOCAB_SIZE))

            total_params, active_params = count_parameters(cfg, VOCAB_SIZE)

            steps = required_training_steps(total_params, BATCH_SIZE_TOKENS, TOKENS_PER_PARAM)
            total_tokens = total_params * TOKENS_PER_PARAM

            t_sec = training_time_seconds(active_params, total_tokens, node_tflops, MFU)
            total_cost = (t_sec / 3600.0) * cost_per_hour

            est_vram = vram_estimate_gb(cfg, total_params, VOCAB_SIZE,
                                        MICRO_BATCH_SIZE, SEQ_LEN,
                                        DTYPE_BYTES, OPTIMIZER_OVERHEAD)
            free_vram = node_vram_gb - est_vram

            if free_vram < 0:
                status = "OOM"
            elif free_vram < node_vram_gb * 0.10:
                status = "Tight (<10%)"
            elif free_vram < node_vram_gb * 0.20:
                status = "Tight (<20%)"
            else:
                status = "OK"

            print(f"{node_name:<20} {depth:>5} {total_params:>14,} {active_params:>14,} "
                  f"{steps:>8,} {fmt_time(t_sec):>12} ${cost_per_hour:>8.2f} "
                  f"${total_cost:>11.2f} {free_vram:>14.1f} {status:>10}")

            rows.append({
                "gpu_node":                node_name,
                "tflops":                  node_tflops,
                "depth":                   depth,
                "total_parameters":        total_params,
                "active_parameters":       active_params,
                "required_training_steps": steps,
                "required_training_time":  fmt_time(t_sec),
                "gpu_node_cost":           cost_per_hour,
                "total_training_cost":     round(total_cost, 2),
                "free_vram_gb":            round(free_vram, 2),
                "vram_status":             status,
            })

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    csv_path = "results.csv"
    fieldnames = [
        "gpu_node", "tflops", "depth", "total_parameters", "active_parameters",
        "required_training_steps", "required_training_time",
        "gpu_node_cost", "total_training_cost", "free_vram_gb", "vram_status",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Saved {csv_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    node_names = list(COMPUTE_RESOURCES.keys())
    colors = cm.tab10(np.linspace(0, 0.9, len(node_names)))

    def get_series(node_name, key):
        return [float(r[key]) for r in rows if r["gpu_node"] == node_name]

    def get_time_series(node_name):
        """Return training time in hours for plotting."""
        times = []
        for node_n, node_info in COMPUTE_RESOURCES.items():
            if node_n != node_name:
                continue
            for depth in DEPTHS:
                cfg = LLMConfig(depth=depth, vocab_size=padded_vocab(VOCAB_SIZE))
                _, active_params = count_parameters(cfg, VOCAB_SIZE)
                total_tokens = active_params / TOKENS_PER_PARAM * TOKENS_PER_PARAM  # same logic
                # recalculate to get seconds
                total_params_loc, _ = count_parameters(cfg, VOCAB_SIZE)
                total_tokens_loc = total_params_loc * TOKENS_PER_PARAM
                t_sec = training_time_seconds(active_params, total_tokens_loc, node_info["tflops"], MFU)
                times.append(t_sec / 3600.0)  # hours
        return times

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Feasibility Analysis", fontsize=16, fontweight="bold")

    # — Plot 1: Training time (hours) by depth per node ——————————————————
    ax = axes[0, 0]
    for i, node_name in enumerate(node_names):
        ax.plot(DEPTHS, get_time_series(node_name), marker="o", label=node_name, color=colors[i])
    ax.set_title("Required Training Time")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Hours")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # — Plot 2: Total training cost by depth per node ————————————————————
    ax = axes[0, 1]
    for i, node_name in enumerate(node_names):
        costs = get_series(node_name, "total_training_cost")
        ax.plot(DEPTHS, costs, marker="o", label=node_name, color=colors[i])
    ax.set_title("Total Training Cost (USD)")
    ax.set_xlabel("Depth")
    ax.set_ylabel("USD")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # — Plot 3: Total vs Active parameters ————————————————————————————————
    ax = axes[1, 0]
    total_p = [count_parameters(LLMConfig(depth=d, vocab_size=padded_vocab(VOCAB_SIZE)), VOCAB_SIZE)[0] / 1e6 for d in DEPTHS]
    active_p = [count_parameters(LLMConfig(depth=d, vocab_size=padded_vocab(VOCAB_SIZE)), VOCAB_SIZE)[1] / 1e6 for d in DEPTHS]
    ax.plot(DEPTHS, total_p,  marker="o", label="Total params", color="steelblue")
    ax.plot(DEPTHS, active_p, marker="s", linestyle="--", label="Active params", color="orange")
    ax.set_title("Parameters by Depth")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Parameters (M)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # — Plot 4: Free VRAM headroom by depth per node ——————————————————————
    ax = axes[1, 1]
    for i, node_name in enumerate(node_names):
        free_vrams = get_series(node_name, "free_vram_gb")
        ax.plot(DEPTHS, free_vrams, marker="o", label=node_name, color=colors[i])
    ax.axhline(0, color="red", linestyle="--", alpha=0.6, label="OOM threshold")
    ax.set_title("Free VRAM During Training (GB)")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Free VRAM (GB)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("✓ Saved results.png")
    plt.show()


if __name__ == "__main__":
    main()
