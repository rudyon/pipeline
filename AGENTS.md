# AGENTS.md - LLM Training Pipeline

## Project Overview

This is a PyTorch-based LLM training pipeline for training Large Language Models from scratch.
It supports distributed training via DDP (Distributed Data Parallel), Weights & Biases logging,
and includes MoE (Mixture of Experts) architecture, rotary embeddings, and SwiGLU activations.

Key files:
- `train.py` - Main training script
- `model.py` - LLM architecture (Transformer, MoE, RMSNorm, etc.)
- `util.py` - DataLoaderLite and utilities
- `hellaswag.py` - Evaluation on HellaSwag benchmark
- `prepare_data.py` - Tokenize and shard datasets

## Development Commands

### Package Management
```bash
# Install dependencies (uses uv - Astral's fast Python package manager)
uv venv && uv pip install -r requirements.txt

# Or with torch on CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Running Training
```bash
# Single GPU / CPU test run (50 steps)
python train.py 50 -d 1 -b 1024 -m 1 -s 256 --cache test_cache

# Full training run (DDP on multiple GPUs)
NUM_GPUS=8 torchrun --standalone --nproc_per_node=$NUM_GPUS train.py 300 --depth 12 --batch 524288 --cache data_cache

# Architecture experiment on Kaggle
torchrun train.py 300 --depth 4 --batch 32768 --micro 4 --cache test_cache --experiment <name>
```

### Linting and Type Checking
```bash
# Install ruff (if not present) and lint all Python files
uv pip install ruff
ruff check .

# Format code with ruff (following existing style)
ruff format .

# Type checking with mypy (if installed)
uv pip install mypy
mypy . --ignore-missing-imports
```

### Running Tests
There is no formal test suite. Test changes manually:
```bash
# Quick smoke test (no GPU needed)
python train.py 10 -d 1 -b 64 -m 1 -s 128 --cache test_cache

# Validate model forward pass
python -c "from model import LLM, LLMConfig; m = LLM(LLMConfig(depth=1)); print(m(torch.zeros(1, 32, dtype=torch.long), torch.zeros(1, 32, dtype=torch.long))[0].shape)"
```

## Code Style Guidelines

### Imports
Order imports as follows with blank lines between groups:
```python
import os           # Standard library
import time
import math

import torch                    # Third-party packages
import numpy as np
from torch.distributed import init_process_group

from model import LLM, LLMConfig  # Local imports
from util import DataLoaderLite
```

### Formatting
- Use 4 spaces for indentation (not tabs)
- Maximum line length: 100 characters (soft guideline, prefer readability)
- Use blank lines sparingly to group related code (2 blank lines between top-level definitions)
- No trailing whitespace
- Use implicit string concatenation where possible: `"foo" "bar"` → `"foobar"`

### Type Hints
- Use type hints for function parameters and return values when types aren't obvious:
  ```python
  def get_lr(it: int) -> float:
      ...
  def configure_optimizers(self, weight_decay: float, learning_rate: float, device: str):
      ...
  ```
- Use `torch.Tensor` for tensor types, not `torch.Tensor` aliases
- Primitive types: `int`, `float`, `str`, `bool`

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `LLM`, `RotaryEmbedding`, `DataLoaderLite` |
| Functions/methods | snake_case | `apply_rotary_pos_emb`, `get_lr`, `next_batch` |
| Constants | SCREAMING_SNAKE_CASE | `MAX_SEQ_LEN`, `GPT_SCALE_INIT` |
| Variables | snake_case | `ddp_rank`, `val_loss_accum`, `grad_accum_steps` |
| Dataclass fields | snake_case | `n_experts`, `n_active_experts`, `block_size` |
| Module-level vars | snake_case | `parser`, `train_loader` |

### Dataclasses for Configuration
Use `@dataclass` for configuration objects:
```python
@dataclass
class LLMConfig:
    depth: int = 12
    block_size: int = 1024
    vocab_size: int = 50257
    n_experts: int = 8
    n_active_experts: int = 2
```

### Error Handling
- Use assertions for invariants and preconditions:
  ```python
  assert total_batch_size % (B * T * ddp_world_size) == 0
  assert T <= self.config.block_size, f"Cannot forward sequence of length {T}..."
  ```
- Use `if master_process:` guards for logging/output that should only run once in DDP
- Handle file I/O with proper error messages:
  ```python
  checkpoint = torch.load(args.resume, map_location=device)
  ```

### PyTorch Conventions
- Always call `super().__init__()` in `nn.Module` subclasses
- Use `torch.no_grad()` for inference:
  ```python
  with torch.no_grad():
      x, y = val_loader.next_batch()
  ```
- Use `torch.autocast` for mixed precision:
  ```python
  with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      logits, loss = model(x, y)
  ```
- Use `torch.nn.utils.clip_grad_norm_` to prevent gradient explosion
- Register buffers with `self.register_buffer("name", tensor)`

### DDP (Distributed Data Parallel)
- Check `RANK` environment variable to determine if running under DDP:
  ```python
  ddp = int(os.environ.get("RANK", -1)) != -1
  ```
- Use `find_unused_parameters=True` in `DDP` constructor if some params may not be used in forward
- Only log/save on `master_process` (rank 0)
- Use `dist.all_reduce` with `dist.ReduceOp.AVG` for averaging metrics across processes

### Performance Notes
- Use `torch.compile()` for model when running on CUDA
- Set `torch.set_float32_matmul_precision("high")` for better performance
- Use `bfloat16` for autocast on newer GPUs
- Use `torch.cuda.synchronize()` before timing if needed

## Architecture Overview

The LLM uses a Transformer architecture with:
- **Rotary Position Embeddings** (RoPE) via `RotaryEmbedding`
- **Grouped Query Attention** (GQA) - separate q and kv heads
- **Causal Conv1D** as local attention mechanism
- **SwiGLU** activation in MLP
- **MoE** (Mixture of Experts) with load balancing auxiliary loss
- **RMSNorm** instead of LayerNorm
- **Weight tying** between `wte` embedding and `lm_head`

## Common Pitfalls

1. **Device mismatch**: Always call `.to(device)` on tensors after moving between CPU/GPU
2. **DDP model access**: Access underlying model via `model.module` when wrapped in DDP
3. **Gradient sync**: Set `model.require_backward_grad_sync = False` for intermediate microsteps
4. **Shard loading**: Ensure `data_root` contains properly sharded `{name}_{split}_{index}` files
5. **Memory**: Lower batch size if OOM; `B * T` is per-GPU micro batch size
