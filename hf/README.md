---
language:
- en
license: mit
tags:
- text-generation
- pytorch
- moe
- gqa
- rope
datasets:
- HuggingFaceFW/fineweb-edu
- mlfoundations/dclm-baseline-1.0
pipeline_tag: text-generation
---

# linnet-497M

A 497M parameter Mixture of Experts language model with 8 experts and 2 active experts per token and 157M active parameters. Trained from scratch using [rudyon/pipeline](https://github.com/rudyon/pipeline) on the [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) and [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) datasets.

Training was done on a single H100 GPU rented on [Prime Intellect](https://www.primeintellect.ai/) for about $17.

## architecture

The model is a 12-layer causal transformer with the following modifications:

| Component | Implementation |
|-----------|---------------|
| Positional encoding | RoPE (base=50000) |
| Attention | GQA + QK Norm + FlashAttention |
| FFN | SwiGLU (8/3 x n_embd hidden dim) |
| Normalization | RMSNorm |
| Sequence mixing | Causal depthwise Conv1d (kernel=3) |
| Sparsity | MoE (8 experts, top-2) |
| Optimizer | Muon + AdamW |

## training

- **Datasets**: HuggingFaceFW/fineweb-edu (~700k docs) + mlfoundations/dclm-baseline-1.0 (~250k docs)
- **Tokenizer**: Custom ByteLevelBPE (vocab size: 32768)
- **Batch size**: 524,288 tokens
- **Sequence length**: 1024

## usage

Download `model.py` from the repository alongside the weights, then:

```python
import torch
from tokenizers import Tokenizer
from model import LLM, LLMConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_pretrained("rudyon/linnet-497M")
model = LLM(LLMConfig(depth=12))
state_dict = torch.load("pytorch_model.bin", map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(model.generate("Hello!", enc=tokenizer))
```