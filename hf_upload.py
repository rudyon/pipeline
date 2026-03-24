"""
hf_upload.py — Upload a trained model + tokenizer to HuggingFace Hub.

Usage:
    HF_TOKEN=hf_... python hf_upload.py --model model_best.pt --tokenizer tokenizer.json

Required env var:
    HF_TOKEN  — your HuggingFace write token

Required args:
    --model      path to the .pt checkpoint (e.g. model_best.pt)
    --tokenizer  path to the tokenizer.json file

The script will:
  1. Load the checkpoint and extract the model state dict + config
  2. Auto-generate hf/config.json from the actual checkpoint metadata
  3. Save pytorch_model.bin (state dict only, no optimizer / training state)
  4. Upload all files in hf/ plus pytorch_model.bin and tokenizer.json to the Hub
"""

import argparse
import json
import os
import sys
import shutil
import tempfile
import importlib.util

import torch


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
parser.add_argument("--repo", default=None,
                    help="HuggingFace repo id, e.g. rudyon/linnet. "
                         "Defaults to the username from the token + '/linnet'.")
parser.add_argument("--hf-dir", default="hf",
                    help="Directory with supplementary HF files (README.md, config.json, etc.)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# huggingface_hub import
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import HfApi, whoami
except ImportError:
    print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# Resolve repo id
if args.repo:
    repo_id = args.repo
else:
    try:
        user_info = whoami(token=HF_TOKEN)
        username = user_info["name"]
    except Exception as e:
        print(f"ERROR: Could not resolve HF username: {e}", file=sys.stderr)
        sys.exit(1)
    repo_id = f"{username}/linnet-497M"

print(f"Target repo: {repo_id}")

# ---------------------------------------------------------------------------
# Load model checkpoint
# ---------------------------------------------------------------------------
print(f"Loading checkpoint: {args.model}")
checkpoint = torch.load(args.model, map_location="cpu")

# Extract just the model weights (strip optimizer, step, etc.)
if "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    # Assume the file is already a raw state dict
    state_dict = checkpoint

# Try to read depth / config from checkpoint if it was saved
saved_config = checkpoint.get("config", None)

# ---------------------------------------------------------------------------
# Dynamically import LLMConfig from model.py to get architecture metadata
# ---------------------------------------------------------------------------
_model_path = os.path.join(os.path.dirname(__file__), "model.py")
_spec = importlib.util.spec_from_file_location("model", _model_path)
_model_module = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_model_module)
    LLMConfig = _model_module.LLMConfig
except Exception as e:
    print(f"WARNING: Could not import LLMConfig from model.py: {e}")
    LLMConfig = None

# ---------------------------------------------------------------------------
# Work in a temp dir — we build the full upload payload there
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:

    # 1. Copy everything from hf/ into tmpdir
    hf_dir = os.path.join(os.path.dirname(__file__), args.hf_dir)
    if not os.path.isdir(hf_dir):
        print(f"ERROR: HF files directory not found: {hf_dir}", file=sys.stderr)
        sys.exit(1)

    for fname in os.listdir(hf_dir):
        src = os.path.join(hf_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tmpdir, fname))
            print(f"  copied {fname}")

    # 2. Auto-generate / overwrite config.json from the real checkpoint
    if LLMConfig is not None:
        # Try to figure out depth from state dict shape
        # The embedding table is wte.weight with shape [vocab_size, n_embd]
        # n_embd = depth * 64, so depth = n_embd // 64
        depth = None
        if "transformer.wte.weight" in state_dict:
            n_embd = state_dict["transformer.wte.weight"].shape[1]
            depth = n_embd // 64
        elif saved_config is not None and hasattr(saved_config, "depth"):
            depth = saved_config.depth

        if depth is not None:
            cfg = LLMConfig(depth=depth)
            config_data = {
                "model_type": "linnet",
                "architectures": ["LLM"],
                "depth": cfg.depth,
                "n_layer": cfg.n_layer,
                "n_head": cfg.n_head,
                "n_kv_head": cfg.n_kv_head,
                "n_embd": cfg.n_embd,
                "ffn_dim": cfg.ffn_dim,
                "block_size": cfg.block_size,
                "vocab_size": cfg.vocab_size,
                "n_experts": cfg.n_experts,
                "n_active_experts": cfg.n_active_experts,
                "rope_base": 50000,
                "torch_dtype": "bfloat16",
                "pipeline_tag": "text-generation",
            }
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            print(f"  generated config.json (depth={depth}, n_embd={cfg.n_embd})")
        else:
            print("  WARNING: could not determine depth from checkpoint; using hf/config.json as-is")

    # 3. Save pytorch_model.bin (weights only)
    bin_path = os.path.join(tmpdir, "pytorch_model.bin")
    torch.save(state_dict, bin_path)
    print(f"  saved pytorch_model.bin ({os.path.getsize(bin_path) / 1e6:.1f} MB)")

    # 4. Copy tokenizer.json
    tok_dst = os.path.join(tmpdir, "tokenizer.json")
    shutil.copy2(args.tokenizer, tok_dst)
    print(f"  copied tokenizer.json")

    # model.py is already included via the hf/ directory copy above (hf/model.py)

    # ---------------------------------------------------------------------------
    # Create repo if it doesn't exist, then upload
    # ---------------------------------------------------------------------------
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        print(f"Repo '{repo_id}' ready.")
    except Exception as e:
        print(f"ERROR: Could not create/access repo: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nUploading all files to {repo_id} ...")
    api.upload_folder(
        folder_path=tmpdir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload model checkpoint and tokenizer",
    )
    print(f"\nDone! Model is live at: https://huggingface.co/{repo_id}")
