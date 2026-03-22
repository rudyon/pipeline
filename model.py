import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import tiktoken


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192, base=50000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_active_experts = config.n_active_experts
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_experts)])

    def forward(self, x):
        B, T, C = x.size()
        logits = self.router(x)  # (B, T, n_experts)
        probs = F.softmax(logits, dim=-1)
        weights, indices = probs.topk(self.n_active_experts, dim=-1)  # (B, T, K)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # renormalize

        # load balancing auxiliary loss (Switch Transformer)
        avg_probs = probs.mean(dim=[0, 1])  # (n_experts,)
        aux_loss = self.n_experts * (avg_probs * avg_probs).sum()

        # compute weighted sum of top-2 expert outputs
        x_flat = x.view(B * T, C)  # (B*T, C)
        out = torch.zeros_like(x_flat)
        indices_flat = indices.view(B * T, self.n_active_experts)  # (B*T, K)
        weights_flat = weights.view(B * T, self.n_active_experts)  # (B*T, K)
        for k in range(self.n_active_experts):
            for e in range(self.n_experts):
                mask = indices_flat[:, k] == e
                if mask.any():
                    out[mask] += weights_flat[mask, k : k + 1] * self.experts[e](
                        x_flat[mask]
                    )

        return out.view(B, T, C), aux_loss


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w_v = nn.Linear(input_dim, 2 * output_dim, bias=False)

    def forward(self, x):
        gate, value = self.w_v(x).chunk(2, dim=-1)
        return F.silu(gate) * value


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_head % config.n_kv_head == 0, (
            f"n_head ({config.n_head}) must be divisible by n_kv_head ({config.n_kv_head})"
        )
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_groups = self.n_head // self.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.kernel_size = 3
        self.l_conv = nn.Conv1d(
            config.n_embd,
            config.n_embd,
            kernel_size=self.kernel_size,
            groups=config.n_embd,
            bias=False,
        )
        self.q_dim = config.n_embd
        self.kv_dim = self.n_kv_head * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, self.q_dim + 2 * self.kv_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.GPT_SCALE_INIT = 1
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size)
        self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x):
        B, T, C = x.size()
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.l_conv(x)
        x = x.transpose(1, 2)
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = self.rotary_emb(T, device=x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = torch.repeat_interleave(k, self.n_groups, dim=2)
        v = torch.repeat_interleave(v, self.n_groups, dim=2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swiglu = SwiGLU(config.n_embd, config.ffn_dim)
        self.c_proj = nn.Linear(config.ffn_dim, config.n_embd, bias=False)
        self.c_proj.GPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.swiglu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.moe = MoE(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        moe_out, aux_loss = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux_loss


@dataclass
class LLMConfig:
    depth: int = 12
    block_size: int = 1024
    vocab_size: int = 50257
    n_experts: int = 8
    n_active_experts: int = 2

    @property
    def n_layer(self):
        return self.depth

    @property
    def n_head(self):
        return self.depth

    @property
    def n_embd(self):
        return self.depth * 64

    @property
    def n_kv_head(self):
        if self.depth % 3 == 0:
            return self.depth // 3
        else:
            return self.depth // 2

    @property
    def ffn_dim(self):
        raw = int(8 / 3 * self.n_embd)
        return (raw + 63) // 64 * 64  # round up to multiple of 64


class LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}."
        )
        tok_emb = self.transformer.wte(idx)
        x = tok_emb
        aux_loss = torch.tensor(0.0, device=idx.device)
        for block in self.transformer.h:
            x, block_aux = block(x)
            aux_loss = aux_loss + block_aux
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            loss = (
                ce_loss + 0.01 * aux_loss / self.config.n_layer
                if self.training
                else ce_loss
            )
        return logits, loss

    def generate(self, prompt, max_new_tokens=20, top_k=50, temperature=1.0, enc=None):
        if enc is None:
            enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(prompt)
        x = (
            torch.tensor(tokens, dtype=torch.long)
            .unsqueeze(0)
            .to(next(self.parameters()).device)
        )
        self.eval()
        with torch.no_grad():
            while x.size(1) < len(tokens) + max_new_tokens:
                logits, _ = self(x)
                logits = logits[:, -1, :] / max(temperature, 0.00001)
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat((x, xcol), dim=1)
        return enc.decode(x[0].tolist())

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        muon_params = [
            p
            for n, p in param_dict.items()
            if p.ndim == 2 and "wte" not in n and "lm_head" not in n
        ]
        muon_set = set(muon_params)
        adamw_params = [p for p in param_dict.values() if p not in muon_set]
        opt1 = torch.optim.Muon(
            muon_params, lr=learning_rate * 10, momentum=0.95, nesterov=True
        )
        opt2 = torch.optim.AdamW(
            adamw_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=("cuda" in device),
        )
        return [opt1, opt2]
