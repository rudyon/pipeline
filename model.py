import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F


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

        # sparse dispatch: route tokens to experts without running unused experts
        x_flat = x.view(B * T, C)  # (N, C)
        indices_flat = indices.view(B * T * self.n_active_experts)  # (N*K,)
        weights_flat = weights.view(B * T * self.n_active_experts, 1)  # (N*K, 1)

        # repeat each token K times (once per active expert slot)
        x_repeated = x_flat.repeat_interleave(self.n_active_experts, dim=0)  # (N*K, C)

        # sort by expert index so each expert processes a contiguous batch
        sort_idx = indices_flat.argsort()
        x_sorted = x_repeated[sort_idx]  # (N*K, C)
        experts_sorted = indices_flat[sort_idx]  # (N*K,)

        # count how many tokens each expert gets
        counts = experts_sorted.bincount(minlength=self.n_experts).tolist()

        # run each expert on its batch
        out_sorted = torch.empty_like(x_sorted)
        start = 0
        for e, count in enumerate(counts):
            if count > 0:
                out_sorted[start : start + count] = self.experts[e](
                    x_sorted[start : start + count]
                )
            start += count

        # unsort and accumulate weighted outputs
        out_repeated = torch.empty_like(x_sorted)
        out_repeated[sort_idx] = out_sorted
        out = (
            (out_repeated * weights_flat)
            .view(B * T, self.n_active_experts, C)
            .sum(dim=1)
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
    probe_every: int = 4  # insert a confidence probe after every N layers

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
            return max(1, self.depth // 3)
        else:
            return max(1, self.depth // 2)

    @property
    def ffn_dim(self):
        raw = int(8 / 3 * self.n_embd)
        return (raw + 63) // 64 * 64  # round up to multiple of 64


class ConfidenceProbe(nn.Module):
    """Lightweight MLP probe on the residual stream → scalar confidence in [0, 1].
    Trained to predict whether the model's top-1 next-token prediction is correct."""

    def __init__(self, n_embd: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, C) → (B, T)
        return self.net(x).squeeze(-1)


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

        # Confidence probes: one after every probe_every layers
        # e.g. depth=12, probe_every=4 → probes after layers 4, 8, 12
        probe_positions = list(range(config.probe_every - 1, config.n_layer, config.probe_every))
        self.probe_positions = probe_positions
        self.probes = nn.ModuleList(
            [ConfidenceProbe(config.n_embd) for _ in probe_positions]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, return_probe_scores=False):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}."
        )
        tok_emb = self.transformer.wte(idx)
        x = tok_emb
        aux_loss = torch.tensor(0.0, device=idx.device)

        # Collect probe outputs during the forward pass
        probe_outputs = []  # list of (B, T) tensors
        probe_idx = 0
        for layer_idx, block in enumerate(self.transformer.h):
            x, block_aux = block(x)
            aux_loss = aux_loss + block_aux
            if probe_idx < len(self.probe_positions) and layer_idx == self.probe_positions[probe_idx]:
                probe_outputs.append(self.probes[probe_idx](x.detach()))  # detach: probe doesn't affect backbone grad
                probe_idx += 1

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        probe_aux_loss_val = torch.tensor(0.0, device=idx.device)
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )

            # Compute probe auxiliary loss: BCE against top-1 correctness
            if probe_outputs:
                with torch.no_grad():
                    top1 = logits.argmax(dim=-1)  # (B, T)
                    correct = (top1 == targets).float()  # (B, T)

                probe_bce = torch.tensor(0.0, device=idx.device)
                for p_scores in probe_outputs:
                    probe_bce = probe_bce + F.binary_cross_entropy(
                        p_scores, correct, reduction="mean"
                    )
                probe_aux_loss_val = probe_bce / len(probe_outputs)

            moe_aux = 0.01 * aux_loss / self.config.n_layer if self.training else torch.tensor(0.0, device=idx.device)
            probe_aux = 0.01 * probe_aux_loss_val if self.training else torch.tensor(0.0, device=idx.device)
            loss = ce_loss + moe_aux + probe_aux

        if return_probe_scores:
            # Average across all probes → (B, T) mean confidence
            if probe_outputs:
                avg_probe = torch.stack(probe_outputs, dim=0).mean(dim=0)
            else:
                avg_probe = torch.zeros(B, T, device=idx.device)
            return logits, loss, avg_probe, probe_aux_loss_val

        return logits, loss

    def generate(self, prompt, max_new_tokens=20, top_k=50, temperature=1.0, enc=None):
        assert enc is not None, "A tokenizer must be provided to generate()"
        tokens = enc.encode(prompt).ids
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
