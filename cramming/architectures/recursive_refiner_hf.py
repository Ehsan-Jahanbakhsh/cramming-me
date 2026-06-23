"""
recursive_refiner_hf.py

A Hugging Face Transformers-compatible implementation of the "Recursive Refiner" model
from your provided code, packaged as:

- RecursiveRefinerConfig (PretrainedConfig)
- RecursiveRefinerModel (PreTrainedModel base model; returns last_hidden_state)
- RecursiveRefinerForMaskedLM (MLM; bidirectional attention)
- RecursiveRefinerForCausalLM (autoregressive; causal attention)

Key differences vs your original file:
- Adds an optional causal attention mask when config.is_causal=True.
- Adds HF-standard forward signatures and returns ModelOutput objects with loss/logits.
- Implements vocab resizing for the factorized embedding (token factor matrix only).

This file is intended to be imported directly, e.g.:

    from recursive_refiner_hf import (
        RecursiveRefinerConfig,
        RecursiveRefinerForMaskedLM,
        RecursiveRefinerForCausalLM,
    )

    cfg = RecursiveRefinerConfig(vocab_size=30522, max_position_embeddings=512)
    model = RecursiveRefinerForMaskedLM(cfg)

For causal LM training with Trainer:
- Use DataCollatorForLanguageModeling(tokenizer, mlm=False)
- Pass labels (Trainer does automatically) and the model shifts internally.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    CausalLMOutputWithPast,
    SequenceClassifierOutput,
)


# ============================================================
# Normalization + FFN
# ============================================================

class RootMeanSquareNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU FFN:
      a,b = Linear(x) split in half
      out = Linear( silu(a) * b )
    """
    def __init__(self, dim: int, expansion: float = 4.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.fc1 = nn.Linear(dim, 2 * hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)


# ============================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Cache cos/sin tables up to a maximum, and extend on demand.
    """
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even.")
        self.dim = dim
        self.base = float(base)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, Hd/2]
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: [B, H, T, Hd]
    cos/sin: [T, Hd/2]
    """
    b, h, t, hd = x.shape
    x_ = x.view(b, h, t, hd // 2, 2)
    x1 = x_[..., 0]
    x2 = x_[..., 1]
    # rotate
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).view(b, h, t, hd)
    return out


# ============================================================
# Factorized Embedding with tied logits
# ============================================================

class LowRankEmbedding(nn.Module):
    """Factorized embedding equivalent to a full VxD matrix.

    We parameterize the implicit embedding matrix as:
        W_full = W_tok (V x r) @ W_mid (r x r) @ W_out (r x D)
    where r = D // embed_factor.

    If embed_factor == 1, use a normal full V x D embedding with tied logits.

    Embedding lookup for embed_factor > 1:
        emb(ids) = W_tok[ids] @ W_mid @ W_out    -> [B, T, D]
    Embedding lookup for embed_factor == 1:
        emb(ids) = W_tok[ids]                    -> [B, T, D]

    Tied logits (no materialization of W_full):
        logits(x) = x @ W_full^T
    """

    def __init__(self, vocab_size: int, dim: int, n: int = 4, init_std: float = 0.02):
        super().__init__()
        if n < 1:
            raise ValueError("embed_factor must be >= 1")
        if dim % n != 0:
            raise ValueError(f"dim ({dim}) must be divisible by embed_factor ({n})")
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.embed_factor = int(n)

        r = dim // n
        self.rank = r

        self.tok = nn.Embedding(self.vocab_size, r)
        self.use_mid = n > 1
        if self.use_mid:
            self.mid = nn.Parameter(torch.empty(r, r))
            self.out = nn.Parameter(torch.empty(r, dim))
        else:
            self.register_parameter("mid", None)
            self.register_parameter("out", None)

        # init
        nn.init.normal_(self.tok.weight, mean=0.0, std=init_std)
        if self.use_mid:
            nn.init.normal_(self.mid, mean=0.0, std=init_std)
            nn.init.normal_(self.out, mean=0.0, std=init_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok(input_ids)  # [B, T, r]
        if self.use_mid:
            x = x @ self.mid      # [B, T, r]
            x = x @ self.out      # [B, T, D]
        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, T, V]
        if not self.use_mid:
            return F.linear(x, self.tok.weight)
        h = x @ self.out.t()               # [B, T, r]
        h = h @ self.mid.t()               # [B, T, r]
        return h @ self.tok.weight.t()     # [B, T, V]

    def resize_vocab(self, new_vocab_size: int) -> None:
        """
        Resize only the token factor matrix W_tok (V x r).
        W_mid and W_out are unaffected.
        """
        new_vocab_size = int(new_vocab_size)
        if new_vocab_size == self.vocab_size:
            return
        if new_vocab_size <= 0:
            raise ValueError("new_vocab_size must be > 0")

        old_weight = self.tok.weight.data
        new_tok = nn.Embedding(new_vocab_size, self.rank, device=old_weight.device, dtype=old_weight.dtype)
        nn.init.normal_(new_tok.weight, mean=0.0, std=old_weight.std().item() if old_weight.numel() > 1 else 0.02)

        n_copy = min(self.vocab_size, new_vocab_size)
        new_tok.weight.data[:n_copy].copy_(old_weight[:n_copy])
        self.tok = new_tok
        self.vocab_size = new_vocab_size


# ============================================================
# Attention (bidirectional or causal)
# ============================================================

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope: Optional[RotaryPositionalEmbedding] = None, is_causal: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = rope
        self.is_causal = bool(is_causal)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, T, D]
        attention_mask (optional): [B, T] with 1 for valid tokens and 0 for padding.
        """
        b, t, d = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Hd]
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope(t)
            cos = cos.to(device=x.device, dtype=x.dtype)
            sin = sin.to(device=x.device, dtype=x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]

        # Causal mask: prevent attending to future positions.
        if self.is_causal:
            causal = torch.tril(torch.ones((t, t), device=att.device, dtype=torch.bool))
            att = att.masked_fill(~causal.view(1, 1, t, t), torch.finfo(att.dtype).min)

        if attention_mask is not None:
            # Mask out *keys* that correspond to padding.
            key_mask = attention_mask.to(dtype=torch.bool, device=att.device).view(b, 1, 1, t)
            att = att.masked_fill(~key_mask, torch.finfo(att.dtype).min)

        att = F.softmax(att, dim=-1)
        y = torch.matmul(att, v)  # [B, H, T, Hd]
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        y = self.out(y)

        if attention_mask is not None:
            # Zero out padded query positions (optional but keeps activations bounded).
            y = y * attention_mask.to(dtype=y.dtype, device=y.device).unsqueeze(-1)

        return y


# ============================================================
# Core recursive blocks
# ============================================================

class RecursiveReasoningBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion: float,
        eps: float,
        rope: Optional[RotaryPositionalEmbedding],
        pre_norm: bool,
        is_causal: bool,
    ):
        super().__init__()
        self.pre_norm = bool(pre_norm)
        self.attn = SelfAttention(dim=dim, num_heads=num_heads, rope=rope, is_causal=is_causal)
        self.ffn = SwiGLUFeedForward(dim=dim, expansion=expansion)

        self.norm1 = RootMeanSquareNorm(dim, eps=eps)
        self.norm2 = RootMeanSquareNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x, attention_mask=attention_mask))
            x = self.norm2(x + self.ffn(x))
        return x


class SharedReasoningStack(nn.Module):
    """
    Shared update operator: inject conditioning, then apply blocks.
    """
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, z: torch.Tensor, inject: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = z + inject
        for blk in self.blocks:
            z = blk(z, attention_mask=attention_mask)
        return z


# ============================================================
# Hugging Face config + models
# ============================================================

class RecursiveRefinerConfig(PretrainedConfig):
    model_type = "recursive_refiner"

    def __init__(
        self,
        vocab_size: int = 50000,
        max_position_embeddings: int = 512,
        hidden_size: int = 256,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 2,
        expansion: float = 4.0,
        hi_cycles: int = 3,
        lo_cycles: int = 2,
        embed_factor: int = 4,
        pre_norm: bool = True,
        rope_theta: float = 10000.0,
        rms_eps: float = 1e-5,
        prefix_len: int = 0,
        is_causal: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        use_cache: bool = False,
        # Downstream / fine-tuning helpers.
        classifier_dropout: float = 0.1,
        pooler_type: str = "cls",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            **kwargs,
        )
        self.vocab_size = int(vocab_size)
        self.max_position_embeddings = int(max_position_embeddings)
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_hidden_layers = int(num_hidden_layers)
        self.expansion = float(expansion)
        self.hi_cycles = int(hi_cycles)
        self.lo_cycles = int(lo_cycles)
        self.embed_factor = int(embed_factor)
        self.pre_norm = bool(pre_norm)
        self.rope_theta = float(rope_theta)
        self.rms_eps = float(rms_eps)
        self.prefix_len = int(prefix_len)
        self.is_causal = bool(is_causal)

        # Fine-tuning helpers.
        self.classifier_dropout = float(classifier_dropout)
        self.pooler_type = str(pooler_type)

        # HF flags: useful for generation utilities.
        # For causal LM heads we will set is_decoder=True externally.
        if "is_decoder" not in kwargs:
            self.is_decoder = bool(is_causal)
        if "tie_word_embeddings" not in kwargs:
            # Our logits are tied by construction (via LowRankEmbedding.logits)
            self.tie_word_embeddings = True


class RecursiveRefinerPreTrainedModel(PreTrainedModel):
    config_class = RecursiveRefinerConfig
    base_model_prefix = "recursive_refiner"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        # Conservative init compatible with most Transformer-style training.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class RecursiveRefinerModel(RecursiveRefinerPreTrainedModel):
    """
    Base model: returns last_hidden_state (z_out) over token positions (prefix removed).
    """

    def __init__(self, config: RecursiveRefinerConfig):
        super().__init__(config)
        cfg = config

        self.embed = LowRankEmbedding(cfg.vocab_size, cfg.hidden_size, n=cfg.embed_factor)

        # Precompute RoPE cache up to maximum runtime length (prefix included).
        max_len = cfg.max_position_embeddings + cfg.prefix_len
        rope = RotaryPositionalEmbedding(dim=cfg.hidden_size // cfg.num_attention_heads, base=cfg.rope_theta, max_seq_len=max_len)

        blocks = nn.ModuleList([
            RecursiveReasoningBlock(
                dim=cfg.hidden_size,
                num_heads=cfg.num_attention_heads,
                expansion=cfg.expansion,
                eps=cfg.rms_eps,
                rope=rope,
                pre_norm=cfg.pre_norm,
                is_causal=cfg.is_causal,
            )
            for _ in range(cfg.num_hidden_layers)
        ])
        self.shared = SharedReasoningStack(blocks)

        self.hi_init = nn.Parameter(torch.randn(cfg.hidden_size) * 0.02)
        self.lo_init = nn.Parameter(torch.randn(cfg.hidden_size) * 0.02)

        if cfg.prefix_len > 0:
            self.prefix = nn.Parameter(torch.randn(cfg.prefix_len, cfg.hidden_size) * 0.02)
        else:
            self.prefix = None

        self.post_init()

    def init_latents(self, batch_size: int, total_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        z_hi = self.hi_init.to(device=device, dtype=dtype).view(1, 1, -1).expand(batch_size, total_len, -1).contiguous()
        z_lo = self.lo_init.to(device=device, dtype=dtype).view(1, 1, -1).expand(batch_size, total_len, -1).contiguous()
        return z_hi, z_lo


    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, LowRankEmbedding):
            raise TypeError("RecursiveRefinerModel expects a LowRankEmbedding.")
        self.embed = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
        b, l = input_ids.shape

        if attention_mask is not None and attention_mask.shape != (b, l):
            raise ValueError(f"attention_mask must have shape {(b, l)}, got {tuple(attention_mask.shape)}")

        x = self.embed(input_ids)  # [B, L, D]

        if self.config.prefix_len > 0:
            prefix = self.prefix.unsqueeze(0).expand(b, -1, -1)  # [B, P, D]
            x = torch.cat([prefix, x], dim=1)  # [B, P+L, D]
            if attention_mask is not None:
                prefix_mask = torch.ones(b, self.config.prefix_len, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, P+L]

        total_len = x.shape[1]
        z_hi, z_lo = self.init_latents(b, total_len=total_len, device=x.device, dtype=x.dtype)

        hi_cycles = int(max(1, self.config.hi_cycles))
        lo_cycles = int(max(1, self.config.lo_cycles))

        for _ in range(hi_cycles):
            for _ in range(lo_cycles):
                z_lo = self.shared(z_lo, inject=(z_hi + x), attention_mask=attention_mask)
            z_hi = self.shared(z_hi, inject=z_lo, attention_mask=attention_mask)

        z_final = z_hi
        z_out = z_final[:, self.config.prefix_len:]  # [B, L, D] (prefix removed)

        if not return_dict:
            return (z_out,)
        return BaseModelOutput(last_hidden_state=z_out, hidden_states=None, attentions=None)


class RecursiveRefinerSingleHighModel(RecursiveRefinerModel):
    """
    Recursive Refiner variant with a single high-level vector per batch item.

    ``z_hi`` is kept as [B, D] internally, while ``z_lo`` remains token-shaped
    as [B, T, D]. Token-level outputs are formed by adding the final high vector
    back to every low-level token state.
    """

    def init_latents(self, batch_size: int, total_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        z_hi = self.hi_init.to(device=device, dtype=dtype).view(1, -1).expand(batch_size, -1).contiguous()
        z_lo = self.lo_init.to(device=device, dtype=dtype).view(1, 1, -1).expand(batch_size, total_len, -1).contiguous()
        return z_hi, z_lo

    def _pool_low_state(self, z_lo: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return z_lo.mean(dim=1)

        mask = attention_mask.to(dtype=z_lo.dtype, device=z_lo.device).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (z_lo * mask).sum(dim=1) / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
        b, l = input_ids.shape

        if attention_mask is not None and attention_mask.shape != (b, l):
            raise ValueError(f"attention_mask must have shape {(b, l)}, got {tuple(attention_mask.shape)}")

        x = self.embed(input_ids)  # [B, L, D]

        if self.config.prefix_len > 0:
            prefix = self.prefix.unsqueeze(0).expand(b, -1, -1)  # [B, P, D]
            x = torch.cat([prefix, x], dim=1)  # [B, P+L, D]
            if attention_mask is not None:
                prefix_mask = torch.ones(b, self.config.prefix_len, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B, P+L]

        total_len = x.shape[1]
        z_hi, z_lo = self.init_latents(b, total_len=total_len, device=x.device, dtype=x.dtype)

        hi_cycles = int(max(1, self.config.hi_cycles))
        lo_cycles = int(max(1, self.config.lo_cycles))

        for _ in range(hi_cycles):
            for _ in range(lo_cycles):
                z_lo = self.shared(z_lo, inject=(z_hi.unsqueeze(1) + x), attention_mask=attention_mask)

            pooled_lo = self._pool_low_state(z_lo, attention_mask)
            z_hi = self.shared(z_hi.unsqueeze(1), inject=pooled_lo.unsqueeze(1), attention_mask=None).squeeze(1)

        z_final = z_lo + z_hi.unsqueeze(1)
        z_out = z_final[:, self.config.prefix_len:]  # [B, L, D] (prefix removed)

        if not return_dict:
            return (z_out,)
        return BaseModelOutput(last_hidden_state=z_out, hidden_states=None, attentions=None)


class RecursiveRefinerForMaskedLM(RecursiveRefinerPreTrainedModel):
    """
    MLM head (bidirectional attention). Use DataCollatorForLanguageModeling(mlm=True).
    """
    model_cls = RecursiveRefinerModel

    def __init__(self, config: RecursiveRefinerConfig):
        # Ensure MLM uses bidirectional attention
        config.is_causal = False
        config.is_decoder = False
        super().__init__(config)
        self.recursive_refiner = self.model_cls(config)
        self.post_init()

    def get_output_embeddings(self) -> Optional[nn.Module]:
        # logits are tied via LowRankEmbedding; there is no separate head module.
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.recursive_refiner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        z_out = outputs.last_hidden_state
        logits = self.recursive_refiner.embed.logits(z_out)

        loss = None
        if labels is not None:
            if attention_mask is not None:
                # Safety: ignore padding positions even if labels were not masked to -100.
                labels = labels.masked_fill(attention_mask == 0, -100)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            out = (logits, z_out)
            return ((loss,) + out) if loss is not None else out

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class RecursiveRefinerForCausalLM(RecursiveRefinerPreTrainedModel):
    """
    Autoregressive (causal) LM head. Use DataCollatorForLanguageModeling(mlm=False).
    The shifting for next-token prediction is done inside forward().
    """
    model_cls = RecursiveRefinerModel

    def __init__(self, config: RecursiveRefinerConfig):
        config.is_causal = True
        config.is_decoder = True
        super().__init__(config)
        self.recursive_refiner = self.model_cls(config)
        self.post_init()

    def get_output_embeddings(self) -> Optional[nn.Module]:
        return None

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # No KV cache implemented; generate() will repeatedly call forward on full sequences.
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.recursive_refiner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        z_out = outputs.last_hidden_state
        logits = self.recursive_refiner.embed.logits(z_out)

        loss = None
        if labels is not None:
            # Ensure padding is ignored.
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)

            # Shift for next-token prediction:
            # Predict token t+1 from positions <= t.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            out = (logits, z_out)
            return ((loss,) + out) if loss is not None else out

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class RecursiveRefinerForSequenceClassification(RecursiveRefinerPreTrainedModel):
    """Sequence classification head for downstream fine-tuning.

    This is intentionally lightweight and compatible with Cramming's training loop,
    which expects the forward pass to return an object supporting ``["loss"]`` and
    ``["logits"]`` access (HF ModelOutput objects satisfy this).

    Pooling:
      * ``pooler_type=\"cls\"`` (default): use first token embedding.
      * ``pooler_type=\"mean\"``: masked mean-pooling over sequence length.
    """
    model_cls = RecursiveRefinerModel

    def __init__(self, config: RecursiveRefinerConfig):
        # Classification is typically done with bidirectional attention, but we do
        # not forcibly override config.is_causal to allow experimentation.
        config.is_decoder = False
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 2))

        self.recursive_refiner = self.model_cls(config)
        p = float(getattr(config, "classifier_dropout", 0.1))
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.problem_type = None
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.recursive_refiner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        z_out = outputs.last_hidden_state  # [B, L, D]

        pooler_type = str(getattr(self.config, "pooler_type", "cls")).lower()
        if pooler_type == "mean":
            if attention_mask is None:
                pooled = z_out.mean(dim=1)
            else:
                mask = attention_mask.to(dtype=z_out.dtype, device=z_out.device).unsqueeze(-1)  # [B, L, 1]
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled = (z_out * mask).sum(dim=1) / denom
        else:  # "cls" or any unknown value
            pooled = z_out[:, 0]

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            # Mirror the widely-used HF/BERT logic.
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:  # multi-label
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            # Cramming expects a tensor-valued loss even when labels are absent.
            loss = logits.new_zeros((1,))

        if not return_dict:
            return (loss, logits)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class RecursiveRefinerSingleHighForMaskedLM(RecursiveRefinerForMaskedLM):
    """Masked-LM head using the single-vector high-level latent variant."""

    model_cls = RecursiveRefinerSingleHighModel


class RecursiveRefinerSingleHighForCausalLM(RecursiveRefinerForCausalLM):
    """Causal-LM head using the single-vector high-level latent variant."""

    model_cls = RecursiveRefinerSingleHighModel


class RecursiveRefinerSingleHighForSequenceClassification(RecursiveRefinerForSequenceClassification):
    """Sequence classification head using the single-vector high-level latent variant."""

    model_cls = RecursiveRefinerSingleHighModel
