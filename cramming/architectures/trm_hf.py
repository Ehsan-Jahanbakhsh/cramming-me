"""TRM-style masked language model.

This module adapts Samsung SAIL Montreal's Tiny Recursive Model (TRM) core to
the local Cramming/Hugging Face style model interface.  The important TRM
mechanics are preserved:

* a single shared reasoning network updates both the latent state ``z_L`` and
  the answer state ``z_H``;
* ``H_cycles - 1`` high-level cycles run under ``torch.no_grad()``;
* the final high-level cycle is fully differentiable;
* optional deep supervision repeats the same refinement process while carrying
  detached ``z_H``/``z_L`` states between refinement steps.

Unlike the upstream puzzle code, this wrapper receives ordinary MLM
``input_ids``/``labels`` and returns objects compatible with Cramming's training
engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import ModelOutput


IGNORE_LABEL_ID = -100


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """JAX/Flax-style truncated normal initialization."""

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
            return tensor

        sqrt2 = math.sqrt(2)
        a = math.erf(lower / sqrt2)
        b = math.erf(upper / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * lower**2)
        pdf_l = c * math.exp(-0.5 * upper**2)
        comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

        tensor.uniform_(a, b)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        tensor.clip_(lower * comp_std, upper * comp_std)
        return tensor


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def stablemax(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    sx = stablemax(x)
    return torch.log(sx / torch.sum(sx, dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = labels != ignore_index

    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
    token_losses = -torch.gather(logprobs, index=safe_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)
    token_losses = torch.where(valid_mask, token_losses, torch.zeros_like(token_losses))
    return token_losses.sum() / valid_mask.sum().clamp_min(1)


def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if valid_mask is None:
        valid_count = (labels != ignore_index).sum().clamp_min(1)
    else:
        valid_count = valid_mask.sum().clamp_min(1)

    loss = F.cross_entropy(
        logits.to(torch.float32).reshape(-1, logits.shape[-1]),
        labels.to(torch.long).reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    return loss / valid_count


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty(out_features, in_features), std=1.0 / math.sqrt(in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias=bias)


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: Optional[torch.dtype]):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(trunc_normal_init_(torch.empty(num_embeddings, embedding_dim), std=init_std))

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding_weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        weight = self.embedding_weight if self.cast_to is None else self.embedding_weight.to(self.cast_to)
        return F.embedding(input_ids, weight)


def find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, multiple_of: int = 256):
        super().__init__()
        inter = find_multiple(round(expansion * hidden_size * 2 / 3), multiple_of)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(0).unsqueeze(2)) + (rotate_half(q) * sin.unsqueeze(0).unsqueeze(2))
    k_embed = (k * cos.unsqueeze(0).unsqueeze(2)) + (rotate_half(k) * sin.unsqueeze(0).unsqueeze(2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            raise ValueError(f"Sequence length {seq_len} exceeds RoPE cache length {self.cos_cached.shape[0]}.")
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = CastedLinear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)

        if cos_sin is not None:
            cos, sin = cos_sin
            cos = cos.to(device=hidden_states.device)
            sin = sin.to(device=hidden_states.device)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query.to(torch.float32), key.to(torch.float32).transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            key_mask = attention_mask.to(device=scores.device, dtype=torch.bool).view(batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1).to(value.dtype)
        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.o_proj(out)

        if attention_mask is not None:
            out = out * attention_mask.to(device=out.device, dtype=out.dtype).unsqueeze(-1)
        return out


class TRMBlock(nn.Module):
    def __init__(self, config: "TRMConfig"):
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        if config.mlp_t:
            self.sequence_mlp = SwiGLU(config.max_position_embeddings + config.prefix_len, config.expansion)
            self.self_attn = None
            self.mlp = None
        else:
            self.self_attn = Attention(config.hidden_size, config.num_attention_heads)
            self.mlp = SwiGLU(config.hidden_size, config.expansion)
            self.sequence_mlp = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.mlp_t:
            expected_len = self.config.max_position_embeddings + self.config.prefix_len
            if hidden_states.shape[1] != expected_len:
                raise ValueError(f"mlp_t requires fixed sequence length {expected_len}, got {hidden_states.shape[1]}.")
            x = hidden_states.transpose(1, 2)
            out = self.sequence_mlp(x)
            hidden_states = rms_norm(x + out, variance_epsilon=self.norm_eps).transpose(1, 2)
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
            return hidden_states

        hidden_states = rms_norm(
            hidden_states + self.self_attn(hidden_states=hidden_states, cos_sin=cos_sin, attention_mask=attention_mask),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        return hidden_states


class TRMReasoningModule(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, cos_sin=cos_sin, attention_mask=attention_mask)
        return hidden_states


@dataclass
class TRMInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TRMActState:
    carry: TRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass
class TRMModelOutput(ModelOutput):
    last_hidden_state: torch.Tensor = None
    halt_state: Optional[torch.Tensor] = None
    carry: Optional[TRMInnerCarry] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class TRMMaskedLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    q_halt_logits: Optional[torch.Tensor] = None
    q_continue_logits: Optional[torch.Tensor] = None
    carry: Optional[TRMInnerCarry] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class TRMConfig(PretrainedConfig):
    model_type = "trm"

    def __init__(
        self,
        vocab_size: int = 50000,
        max_position_embeddings: int = 512,
        seq_len: Optional[int] = None,
        hidden_size: int = 512,
        num_attention_heads: int = 8,
        num_heads: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        L_layers: Optional[int] = None,
        H_layers: int = 0,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
        hi_cycles: Optional[int] = None,
        lo_cycles: Optional[int] = None,
        expansion: float = 4.0,
        pos_encodings: str = "rope",
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        prefix_len: int = 0,
        mlp_t: bool = False,
        forward_dtype: str = "bfloat16",
        halt_max_steps: Optional[int] = None,
        deep_supervision_steps: Optional[int] = None,
        inference_steps: Optional[int] = None,
        act_training: bool = False,
        halt_exploration_prob: float = 0.0,
        no_ACT_continue: bool = True,
        q_halt_loss_weight: float = 0.5,
        loss_type: str = "stablemax_cross_entropy",
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        classifier_dropout: float = 0.1,
        pooler_type: str = "cls",
        use_cache: bool = False,
        puzzle_emb_ndim: int = 0,
        num_puzzle_identifiers: int = 1,
        puzzle_emb_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            **kwargs,
        )

        if seq_len is not None:
            max_position_embeddings = seq_len
        if num_heads is not None:
            num_attention_heads = num_heads

        resolved_layers = L_layers if L_layers is not None else num_hidden_layers
        if resolved_layers is None:
            resolved_layers = 2

        resolved_H_cycles = H_cycles if H_cycles is not None else hi_cycles
        resolved_L_cycles = L_cycles if L_cycles is not None else lo_cycles
        if resolved_H_cycles is None:
            resolved_H_cycles = 3
        if resolved_L_cycles is None:
            resolved_L_cycles = 6

        if deep_supervision_steps is None:
            deep_supervision_steps = halt_max_steps if halt_max_steps is not None else 1
        if halt_max_steps is None:
            halt_max_steps = deep_supervision_steps
        if inference_steps is None:
            inference_steps = deep_supervision_steps

        self.vocab_size = int(vocab_size)
        self.max_position_embeddings = int(max_position_embeddings)
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_hidden_layers = int(resolved_layers)
        self.L_layers = int(resolved_layers)
        self.H_layers = int(H_layers)
        self.H_cycles = int(resolved_H_cycles)
        self.L_cycles = int(resolved_L_cycles)
        self.hi_cycles = int(resolved_H_cycles)
        self.lo_cycles = int(resolved_L_cycles)
        self.expansion = float(expansion)
        self.pos_encodings = str(pos_encodings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.rope_theta = float(rope_theta)
        self.prefix_len = int(prefix_len)
        self.mlp_t = bool(mlp_t)
        self.forward_dtype = str(forward_dtype)
        self.halt_max_steps = int(halt_max_steps)
        self.deep_supervision_steps = int(deep_supervision_steps)
        self.inference_steps = int(inference_steps)
        self.act_training = bool(act_training)
        self.halt_exploration_prob = float(halt_exploration_prob)
        self.no_ACT_continue = bool(no_ACT_continue)
        self.q_halt_loss_weight = float(q_halt_loss_weight)
        self.loss_type = str(loss_type)
        self.classifier_dropout = float(classifier_dropout)
        self.pooler_type = str(pooler_type)
        self.seq_len = self.max_position_embeddings
        self.num_heads = self.num_attention_heads
        self.puzzle_emb_ndim = int(puzzle_emb_ndim)
        self.num_puzzle_identifiers = int(num_puzzle_identifiers)
        self.puzzle_emb_len = int(puzzle_emb_len or 0)
        self.batch_size = batch_size
        self.is_decoder = False
        self.tie_word_embeddings = False


class TRMPreTrainedModel(PreTrainedModel):
    config_class = TRMConfig
    base_model_prefix = "trm"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class TRMModel(TRMPreTrainedModel):
    def __init__(self, config: TRMConfig):
        super().__init__(config)
        self.config = config
        self.forward_dtype = None if config.forward_dtype in ("", "auto", "none") else getattr(torch, config.forward_dtype)
        embed_init_std = 1.0 / math.sqrt(config.hidden_size)

        self.embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        if config.prefix_len > 0:
            self.prefix = nn.Parameter(trunc_normal_init_(torch.empty(config.prefix_len, config.hidden_size), std=embed_init_std))
        else:
            self.prefix = None

        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings + config.prefix_len,
                base=config.rope_theta,
            )
            self.embed_pos = None
        elif config.pos_encodings == "learned":
            self.rotary_emb = None
            self.embed_pos = CastedEmbedding(
                config.max_position_embeddings + config.prefix_len,
                config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        elif config.pos_encodings in ("none", "None", ""):
            self.rotary_emb = None
            self.embed_pos = None
        else:
            raise ValueError(f"Unsupported pos_encodings={config.pos_encodings!r}.")

        self.L_level = TRMReasoningModule(nn.ModuleList([TRMBlock(config) for _ in range(config.L_layers)]))
        init_dtype = self.forward_dtype if self.forward_dtype is not None else torch.float32
        self.register_buffer("H_init", trunc_normal_init_(torch.empty(config.hidden_size, dtype=init_dtype), std=1), persistent=True)
        self.register_buffer("L_init", trunc_normal_init_(torch.empty(config.hidden_size, dtype=init_dtype), std=1), persistent=True)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, CastedEmbedding):
            raise TypeError("TRMModel expects a CastedEmbedding.")
        self.embed_tokens = value

    def _input_embeddings(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")

        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds max_position_embeddings={self.config.max_position_embeddings}.")

        embedding = self.embed_tokens(input_ids.to(torch.long))

        if self.prefix is not None:
            prefix = self.prefix.to(dtype=embedding.dtype, device=embedding.device).unsqueeze(0).expand(bsz, -1, -1)
            embedding = torch.cat((prefix, embedding), dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(bsz, self.config.prefix_len, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((prefix_mask, attention_mask), dim=1)

        total_len = embedding.shape[1]
        if self.embed_pos is not None:
            pos_ids = torch.arange(total_len, device=input_ids.device, dtype=torch.long)
            pos = self.embed_pos(pos_ids).unsqueeze(0)
            embedding = 0.7071067811865476 * (embedding + pos)

        if attention_mask is None:
            attention_mask = torch.ones(bsz, total_len, dtype=torch.long, device=input_ids.device)

        return self.embed_scale * embedding, attention_mask

    def empty_carry(self, batch_size: int, total_len: int, device: torch.device, dtype: torch.dtype) -> TRMInnerCarry:
        z_H = self.H_init.to(device=device, dtype=dtype).view(1, 1, -1).expand(batch_size, total_len, -1).contiguous()
        z_L = self.L_init.to(device=device, dtype=dtype).view(1, 1, -1).expand(batch_size, total_len, -1).contiguous()
        return TRMInnerCarry(z_H=z_H, z_L=z_L)

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry) -> TRMInnerCarry:
        reset = reset_flag.to(device=carry.z_H.device, dtype=torch.bool).view(-1, 1, 1)
        z_H_init = self.H_init.to(device=carry.z_H.device, dtype=carry.z_H.dtype).view(1, 1, -1)
        z_L_init = self.L_init.to(device=carry.z_L.device, dtype=carry.z_L.dtype).view(1, 1, -1)
        return TRMInnerCarry(
            z_H=torch.where(reset, z_H_init, carry.z_H),
            z_L=torch.where(reset, z_L_init, carry.z_L),
        )

    def refine_once(
        self,
        carry: TRMInnerCarry,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[TRMInnerCarry, torch.Tensor, torch.Tensor]:
        seq_len = input_embeddings.shape[1]
        cos_sin = self.rotary_emb(seq_len) if self.rotary_emb is not None else None
        if cos_sin is not None:
            cos_sin = tuple(x.to(device=input_embeddings.device) for x in cos_sin)

        z_H, z_L = carry.z_H, carry.z_L

        with torch.no_grad():
            for _ in range(max(0, self.config.H_cycles - 1)):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin, attention_mask=attention_mask)
                z_H = self.L_level(z_H, z_L, cos_sin=cos_sin, attention_mask=attention_mask)

        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin, attention_mask=attention_mask)
        z_H = self.L_level(z_H, z_L, cos_sin=cos_sin, attention_mask=attention_mask)

        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        halt_state = z_H[:, 0]
        return new_carry, z_H, halt_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        carry: Optional[TRMInnerCarry] = None,
        num_steps: Optional[int] = None,
        return_carry: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[TRMModelOutput, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        input_embeddings, attention_mask = self._input_embeddings(input_ids, attention_mask)
        if carry is None:
            carry = self.empty_carry(
                batch_size=input_ids.shape[0],
                total_len=input_embeddings.shape[1],
                device=input_embeddings.device,
                dtype=input_embeddings.dtype,
            )

        steps = max(1, int(num_steps if num_steps is not None else self.config.inference_steps))
        hidden_history = [] if output_hidden_states else None
        halt_state = None
        z_H = carry.z_H
        for _ in range(steps):
            carry, z_H, step_halt_state = self.refine_once(carry, input_embeddings, attention_mask)
            halt_state = step_halt_state
            if hidden_history is not None:
                hidden_history.append(z_H[:, self.config.prefix_len :])

        last_hidden_state = z_H[:, self.config.prefix_len :]
        if not return_dict:
            out = (last_hidden_state, halt_state)
            return out + (carry,) if return_carry else out

        return TRMModelOutput(
            last_hidden_state=last_hidden_state,
            halt_state=halt_state,
            carry=carry if return_carry else None,
            hidden_states=tuple(hidden_history) if hidden_history is not None else None,
        )


class TRMForMaskedLM(TRMPreTrainedModel):
    def __init__(self, config: TRMConfig):
        super().__init__(config)
        self.trm = TRMModel(config)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        self._act_state: Optional[TRMActState] = None
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.trm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.trm.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.lm_head = value

    def _token_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.config.loss_type == "stablemax_cross_entropy":
            logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
            return -torch.gather(logprobs, index=labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)
        if self.config.loss_type in ("cross_entropy", "softmax_cross_entropy"):
            return F.cross_entropy(logits.to(torch.float32), labels.to(torch.long), reduction="none")
        raise ValueError(f"Unsupported TRM MLM loss_type={self.config.loss_type!r}.")

    def _lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid_mask = labels != IGNORE_LABEL_ID
        if not valid_mask.any():
            return logits.sum() * 0.0

        flat_logits = logits[valid_mask]
        flat_labels = labels[valid_mask]
        token_loss = self._token_loss(flat_logits, flat_labels)
        seq_ids = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(labels)[valid_mask]
        per_seq_loss = token_loss.new_zeros((labels.shape[0],))
        per_seq_loss.scatter_add_(0, seq_ids, token_loss)
        loss_counts = valid_mask.sum(dim=-1).clamp_min(1).to(token_loss.dtype)
        return (per_seq_loss / loss_counts).mean()

    def _seq_is_correct(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid_mask = labels != IGNORE_LABEL_ID
        loss_counts = valid_mask.sum(dim=-1)
        safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
        preds = logits.argmax(dim=-1)
        is_correct = valid_mask & (preds == safe_labels)
        return (is_correct.sum(dim=-1) == loss_counts) & (loss_counts > 0)

    def _split_q_logits(self, halt_state: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_logits = self.q_head(halt_state).to(torch.float32)
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1] if q_logits.shape[-1] > 1 else None
        return q_halt_logits, q_continue_logits

    def _q_binary_loss(self, q_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(q_logits, targets.to(q_logits.dtype), reduction="mean")

    def reset_act_state(self) -> None:
        self._act_state = None

    def _act_state_matches(self, input_ids: torch.Tensor, labels: torch.Tensor) -> bool:
        state = self._act_state
        return (
            state is not None
            and state.input_ids.shape == input_ids.shape
            and state.labels.shape == labels.shape
            and state.input_ids.device == input_ids.device
        )

    def _forward_act_training(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        return_carry: bool,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> Union[TRMMaskedLMOutput, Tuple[torch.Tensor]]:
        raw_attention_mask = attention_mask
        if raw_attention_mask is None:
            raw_attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        if not self._act_state_matches(input_ids, labels):
            self._act_state = TRMActState(
                carry=self.trm.empty_carry(
                    batch_size=input_ids.shape[0],
                    total_len=input_ids.shape[1] + self.config.prefix_len,
                    device=input_ids.device,
                    dtype=self.trm.H_init.dtype,
                ),
                steps=torch.zeros((input_ids.shape[0],), dtype=torch.int32, device=input_ids.device),
                halted=torch.ones((input_ids.shape[0],), dtype=torch.bool, device=input_ids.device),
                input_ids=torch.empty_like(input_ids),
                attention_mask=torch.empty_like(raw_attention_mask),
                labels=torch.empty_like(labels),
            )

        state = self._act_state
        reset_flag = state.halted.to(device=input_ids.device, dtype=torch.bool)
        reset_data = reset_flag.view(-1, 1)

        current_input_ids = torch.where(reset_data, input_ids, state.input_ids.to(input_ids.device))
        current_attention_mask = torch.where(reset_data, raw_attention_mask, state.attention_mask.to(input_ids.device))
        current_labels = torch.where(reset_data, labels, state.labels.to(input_ids.device))

        input_embeddings, trm_attention_mask = self.trm._input_embeddings(current_input_ids, current_attention_mask)
        carry = TRMInnerCarry(
            z_H=state.carry.z_H.to(device=input_embeddings.device, dtype=input_embeddings.dtype),
            z_L=state.carry.z_L.to(device=input_embeddings.device, dtype=input_embeddings.dtype),
        )
        carry = self.trm.reset_carry(reset_flag, carry)
        steps = torch.where(reset_flag, torch.zeros_like(state.steps), state.steps).to(device=input_ids.device)

        new_carry, z_H, q_halt_hidden = self.trm.refine_once(carry, input_embeddings, trm_attention_mask)
        token_hidden = z_H[:, self.config.prefix_len :]
        logits = self.lm_head(token_hidden)
        q_halt_logits, q_continue_logits = self._split_q_logits(q_halt_hidden)

        with torch.no_grad():
            seq_is_correct = self._seq_is_correct(logits, current_labels)

        lm_loss = self._lm_loss(logits, current_labels)
        q_halt_loss = self._q_binary_loss(q_halt_logits, seq_is_correct)
        loss = lm_loss + self.config.q_halt_loss_weight * q_halt_loss

        with torch.no_grad():
            new_steps = steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step
            if self.config.halt_max_steps > 1:
                if self.config.no_ACT_continue or q_continue_logits is None:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                if self.config.halt_exploration_prob > 0:
                    explore = torch.rand_like(q_halt_logits, dtype=torch.float32) < self.config.halt_exploration_prob
                    min_halt_steps = torch.where(
                        explore,
                        torch.randint(
                            low=2,
                            high=self.config.halt_max_steps + 1,
                            size=new_steps.shape,
                            device=new_steps.device,
                            dtype=new_steps.dtype,
                        ),
                        torch.zeros_like(new_steps),
                    )
                    halted = halted & (new_steps >= min_halt_steps)

        if not self.config.no_ACT_continue and q_continue_logits is not None:
            with torch.no_grad():
                _, _next_z_H, next_q_halt_hidden = self.trm.refine_once(new_carry, input_embeddings, trm_attention_mask)
                next_q_halt_logits, next_q_continue_logits = self._split_q_logits(next_q_halt_hidden)
                next_q_value = next_q_halt_logits
                if next_q_continue_logits is not None:
                    next_q_value = torch.maximum(next_q_halt_logits, next_q_continue_logits)
                target_q_continue = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, next_q_value))
            loss = loss + self.config.q_halt_loss_weight * self._q_binary_loss(q_continue_logits, target_q_continue)

        self._act_state = TRMActState(
            carry=new_carry,
            steps=new_steps.detach(),
            halted=halted.detach(),
            input_ids=current_input_ids.detach(),
            attention_mask=current_attention_mask.detach(),
            labels=current_labels.detach(),
        )

        hidden_history = (token_hidden,) if output_hidden_states else None

        if not return_dict:
            out = (logits, q_halt_logits, q_continue_logits)
            if return_carry:
                out = out + (new_carry,)
            return (loss,) + out

        return TRMMaskedLMOutput(
            loss=loss,
            logits=logits,
            q_halt_logits=q_halt_logits,
            q_continue_logits=q_continue_logits,
            carry=new_carry if return_carry else None,
            hidden_states=hidden_history,
            attentions=None,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        carry: Optional[TRMInnerCarry] = None,
        return_carry: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[TRMMaskedLMOutput, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(f"attention_mask must match input_ids shape, got {tuple(attention_mask.shape)}.")
            if labels is not None:
                labels = labels.masked_fill(attention_mask == 0, IGNORE_LABEL_ID)

        if self.training and labels is not None and self.config.act_training:
            return self._forward_act_training(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_carry=return_carry,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if not self.training:
            self.reset_act_state()

        input_embeddings, trm_attention_mask = self.trm._input_embeddings(input_ids, attention_mask)
        if carry is None:
            carry = self.trm.empty_carry(
                batch_size=input_ids.shape[0],
                total_len=input_embeddings.shape[1],
                device=input_embeddings.device,
                dtype=input_embeddings.dtype,
            )

        steps = self.config.deep_supervision_steps if self.training and labels is not None else self.config.inference_steps
        steps = max(1, int(steps))

        losses = []
        hidden_history = [] if output_hidden_states else None
        logits = None
        q_halt_logits = None
        q_continue_logits = None

        for _ in range(steps):
            carry, z_H, q_halt_hidden = self.trm.refine_once(carry, input_embeddings, trm_attention_mask)
            token_hidden = z_H[:, self.config.prefix_len :]
            logits = self.lm_head(token_hidden)
            q_halt_logits, q_continue_logits = self._split_q_logits(q_halt_hidden)

            if hidden_history is not None:
                hidden_history.append(token_hidden)

            if labels is not None:
                step_loss = self._lm_loss(logits, labels)
                if self.config.q_halt_loss_weight > 0:
                    with torch.no_grad():
                        seq_is_correct = self._seq_is_correct(logits, labels)
                    step_loss = step_loss + self.config.q_halt_loss_weight * self._q_binary_loss(q_halt_logits, seq_is_correct)
                losses.append(step_loss)

        loss = torch.stack(losses).mean() if losses else None

        if not return_dict:
            out = (logits, q_halt_logits, q_continue_logits)
            if return_carry:
                out = out + (carry,)
            return ((loss,) + out) if loss is not None else out

        return TRMMaskedLMOutput(
            loss=loss,
            logits=logits,
            q_halt_logits=q_halt_logits,
            q_continue_logits=q_continue_logits,
            carry=carry if return_carry else None,
            hidden_states=tuple(hidden_history) if hidden_history is not None else None,
            attentions=None,
        )


class TRMForSequenceClassification(TRMPreTrainedModel):
    def __init__(self, config: TRMConfig):
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.trm = TRMModel(config)
        self.dropout = nn.Dropout(float(getattr(config, "classifier_dropout", 0.1)))
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.problem_type = None
        self.post_init()

    def _pool(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        pooler_type = str(getattr(self.config, "pooler_type", "cls")).lower()
        if pooler_type == "mean":
            if attention_mask is None:
                return hidden.mean(dim=1)
            mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return hidden[:, 0]

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.problem_type is None:
            if self.num_labels == 1:
                self.problem_type = "regression"
            elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                self.problem_type = "single_label_classification"
            else:
                self.problem_type = "multi_label_classification"

        if self.problem_type == "regression":
            return F.mse_loss(logits.squeeze(), labels.squeeze())
        if self.problem_type == "single_label_classification":
            return F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return F.binary_cross_entropy_with_logits(logits, labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeddings, trm_attention_mask = self.trm._input_embeddings(input_ids, attention_mask)
        carry = self.trm.empty_carry(
            batch_size=input_ids.shape[0],
            total_len=input_embeddings.shape[1],
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
        )

        steps = self.config.deep_supervision_steps if self.training and labels is not None else self.config.inference_steps
        steps = max(1, int(steps))
        losses = []
        logits = None

        for _ in range(steps):
            carry, z_H, _halt_state = self.trm.refine_once(carry, input_embeddings, trm_attention_mask)
            hidden = z_H[:, self.config.prefix_len :]
            pooled = self._pool(hidden, attention_mask)
            logits = self.classifier(self.dropout(pooled))
            if labels is not None:
                losses.append(self._loss(logits, labels))

        if losses:
            loss = torch.stack(losses).mean()
        else:
            loss = logits.new_zeros((1,))

        if not return_dict:
            return (loss, logits)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)


# Keep the user's requested spelling available as an alias.
TRMforMaskedLM = TRMForMaskedLM
