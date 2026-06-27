"""Early-exit Recursive Refiner models.

This module intentionally lives next to, rather than inside,
``recursive_refiner_hf.py`` so the existing Recursive Refiner implementation and
configs remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from .recursive_refiner_hf import (
    LowRankEmbedding,
    RecursiveReasoningBlock,
    RotaryPositionalEmbedding,
    SharedReasoningStack,
)


_EARLY_EXIT_DEFAULTS = {
    "enabled": True,
    "max_depth": None,
    "min_depth": 1,
    "halt_threshold": 0.5,
    "aux_loss_weight": 0.5,
    "halt_loss_weight": 1.0,
    "ponder_kl_weight": 0.01,
    "halt_prior_lambda": 0.5,
    "inference_enabled": True,
    "fail_if_all_max": False,
}

_PREDICTION_FEEDBACK_DEFAULTS = {
    "enabled": False,
    "top_k": 16,
    "temperature": 1.0,
    "detach": True,
}


def _plain_dict(value: Optional[Union[Dict[str, Any], Any]]) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {k: v for k, v in value.items()}
    return {}


def _merge_settings(value: Optional[Union[Dict[str, Any], Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(_plain_dict(value))
    return merged


def get_early_exit_settings(config) -> Dict[str, Any]:
    defaults = dict(_EARLY_EXIT_DEFAULTS)
    fallback_depth = int(getattr(config, "hi_cycles", getattr(config, "num_hidden_layers", 1)))
    settings = _merge_settings(getattr(config, "early_exit", None), defaults)
    if settings.get("max_depth") is None:
        settings["max_depth"] = fallback_depth
    settings["max_depth"] = max(1, int(settings["max_depth"]))
    settings["min_depth"] = max(1, min(int(settings["min_depth"]), settings["max_depth"]))
    settings["halt_threshold"] = float(settings["halt_threshold"])
    settings["aux_loss_weight"] = float(settings["aux_loss_weight"])
    settings["halt_loss_weight"] = float(settings["halt_loss_weight"])
    settings["ponder_kl_weight"] = float(settings["ponder_kl_weight"])
    settings["halt_prior_lambda"] = float(settings["halt_prior_lambda"])
    settings["enabled"] = bool(settings["enabled"])
    settings["inference_enabled"] = bool(settings["inference_enabled"])
    settings["fail_if_all_max"] = bool(settings["fail_if_all_max"])
    return settings


def get_prediction_feedback_settings(config) -> Dict[str, Any]:
    settings = _merge_settings(getattr(config, "prediction_feedback", None), _PREDICTION_FEEDBACK_DEFAULTS)
    settings["enabled"] = bool(settings["enabled"])
    settings["top_k"] = max(1, int(settings["top_k"]))
    settings["temperature"] = max(float(settings["temperature"]), 1e-6)
    settings["detach"] = bool(settings["detach"])
    return settings


def pool_sequence(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, pooler_type: str = "mean") -> torch.Tensor:
    pooler_type = str(pooler_type).lower()
    if pooler_type == "cls" or attention_mask is None:
        return hidden_states[:, 0]
    mask = attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hidden_states * mask).sum(dim=1) / denom


def select_exit_depths(halt_logits: torch.Tensor, settings: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    halt_probs = torch.sigmoid(halt_logits)
    max_depth = halt_probs.shape[1]
    min_depth = max(1, min(int(settings["min_depth"]), max_depth))

    triggered = halt_probs >= float(settings["halt_threshold"])
    if min_depth > 1:
        triggered[:, : min_depth - 1] = False
    triggered[:, -1] = True

    exit_indices = triggered.to(dtype=torch.float32).argmax(dim=1)
    exit_depths = exit_indices + 1
    exit_probs = halt_probs.gather(1, exit_indices.view(-1, 1)).squeeze(1)
    return exit_depths, exit_probs, halt_probs


def compute_exit_statistics(exit_depths: Optional[torch.Tensor], max_depth: int) -> Dict[str, torch.Tensor]:
    if exit_depths is None or exit_depths.numel() == 0:
        return {}
    depth_float = exit_depths.to(dtype=torch.float32)
    stats = {
        "exit/avg_depth": depth_float.mean().detach(),
        "exit/max_depth_frac": (exit_depths == int(max_depth)).to(dtype=torch.float32).mean().detach(),
        "exit/effective_depth_ratio": (depth_float.mean() / float(max_depth)).detach(),
    }
    for depth in range(1, int(max_depth) + 1):
        stats[f"exit/depth_{depth}_frac"] = (exit_depths == depth).to(dtype=torch.float32).mean().detach()
    return stats


def halting_distribution(halt_logits: torch.Tensor) -> torch.Tensor:
    halt_probs = torch.sigmoid(halt_logits)
    forced_last = torch.ones_like(halt_probs[:, -1:])
    halt_probs = torch.cat([halt_probs[:, :-1], forced_last], dim=1)
    not_halted = 1.0 - halt_probs
    prefix = torch.cumprod(
        torch.cat([torch.ones_like(not_halted[:, :1]), not_halted[:, :-1]], dim=1),
        dim=1,
    )
    dist = halt_probs * prefix
    return dist / dist.sum(dim=1, keepdim=True).clamp_min(1e-6)


def geometric_prior(max_depth: int, prior_lambda: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    prior_lambda = min(max(float(prior_lambda), 1e-6), 1.0 - 1e-6)
    depths = torch.arange(max_depth, device=device, dtype=dtype)
    prior = prior_lambda * torch.pow(torch.as_tensor(1.0 - prior_lambda, device=device, dtype=dtype), depths)
    prior[-1] = torch.pow(torch.as_tensor(1.0 - prior_lambda, device=device, dtype=dtype), max_depth - 1)
    return prior / prior.sum().clamp_min(1e-6)


def compute_halting_loss(
    per_sample_losses: torch.Tensor,
    halt_logits: Optional[torch.Tensor],
    settings: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if halt_logits is None:
        zero = per_sample_losses.new_zeros(())
        return zero, {}

    dist = halting_distribution(halt_logits)
    weighted_task_loss = (dist * per_sample_losses.detach()).sum(dim=1).mean()

    prior = geometric_prior(dist.shape[1], settings["halt_prior_lambda"], dist.device, dist.dtype)
    kl = (dist * (dist.clamp_min(1e-6).log() - prior.view(1, -1).clamp_min(1e-6).log())).sum(dim=1).mean()
    depths = torch.arange(1, dist.shape[1] + 1, device=dist.device, dtype=dist.dtype)
    expected_depth = (dist * depths.view(1, -1)).sum(dim=1).mean()

    loss = weighted_task_loss + float(settings["ponder_kl_weight"]) * kl
    stats = {
        "exit/halt_weighted_loss": weighted_task_loss.detach(),
        "exit/halt_kl": kl.detach(),
        "exit/expected_depth": expected_depth.detach(),
    }
    return loss, stats


def masked_lm_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    flat_loss = F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100, reduction="none")
    token_loss = flat_loss.view(labels.shape)
    valid = labels.ne(-100).to(dtype=token_loss.dtype)
    per_sample = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    mean_loss = (token_loss * valid).sum() / valid.sum().clamp_min(1.0)
    return mean_loss, per_sample


def sequence_classification_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
    problem_type: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    if problem_type is None:
        if num_labels == 1:
            problem_type = "regression"
        elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            problem_type = "single_label_classification"
        else:
            problem_type = "multi_label_classification"

    if problem_type == "regression":
        per_sample = F.mse_loss(logits.view(logits.shape[0], -1), labels.view(labels.shape[0], -1), reduction="none").mean(dim=1)
    elif problem_type == "single_label_classification":
        per_sample = F.cross_entropy(logits.view(-1, num_labels), labels.view(-1), reduction="none")
    else:
        per_label = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        per_sample = per_label.mean(dim=1)
    return per_sample.mean(), per_sample, problem_type


@dataclass
class EarlyExitModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    pooler_output: Optional[torch.FloatTensor] = None
    exit_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    halt_logits: Optional[torch.FloatTensor] = None
    exit_depths: Optional[torch.LongTensor] = None
    exit_probs: Optional[torch.FloatTensor] = None
    exit_stats: Optional[Dict[str, torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class EarlyExitMaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    exit_depths: Optional[torch.LongTensor] = None
    exit_probs: Optional[torch.FloatTensor] = None
    exit_stats: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class EarlyExitSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    exit_depths: Optional[torch.LongTensor] = None
    exit_probs: Optional[torch.FloatTensor] = None
    exit_stats: Optional[Dict[str, torch.Tensor]] = None


class RecursiveRefinerEarlyExitConfig(PretrainedConfig):
    model_type = "recursive_refiner_early_exit"

    def __init__(
        self,
        vocab_size: int = 50000,
        max_position_embeddings: int = 512,
        hidden_size: int = 256,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 2,
        expansion: float = 4.0,
        hi_cycles: int = 2,
        lo_cycles: int = 3,
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
        classifier_dropout: float = 0.1,
        pooler_type: str = "cls",
        early_exit: Optional[Dict[str, Any]] = None,
        prediction_feedback: Optional[Dict[str, Any]] = None,
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
        self.classifier_dropout = float(classifier_dropout)
        self.pooler_type = str(pooler_type)

        self.early_exit = get_early_exit_settings(self)
        self.early_exit.update(_plain_dict(early_exit))
        if self.early_exit.get("max_depth") is None:
            self.early_exit["max_depth"] = self.hi_cycles
        self.early_exit = get_early_exit_settings(self)

        self.prediction_feedback = _merge_settings(prediction_feedback, _PREDICTION_FEEDBACK_DEFAULTS)

        if "is_decoder" not in kwargs:
            self.is_decoder = bool(is_causal)
        if "tie_word_embeddings" not in kwargs:
            self.tie_word_embeddings = True


class RecursiveRefinerEarlyExitPreTrainedModel(PreTrainedModel):
    config_class = RecursiveRefinerEarlyExitConfig
    base_model_prefix = "recursive_refiner"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class RecursiveRefinerEarlyExitModel(RecursiveRefinerEarlyExitPreTrainedModel):
    def __init__(self, config: RecursiveRefinerEarlyExitConfig):
        super().__init__(config)
        cfg = config

        self.embed = LowRankEmbedding(cfg.vocab_size, cfg.hidden_size, n=cfg.embed_factor)
        max_len = cfg.max_position_embeddings + cfg.prefix_len
        rope = RotaryPositionalEmbedding(dim=cfg.hidden_size // cfg.num_attention_heads, base=cfg.rope_theta, max_seq_len=max_len)

        blocks = nn.ModuleList(
            [
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
            ]
        )
        self.shared = SharedReasoningStack(blocks)

        self.hi_init = nn.Parameter(torch.randn(cfg.hidden_size) * 0.02)
        self.lo_init = nn.Parameter(torch.randn(cfg.hidden_size) * 0.02)
        self.halt_head = nn.Linear(cfg.hidden_size, 1)

        self.feedback_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.feedback_gate = nn.Parameter(torch.zeros(()))

        if cfg.prefix_len > 0:
            self.prefix = nn.Parameter(torch.randn(cfg.prefix_len, cfg.hidden_size) * 0.02)
        else:
            self.prefix = None

        self.post_init()
        self.reset_early_exit_parameters()

    def reset_early_exit_parameters(self) -> None:
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -2.0)

    def max_depth(self) -> int:
        settings = get_early_exit_settings(self.config)
        return int(settings["max_depth"]) if settings["enabled"] else max(1, int(self.config.hi_cycles))

    def init_latents(self, batch_size: int, total_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        z_hi = self.hi_init.to(device=device, dtype=dtype).view(1, 1, -1).expand(batch_size, total_len, -1).contiguous()
        z_lo = self.lo_init.to(device=device, dtype=dtype).view(1, 1, -1).expand(batch_size, total_len, -1).contiguous()
        return z_hi, z_lo

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, LowRankEmbedding):
            raise TypeError("RecursiveRefinerEarlyExitModel expects a LowRankEmbedding.")
        self.embed = value

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], int, int]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
        b, l = input_ids.shape
        if attention_mask is not None and attention_mask.shape != (b, l):
            raise ValueError(f"attention_mask must have shape {(b, l)}, got {tuple(attention_mask.shape)}")

        token_attention_mask = attention_mask
        x = self.embed(input_ids)
        if self.config.prefix_len > 0:
            prefix = self.prefix.unsqueeze(0).expand(b, -1, -1)
            x = torch.cat([prefix, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(b, self.config.prefix_len, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return x, attention_mask, token_attention_mask, b, l

    def _embed_token_indices(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embed.tok(token_ids)
        if self.embed.use_mid:
            emb = emb @ self.embed.mid
            emb = emb @ self.embed.out
        return emb

    def _prediction_feedback(self, z_out: torch.Tensor) -> torch.Tensor:
        feedback_cfg = get_prediction_feedback_settings(self.config)
        if not feedback_cfg["enabled"]:
            return z_out.new_zeros(z_out.shape)

        logits = self.embed.logits(z_out)
        k = min(int(feedback_cfg["top_k"]), logits.shape[-1])
        top_values, top_indices = torch.topk(logits, k=k, dim=-1)
        probs = F.softmax(top_values / float(feedback_cfg["temperature"]), dim=-1)
        token_embeddings = self._embed_token_indices(top_indices)
        feedback = (probs.unsqueeze(-1) * token_embeddings).sum(dim=-2)
        if feedback_cfg["detach"]:
            feedback = feedback.detach()
        return torch.tanh(self.feedback_gate) * self.feedback_proj(feedback)

    def _feedback_with_prefix(self, feedback_tokens: torch.Tensor, total_len: int) -> torch.Tensor:
        if self.config.prefix_len <= 0:
            return feedback_tokens
        prefix_feedback = feedback_tokens.new_zeros(feedback_tokens.shape[0], self.config.prefix_len, feedback_tokens.shape[-1])
        return torch.cat([prefix_feedback, feedback_tokens], dim=1)[:, :total_len]

    def _one_high_cycle(
        self,
        z_hi: torch.Tensor,
        z_lo: torch.Tensor,
        conditioning: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lo_cycles = int(max(1, self.config.lo_cycles))
        for _ in range(lo_cycles):
            z_lo = self.shared(z_lo, inject=(z_hi + conditioning), attention_mask=attention_mask)
        z_hi = self.shared(z_hi, inject=z_lo, attention_mask=attention_mask)
        return z_hi, z_lo

    def _forward_all_depths(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> EarlyExitModelOutput:
        settings = get_early_exit_settings(self.config)
        x, full_attention_mask, token_attention_mask, b, _ = self._prepare_inputs(input_ids, attention_mask)
        total_len = x.shape[1]
        z_hi, z_lo = self.init_latents(b, total_len=total_len, device=x.device, dtype=x.dtype)

        max_depth = self.max_depth()
        feedback = x.new_zeros(x.shape)
        exit_states = []
        halt_logits = []

        for depth in range(max_depth):
            conditioning = x + feedback
            z_hi, z_lo = self._one_high_cycle(z_hi, z_lo, conditioning, full_attention_mask)
            z_out = z_hi[:, self.config.prefix_len :]
            exit_states.append(z_out)

            pooled = pool_sequence(z_out, token_attention_mask, pooler_type="mean")
            halt_logits.append(self.halt_head(pooled).squeeze(-1))

            if depth < max_depth - 1:
                feedback_tokens = self._prediction_feedback(z_out)
                feedback = self._feedback_with_prefix(feedback_tokens, total_len)

        halt_logits_tensor = torch.stack(halt_logits, dim=1)
        exit_depths, exit_probs, _ = select_exit_depths(halt_logits_tensor, settings)
        exit_stats = compute_exit_statistics(exit_depths, max_depth)

        return EarlyExitModelOutput(
            last_hidden_state=exit_states[-1],
            exit_hidden_states=tuple(exit_states),
            halt_logits=halt_logits_tensor,
            exit_depths=exit_depths,
            exit_probs=exit_probs,
            exit_stats=exit_stats,
            hidden_states=None,
            attentions=None,
        )

    @torch.no_grad()
    def _forward_dynamic(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> EarlyExitModelOutput:
        settings = get_early_exit_settings(self.config)
        x, full_attention_mask, token_attention_mask, b, l = self._prepare_inputs(input_ids, attention_mask)
        total_len = x.shape[1]
        max_depth = self.max_depth()

        z_hi, z_lo = self.init_latents(b, total_len=total_len, device=x.device, dtype=x.dtype)
        active_indices = torch.arange(b, device=x.device)
        active_x = x
        active_full_mask = full_attention_mask
        active_token_mask = token_attention_mask
        active_feedback = x.new_zeros(x.shape)

        final_hidden = x.new_zeros(b, l, x.shape[-1])
        exit_depths = torch.full((b,), max_depth, dtype=torch.long, device=x.device)
        exit_probs = x.new_zeros(b)

        for depth in range(1, max_depth + 1):
            conditioning = active_x + active_feedback
            z_hi, z_lo = self._one_high_cycle(z_hi, z_lo, conditioning, active_full_mask)
            z_out = z_hi[:, self.config.prefix_len :]

            pooled = pool_sequence(z_out, active_token_mask, pooler_type="mean")
            probs = torch.sigmoid(self.halt_head(pooled).squeeze(-1))
            should_exit = probs >= float(settings["halt_threshold"])
            if depth < int(settings["min_depth"]):
                should_exit = torch.zeros_like(should_exit, dtype=torch.bool)
            if depth == max_depth:
                should_exit = torch.ones_like(should_exit, dtype=torch.bool)

            if should_exit.any():
                exiting_indices = active_indices[should_exit]
                final_hidden[exiting_indices] = z_out[should_exit]
                exit_depths[exiting_indices] = depth
                exit_probs[exiting_indices] = probs[should_exit]

            keep = ~should_exit
            if not keep.any():
                break

            if depth < max_depth:
                feedback_tokens = self._prediction_feedback(z_out[keep])
                active_feedback = self._feedback_with_prefix(feedback_tokens, total_len)

            active_indices = active_indices[keep]
            active_x = active_x[keep]
            z_hi = z_hi[keep]
            z_lo = z_lo[keep]
            if active_full_mask is not None:
                active_full_mask = active_full_mask[keep]
            if active_token_mask is not None:
                active_token_mask = active_token_mask[keep]

        exit_stats = compute_exit_statistics(exit_depths, max_depth)
        if settings["fail_if_all_max"] and bool((exit_depths == max_depth).all().item()):
            raise RuntimeError("All samples exited at max depth; adjust halting threshold or regularization.")

        return EarlyExitModelOutput(
            last_hidden_state=final_hidden,
            exit_hidden_states=(final_hidden,),
            halt_logits=None,
            exit_depths=exit_depths,
            exit_probs=exit_probs,
            exit_stats=exit_stats,
            hidden_states=None,
            attentions=None,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        force_all_depths: Optional[bool] = None,
        **kwargs,
    ) -> Union[EarlyExitModelOutput, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        settings = get_early_exit_settings(self.config)
        if force_all_depths is None:
            force_all_depths = bool(torch.is_grad_enabled())

        use_dynamic = settings["enabled"] and settings["inference_enabled"] and not force_all_depths
        outputs = self._forward_dynamic(input_ids, attention_mask) if use_dynamic else self._forward_all_depths(input_ids, attention_mask)

        if not return_dict:
            return (outputs.last_hidden_state,)
        return outputs


class RecursiveRefinerEarlyExitForMaskedLM(RecursiveRefinerEarlyExitPreTrainedModel):
    def __init__(self, config: RecursiveRefinerEarlyExitConfig):
        config.is_causal = False
        config.is_decoder = False
        super().__init__(config)
        self.recursive_refiner = RecursiveRefinerEarlyExitModel(config)
        self.post_init()
        self.recursive_refiner.reset_early_exit_parameters()

    def get_output_embeddings(self) -> Optional[nn.Module]:
        return None

    def _loss_from_exit_states(
        self,
        exit_states: Tuple[torch.Tensor, ...],
        halt_logits: Optional[torch.Tensor],
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        settings = get_early_exit_settings(self.config)
        mean_losses = []
        per_sample_losses = []
        final_logits = None

        for state in exit_states:
            logits = self.recursive_refiner.embed.logits(state)
            final_logits = logits
            mean_loss, per_sample_loss = masked_lm_losses(logits, labels, self.config.vocab_size, attention_mask)
            mean_losses.append(mean_loss)
            per_sample_losses.append(per_sample_loss)

        final_loss = mean_losses[-1]
        if len(mean_losses) > 1:
            aux_loss = torch.stack(mean_losses[:-1]).mean()
        else:
            aux_loss = final_loss.new_zeros(())

        loss_matrix = torch.stack(per_sample_losses, dim=1)
        halt_loss, halt_stats = compute_halting_loss(loss_matrix, halt_logits, settings)
        total_loss = final_loss + settings["aux_loss_weight"] * aux_loss + settings["halt_loss_weight"] * halt_loss
        loss_stats = {
            "exit/final_loss": final_loss.detach(),
            "exit/aux_loss": aux_loss.detach(),
            "exit/halt_loss": halt_loss.detach(),
        }
        loss_stats.update(halt_stats)
        return total_loss, final_logits, loss_stats

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[EarlyExitMaskedLMOutput, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        force_all_depths = labels is not None and torch.is_grad_enabled()

        outputs = self.recursive_refiner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            force_all_depths=force_all_depths,
        )

        exit_stats = dict(outputs.exit_stats or {})
        loss = None
        if labels is not None and force_all_depths and outputs.exit_hidden_states is not None:
            loss, logits, loss_stats = self._loss_from_exit_states(outputs.exit_hidden_states, outputs.halt_logits, labels, attention_mask)
            exit_stats.update(loss_stats)
        else:
            logits = self.recursive_refiner.embed.logits(outputs.last_hidden_state)
            if labels is not None:
                loss, _ = masked_lm_losses(logits, labels, self.config.vocab_size, attention_mask)

        if not return_dict:
            out = (logits, outputs.last_hidden_state)
            return ((loss,) + out) if loss is not None else out

        return EarlyExitMaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
            exit_depths=outputs.exit_depths,
            exit_probs=outputs.exit_probs,
            exit_stats=exit_stats,
        )


class RecursiveRefinerEarlyExitForSequenceClassification(RecursiveRefinerEarlyExitPreTrainedModel):
    def __init__(self, config: RecursiveRefinerEarlyExitConfig):
        config.is_decoder = False
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.recursive_refiner = RecursiveRefinerEarlyExitModel(config)
        self.dropout = nn.Dropout(float(getattr(config, "classifier_dropout", 0.1)))
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.problem_type = None
        self.post_init()
        self.recursive_refiner.reset_early_exit_parameters()

    def _classify(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        pooled = pool_sequence(hidden_states, attention_mask, pooler_type=getattr(self.config, "pooler_type", "cls"))
        return self.classifier(self.dropout(pooled))

    def _loss_from_exit_states(
        self,
        exit_states: Tuple[torch.Tensor, ...],
        halt_logits: Optional[torch.Tensor],
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        settings = get_early_exit_settings(self.config)
        mean_losses = []
        per_sample_losses = []
        final_logits = None
        problem_type = self.problem_type

        for state in exit_states:
            logits = self._classify(state, attention_mask)
            final_logits = logits
            mean_loss, per_sample_loss, problem_type = sequence_classification_losses(logits, labels, self.num_labels, problem_type)
            mean_losses.append(mean_loss)
            per_sample_losses.append(per_sample_loss)

        self.problem_type = problem_type
        final_loss = mean_losses[-1]
        aux_loss = torch.stack(mean_losses[:-1]).mean() if len(mean_losses) > 1 else final_loss.new_zeros(())
        halt_loss, halt_stats = compute_halting_loss(torch.stack(per_sample_losses, dim=1), halt_logits, settings)
        total_loss = final_loss + settings["aux_loss_weight"] * aux_loss + settings["halt_loss_weight"] * halt_loss
        loss_stats = {
            "exit/final_loss": final_loss.detach(),
            "exit/aux_loss": aux_loss.detach(),
            "exit/halt_loss": halt_loss.detach(),
        }
        loss_stats.update(halt_stats)
        return total_loss, final_logits, loss_stats

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[EarlyExitSequenceClassifierOutput, Tuple[torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        force_all_depths = labels is not None and torch.is_grad_enabled()

        outputs = self.recursive_refiner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            force_all_depths=force_all_depths,
        )

        exit_stats = dict(outputs.exit_stats or {})
        if labels is not None and force_all_depths and outputs.exit_hidden_states is not None:
            loss, logits, loss_stats = self._loss_from_exit_states(outputs.exit_hidden_states, outputs.halt_logits, labels, attention_mask)
            exit_stats.update(loss_stats)
        else:
            logits = self._classify(outputs.last_hidden_state, attention_mask)
            if labels is not None:
                loss, _, self.problem_type = sequence_classification_losses(logits, labels, self.num_labels, self.problem_type)
            else:
                loss = logits.new_zeros((1,))

        if not return_dict:
            return (loss, logits)

        return EarlyExitSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
            exit_depths=outputs.exit_depths,
            exit_probs=outputs.exit_probs,
            exit_stats=exit_stats,
        )
