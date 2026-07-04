"""TRM as Recursive Refiner plus deep supervision and halting.

This module intentionally reuses the Recursive Refiner implementation instead
of maintaining a second nearly-identical recurrent Transformer stack.  The TRM
variant keeps the same core variables and update rule:

    z_lo <- shared(z_lo, inject=z_hi + x)
    z_hi <- shared(z_hi, inject=z_lo)

The additions here are:

* a detached carry so refinement can continue across supervision/inference steps;
* optional deep supervision over repeated refinement steps;
* a lightweight halt head trained from whether the current prediction is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import ModelOutput

from .recursive_refiner_hf import (
    RecursiveRefinerConfig,
    RecursiveRefinerModel,
    RecursiveRefinerPreTrainedModel,
)


IGNORE_LABEL_ID = -100


@dataclass
class TRMCarry:
    z_hi: torch.Tensor
    z_lo: torch.Tensor

    @property
    def z_H(self) -> torch.Tensor:
        return self.z_hi

    @property
    def z_L(self) -> torch.Tensor:
        return self.z_lo


TRMInnerCarry = TRMCarry


@dataclass
class TRMActState:
    carry: TRMCarry
    steps: torch.Tensor
    halted: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass
class TRMModelOutput(ModelOutput):
    last_hidden_state: torch.Tensor = None
    halt_state: Optional[torch.Tensor] = None
    carry: Optional[TRMCarry] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class TRMMaskedLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    q_halt_logits: Optional[torch.Tensor] = None
    q_continue_logits: Optional[torch.Tensor] = None
    carry: Optional[TRMCarry] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class TRMConfig(RecursiveRefinerConfig):
    model_type = "trm"

    def __init__(
        self,
        vocab_size: int = 50000,
        max_position_embeddings: int = 512,
        seq_len: Optional[int] = None,
        hidden_size: int = 256,
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
        grad_last_cycle_only: bool = True,
        embed_factor: int = 1,
        pre_norm: bool = True,
        rms_eps: Optional[float] = None,
        rms_norm_eps: Optional[float] = None,
        rope_theta: float = 10000.0,
        prefix_len: int = 0,
        is_causal: bool = False,
        deep_supervision_steps: Optional[int] = None,
        inference_steps: Optional[int] = None,
        halt_max_steps: Optional[int] = None,
        act_training: bool = False,
        halt_exploration_prob: float = 0.0,
        no_ACT_continue: bool = True,
        q_halt_loss_weight: float = 0.5,
        # Compatibility-only fields accepted from older TRM configs.
        pos_encodings: str = "rope",
        mlp_t: bool = False,
        forward_dtype: str = "auto",
        loss_type: str = "cross_entropy",
        puzzle_emb_ndim: int = 0,
        num_puzzle_identifiers: int = 1,
        puzzle_emb_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        classifier_dropout: float = 0.1,
        pooler_type: str = "cls",
        use_cache: bool = False,
        **kwargs,
    ):
        if seq_len is not None:
            max_position_embeddings = seq_len
        if num_heads is not None:
            num_attention_heads = num_heads

        resolved_layers = L_layers if L_layers is not None else num_hidden_layers
        if resolved_layers is None:
            resolved_layers = 2

        resolved_hi_cycles = H_cycles if H_cycles is not None else hi_cycles
        resolved_lo_cycles = L_cycles if L_cycles is not None else lo_cycles
        if resolved_hi_cycles is None:
            resolved_hi_cycles = 3
        if resolved_lo_cycles is None:
            resolved_lo_cycles = 6

        resolved_rms_eps = rms_eps if rms_eps is not None else rms_norm_eps
        if resolved_rms_eps is None:
            resolved_rms_eps = 1e-5

        if deep_supervision_steps is None:
            deep_supervision_steps = halt_max_steps if halt_max_steps is not None else 1
        if inference_steps is None:
            inference_steps = deep_supervision_steps
        if halt_max_steps is None:
            halt_max_steps = max(deep_supervision_steps, inference_steps)

        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=resolved_layers,
            expansion=expansion,
            hi_cycles=resolved_hi_cycles,
            lo_cycles=resolved_lo_cycles,
            grad_last_cycle_only=grad_last_cycle_only,
            embed_factor=embed_factor,
            pre_norm=pre_norm,
            rope_theta=rope_theta,
            rms_eps=resolved_rms_eps,
            prefix_len=prefix_len,
            is_causal=is_causal,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            use_cache=use_cache,
            classifier_dropout=classifier_dropout,
            pooler_type=pooler_type,
            **kwargs,
        )

        # TRM/older-config aliases.  The canonical variables are the Recursive
        # Refiner names: hi_cycles, lo_cycles, rms_eps, num_hidden_layers.
        self.H_cycles = self.hi_cycles
        self.L_cycles = self.lo_cycles
        self.H_layers = int(H_layers)
        self.L_layers = self.num_hidden_layers
        self.seq_len = self.max_position_embeddings
        self.num_heads = self.num_attention_heads
        self.rms_norm_eps = self.rms_eps

        self.deep_supervision_steps = int(deep_supervision_steps)
        self.inference_steps = int(inference_steps)
        self.halt_max_steps = int(halt_max_steps)
        self.act_training = bool(act_training)
        self.halt_exploration_prob = float(halt_exploration_prob)
        self.no_ACT_continue = bool(no_ACT_continue)
        self.q_halt_loss_weight = float(q_halt_loss_weight)

        self.pos_encodings = str(pos_encodings)
        self.mlp_t = bool(mlp_t)
        self.forward_dtype = str(forward_dtype)
        self.loss_type = str(loss_type)
        self.puzzle_emb_ndim = int(puzzle_emb_ndim)
        self.num_puzzle_identifiers = int(num_puzzle_identifiers)
        self.puzzle_emb_len = int(puzzle_emb_len or 0)
        self.batch_size = batch_size


class TRMPreTrainedModel(RecursiveRefinerPreTrainedModel):
    config_class = TRMConfig
    base_model_prefix = "trm"
    supports_gradient_checkpointing = False


class TRMModel(RecursiveRefinerModel):
    config_class = TRMConfig
    base_model_prefix = "trm"

    def _input_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
            raise ValueError(f"attention_mask must have shape {(batch_size, seq_len)}, got {tuple(attention_mask.shape)}")

        input_embeddings = self.embed(input_ids)
        if self.config.prefix_len > 0:
            prefix = self.prefix.unsqueeze(0).expand(batch_size, -1, -1)
            input_embeddings = torch.cat([prefix, input_embeddings], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    batch_size,
                    self.config.prefix_len,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return input_embeddings, attention_mask

    def empty_carry(self, batch_size: int, total_len: int, device: torch.device, dtype: torch.dtype) -> TRMCarry:
        z_hi, z_lo = self.init_latents(batch_size, total_len=total_len, device=device, dtype=dtype)
        return TRMCarry(z_hi=z_hi, z_lo=z_lo)

    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMCarry) -> TRMCarry:
        reset = reset_flag.to(device=carry.z_hi.device, dtype=torch.bool).view(-1, 1, 1)
        z_hi_init = self.hi_init.to(device=carry.z_hi.device, dtype=carry.z_hi.dtype).view(1, 1, -1)
        z_lo_init = self.lo_init.to(device=carry.z_lo.device, dtype=carry.z_lo.dtype).view(1, 1, -1)
        return TRMCarry(
            z_hi=torch.where(reset, z_hi_init, carry.z_hi),
            z_lo=torch.where(reset, z_lo_init, carry.z_lo),
        )

    def refine_once(
        self,
        carry: TRMCarry,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[TRMCarry, torch.Tensor, torch.Tensor]:
        z_hi, z_lo = carry.z_hi, carry.z_lo
        hi_cycles = int(max(1, self.config.hi_cycles))
        lo_cycles = int(max(1, self.config.lo_cycles))

        if self.config.grad_last_cycle_only:
            with torch.no_grad():
                for _ in range(hi_cycles - 1):
                    for _ in range(lo_cycles):
                        z_lo = self.shared(z_lo, inject=(z_hi + input_embeddings), attention_mask=attention_mask)
                    z_hi = self.shared(z_hi, inject=z_lo, attention_mask=attention_mask)

        grad_cycles = 1 if self.config.grad_last_cycle_only else hi_cycles
        for _ in range(grad_cycles):
            for _ in range(lo_cycles):
                z_lo = self.shared(z_lo, inject=(z_hi + input_embeddings), attention_mask=attention_mask)
            z_hi = self.shared(z_hi, inject=z_lo, attention_mask=attention_mask)

        new_carry = TRMCarry(z_hi=z_hi.detach(), z_lo=z_lo.detach())
        halt_state = z_hi[:, 0]
        return new_carry, z_hi, halt_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        carry: Optional[TRMCarry] = None,
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
        z_hi = carry.z_hi
        halt_state = None

        for _ in range(steps):
            carry, z_hi, halt_state = self.refine_once(carry, input_embeddings, attention_mask)
            if hidden_history is not None:
                hidden_history.append(z_hi[:, self.config.prefix_len :])

        last_hidden_state = z_hi[:, self.config.prefix_len :]
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
        config.is_causal = False
        config.is_decoder = False
        super().__init__(config)
        self.recursive_refiner = TRMModel(config)
        self.trm = self.recursive_refiner
        self.q_head = nn.Linear(config.hidden_size, 1)
        self._act_state: Optional[TRMActState] = None
        self.post_init()
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def get_input_embeddings(self) -> nn.Module:
        return self.recursive_refiner.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.recursive_refiner.set_input_embeddings(value)

    def get_output_embeddings(self) -> Optional[nn.Module]:
        return None

    def _lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL_ID)
        return loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

    def _seq_is_correct(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid_mask = labels != IGNORE_LABEL_ID
        loss_counts = valid_mask.sum(dim=-1)
        safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
        preds = logits.argmax(dim=-1)
        is_correct = valid_mask & (preds == safe_labels)
        return (is_correct.sum(dim=-1) == loss_counts) & (loss_counts > 0)

    def _q_halt_logits(self, halt_state: torch.Tensor) -> torch.Tensor:
        return self.q_head(halt_state).squeeze(-1).to(torch.float32)

    def _q_halt_loss(self, q_halt_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(q_halt_logits, targets.to(q_halt_logits.dtype), reduction="mean")

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
                carry=self.recursive_refiner.empty_carry(
                    batch_size=input_ids.shape[0],
                    total_len=input_ids.shape[1] + self.config.prefix_len,
                    device=input_ids.device,
                    dtype=self.recursive_refiner.hi_init.dtype,
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

        input_embeddings, trm_attention_mask = self.recursive_refiner._input_embeddings(current_input_ids, current_attention_mask)
        carry = TRMCarry(
            z_hi=state.carry.z_hi.to(device=input_embeddings.device, dtype=input_embeddings.dtype),
            z_lo=state.carry.z_lo.to(device=input_embeddings.device, dtype=input_embeddings.dtype),
        )
        carry = self.recursive_refiner.reset_carry(reset_flag, carry)
        steps = torch.where(reset_flag, torch.zeros_like(state.steps), state.steps).to(device=input_ids.device)

        new_carry, z_hi, halt_state = self.recursive_refiner.refine_once(carry, input_embeddings, trm_attention_mask)
        token_hidden = z_hi[:, self.config.prefix_len :]
        logits = self.recursive_refiner.embed.logits(token_hidden)
        q_halt_logits = self._q_halt_logits(halt_state)

        with torch.no_grad():
            seq_is_correct = self._seq_is_correct(logits, current_labels)

        loss = self._lm_loss(logits, current_labels)
        if self.config.q_halt_loss_weight > 0:
            loss = loss + self.config.q_halt_loss_weight * self._q_halt_loss(q_halt_logits, seq_is_correct)

        with torch.no_grad():
            new_steps = steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step
            if self.config.halt_max_steps > 1:
                halted = halted | (q_halt_logits > 0)
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
            out = (logits, q_halt_logits, None)
            if return_carry:
                out = out + (new_carry,)
            return (loss,) + out

        return TRMMaskedLMOutput(
            loss=loss,
            logits=logits,
            q_halt_logits=q_halt_logits,
            q_continue_logits=None,
            carry=new_carry if return_carry else None,
            hidden_states=hidden_history,
            attentions=None,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        carry: Optional[TRMCarry] = None,
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

        input_embeddings, trm_attention_mask = self.recursive_refiner._input_embeddings(input_ids, attention_mask)
        if carry is None:
            carry = self.recursive_refiner.empty_carry(
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

        for step_idx in range(steps):
            carry, z_hi, halt_state = self.recursive_refiner.refine_once(carry, input_embeddings, trm_attention_mask)
            token_hidden = z_hi[:, self.config.prefix_len :]
            logits = self.recursive_refiner.embed.logits(token_hidden)
            q_halt_logits = self._q_halt_logits(halt_state)

            if hidden_history is not None:
                hidden_history.append(token_hidden)

            if labels is not None:
                step_loss = self._lm_loss(logits, labels)
                if self.config.q_halt_loss_weight > 0:
                    with torch.no_grad():
                        seq_is_correct = self._seq_is_correct(logits, labels)
                    step_loss = step_loss + self.config.q_halt_loss_weight * self._q_halt_loss(q_halt_logits, seq_is_correct)
                losses.append(step_loss)
            elif step_idx + 1 < steps and self.config.halt_max_steps > 1 and bool((q_halt_logits > 0).all().item()):
                break

        loss = torch.stack(losses).mean() if losses else None

        if not return_dict:
            out = (logits, q_halt_logits, None)
            if return_carry:
                out = out + (carry,)
            return ((loss,) + out) if loss is not None else out

        return TRMMaskedLMOutput(
            loss=loss,
            logits=logits,
            q_halt_logits=q_halt_logits,
            q_continue_logits=None,
            carry=carry if return_carry else None,
            hidden_states=tuple(hidden_history) if hidden_history is not None else None,
            attentions=None,
        )


class TRMForSequenceClassification(TRMPreTrainedModel):
    def __init__(self, config: TRMConfig):
        config.is_decoder = False
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.recursive_refiner = TRMModel(config)
        self.trm = self.recursive_refiner
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

        input_embeddings, trm_attention_mask = self.recursive_refiner._input_embeddings(input_ids, attention_mask)
        carry = self.recursive_refiner.empty_carry(
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
            carry, z_hi, _halt_state = self.recursive_refiner.refine_once(carry, input_embeddings, trm_attention_mask)
            hidden = z_hi[:, self.config.prefix_len :]
            pooled = self._pool(hidden, attention_mask)
            logits = self.classifier(self.dropout(pooled))
            if labels is not None:
                losses.append(self._loss(logits, labels))

        loss = torch.stack(losses).mean() if losses else logits.new_zeros((1,))
        if not return_dict:
            return (loss, logits)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)


TRMforMaskedLM = TRMForMaskedLM
