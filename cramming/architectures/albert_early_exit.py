"""ALBERT-style shared-parameter model with sequence-level early exits."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from omegaconf import OmegaConf
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.models.albert.modeling_albert import (
    AlbertEmbeddings,
    AlbertMLMHead,
    AlbertPreTrainedModel,
    AlbertTransformer,
)

from .recursive_refiner_early_exit_hf import (
    EarlyExitMaskedLMOutput,
    EarlyExitModelOutput,
    EarlyExitSequenceClassifierOutput,
    compute_exit_statistics,
    compute_halting_loss,
    get_early_exit_settings,
    masked_lm_losses,
    pool_sequence,
    select_exit_depths,
    sequence_classification_losses,
)


class AlbertEarlyExitModel(AlbertPreTrainedModel):
    config_class = transformers.AlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config: transformers.AlbertConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.halt_head = nn.Linear(config.hidden_size, 1)
        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        self.attn_implementation = getattr(config, "_attn_implementation", "eager")
        self.position_embedding_type = config.position_embedding_type
        self.post_init()
        self.reset_early_exit_parameters()

    def reset_early_exit_parameters(self) -> None:
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -2.0)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def max_depth(self) -> int:
        return int(get_early_exit_settings(self.config)["max_depth"])

    def exit_interval(self) -> int:
        fallback = max(1, self.config.num_hidden_layers // max(1, self.max_depth()))
        return max(1, int(getattr(self.config, "exit_interval", fallback)))

    def exit_layers(self) -> Tuple[int, ...]:
        max_depth = self.max_depth()
        interval = self.exit_interval()
        final_layer = min(int(self.config.num_hidden_layers), max_depth * interval)
        layers = [min(final_layer, depth * interval) for depth in range(1, max_depth + 1)]
        layers[-1] = final_layer
        return tuple(dict.fromkeys(layers))

    def _prepare_inputs(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        output_attentions: bool,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        use_sdpa_attention_mask = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )
        if use_sdpa_attention_mask:
            extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask,
                embedding_output.dtype,
                tgt_len=seq_length,
            )
        else:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        hidden_states = self.encoder.embedding_hidden_mapping_in(embedding_output)
        return hidden_states, extended_attention_mask, attention_mask, head_mask

    def _layer_group_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask,
        layer_index: int,
        output_attentions: bool,
        output_hidden_states: bool,
    ):
        layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
        group_idx = int(layer_index / (self.config.num_hidden_layers / self.config.num_hidden_groups))
        group_head_mask = head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group]
        return self.encoder.albert_layer_groups[group_idx](
            hidden_states,
            attention_mask,
            group_head_mask,
            output_attentions,
            output_hidden_states,
        )

    def _pool(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        if self.pooler is None:
            return None
        return self.pooler_activation(self.pooler(hidden_states[:, 0]))

    def _forward_all_depths(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> EarlyExitModelOutput:
        settings = get_early_exit_settings(self.config)
        hidden_states, extended_attention_mask, token_attention_mask, head_mask = self._prepare_inputs(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
        )
        exit_layers = set(self.exit_layers())
        final_layer = max(exit_layers)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        exit_states = []
        halt_logits = []

        for layer_index in range(final_layer):
            layer_group_output = self._layer_group_forward(
                hidden_states,
                extended_attention_mask,
                head_mask,
                layer_index,
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if layer_index + 1 in exit_layers:
                exit_states.append(hidden_states)
                pooled_for_halt = pool_sequence(hidden_states, token_attention_mask, pooler_type="mean")
                halt_logits.append(self.halt_head(pooled_for_halt).squeeze(-1))

        halt_logits_tensor = torch.stack(halt_logits, dim=1)
        exit_depths, exit_probs, _ = select_exit_depths(halt_logits_tensor, settings)
        exit_stats = compute_exit_statistics(exit_depths, len(exit_states))

        return EarlyExitModelOutput(
            last_hidden_state=exit_states[-1],
            pooler_output=self._pool(exit_states[-1]),
            exit_hidden_states=tuple(exit_states),
            halt_logits=halt_logits_tensor,
            exit_depths=exit_depths,
            exit_probs=exit_probs,
            exit_stats=exit_stats,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    @torch.no_grad()
    def _forward_dynamic(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        head_mask: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> EarlyExitModelOutput:
        settings = get_early_exit_settings(self.config)
        hidden_states, extended_attention_mask, token_attention_mask, head_mask = self._prepare_inputs(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
        )
        batch_size, seq_len, hidden_size = hidden_states.shape
        exit_layers = tuple(self.exit_layers())
        exit_layer_to_depth = {layer: idx + 1 for idx, layer in enumerate(exit_layers)}
        final_layer = max(exit_layers)
        max_depth = len(exit_layers)

        final_hidden = hidden_states.new_zeros(batch_size, seq_len, hidden_size)
        exit_depths = torch.full((batch_size,), max_depth, dtype=torch.long, device=hidden_states.device)
        exit_probs = hidden_states.new_zeros(batch_size)

        active_indices = torch.arange(batch_size, device=hidden_states.device)
        active_hidden = hidden_states
        active_attention_mask = extended_attention_mask
        active_token_mask = token_attention_mask

        for layer_index in range(final_layer):
            layer_group_output = self._layer_group_forward(
                active_hidden,
                active_attention_mask,
                head_mask,
                layer_index,
                output_attentions=False,
                output_hidden_states=False,
            )
            active_hidden = layer_group_output[0]

            layer_num = layer_index + 1
            if layer_num not in exit_layer_to_depth:
                continue

            depth = exit_layer_to_depth[layer_num]
            pooled_for_halt = pool_sequence(active_hidden, active_token_mask, pooler_type="mean")
            probs = torch.sigmoid(self.halt_head(pooled_for_halt).squeeze(-1))
            should_exit = probs >= float(settings["halt_threshold"])
            if depth < int(settings["min_depth"]):
                should_exit = torch.zeros_like(should_exit, dtype=torch.bool)
            if depth == max_depth:
                should_exit = torch.ones_like(should_exit, dtype=torch.bool)

            if should_exit.any():
                exiting_indices = active_indices[should_exit]
                final_hidden[exiting_indices] = active_hidden[should_exit]
                exit_depths[exiting_indices] = depth
                exit_probs[exiting_indices] = probs[should_exit]

            keep = ~should_exit
            if not keep.any():
                break

            active_indices = active_indices[keep]
            active_hidden = active_hidden[keep]
            active_attention_mask = active_attention_mask[keep]
            active_token_mask = active_token_mask[keep]

        exit_stats = compute_exit_statistics(exit_depths, max_depth)
        if settings["fail_if_all_max"] and bool((exit_depths == max_depth).all().item()):
            raise RuntimeError("All samples exited at max depth; adjust halting threshold or regularization.")

        return EarlyExitModelOutput(
            last_hidden_state=final_hidden,
            pooler_output=self._pool(final_hidden),
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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        force_all_depths: Optional[bool] = None,
    ) -> Union[EarlyExitModelOutput, Tuple]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if force_all_depths is None:
            force_all_depths = bool(torch.is_grad_enabled())

        settings = get_early_exit_settings(self.config)
        use_dynamic = settings["enabled"] and settings["inference_enabled"] and not force_all_depths
        if use_dynamic:
            outputs = self._forward_dynamic(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                output_attentions,
                output_hidden_states,
            )
        else:
            outputs = self._forward_all_depths(
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                output_attentions,
                output_hidden_states,
            )

        if not return_dict:
            return (outputs.last_hidden_state, outputs.pooler_output, outputs.hidden_states, outputs.attentions)
        return outputs


class AlbertEarlyExitForMaskedLM(AlbertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "predictions.decoder.weight"]

    def __init__(self, config: transformers.AlbertConfig):
        super().__init__(config)
        self.albert = AlbertEarlyExitModel(config, add_pooling_layer=False)
        self.predictions = AlbertMLMHead(config)
        self.post_init()
        self.albert.reset_early_exit_parameters()

    def get_output_embeddings(self) -> nn.Linear:
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.predictions.decoder = new_embeddings
        self.predictions.bias = new_embeddings.bias

    def get_input_embeddings(self) -> nn.Embedding:
        return self.albert.embeddings.word_embeddings

    def _loss_from_exit_states(
        self,
        exit_states: Tuple[torch.Tensor, ...],
        halt_logits: Optional[torch.Tensor],
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        settings = get_early_exit_settings(self.config)
        mean_losses = []
        per_sample_losses = []
        final_logits = None

        for state in exit_states:
            logits = self.predictions(state)
            final_logits = logits
            mean_loss, per_sample_loss = masked_lm_losses(logits, labels, self.config.vocab_size, attention_mask)
            mean_losses.append(mean_loss)
            per_sample_losses.append(per_sample_loss)

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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[EarlyExitMaskedLMOutput, Tuple]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        force_all_depths = labels is not None and torch.is_grad_enabled()
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            force_all_depths=force_all_depths,
        )

        exit_stats = dict(outputs.exit_stats or {})
        if labels is not None and force_all_depths and outputs.exit_hidden_states is not None:
            loss, logits, loss_stats = self._loss_from_exit_states(outputs.exit_hidden_states, outputs.halt_logits, labels, attention_mask)
            exit_stats.update(loss_stats)
        else:
            logits = self.predictions(outputs.last_hidden_state)
            loss = None
            if labels is not None:
                loss, _ = masked_lm_losses(logits, labels, self.config.vocab_size, attention_mask)

        if not return_dict:
            output = (logits,) + (() if outputs.hidden_states is None else (outputs.hidden_states,))
            return ((loss,) + output) if loss is not None else output

        return EarlyExitMaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            exit_depths=outputs.exit_depths,
            exit_probs=outputs.exit_probs,
            exit_stats=exit_stats,
        )


class AlbertEarlyExitForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config: transformers.AlbertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.albert = AlbertEarlyExitModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.problem_type = None
        self.post_init()
        self.albert.reset_early_exit_parameters()

    def _classify(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = self.albert._pool(hidden_states)
        return self.classifier(self.dropout(pooled))

    def _loss_from_exit_states(
        self,
        exit_states: Tuple[torch.Tensor, ...],
        halt_logits: Optional[torch.Tensor],
        labels: torch.Tensor,
    ):
        settings = get_early_exit_settings(self.config)
        mean_losses = []
        per_sample_losses = []
        final_logits = None
        problem_type = self.problem_type

        for state in exit_states:
            logits = self._classify(state)
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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[EarlyExitSequenceClassifierOutput, Tuple]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        force_all_depths = labels is not None and torch.is_grad_enabled()
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            force_all_depths=force_all_depths,
        )

        exit_stats = dict(outputs.exit_stats or {})
        if labels is not None and force_all_depths and outputs.exit_hidden_states is not None:
            loss, logits, loss_stats = self._loss_from_exit_states(outputs.exit_hidden_states, outputs.halt_logits, labels)
            exit_stats.update(loss_stats)
        else:
            logits = self._classify(outputs.last_hidden_state)
            if labels is not None:
                loss, _, self.problem_type = sequence_classification_losses(logits, labels, self.num_labels, self.problem_type)
            else:
                loss = logits.new_zeros((1,))

        if not return_dict:
            return (loss, logits)

        return EarlyExitSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            exit_depths=outputs.exit_depths,
            exit_probs=outputs.exit_probs,
            exit_stats=exit_stats,
        )


def construct_albert_early_exit(cfg_arch, vocab_size: int, downstream_classes: Optional[int] = None):
    config_dict = OmegaConf.to_container(cfg_arch, resolve=True)
    config_dict["vocab_size"] = int(vocab_size)
    configuration = transformers.AlbertConfig(**config_dict)
    if downstream_classes is not None:
        configuration.num_labels = int(downstream_classes)
        model = AlbertEarlyExitForSequenceClassification(configuration)
    else:
        model = AlbertEarlyExitForMaskedLM(configuration)
    model.vocab_size = configuration.vocab_size
    return model
