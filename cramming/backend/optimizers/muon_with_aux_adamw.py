"""Muon + AdamW combined optimizer for Cramming.

PyTorch 2.9 provides `torch.optim.Muon`, which is intended for *2D* parameters of
neural network hidden layers. Other parameters (biases, LayerNorm scale/bias,
embeddings, output heads) should typically be optimized with a standard method
such as AdamW.

Cramming's torch backend expects a single optimizer object (for AMP `GradScaler`,
gradient clipping, checkpointing, and schedulers). This wrapper exposes a single
`torch.optim.Optimizer` interface while internally stepping a Muon optimizer for
selected parameters and an AdamW optimizer for the remainder.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.optim import Optimizer

log = logging.getLogger(__name__)


def _matches_any_substring(name: str, substrings: Sequence[str]) -> bool:
    lowered_name = name.lower()
    return any(s and s.lower() in lowered_name for s in substrings)


def _optional_int(value: Any, default: Optional[int]) -> Optional[int]:
    if value is None or value == "":
        return default
    return int(value)


def _filter_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any], label: str) -> Dict[str, Any]:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return kwargs

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs

    supported = set(parameters)
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        log.warning("Dropping unsupported %s optimizer args for this PyTorch build: %s", label, dropped)
    return filtered


@dataclass(frozen=True)
class MuonSplit:
    muon_params: List[torch.nn.Parameter]
    aux_params_decay: List[torch.nn.Parameter]
    aux_params_no_decay: List[torch.nn.Parameter]


def split_parameters_for_muon(
    model: torch.nn.Module,
    *,
    limited_decay_keys: Sequence[str],
    muon_exclude_name_substrings: Sequence[str],
    muon_min_ndim: int = 2,
    muon_max_ndim: Optional[int] = 2,
) -> MuonSplit:
    """Split model parameters into Muon vs auxiliary (AdamW) groups.

    Heuristic:
      - Muon: hidden matrix parameters whose ndim is in the configured range
        (2D by default) and whose name does not match any exclusion substring.
      - Aux/AdamW: everything else.
      - For aux params, apply Cramming's `limited_decay_keys` convention to create
        a no-decay group.
    """

    muon_params: List[torch.nn.Parameter] = []
    aux_params_decay: List[torch.nn.Parameter] = []
    aux_params_no_decay: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        in_ndim_range = p.ndim >= muon_min_ndim and (muon_max_ndim is None or p.ndim <= muon_max_ndim)
        is_muon_candidate = in_ndim_range and not _matches_any_substring(name, muon_exclude_name_substrings)

        if is_muon_candidate:
            muon_params.append(p)
        else:
            if len(limited_decay_keys) > 0 and any(k in name for k in limited_decay_keys):
                aux_params_no_decay.append(p)
            else:
                aux_params_decay.append(p)

    return MuonSplit(muon_params=muon_params, aux_params_decay=aux_params_decay, aux_params_no_decay=aux_params_no_decay)


class MuonWithAuxAdamW(Optimizer):
    """A single-optimizer facade that steps Muon (hidden matrices) + AdamW (aux).

    Notes:
      - This optimizer intentionally keeps *wrapper* param_groups separate from
        the internal optimizers' param_groups. A scheduler mutates the wrapper
        param_groups, and we sync the learning rates into the internal optimizers
        right before stepping.
      - This design keeps compatibility with `torch.cuda.amp.GradScaler`.
    """

    def __init__(
        self,
        param_groups: List[Dict[str, Any]],
        *,
        muon_group_indices: Sequence[int],
        aux_group_indices: Sequence[int],
        muon_kwargs: Mapping[str, Any],
        aux_kwargs: Mapping[str, Any],
    ):
        # Union defaults for the wrapper Optimizer (only keys present here are
        # allowed in param groups).
        defaults: Dict[str, Any] = {
            "lr": 0.0,
            "weight_decay": 0.0,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "amsgrad": False,
            "fused": False,
        }

        super().__init__(param_groups, defaults)

        if len(param_groups) == 0:
            raise ValueError("MuonWithAuxAdamW received no trainable parameters.")

        if len(muon_group_indices) > 0 and not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "torch.optim.Muon was not found. Install PyTorch >= 2.9 or vendor a Muon implementation."
            )

        self._muon_group_indices = list(muon_group_indices)
        self._aux_group_indices = list(aux_group_indices)
        self._muon_kwargs = dict(muon_kwargs)
        self._aux_kwargs = dict(aux_kwargs)

        # Build internal optimizers with sanitized param group dicts.
        muon_param_groups: List[Dict[str, Any]] = []
        self._wrapper_to_muon: Dict[int, int] = {}
        for j, idx in enumerate(self._muon_group_indices):
            g = self.param_groups[idx]
            muon_param_groups.append({"params": g["params"], "lr": g["lr"], "weight_decay": g.get("weight_decay", 0.0)})
            self._wrapper_to_muon[idx] = j

        aux_param_groups: List[Dict[str, Any]] = []
        self._wrapper_to_aux: Dict[int, int] = {}
        for j, idx in enumerate(self._aux_group_indices):
            g = self.param_groups[idx]
            aux_param_groups.append(
                {
                    "params": g["params"],
                    "lr": g["lr"],
                    "betas": g.get("betas", defaults["betas"]),
                    "eps": g.get("eps", defaults["eps"]),
                    "weight_decay": g.get("weight_decay", 0.0),
                    "amsgrad": g.get("amsgrad", defaults["amsgrad"]),
                }
            )
            self._wrapper_to_aux[idx] = j

        # Initialize internal optimizers.
        if len(muon_param_groups) > 0:
            muon_init_kwargs = _filter_supported_kwargs(torch.optim.Muon, self._muon_kwargs, "Muon")
            self._muon: Optional[Optimizer] = torch.optim.Muon(muon_param_groups, **muon_init_kwargs)
        else:
            self._muon = None

        # AdamW supports optional fused/foreach flags in newer PyTorch versions.
        # Only pass keys that are not None / not empty.
        if len(aux_param_groups) > 0:
            aux_init_kwargs = _filter_supported_kwargs(torch.optim.AdamW, dict(self._aux_kwargs), "AdamW")
            self._aux: Optional[Optimizer] = torch.optim.AdamW(aux_param_groups, **aux_init_kwargs)
        else:
            self._aux = None

    def _sync_group_lrs(self) -> None:
        """Sync wrapper learning rates (and weight decay) into internal optimizers."""

        for wrapper_idx, muon_idx in self._wrapper_to_muon.items():
            if self._muon is None:
                continue
            wg = self.param_groups[wrapper_idx]
            ig = self._muon.param_groups[muon_idx]
            ig["lr"] = wg["lr"]
            ig["weight_decay"] = wg.get("weight_decay", 0.0)

        for wrapper_idx, aux_idx in self._wrapper_to_aux.items():
            if self._aux is None:
                continue
            wg = self.param_groups[wrapper_idx]
            ig = self._aux.param_groups[aux_idx]
            ig["lr"] = wg["lr"]
            ig["weight_decay"] = wg.get("weight_decay", 0.0)

    @torch.no_grad()
    def step(self, closure=None):
        self._sync_group_lrs()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Order is not mathematically important for these two optimizers.
        if self._muon is not None:
            self._muon.step()
        if self._aux is not None:
            self._aux.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        # Keep wrapper semantics consistent with Optimizer.
        if self._muon is not None:
            self._muon.zero_grad(set_to_none=set_to_none)
        if self._aux is not None:
            self._aux.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "wrapper": super().state_dict(),
            "muon": self._muon.state_dict() if self._muon is not None else None,
            "aux": self._aux.state_dict() if self._aux is not None else None,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if "wrapper" in state_dict:
            super().load_state_dict(state_dict["wrapper"])

        muon_state = state_dict.get("muon")
        aux_state = state_dict.get("aux")
        if self._muon is not None and muon_state is not None:
            self._muon.load_state_dict(muon_state)
        if self._aux is not None and aux_state is not None:
            self._aux.load_state_dict(aux_state)
        self._sync_group_lrs()


def build_muon_with_aux_adamw(model: torch.nn.Module, cfg_train, cfg_impl) -> MuonWithAuxAdamW:
    """Factory used by Cramming's torch backend."""

    # Reasonable defaults for transformer-style models.
    default_excludes = [
        "embeddings",
        "embedding",
        "embed",
        "word_embeddings",
        "position_embeddings",
        "pos_embedding",
        "lm_head",
        "q_head",
        "head",
        "cls",
        "classifier",
        "decoder",
        "pooler",
        "score",
    ]

    muon_excludes = list(getattr(cfg_train.optim, "muon_exclude_name_substrings", default_excludes) or default_excludes)
    muon_min_ndim = int(getattr(cfg_train.optim, "muon_min_ndim", 2))
    muon_max_ndim = _optional_int(getattr(cfg_train.optim, "muon_max_ndim", 2), 2)

    split = split_parameters_for_muon(
        model,
        limited_decay_keys=getattr(cfg_train, "limited_decay_keys", []),
        muon_exclude_name_substrings=muon_excludes,
        muon_min_ndim=muon_min_ndim,
        muon_max_ndim=muon_max_ndim,
    )

    # Hyperparameters.
    aux_lr = float(cfg_train.optim.lr)
    aux_betas = tuple(getattr(cfg_train.optim, "betas", (0.9, 0.999)))
    aux_eps = float(getattr(cfg_train.optim, "eps", 1e-8))
    aux_weight_decay = float(getattr(cfg_train.optim, "weight_decay", 0.0))
    aux_amsgrad = bool(getattr(cfg_train.optim, "amsgrad", False))

    muon_lr = float(getattr(cfg_train.optim, "muon_lr", 0.02))
    muon_weight_decay = float(getattr(cfg_train.optim, "muon_weight_decay", aux_weight_decay))
    muon_momentum = float(getattr(cfg_train.optim, "muon_momentum", 0.95))
    muon_nesterov = bool(getattr(cfg_train.optim, "muon_nesterov", True))
    muon_ns_steps = int(getattr(cfg_train.optim, "muon_ns_steps", 5))
    muon_eps = float(getattr(cfg_train.optim, "muon_eps", 1e-7))
    muon_adjust_lr_fn = getattr(cfg_train.optim, "muon_adjust_lr_fn", None)
    muon_ns_coefficients = getattr(cfg_train.optim, "muon_ns_coefficients", None)

    # Wrapper param groups.
    param_groups: List[Dict[str, Any]] = []
    muon_group_indices: List[int] = []
    aux_group_indices: List[int] = []

    if len(split.muon_params) > 0:
        muon_group_indices.append(len(param_groups))
        param_groups.append({"params": split.muon_params, "lr": muon_lr, "weight_decay": muon_weight_decay})

    if len(split.aux_params_decay) > 0:
        aux_group_indices.append(len(param_groups))
        param_groups.append(
            {
                "params": split.aux_params_decay,
                "lr": aux_lr,
                "betas": aux_betas,
                "eps": aux_eps,
                "weight_decay": aux_weight_decay,
                "amsgrad": aux_amsgrad,
            }
        )

    if len(split.aux_params_no_decay) > 0:
        aux_group_indices.append(len(param_groups))
        param_groups.append(
            {
                "params": split.aux_params_no_decay,
                "lr": aux_lr,
                "betas": aux_betas,
                "eps": aux_eps,
                "weight_decay": 0.0,
                "amsgrad": aux_amsgrad,
            }
        )

    # Internal optimizer kwargs.
    muon_kwargs: Dict[str, Any] = {
        "momentum": muon_momentum,
        "nesterov": muon_nesterov,
        "ns_steps": muon_ns_steps,
        "eps": muon_eps,
    }
    if muon_adjust_lr_fn is not None and muon_adjust_lr_fn != "":
        muon_kwargs["adjust_lr_fn"] = muon_adjust_lr_fn
    if muon_ns_coefficients is not None and muon_ns_coefficients != "":
        muon_kwargs["ns_coefficients"] = tuple(float(x) for x in muon_ns_coefficients)

    aux_kwargs: Dict[str, Any] = {}
    # Cramming optionally enables foreach mode for optimizers.
    if getattr(cfg_impl, "foreach_optimizer", False):
        aux_kwargs["foreach"] = True
    # Optional fused flag (present in some PyTorch builds). If it is None/empty,
    # do not pass it.
    fused = getattr(cfg_train.optim, "fused", None)
    if fused is not None and fused != "":
        aux_kwargs["fused"] = bool(fused)

    return MuonWithAuxAdamW(
        param_groups,
        muon_group_indices=muon_group_indices,
        aux_group_indices=aux_group_indices,
        muon_kwargs=muon_kwargs,
        aux_kwargs=aux_kwargs,
    )
