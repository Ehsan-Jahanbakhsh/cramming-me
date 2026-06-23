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

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch.optim import Optimizer


def _matches_any_substring(name: str, substrings: Sequence[str]) -> bool:
    return any(s in name for s in substrings)


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
) -> MuonSplit:
    """Split model parameters into Muon vs auxiliary (AdamW) groups.

    Heuristic:
      - Muon: parameters with ndim >= `muon_min_ndim` (typically 2D matrices)
        whose *name* does not match any exclusion substring.
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

        is_muon_candidate = (p.ndim >= muon_min_ndim) and not _matches_any_substring(name, muon_exclude_name_substrings)

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

        if not hasattr(torch.optim, "Muon"):
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
        self._muon = torch.optim.Muon(muon_param_groups, **self._muon_kwargs)

        # AdamW supports optional fused/foreach flags in newer PyTorch versions.
        # Only pass keys that are not None / not empty.
        aux_init_kwargs = dict(self._aux_kwargs)
        self._aux = torch.optim.AdamW(aux_param_groups, **aux_init_kwargs)

    def _sync_group_lrs(self) -> None:
        """Sync wrapper learning rates (and weight decay) into internal optimizers."""

        for wrapper_idx, muon_idx in self._wrapper_to_muon.items():
            wg = self.param_groups[wrapper_idx]
            ig = self._muon.param_groups[muon_idx]
            ig["lr"] = wg["lr"]
            ig["weight_decay"] = wg.get("weight_decay", 0.0)

        for wrapper_idx, aux_idx in self._wrapper_to_aux.items():
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
        self._muon.step()
        self._aux.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        # Keep wrapper semantics consistent with Optimizer.
        self._muon.zero_grad(set_to_none=set_to_none)
        self._aux.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "muon": self._muon.state_dict(),
            "aux": self._aux.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._muon.load_state_dict(state_dict["muon"])
        self._aux.load_state_dict(state_dict["aux"])


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
        "cls",
        "classifier",
    ]

    muon_excludes = list(getattr(cfg_train.optim, "muon_exclude_name_substrings", default_excludes) or default_excludes)
    muon_min_ndim = int(getattr(cfg_train.optim, "muon_min_ndim", 2))

    split = split_parameters_for_muon(
        model,
        limited_decay_keys=getattr(cfg_train, "limited_decay_keys", []),
        muon_exclude_name_substrings=muon_excludes,
        muon_min_ndim=muon_min_ndim,
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
