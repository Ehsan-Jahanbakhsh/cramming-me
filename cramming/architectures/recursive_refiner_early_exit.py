"""Cramming constructor for the early-exit Recursive Refiner."""

from __future__ import annotations

from typing import Optional

from omegaconf import OmegaConf

from .recursive_refiner_early_exit_hf import (
    RecursiveRefinerEarlyExitConfig,
    RecursiveRefinerEarlyExitForMaskedLM,
    RecursiveRefinerEarlyExitForSequenceClassification,
)


def construct_recursive_refiner_early_exit(cfg_arch, vocab_size: int, downstream_classes: Optional[int] = None):
    cfg = OmegaConf.to_container(cfg_arch, resolve=True)
    cfg.pop("architectures", None)
    cfg["vocab_size"] = int(vocab_size)

    config = RecursiveRefinerEarlyExitConfig(**cfg)
    if downstream_classes is not None:
        config.num_labels = int(downstream_classes)
        config.is_causal = False
        config.is_decoder = False
        model = RecursiveRefinerEarlyExitForSequenceClassification(config)
    else:
        config.is_causal = False
        config.is_decoder = False
        model = RecursiveRefinerEarlyExitForMaskedLM(config)

    model.vocab_size = config.vocab_size
    return model
