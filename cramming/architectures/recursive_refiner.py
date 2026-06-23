"""Recursive Refiner architecture integration.

This module wires a Hugging Face Transformers-compatible implementation of the
"Recursive Refiner" model into Cramming's local architecture construction
pipeline.

The model implementation itself lives in :mod:`cramming.architectures.recursive_refiner_hf`.
"""

from __future__ import annotations

from typing import Optional

from omegaconf import OmegaConf

from .recursive_refiner_hf import (
    RecursiveRefinerConfig,
    RecursiveRefinerForMaskedLM,
    RecursiveRefinerForCausalLM,
    RecursiveRefinerForSequenceClassification,
    RecursiveRefinerSingleHighForMaskedLM,
    RecursiveRefinerSingleHighForCausalLM,
    RecursiveRefinerSingleHighForSequenceClassification,
)


def construct_recursive_refiner(cfg_arch, vocab_size: int, downstream_classes: Optional[int] = None):
    """Construct a Recursive Refiner model.

    Notes:
      * Pretraining in this repo is MLM-based by default. Set ``arch.is_causal=true``
        and choose ``RecursiveRefinerForCausalLM`` in the ``architectures`` list if
        you want autoregressive training.
      * Downstream sequence classification is supported via ``RecursiveRefinerForSequenceClassification``.
    """

    cfg = OmegaConf.to_container(cfg_arch, resolve=True)
    cfg.pop("architectures", None)

    # Populate vocab size from the tokenizer/dataset.
    cfg["vocab_size"] = int(vocab_size)

    config = RecursiveRefinerConfig(**cfg)

    # Populate downstream task head size if requested.
    if downstream_classes is not None:
        config.num_labels = int(downstream_classes)

    # Pick head based on explicit config and/or architecture name.
    arch_names = set(getattr(cfg_arch, "architectures", []) or [])
    single_high = any(name.startswith("RecursiveRefinerSingleHigh") for name in arch_names)
    force_causal = (
        "RecursiveRefinerForCausalLM" in arch_names
        or "RecursiveRefinerSingleHighForCausalLM" in arch_names
        or bool(getattr(config, "is_causal", False))
    )

    # Downstream sequence classification uses a dedicated head.
    if downstream_classes is not None:
        # For classification, we generally want bidirectional attention.
        # If users explicitly requested causal mode, keep it.
        if not force_causal:
            config.is_causal = False
            config.is_decoder = False
        if single_high:
            model = RecursiveRefinerSingleHighForSequenceClassification(config)
        else:
            model = RecursiveRefinerForSequenceClassification(config)
        model.vocab_size = config.vocab_size
        return model

    if force_causal:
        config.is_causal = True
        if single_high:
            model = RecursiveRefinerSingleHighForCausalLM(config)
        else:
            model = RecursiveRefinerForCausalLM(config)
    else:
        config.is_causal = False
        if single_high:
            model = RecursiveRefinerSingleHighForMaskedLM(config)
        else:
            model = RecursiveRefinerForMaskedLM(config)

    # Cramming expects a vocab_size attribute in some utilities.
    model.vocab_size = config.vocab_size
    return model
