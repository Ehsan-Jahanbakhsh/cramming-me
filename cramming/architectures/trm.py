"""Construction helpers for the TRM masked language model."""

from __future__ import annotations

from typing import Optional

from omegaconf import OmegaConf

from .trm_hf import TRMConfig, TRMForMaskedLM, TRMForSequenceClassification


def construct_trm(cfg_arch, vocab_size: int, downstream_classes: Optional[int] = None):
    cfg = OmegaConf.to_container(cfg_arch, resolve=True)
    cfg.pop("architectures", None)
    cfg["vocab_size"] = int(vocab_size)

    config = TRMConfig(**cfg)
    if downstream_classes is not None:
        config.num_labels = int(downstream_classes)
        model = TRMForSequenceClassification(config)
    else:
        model = TRMForMaskedLM(config)

    model.vocab_size = config.vocab_size
    return model
