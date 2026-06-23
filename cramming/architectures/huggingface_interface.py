"""Hugging Face model variations."""

import transformers
from omegaconf import OmegaConf


def _build_hf_config(cfg_arch, downstream_classes=None):
    """Build a HF config from a Hydra architecture config.

    Historically this repo treated every HF-style YAML as BERT. For paper
    baselines we also want ALBERT-style parameter sharing, selected explicitly by
    ``model_type: albert`` or an ``Albert...`` architecture name.
    """

    if isinstance(cfg_arch, transformers.PretrainedConfig):
        configuration = cfg_arch
    else:
        config_dict = OmegaConf.to_container(cfg_arch, resolve=True)
        arch_names = config_dict.get("architectures", []) or []
        model_type = str(config_dict.get("model_type", "")).lower()

        if model_type == "albert" or any(str(name).startswith("Albert") for name in arch_names):
            configuration = transformers.AlbertConfig(**config_dict)
        else:
            configuration = transformers.BertConfig(**config_dict)

    if downstream_classes is not None:
        configuration.num_labels = downstream_classes
        if hasattr(configuration, "arch"):
            configuration.arch["num_labels"] = downstream_classes

    return configuration


def construct_huggingface_model(cfg_arch, vocab_size, downstream_classes=None):
    """construct model from given configuration. Only works if this arch exists on the hub."""
    if downstream_classes is None:
        configuration = _build_hf_config(cfg_arch)
        configuration.pad_token_id = None  # Need to drop this during pretraining, otherwise leads to a graph break in a HF warning
        configuration.vocab_size = vocab_size
        model = transformers.AutoModelForMaskedLM.from_config(configuration)
        model.vocab_size = model.config.vocab_size
    else:
        configuration = _build_hf_config(cfg_arch, downstream_classes=downstream_classes)
        configuration.vocab_size = vocab_size

        configuration.problem_type = None  # always reset this!
        model = transformers.AutoModelForSequenceClassification.from_config(configuration)
        model.vocab_size = vocab_size
    return model
