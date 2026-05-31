"""
Model registry for EMtrain.

Each architecture exposes a `build(model_config: dict)` function that returns
a `(model, loss, output_keys)` triple, where:

- model: torch.nn.Module
- loss: torch.nn.Module compatible with the gunpowder Train node's
  `loss_inputs` mapping (kwargs by name)
- output_keys: ordered list of strings naming the model's forward outputs.
  These names are the canonical handles used by `train.py` to allocate
  gunpowder ArrayKeys and wire the Train node's `outputs` and `loss_inputs`.

To add a new architecture, drop a module in this package that exports a
`build(model_config)` function and register it in `_REGISTRY` below.
"""

from .affs_lsd import build as _build_affs_lsd
from .affs_lsd_sngp import build as _build_affs_lsd_sngp


_REGISTRY = {
    'affs_lsd':      _build_affs_lsd,
    'affs_lsd_sngp': _build_affs_lsd_sngp,
}


def build_model(model_config):
    """
    Build a model from a model_config dict.

    Required key: `architecture` — one of the registered names. If absent,
    defaults to `affs_lsd` to preserve the historical behavior.

    Returns (model, loss, output_keys).
    """
    arch = model_config.get('architecture', 'affs_lsd')
    if arch not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture {arch!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[arch](model_config)


def available_architectures():
    return sorted(_REGISTRY)
