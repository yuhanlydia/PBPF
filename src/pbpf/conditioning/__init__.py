"""Optional Torch actor conditioning; importing this package needs no ML stack."""

from importlib import import_module

_EXPORTS = {
    "SoftPrefixProjector": ".soft_prompt",
    "LowRankKVProjector": ".kv_delta",
    "whole_sequence_mixture_loss": ".mixture",
    "sample_components_once": ".mixture",
    "serial_mixture_backward": ".mixture",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
