"""Candidate-local piecewise-static particle filtering."""

from .filter import BeliefStore
from .mala import MALAResult, mala_resample_move
from .types import ParticleSet, SMCStep
from importlib import import_module

__all__ = ["BeliefStore", "ParticleSet", "SMCStep", "MALAResult", "mala_resample_move"]

_NEURAL_EXPORTS = {
    "NeuralBeliefModel": ".model", "GaussianParams": ".model",
    "BeliefBatch": ".features", "SemanticEncoder": ".features",
    "fivo_future_loss": ".losses", "temperature_scale": ".losses",
}
__all__ += list(_NEURAL_EXPORTS)


def __getattr__(name):
    if name in _NEURAL_EXPORTS:
        value = getattr(import_module(_NEURAL_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
