"""Query-conditioned outcome propagation, a deterministic association control.

This retains paired evidence until the future query is known. It provides no
latent bug identities, posterior diagnostic entropy, or particle inference.
"""

import math
import numpy as np


def kernel_predict(tests, visible_outcomes, *, visible_steps, alpha, temperature, mix):
    """Return [candidate,future_test,5] probabilities without future labels.

    Weights mix uniform history counts with cosine-kernel attention. Dirichlet
    pseudocount alpha prevents zero probabilities for unseen outcome classes.
    """
    x = np.asarray(tests)
    y = np.asarray(visible_outcomes)
    if (x.ndim != 3 or type(visible_steps) is not int or not 0 < visible_steps < x.shape[1]
            or y.shape != (x.shape[0], visible_steps)):
        raise ValueError('features and visible-only outcomes must have compatible shapes')
    if (not np.isfinite(x).all() or not np.issubdtype(y.dtype, np.integer)
            or np.any((y < 0) | (y >= 5))):
        raise ValueError('finite features and valid integer outcome classes required')
    if (not all(math.isfinite(v) for v in (alpha, temperature, mix))
            or alpha <= 0 or temperature <= 0 or not 0 <= mix <= 1):
        raise ValueError('positive alpha/temperature and mixture in [0,1] required')
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    scores = np.einsum('btf,bsf->bts', x[:, visible_steps:], x[:, :visible_steps]) / temperature
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    weights = mix * weights + (1 - mix) / visible_steps
    counts = visible_steps * np.einsum('bts,bsc->btc', weights, np.eye(5)[y])
    return (counts + alpha) / (visible_steps + 5 * alpha)
