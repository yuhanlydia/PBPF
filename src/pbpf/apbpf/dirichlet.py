"""Exploratory statistical-feedback primitives, not a validated repair policy.

Weighted and discounted counts are heuristic evidence updates. Neither is
claimed to be an exact Bayesian posterior for correlated program executions.
"""
import numpy as np


def _categories(outcomes, classes):
    values = np.asarray(outcomes)
    if (type(classes) is not int or classes < 2 or values.ndim != 1
            or (values.size and (not np.issubdtype(values.dtype, np.integer)
                                or (values < 0).any() or (values >= classes).any()))):
        raise ValueError('outcomes must be integer category indices')
    return values.astype(np.int64)


def predict_rate(outcomes, alpha, *, weights=None, classes=5):
    """Smoothed outcome counts; supplied weights change evidence, not priors."""
    values = _categories(outcomes, classes)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive and finite')
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != values.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('weights must be matching finite nonnegative values')
    posterior = np.bincount(values, weights=weights, minlength=classes) + alpha
    return posterior / posterior.sum()


def similarity_weights(query, history, strength):
    """Cosine kernel with fixed total evidence, nested at strength zero."""
    query, history = np.asarray(query, dtype=float), np.asarray(history, dtype=float)
    if (query.ndim != 1 or history.ndim != 2 or history.shape[1] != len(query)
            or not len(history) or not len(query) or not np.isfinite(query).all()
            or not np.isfinite(history).all() or not np.isfinite(strength) or strength < 0):
        raise ValueError('invalid feature shapes or kernel strength')
    norms = np.linalg.norm(history, axis=1) * np.linalg.norm(query)
    similarity = np.divide(history @ query, norms, out=np.zeros(len(history)), where=norms > 0)
    logits = strength * np.clip(similarity, -1, 1)
    weights = np.exp(logits - logits.max())
    return len(history) * weights / weights.sum()


def effective_evidence_weights(weights):
    """Heuristic ESS mass: normalized weights times 1/sum(p**2).

    This discounts weight concentration, not execution correlation, and is not
    an exact posterior or a calibrated count of independent observations.
    """
    values = np.asarray(weights, dtype=float)
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or (values < 0).any() or values.max() <= 0):
        raise ValueError('weights must be a nonempty nonnegative vector with positive mass')
    scaled = values / values.max()
    probabilities = scaled / scaled.sum()
    return probabilities / np.square(probabilities).sum()


def inherit_counts(parent_counts, child_outcomes, retention):
    """Discount parent evidence once at a code change; never inherit alpha."""
    parent = np.asarray(parent_counts, dtype=float)
    if (parent.ndim != 1 or len(parent) < 2 or not np.isfinite(parent).all()
            or (parent < 0).any() or not np.isfinite(retention) or not 0 <= retention <= 1):
        raise ValueError('invalid evidence or retention')
    values = _categories(child_outcomes, len(parent))
    return retention * parent + np.bincount(values, minlength=len(parent))


def allocate_rollouts(counts, budget, observe, *, seed):
    """Posterior-sampling allocation with one real observation per action.

    Counts describe previous *repair attempts*, not tests of unchanged code.
    observe(candidate_index) performs one budgeted action and returns its public
    outcome category (PASS=0). Hidden evaluator labels must never be returned.
    This is a standard heuristic baseline, not a claim of a new bandit algorithm.
    """
    counts = np.array(counts, dtype=float, copy=True)
    if (counts.ndim != 2 or counts.shape[0] < 1 or counts.shape[1] < 2
            or not np.isfinite(counts).all() or (counts < 0).any()
            or type(budget) is not int or budget < 0):
        raise ValueError('invalid repair counts or budget')
    rng, choices = np.random.default_rng(seed), []
    for _ in range(budget):
        index = int(np.argmax([rng.dirichlet(row + 1)[0] for row in counts]))
        outcome = _categories([observe(index)], counts.shape[1])[0]
        counts[index, outcome] += 1
        choices.append(index)
    return choices, counts
