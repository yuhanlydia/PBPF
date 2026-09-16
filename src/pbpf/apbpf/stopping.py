"""Development-locked stopping rules on public remaining-test information."""
import numpy as np


def stopped_budgets(remaining_information, threshold):
    values = np.asarray(remaining_information, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('three finite nonnegative public-information prefixes required')
    if threshold is None:
        return np.full(len(values), 4, dtype=int)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('invalid stopping threshold')
    crossed = np.concatenate((values <= threshold, np.ones((len(values), 1), dtype=bool)), axis=1)
    return crossed.argmax(1) + 1


def lock_threshold(remaining_information, future_nll, thresholds):
    """Use calibration sources only; None is the fixed-four fallback."""
    nll = np.asarray(future_nll, dtype=float)
    if nll.shape != (len(remaining_information), 4) or not len(nll) or not np.isfinite(nll).all():
        raise ValueError('finite calibration NLLs at four prefixes required')
    fixed = nll[:, -1].mean()
    choices = []
    for threshold in (*thresholds, None):
        budgets = stopped_budgets(remaining_information, threshold)
        quality = nll[np.arange(len(nll)), budgets - 1].mean()
        choices.append({'threshold': threshold, 'mean_tests': float(budgets.mean()),
                        'nll': float(quality), 'matched_quality': bool(quality <= fixed)})
    eligible = [row for row in choices if row['matched_quality']]
    chosen = min(eligible, key=lambda row: (row['mean_tests'], row['nll'],
                                          -1 if row['threshold'] is None else row['threshold']))
    return {'threshold': chosen['threshold'], 'calibration_curve': choices,
            'rule': 'minimum mean tests subject to calibration NLL <= same-policy fixed-four; no tolerance'}
