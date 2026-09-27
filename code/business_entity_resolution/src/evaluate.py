"""Exact competition metric: macro-averaged F0.5 over Source-1 entities (singletons included)."""
import numpy as np


def f05(pred, true):
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    fp, fn = len(pred - true), len(true - pred)
    return 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)


def macro_f05(pred, truth, ids=None):
    """pred/truth: dict s1_id -> iterable of ids. ids: the S1 ids to average over (default: truth keys)."""
    ids = list(truth.keys()) if ids is None else list(ids)
    scores = np.array([f05(set(pred.get(s, ())), set(truth.get(s, ()))) for s in ids])
    return float(scores.mean()), scores


def pair_recall(cands, truth, ids):
    tot = hit = 0
    for s in ids:
        t = set(truth.get(s, ()))
        tot += len(t)
        hit += len(t & set(cands.get(s, ())))
    return hit / max(tot, 1)


assert abs(f05({"a", "b", "c"}, {"a", "c"}) - 0.7142857) < 1e-6
