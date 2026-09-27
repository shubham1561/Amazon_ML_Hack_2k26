"""Decision layer: turn calibrated pair probabilities into per-S1 match sets.

1. Ownership: a S2/S3 record belongs to at most one S1 (true for 100 % of train
   labels), so each pool record keeps only its best-scoring S1 claim (optionally
   also claims within `margin` of the best).
2. Per-S1 selection: take every candidate with p >= tau_hi; if none qualifies,
   take the single best one when p >= tau_lo (an empty prediction only pays off
   for singletons, 5.6 % of S1). Caps: <= 5 S2 and <= 6 S3 per S1 (train maxima).
3. Thresholds are tuned on out-of-fold train predictions with the exact
   macro-F0.5, computed in vectorised form.
"""
import numpy as np

from models import group_rank


def ownership_mask(gpool, p, margin=0.0):
    """Keep a claim only if it is the best claim on that pool record (within `margin`)."""
    order = np.lexsort((-p, gpool))
    g, s = gpool[order], p[order]
    first = np.r_[True, g[1:] != g[:-1]]
    start = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    best = s[start][grp]
    keep = np.empty(len(p), bool)
    keep[order] = s >= best - margin
    return keep


class Selector:
    """Precomputes per-S1 ranks for one eligibility mask; thresholds are then cheap."""

    def __init__(self, g1, src, p, eligible=None, cap2=5, cap3=6):
        self.g1 = g1
        self.elig = np.ones(len(p), bool) if eligible is None else eligible
        self.pe = np.where(self.elig, p, -1.0)
        self.r_all = group_rank(g1, self.pe)
        r_src = group_rank(g1 * 4 + src.astype(np.int64), self.pe)
        self.in_cap = r_src < np.where(src == 2, cap2, cap3)
        self.n_groups = int(g1.max()) + 1 if len(g1) else 1

    def mask(self, tau_lo, tau_hi):
        hi = self.elig & (self.pe >= tau_hi) & self.in_cap
        has_hi = np.zeros(self.n_groups, bool)
        has_hi[self.g1[hi]] = True
        top = self.elig & (self.r_all == 0) & (self.pe >= tau_lo) & ~has_hi[self.g1]
        return hi | top

    def mask_ranked(self, taus):
        """Rank-specific thresholds: the r-th best candidate of an S1 (0-based) is kept when
        p >= taus[min(r, len(taus) - 1)]; with non-decreasing taus the kept set is a prefix."""
        t = np.asarray(taus, dtype=np.float64)[np.minimum(self.r_all, len(taus) - 1)]
        return self.elig & self.in_cap & (self.pe >= t)


def select_mask(g1, src, p, tau_lo, tau_hi, cap2=5, cap3=6, eligible=None):
    return Selector(g1, src, p, eligible, cap2, cap3).mask(tau_lo, tau_hi)


def macro_f05_vec(g1, sel, label, n_true, n_groups):
    """Exact macro-F0.5. g1 in [0, n_groups); n_true: true-match count per S1 group
    (including matches missing from the candidate set)."""
    tp = np.bincount(g1[sel], weights=label[sel], minlength=n_groups)
    fp = np.bincount(g1[sel], weights=1 - label[sel], minlength=n_groups)
    fn = n_true - tp
    denom = 1.25 * tp + 0.25 * fn + fp
    f = np.where(tp > 0, 1.25 * tp / np.maximum(denom, 1e-9), 0.0)
    f = np.where((n_true == 0) & (fp == 0), 1.0, f)
    return float(f.mean()), f


def tune(g1, gpool, src, p, label, n_true, n_groups, verbose=True):
    """Grid-search ownership margin and (tau_lo, tau_hi) on OOF predictions."""
    best = (-1, None)
    for margin in (None, 0.0, 0.02, 0.1):
        elig = None if margin is None else ownership_mask(gpool, p, margin)
        sel = Selector(g1, src, p, elig)
        for tau_hi in (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
            for tau_lo in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
                if tau_lo > tau_hi:
                    continue
                score, _ = macro_f05_vec(g1, sel.mask(tau_lo, tau_hi), label, n_true, n_groups)
                if score > best[0]:
                    best = (score, {"margin": margin, "tau_lo": tau_lo, "tau_hi": tau_hi})
        if verbose:
            print(f"    margin={margin}: best so far {best[0]:.5f} {best[1]}", flush=True)
    # refinement: rank-specific thresholds around the best (tau_lo, tau_hi)
    b = best[1]
    elig = None if b["margin"] is None else ownership_mask(gpool, p, b["margin"])
    sel = Selector(g1, src, p, elig)
    grid = np.round(np.arange(0.02, 0.96, 0.02), 2)
    taus = [b["tau_lo"], b["tau_hi"], b["tau_hi"]]
    for _ in range(2):                                   # coordinate ascent, 2 sweeps
        for i in range(3):
            lo = taus[i - 1] if i > 0 else 0.0
            hi = taus[i + 1] if i < 2 else 1.0
            for t in grid[(grid >= lo) & (grid <= hi)]:
                cand = list(taus)
                cand[i] = float(t)
                score, _ = macro_f05_vec(g1, sel.mask_ranked(cand), label, n_true, n_groups)
                if score > best[0]:
                    best = (score, {"margin": b["margin"], "taus": cand})
                    taus = cand
    if verbose:
        print(f"    rank-specific: {best[0]:.5f} {best[1]}", flush=True)
    return best


def apply_params(g1, gpool, src, p, params):
    elig = None if params["margin"] is None else ownership_mask(gpool, p, params["margin"])
    sel = Selector(g1, src, p, elig)
    if "taus" in params:
        return sel.mask_ranked(params["taus"])
    return sel.mask(params["tau_lo"], params["tau_hi"])
