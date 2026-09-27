"""Gradient-boosting helpers shared by stage 1 (pruner), stage 2 (matcher) and stage 3.

Backend: XGBoost on the GPU (device="cuda", Apache-2.0) when available, otherwise
LightGBM on the CPU (MIT). Force one with ER_BACKEND=xgb|lgb. Parameters are written
LightGBM-style and translated for XGBoost.
"""
import os

import numpy as np
import pandas as pd

import config
from blocking import KEY_BITS

_BACKEND = None


def backend():
    global _BACKEND
    if _BACKEND is None:
        want = os.environ.get("ER_BACKEND", "auto")
        _BACKEND = "lgb"
        if want in ("auto", "xgb"):
            try:
                import xgboost as xgb
                X = np.random.rand(256, 4).astype(np.float32)
                xgb.train({"tree_method": "hist", "device": "cuda", "verbosity": 0},
                          xgb.DMatrix(X, label=(X[:, 0] > .5).astype(int)), 2)
                _BACKEND = "xgb"
            except Exception as e:          # no GPU / no xgboost -> CPU LightGBM
                if want == "xgb":
                    raise
                print(f"[models] XGBoost-CUDA unavailable ({e}); using LightGBM CPU", flush=True)
    return _BACKEND


class Model:
    def __init__(self, kind, booster, names):
        self.kind, self.booster, self.names = kind, booster, names

    def predict_np(self, X):
        if self.kind == "xgb":
            return self.booster.inplace_predict(X)
        return self.booster.predict(X, num_threads=config.N_JOBS)

    def importance(self):
        if self.kind == "xgb":
            s = pd.Series(self.booster.get_score(importance_type="total_gain"), dtype=float)
            return s.reindex(self.names).fillna(0.0)
        return pd.Series(self.booster.feature_importance("gain"), index=self.names)


def expand_keymask(df):
    km = df["keymask"].values
    for bit, name in KEY_BITS:
        df[name] = ((km & bit) > 0).astype(np.uint8)
    return df


def _xgb_params(p):
    q = {"objective": "binary:logistic", "tree_method": "hist", "device": "cuda", "max_bin": 256,
         "eta": p.get("learning_rate", 0.08), "subsample": p.get("bagging_fraction", 0.8),
         "colsample_bytree": p.get("feature_fraction", 0.8), "lambda": p.get("lambda_l2", 1.0),
         "min_child_weight": max(1.0, p.get("min_data_in_leaf", 100) / 20.0),
         "grow_policy": "lossguide", "max_leaves": p.get("num_leaves", 127), "max_depth": 0,
         "seed": config.SEED, "verbosity": 0}
    return q


def train_lgb(X, y, params=None, rounds=400, w=None):
    """Train a binary classifier on DataFrame X (name kept for backward compatibility)."""
    p = {"objective": "binary", "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 200,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
         "max_bin": 255, "num_threads": config.N_JOBS, "verbose": -1, "seed": config.SEED}
    if params:
        p.update(params)
    names = list(X.columns)
    Xn = X.to_numpy(dtype=np.float32)
    if backend() == "xgb":
        import xgboost as xgb
        dm = xgb.QuantileDMatrix(Xn, label=y, weight=w, feature_names=names, max_bin=256)
        del Xn
        return Model("xgb", xgb.train(_xgb_params(p), dm, num_boost_round=rounds), names)
    import lightgbm as lgb
    ds = lgb.Dataset(Xn, label=y, weight=w, feature_name=names, free_raw_data=True)
    return Model("lgb", lgb.train(p, ds, num_boost_round=rounds), names)


def predict(model, X, chunk=2_000_000):
    out = np.empty(len(X), dtype=np.float32)
    for lo in range(0, len(X), chunk):
        out[lo:lo + chunk] = model.predict_np(X.iloc[lo:lo + chunk].to_numpy(dtype=np.float32))
    return out


def group_rank(keys, score):
    """Rank (0 = best) of each row within its group `keys` by descending score."""
    order = np.lexsort((-score, keys))
    k = keys[order]
    first = np.r_[True, k[1:] != k[:-1]]
    start = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    rank_sorted = np.arange(len(k)) - start[grp]
    rank = np.empty(len(k), dtype=np.int32)
    rank[order] = rank_sorted
    return rank


def group_stats(keys, score):
    """Per-row: max score of group, second max of group, group size."""
    order = np.lexsort((-score, keys))
    k, s = keys[order], score[order]
    first = np.r_[True, k[1:] != k[:-1]]
    start = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    size = np.diff(np.r_[start, len(k)])
    best = s[start]
    second = np.where(size > 1, s[np.minimum(start + 1, len(s) - 1)], -1.0)
    out_best = np.empty(len(k), np.float32)
    out_second = np.empty(len(k), np.float32)
    out_size = np.empty(len(k), np.int32)
    out_best[order] = best[grp]
    out_second[order] = second[grp]
    out_size[order] = size[grp]
    return out_best, out_second, out_size


def importance(model, top=40):
    imp = model.importance()
    return (imp / max(imp.sum(), 1e-12)).sort_values(ascending=False).head(top)
