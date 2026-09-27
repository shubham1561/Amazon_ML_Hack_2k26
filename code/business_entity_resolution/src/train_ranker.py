"""Learn the blocking-cap ranker (v2): a small GBDT on the fast similarity features of the
pre-capped union, replacing the hand-set formula that orders candidates before the
per-(S1, source) cap. Trained on a small S1 sample of train with ground truth.

    python src/train_ranker.py [--frac 0.03]
"""
import argparse
import gc
import os
import pickle
import time

import numpy as np
import pandas as pd

import config
from blocking import CountryBlocker
from data import load_table, split_by_country
from models import group_rank, importance, train_lgb
from preprocess import load_gt
from stage1_pairs import gt_codes


def collect(frac, chunk=100_000):
    gt = load_gt()
    rng = np.random.default_rng(config.SEED + 7)
    frames = []
    for c in split_by_country("train"):
        t = time.time()
        s1, s1n = load_table("train", c, "s1")
        pool, pn = load_table("train", c, "pool")
        sel = np.zeros(len(s1), bool)
        sel[rng.choice(len(s1), int(len(s1) * frac), replace=False)] = True
        b = CountryBlocker(s1, s1n, pool, pn, {"no_ranker": 1})
        codes = gt_codes(s1, pool, gt)
        b.truth = codes[sel[codes // len(pool)]]
        b.dump = []
        n_true = len(b.truth)
        for lo in range(0, len(s1), chunk):
            hi = min(lo + chunk, len(s1))
            if sel[lo:hi].any():
                b.block(lo, hi, sel)
        d = pd.concat([pd.DataFrame(x) for x in b.dump], ignore_index=True)
        d["n_true_country"] = n_true
        d["country"] = c
        # S1 id for grouping: recover from the order of dumps is not needed; use a running key
        frames.append(d)
        print(f"[ranker] {c}: {len(d):,} rows, {int(d['label'].sum()):,}/{n_true:,} true in pre-cap "
              f"({time.time() - t:.0f}s)", flush=True)
        del b, s1, pool
        gc.collect()
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.03)
    args = ap.parse_args()
    frames = collect(args.frac)
    d = pd.concat(frames, ignore_index=True)
    d.to_pickle(config.work_path("ranker_data.pkl"))
    feats = CountryBlocker.RANK_FEATS
    tr = (d["i1"].values % 10) < 7                       # split by S1
    m = train_lgb(d.loc[tr, feats], d.loc[tr, "label"].values,
                  {"num_leaves": 63, "learning_rate": 0.1, "min_data_in_leaf": 200}, 300)
    print(importance(m, 12).round(4).to_string(), flush=True)
    # held-out comparison: recall inside the pre-cap union at the per-(S1, source) cap
    ev = d[~tr].copy()
    ev["formula"] = ev["q_name"] + 0.8 * ev["q_addr"] + 20.0 * np.minimum(ev["q_meta"], 3.0)
    ev["ranker"] = m.predict_np(ev[feats].to_numpy(np.float32))
    cc = pd.factorize(ev["country"])[0].astype(np.int64)
    g = (cc << 40) | (ev["i1"].values.astype(np.int64) << 2) | ev["q_src"].values.astype(np.int64)
    tot = ev["label"].sum()
    for col in ("formula", "ranker"):
        r = group_rank(g, ev[col].values)
        print(col, " ".join(f"@{k}={ev['label'].values[r < k].sum() / tot:.4f}" for k in (10, 20, 30, 40)),
              flush=True)
    with open(config.work_path("ranker.pkl"), "wb") as f:
        pickle.dump(m, f)
    print("[ranker] saved", flush=True)
