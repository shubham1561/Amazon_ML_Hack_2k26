"""Stage 3: re-rank with competition and cluster-consistency features computed from the
stage-2 probabilities (Foursquare-style iterative refinement + TransClean-style consistency).

For every candidate of an S1:
  * competition from p2: rank within the S1 / source, gap to the best, best competing S1
    claim on the same pool record, margin;
  * consistency: best name / address similarity to the S1's *other* confident candidates
    (p2 >= 0.5) and to its top candidate: records of one real entity resemble each other.
The stage-2 features are kept, so stage 3 is a full re-ranker.

    python src/stage3.py          # OOF p3 on train (2-fold by S1) + p3 on test
"""
import gc
import pickle
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import config
from data import load_table, split_by_country
from models import group_rank, importance, predict, train_lgb
from stage2 import FEATS2, context_features, feat_path

CONS = ["c_n_conf", "c_best_name", "c_best_addr", "c_mean_name", "c_top_name", "c_top_addr", "c_top_p2"]
MODEL3 = "stage3_lgb.pkl"


def consistency(d, pool, p, conf=0.5, max_sib=6):
    """Similarity of each candidate to its S1's other confident candidates (by p)."""
    n = len(d)
    g1 = d["i1"].values.astype(np.int64)
    ip = d["ip"].values
    r = group_rank(g1, p)
    order = np.lexsort((-p, g1))
    gs = g1[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.flatnonzero(first)
    grp_sorted = np.cumsum(first) - 1
    size = np.diff(np.r_[start, n])
    grp = np.empty(n, np.int64)
    grp[order] = grp_sorted
    core, addr = pool["core"].values, pool["addr"].values
    out = {k: np.zeros(n, np.float32) for k in CONS}
    out["c_top_p2"][:] = -1
    best_n = np.zeros(n, np.float32)
    best_a = np.zeros(n, np.float32)
    sum_n = np.zeros(n, np.float32)
    cnt = np.zeros(n, np.float32)
    for j in range(max_sib):                         # j-th best sibling of the group
        has = size[grp] > j
        sib_row = np.full(n, -1, np.int64)
        sib_row[has] = order[start[grp[has]] + j]
        ok = has & (sib_row != np.arange(n)) & (p[np.maximum(sib_row, 0)] >= conf)
        if not ok.any():
            continue
        rows = np.flatnonzero(ok)
        a_ip, b_ip = ip[rows], ip[sib_row[rows]]
        sn = cpdist(core[a_ip].tolist(), core[b_ip].tolist(), scorer=fuzz.token_set_ratio,
                    workers=-1, dtype=np.float32)
        sa = cpdist(addr[a_ip].tolist(), addr[b_ip].tolist(), scorer=fuzz.token_set_ratio,
                    workers=-1, dtype=np.float32)
        best_n[rows] = np.maximum(best_n[rows], sn)
        best_a[rows] = np.maximum(best_a[rows], sa)
        sum_n[rows] += sn
        cnt[rows] += 1
        if j == 0:
            out["c_top_name"][rows], out["c_top_addr"][rows] = sn, sa
    # the top row's own "top" reference is the runner-up (j == 1)
    top_is_self = r == 0
    has2 = top_is_self & (size[grp] > 1)
    rows = np.flatnonzero(has2)
    if len(rows):
        sib = order[start[grp[rows]] + 1]
        out["c_top_name"][rows] = cpdist(core[ip[rows]].tolist(), core[ip[sib]].tolist(),
                                         scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        out["c_top_addr"][rows] = cpdist(addr[ip[rows]].tolist(), addr[ip[sib]].tolist(),
                                         scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        out["c_top_p2"][rows] = p[sib]
    rows = np.flatnonzero(~top_is_self)
    out["c_top_p2"][rows] = p[order[start[grp[rows]]]]
    out["c_n_conf"], out["c_best_name"], out["c_best_addr"] = cnt, best_n, best_a
    out["c_mean_name"] = np.where(cnt > 0, sum_n / np.maximum(cnt, 1), -1).astype(np.float32)
    return pd.DataFrame(out, index=d.index)


def build(split, p2_frame):
    """Stage-3 table for one split: stage-2 features + p2 + competition + consistency."""
    parts = []
    off = 0
    for c in split_by_country(split):
        t = time.time()
        d = pd.read_pickle(feat_path(split, c))
        n = len(d)
        pf = p2_frame.iloc[off:off + n]
        off += n
        assert (pf["i1"].values == d["i1"].values).all() and (pf["ip"].values == d["ip"].values).all()
        cols = list(dict.fromkeys(FEATS2 + ["i1", "ip", "src"] + (["label"] if "label" in d else [])))
        d = d[cols].copy()
        d["p2"] = pf["p2"].values
        q = context_features(d, score_col="p2").add_prefix("q_")
        pool, _ = load_table(split, c, "pool")
        cons = consistency(d, pool, d["p2"].values)
        del pool
        d = pd.concat([d, q, cons], axis=1)
        d["s1_id"], d["pool_id"], d["country"] = pf["s1_id"].values, pf["pool_id"].values, c
        parts.append(d)
        print(f"[stage3] {split} {c}: {n:,} rows ({time.time() - t:.0f}s)", flush=True)
        gc.collect()
    return pd.concat(parts, ignore_index=True)


def fold3(i1):
    return ((i1 // 20) % 2).astype(np.int8)


def main(rounds=1200):
    t = time.time()
    oof = pd.read_pickle(config.work_path("oof_train.pkl"))
    tr = build("train", oof)
    del oof
    feats = FEATS2 + ["p2"] + [c for c in tr.columns if c.startswith("q_")] + CONS
    f = fold3(tr["i1"].values)
    p3 = np.zeros(len(tr), np.float32)
    models = []
    params = {"num_leaves": 255, "learning_rate": 0.04, "min_data_in_leaf": 100, "feature_fraction": 0.7}
    for k in (0, 1):
        m = train_lgb(tr.loc[f != k, feats], tr.loc[f != k, "label"].values, params, rounds)
        p3[f == k] = predict(m, tr.loc[f == k, feats])
        models.append(m)
        print(f"[stage3] fold {k} {time.time() - t:.0f}s", flush=True)
    print(importance(models[0], 25).round(4).to_string(), flush=True)
    keep = ["s1_id", "pool_id", "country", "i1", "ip", "src", "p2"]
    out = tr[keep + ["label"]].copy()
    out["p1"] = tr["p1"].values
    out["p3"] = p3
    out.to_pickle(config.work_path("oof_train.pkl"))
    del tr
    gc.collect()
    pred = pd.read_pickle(config.work_path("pred_test.pkl"))
    te = build("test", pred)
    del pred
    te_out = te[keep].copy()
    te_out["p1"] = te["p1"].values
    te_out["p3"] = np.mean([predict(m, te[feats]) for m in models], axis=0).astype(np.float32)
    te_out.to_pickle(config.work_path("pred_test.pkl"))
    with open(config.work_path(MODEL3), "wb") as fh:
        pickle.dump({"models": models, "feats": feats}, fh)
    print(f"[stage3] done {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
