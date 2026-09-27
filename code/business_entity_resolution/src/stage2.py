"""Stage 2: rich pair features + context (competition / consistency) features -> LightGBM matcher.

    python src/stage2.py features train|test   # rich + context features for the candidate set
    python src/stage2.py fit                   # 2-fold (by S1) models + out-of-fold p2 on train
    python src/stage2.py predict test          # p2 on test = mean of the fold models

Memory-safe: country files are processed one at a time; the fold models are trained on
an S1-level sample capped at MAX_TRAIN_ROWS rows.
"""
import argparse
import gc
import os
import pickle
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import config
from data import entity_ids, load_table, split_by_country
from features import RICH_EXTRA, CountryContext, rich_features
from models import group_rank, group_stats, importance, predict, train_lgb
from stage1_model import FEATS1, cand_path, is_fit_row

CTX = ["p1", "r1_all", "r1_src", "n1", "best1", "second1", "gap1", "n1_hi", "n1_src_hi",
       "rp", "np", "p_other", "p_margin"]
ANCHOR = ["anc_p1", "anc_name", "anc_addr", "anc_same_src"]
FEATS2 = FEATS1 + RICH_EXTRA + CTX + ANCHOR
MODEL2 = "stage2_lgb.pkl"
MAX_TRAIN_ROWS = int(os.environ.get("ER_MAX_TRAIN_ROWS", 7_000_000))


def feat_path(split, country):
    safe = "".join(ch for ch in country if ch.isalnum())
    return config.work_path("feats", f"{split}_{safe}.pkl")


def context_features(d, score_col="p1"):
    """Rank / competition features of a pair among its S1's candidates and among the S1s
    competing for the same pool record."""
    p = d[score_col].values.astype(np.float32)
    g1 = d["i1"].values.astype(np.int64)
    gs = g1 * 4 + d["src"].values
    gp = d["ip"].values.astype(np.int64)
    out = {}
    out["r1_all"] = group_rank(g1, p)
    out["r1_src"] = group_rank(gs, p)
    best1, second1, n1 = group_stats(g1, p)
    out["n1"], out["best1"], out["second1"] = n1, best1, second1
    out["gap1"] = best1 - p
    hi = (p >= 0.5).astype(np.float32)
    out["n1_hi"] = np.bincount(g1, weights=hi, minlength=g1.max() + 1)[g1].astype(np.float32)
    _, inv = np.unique(gs, return_inverse=True)
    out["n1_src_hi"] = np.bincount(inv, weights=hi)[inv].astype(np.float32)
    rp = group_rank(gp, p)
    bestp, secondp, npc = group_stats(gp, p)
    other = np.where(rp == 0, secondp, bestp)
    out["rp"], out["np"] = rp, npc
    out["p_other"] = other
    out["p_margin"] = p - other
    return pd.DataFrame(out, index=d.index)


def anchor_features(d, pool, score_col="p1"):
    """Consistency with the S1's most confident candidate (the 'anchor'; for the anchor
    itself, the runner-up): records of one real entity resemble each other."""
    p = d[score_col].values
    g1 = d["i1"].values.astype(np.int64)
    r = group_rank(g1, p)
    n = len(d)
    order = np.lexsort((-p, g1))
    gs = g1[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.flatnonzero(first)
    size = np.diff(np.r_[start, n])
    grp = np.cumsum(first) - 1
    top_row = order[start]                                  # anchor row per S1 group
    second_row = np.where(size > 1, order[np.minimum(start + 1, n - 1)], -1)
    anc = np.empty(n, np.int64)
    anc[order] = top_row[grp]
    sec = np.empty(n, np.int64)
    sec[order] = second_row[grp]
    anc = np.where(r == 0, sec, anc)                        # anchor for the top row = runner-up
    ok = anc >= 0
    ip = d["ip"].values
    out = {"anc_p1": np.full(n, -1, np.float32), "anc_name": np.zeros(n, np.uint8),
           "anc_addr": np.zeros(n, np.uint8), "anc_same_src": np.zeros(n, np.int8)}
    if ok.any():
        a_ip = ip[anc[ok]]
        out["anc_p1"][ok] = p[anc[ok]]
        out["anc_name"][ok] = cpdist(pool["core"].values[ip[ok]].tolist(), pool["core"].values[a_ip].tolist(),
                                     scorer=fuzz.token_set_ratio, workers=-1, dtype=np.uint8)
        out["anc_addr"][ok] = cpdist(pool["addr"].values[ip[ok]].tolist(), pool["addr"].values[a_ip].tolist(),
                                     scorer=fuzz.token_set_ratio, workers=-1, dtype=np.uint8)
        out["anc_same_src"][ok] = (d["src"].values[ok] == d["src"].values[anc[ok]]).astype(np.int8)
    return pd.DataFrame(out, index=d.index)


def build_features(split, chunk=2_000_000):
    for c in split_by_country(split):
        t = time.time()
        with open(cand_path(split, c), "rb") as f:
            blob = pickle.load(f)
        d = blob["cands"].reset_index(drop=True)
        d = d.drop(columns=[x for x in CTX + ANCHOR if x in d.columns and x != "p1"])
        s1, s1n = load_table(split, c, "s1")
        pool, pn = load_table(split, c, "pool")
        ctx = CountryContext(s1, s1n, pool, pn, with_tfidf=True)
        t1 = time.time()
        parts = []
        for lo in range(0, len(d), chunk):
            i1 = d["i1"].values[lo:lo + chunk].astype(np.int64)
            ip = d["ip"].values[lo:lo + chunk].astype(np.int64)
            parts.append(pd.DataFrame(rich_features(ctx, i1, ip)))
        rich = pd.concat(parts, ignore_index=True)
        d = pd.concat([d, rich, context_features(d), anchor_features(d, pool)], axis=1)
        for col in d.columns:
            if d[col].dtype == np.float64:
                d[col] = d[col].astype(np.float32)
        with open(feat_path(split, c), "wb") as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        fold = fold2(d["i1"].values)
        with open(feat_path(split, c) + ".meta", "wb") as f:
            pickle.dump({"n": len(d), "n_fold": [int((fold == 0).sum()), int((fold == 1).sum())]}, f)
        print(f"[stage2] features {split} {c}: {len(d):,} pairs, ctx {t1 - t:.0f}s, "
              f"feats {time.time() - t1:.0f}s", flush=True)
        del d, rich, parts, ctx, s1, pool
        gc.collect()


def fold2(i1):
    return ((i1 // 10) % 2).astype(np.int8)


def _pred_frame(d, split, country, p2, with_label):
    s1_ids, pool_ids, _ = entity_ids(split, country)
    cols = {"s1_id": s1_ids[d["i1"].values], "pool_id": pool_ids[d["ip"].values], "country": country,
            "i1": d["i1"].values, "ip": d["ip"].values, "src": d["src"].values, "p1": d["p1"].values,
            "p2": p2.astype(np.float32)}
    if with_label:
        cols["label"] = d["label"].values
    return pd.DataFrame(cols)


def fit(rounds=1200):
    t = time.time()
    countries = split_by_country("train")
    rng = np.random.default_rng(config.SEED)
    metas = {c: pickle.load(open(feat_path("train", c) + ".meta", "rb")) for c in countries}
    frac = [min(1.0, MAX_TRAIN_ROWS / max(sum(m["n_fold"][k] for m in metas.values()), 1)) for k in (0, 1)]
    samples = {0: [], 1: []}
    for c in countries:
        d = pd.read_pickle(feat_path("train", c))
        f = fold2(d["i1"].values)
        for k in (0, 1):
            m = (f == k) & (rng.random(len(d)) < frac[k])
            samples[k].append(d.loc[m, FEATS2 + ["label"]])
        del d
        gc.collect()
    models = []
    params = {"num_leaves": 255, "learning_rate": 0.04, "min_data_in_leaf": 100, "feature_fraction": 0.7}
    for k in (0, 1):
        tr = pd.concat(samples[1 - k], ignore_index=True)        # model k is trained on the other fold
        m = train_lgb(tr[FEATS2], tr["label"].values, params, rounds)
        models.append(m)
        print(f"[stage2] model {k} (trained on fold {1 - k}, {len(tr):,} rows) {time.time() - t:.0f}s", flush=True)
        del tr
        gc.collect()
    del samples
    print(importance(models[0], 30).round(4).to_string(), flush=True)
    with open(config.work_path(MODEL2), "wb") as fh:
        pickle.dump(models, fh)
    outs = []
    for c in countries:
        d = pd.read_pickle(feat_path("train", c))
        f = fold2(d["i1"].values)
        oof = np.zeros(len(d), np.float32)
        for k in (0, 1):
            m = f == k
            oof[m] = predict(models[k], d.loc[m, FEATS2])
        outs.append(_pred_frame(d, "train", c, oof, True))
        del d
        gc.collect()
    pd.concat(outs, ignore_index=True).to_pickle(config.work_path("oof_train.pkl"))
    print(f"[stage2] fit done {time.time() - t:.0f}s", flush=True)


def predict_split(split):
    with open(config.work_path(MODEL2), "rb") as f:
        models = pickle.load(f)
    outs = []
    for c in split_by_country(split):
        d = pd.read_pickle(feat_path(split, c))
        p = np.mean([predict(m, d[FEATS2]) for m in models], axis=0)
        outs.append(_pred_frame(d, split, c, p, False))
        del d
        gc.collect()
    out = pd.concat(outs, ignore_index=True)
    out.to_pickle(config.work_path(f"pred_{split}.pkl"))
    print(f"[stage2] predicted {split}: {len(out):,} pairs", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "fit", "predict"])
    ap.add_argument("split", nargs="?", default="train")
    a = ap.parse_args()
    if a.cmd == "features":
        build_features(a.split)
    elif a.cmd == "fit":
        fit()
    else:
        predict_split(a.split)
