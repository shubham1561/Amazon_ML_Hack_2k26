"""Stage 1b: LightGBM pruner on cheap features -> final candidate set (candidate_pairs.tsv).

    python src/stage1_model.py fit                 # trains on 10 % of train S1 (i1 % 10 == 0)
    python src/stage1_model.py sweep               # recall / size trade-off of (K, p_min) on train
    python src/stage1_model.py select train|test   # keep top-K per (S1, source) with p1 >= p_min

The 10 % of S1 used to fit this model are excluded from all later validation
(their p1 is in-sample); every other train S1 gets an honest out-of-sample p1.
"""
import argparse
import gc
import os
import pickle
import time

import numpy as np
import pandas as pd

import config
from blocking import KEY_BITS
from data import entity_ids, split_by_country
from features import CHEAP
from models import expand_keymask, group_rank, importance, predict, train_lgb
from stage1_pairs import iter_chunks, meta_info

FEATS1 = [f for f in CHEAP if f != "keymask"] + ["meta", "quick"] + [n for _, n in KEY_BITS]
MODEL1 = "stage1_lgb.pkl"
K_PER_SRC = int(os.environ.get("ER_K_PER_SRC", 8))
P_MIN = float(os.environ.get("ER_P_MIN", 0.003))


def is_fit_row(i1):
    return (np.asarray(i1) % 10) == 0


def cand_path(split, country):
    safe = "".join(ch for ch in country if ch.isalnum())
    return config.work_path("cands", f"{split}_{safe}.pkl")


def fit():
    t = time.time()
    frames = []
    rng = np.random.default_rng(config.SEED)
    for c in split_by_country("train"):
        for d in iter_chunks("train", c):
            m = is_fit_row(d["i1"].values) & (rng.random(len(d)) < 0.5)
            frames.append(expand_keymask(d[m].copy())[FEATS1 + ["label"]])
            del d
    tr = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"[stage1] fit rows={len(tr):,} pos={tr['label'].mean():.4f}", flush=True)
    model = train_lgb(tr[FEATS1], tr["label"].values, {"num_leaves": 63, "learning_rate": 0.1}, rounds=300)
    with open(config.work_path(MODEL1), "wb") as f:
        pickle.dump(model, f)
    print(importance(model, 15).round(4).to_string(), flush=True)
    print(f"[stage1] fit done {time.time() - t:.0f}s", flush=True)


def _scored_chunks(split, c, model):
    for d in iter_chunks(split, c):
        d = expand_keymask(d)
        d["p1"] = predict(model, d[FEATS1])
        g = d["i1"].values.astype(np.int64) * 4 + d["src"].values
        d["r1_src"] = group_rank(g, d["p1"].values)
        yield d


def sweep(ks=(3, 4, 5, 6, 8, 10), pmins=(0.0, 0.002, 0.005, 0.01, 0.02, 0.05)):
    """Pair recall (vs. all true pairs of honest S1s) and mean candidates per S1."""
    from preprocess import load_gt
    with open(config.work_path(MODEL1), "rb") as f:
        model = pickle.load(f)
    gt = load_gt()
    rows = []
    n_s1 = n_true = 0
    for c in split_by_country("train"):
        s1 = entity_ids("train", c)[0]
        honest = ~is_fit_row(np.arange(len(s1)))
        n_s1 += int(honest.sum())
        n_true += sum(len(gt.get(s, ())) for s in s1[honest])
        for d in _scored_chunks("train", c, model):
            d = d[(~is_fit_row(d["i1"].values)) & (d["r1_src"].values < max(ks))]
            rows.append(d[["p1", "r1_src", "label"]])
    a = pd.concat(rows, ignore_index=True)
    print(f"[stage1] sweep over {n_s1:,} honest S1, {n_true:,} true pairs", flush=True)
    for k in ks:
        line = []
        for pm in pmins:
            m = (a["r1_src"].values < k) & (a["p1"].values >= pm)
            line.append(f"p>={pm:<5}: R={a['label'].values[m].sum() / n_true:.4f} C={m.sum() / n_s1:5.2f}")
        print(f"  K={k:2d} | " + " | ".join(line), flush=True)


def select(split, k_per_src=K_PER_SRC, p_min=P_MIN):
    with open(config.work_path(MODEL1), "rb") as f:
        model = pickle.load(f)
    for c in split_by_country(split):
        t = time.time()
        keep = []
        for d in _scored_chunks(split, c, model):
            m = (d["r1_src"].values < k_per_src) & (d["p1"].values >= p_min)
            keep.append(d[m])
        cand = pd.concat(keep, ignore_index=True)
        mi = meta_info(split, c)
        with open(cand_path(split, c), "wb") as f:
            pickle.dump({"cands": cand, "n_s1": mi["n_s1"], "n_pool": mi["n_pool"]}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        msg = f"{len(cand):,} candidates, {len(cand) / mi['n_s1']:.2f} per S1"
        if "label" in cand:
            msg += f", true kept={int(cand['label'].sum()):,}"
        print(f"[stage1] select {split} {c}: {msg} ({time.time() - t:.0f}s)", flush=True)
        del keep, cand
        gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "sweep", "select"])
    ap.add_argument("split", nargs="?", default="train")
    a = ap.parse_args()
    {"fit": fit, "sweep": sweep}.get(a.cmd, lambda: select(a.split))()
