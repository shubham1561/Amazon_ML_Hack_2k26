"""Stage 1a runner: blocking + cheap pair features for every country of a split.
Chunks of S1 rows are streamed to disk (work/pairs/<split>_<country>/cNNNN.pkl) to keep RAM flat.

    python src/stage1_pairs.py train [--frac 0.1]     # --frac: dev subset of S1 rows
    python src/stage1_pairs.py test
"""
import argparse
import gc
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd

import config
from blocking import CountryBlocker, KEY_BITS
from data import load_table, split_by_country
from features import CountryContext, cheap_features
from models import group_rank
from preprocess import load_gt


def gt_codes(s1, pool, gt):
    """int64 codes (i1 * npool + ip) of the true pairs of this country."""
    pidx = pd.Index(pool["entity_id"].values)
    rows, cols = [], []
    for i, sid in enumerate(s1["entity_id"].values):
        ms = gt.get(sid)
        if ms:
            rows.extend([i] * len(ms))
            cols.extend(ms)
    ip = pidx.get_indexer(cols)
    rows = np.asarray(rows, dtype=np.int64)
    ok = ip >= 0
    return np.sort(rows[ok] * len(pool) + ip[ok])


def pairs_dir(split, country, tag=""):
    safe = "".join(ch for ch in country if ch.isalnum())
    d = config.work_path("pairs", f"{split}_{safe}{tag}", "x")
    return os.path.dirname(d)


def iter_chunks(split, country, tag=""):
    for p in sorted(glob.glob(os.path.join(pairs_dir(split, country, tag), "c*.pkl"))):
        yield pd.read_pickle(p)


def meta_info(split, country, tag=""):
    with open(os.path.join(pairs_dir(split, country, tag), "meta.pkl"), "rb") as f:
        return pickle.load(f)


def run_country(split, country, gt=None, frac=None, chunk=100_000, cfg=None, tag=""):
    t0 = time.time()
    out_dir = pairs_dir(split, country, tag)
    if os.path.exists(os.path.join(out_dir, "meta.pkl")) and not frac and not os.environ.get("ER_FORCE"):
        print(f"  [{country}] cached, skipping", flush=True)
        return meta_info(split, country, tag)["info"]
    if frac or os.environ.get("ER_FORCE"):
        for p in glob.glob(os.path.join(out_dir, "*.pkl")):
            os.remove(p)
    s1, s1_nums = load_table(split, country, "s1")
    pool, pool_nums = load_table(split, country, "pool")
    sel = np.ones(len(s1), bool)
    if frac:
        rng = np.random.default_rng(config.SEED)
        sel[:] = False
        sel[rng.choice(len(s1), int(len(s1) * frac), replace=False)] = True
    blocker = CountryBlocker(s1, s1_nums, pool, pool_nums, cfg)
    ctx = CountryContext(s1, s1_nums, pool, pool_nums)
    codes = gt_codes(s1, pool, gt) if gt is not None else None
    if codes is not None:
        codes = codes[sel[codes // len(pool)]]
        blocker.truth = codes
    print(f"  [{country}] tables+indexes {time.time() - t0:.0f}s  s1={len(s1):,} pool={len(pool):,}", flush=True)

    t1 = time.time()
    n_pairs = 0
    hits = {"all": 0, **{n: 0 for _, n in KEY_BITS}, **{f"q{k}": 0 for k in (5, 10, 20, 30, 40)}}
    for lo in range(0, len(s1), chunk):
        hi = min(lo + chunk, len(s1))
        if not sel[lo:hi].any():
            continue
        chunk_file = os.path.join(out_dir, f"c{lo // chunk:04d}.pkl")
        if os.path.exists(chunk_file):              # resume after an interruption
            n_pairs += len(pd.read_pickle(chunk_file))
            continue
        i1, ip, mask, meta, quick = blocker.block(lo, hi, sel if frac else None)
        tf = time.time()
        f = cheap_features(ctx, i1, ip, mask)
        blocker._stat("t_cheap", time.time() - tf)
        f["meta"] = meta.astype(np.float32)
        f["quick"] = quick.astype(np.float32)
        df = pd.DataFrame(f)
        df.insert(0, "i1", i1.astype(np.int32))
        df.insert(1, "ip", ip.astype(np.int32))
        if codes is not None:
            pc = i1 * len(pool) + ip
            lab = np.isin(pc, codes)
            df["label"] = lab.astype(np.int8)
            hits["all"] += int(lab.sum())
            for bit, name in KEY_BITS:
                hits[name] += int(lab[(mask & bit) > 0].sum())
            r = group_rank(i1 * 4 + df["src"].values, quick)
            for k in (5, 10, 20, 30, 40):
                hits[f"q{k}"] += int(lab[r < k].sum())
        df.to_pickle(chunk_file + ".tmp")
        os.replace(chunk_file + ".tmp", chunk_file)     # atomic: no half-written chunks
        n_pairs += len(df)
        del df, f
        gc.collect()
    info = {"country": country, "n_s1": int(sel.sum()), "n_pool": len(pool), "n_pairs": n_pairs,
            "pairs_per_s1": n_pairs / max(sel.sum(), 1), "secs": round(time.time() - t0)}
    st = blocker.stats or {}
    info.update({k: round(v, 1) for k, v in st.items() if k.startswith("t_")})
    info["union_per_s1"] = round(st.get("n_union", 0) / max(sel.sum(), 1), 1)
    info["pre_per_s1"] = round(st.get("n_pre", 0) / max(sel.sum(), 1), 1)
    if codes is not None:
        n_true = max(len(codes), 1)
        info["n_true"] = len(codes)
        info["R_union"] = round(st.get("hit_union", 0) / n_true, 4)
        info["R_pre"] = round(st.get("hit_pre", 0) / n_true, 4)
        info.update({f"R_{k}": round(v / n_true, 4) for k, v in hits.items()})
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as fh:
        pickle.dump({"n_s1": len(s1), "n_pool": len(pool), "info": info}, fh)
    print(f"  [{country}] block+features {time.time() - t1:.0f}s :: {info}", flush=True)
    del blocker, ctx, s1, pool
    gc.collect()
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cfg", default="", help="blocker overrides, e.g. pre_cap=100,addr_cap=150")
    args = ap.parse_args()
    cfg = {k: int(v) for k, v in (kv.split("=") for kv in args.cfg.split(",") if kv)}
    countries = split_by_country(args.split)
    gt = load_gt() if args.split == "train" else None
    for c in (args.countries or countries):
        run_country(args.split, c, gt, args.frac, cfg=cfg, tag=args.tag)
