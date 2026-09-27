"""Stage 0: read the raw TSVs, normalise every record (multi-process) and cache to work/.

    python src/preprocess.py            # all splits
    python src/preprocess.py test       # one split
"""
import csv
import gc
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import config
from normalize import normalize_records

COLS = ["core", "legal", "alias", "indic", "addr", "nums", "unit", "pobox"]


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       keep_default_na=False, na_filter=False, encoding="utf-8")


def _work(args):
    names, addrs = args
    return normalize_records(names, addrs)


def normalise_frame(df, n_jobs, chunk=100_000):
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    tasks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    out = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for res in ex.map(_work, tasks):
            out.extend(res)
    norm = pd.DataFrame(out, columns=COLS)
    norm["indic"] = norm["indic"].astype(np.int8)
    return norm


def norm_path(split, src):
    return config.work_path(f"norm_{split}_s{src}.pkl")


def run_split(split):
    for src in config.SOURCES:
        path = norm_path(split, src)
        if os.path.exists(path):
            print(f"[preprocess] {split} s{src}: cached", flush=True)
            continue
        t = time.time()
        raw = read_tsv(config.source_path(split, src))
        norm = normalise_frame(raw, config.N_JOBS)
        norm.insert(0, "entity_id", raw["entity_id"].values)
        norm.insert(1, "country", raw["country"].values)
        norm.insert(2, "raw_name", raw["business_name"].values)
        norm.insert(3, "raw_addr", raw["business_address"].values)
        norm["src"] = np.int8(src)
        with open(path, "wb") as f:
            pickle.dump(norm, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[preprocess] {split} s{src}: {len(norm):,} rows in {time.time() - t:.0f}s", flush=True)
        del raw, norm
        gc.collect()


def load_gt():
    path = config.work_path("gt.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    gt = read_tsv(config.gt_path())
    mapping = {s: (m.split(",") if m else []) for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}
    with open(path, "wb") as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    return mapping


def load_norm(split, src):
    with open(norm_path(split, src), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    splits = sys.argv[1:] or list(config.SPLITS)
    for sp in splits:
        run_split(sp)
    if "train" in splits:
        load_gt()
    print("[preprocess] done", flush=True)
