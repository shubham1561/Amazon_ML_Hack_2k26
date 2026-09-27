"""Per-country record tables built from the normalised caches.

A "table" is a DataFrame for one (split, country) holding either the S1 records
or the pool (S2 + S3) records, plus derived arrays used by blocking/features.
"""
import gc
import os
import pickle
import re

import numpy as np
import pandas as pd

import config
from normalize import _ocr_fix, skeleton
from preprocess import load_norm

KEEP = ["entity_id", "country", "raw_name", "raw_addr", "core", "legal", "alias", "indic",
        "addr", "nums", "unit", "pobox", "src"]


def _country_path(split, country, side):
    safe = "".join(ch for ch in country if ch.isalnum()) or "unknown"
    return config.work_path("country", f"{split}_{safe}_{side}.pkl")


def split_by_country(split):
    """Write per-country S1 and pool pickles (idempotent). Returns sorted country list."""
    marker = config.work_path("country", f"{split}_countries.pkl")
    if os.path.exists(marker):
        with open(marker, "rb") as f:
            return pickle.load(f)
    countries = set()
    parts = {}
    for src in config.SOURCES:
        df = load_norm(split, src)[KEEP]
        for c, g in df.groupby("country", sort=False):
            countries.add(c)
            side = "s1" if src == 1 else "pool"
            parts.setdefault((c, side), []).append(g.reset_index(drop=True))
        del df
        gc.collect()
    for (c, side), frames in parts.items():
        tab = pd.concat(frames, ignore_index=True)
        with open(_country_path(split, c, side), "wb") as f:
            pickle.dump(tab, f, protocol=pickle.HIGHEST_PROTOCOL)
    countries = sorted(countries)
    with open(marker, "wb") as f:
        pickle.dump(countries, f)
    return countries


def entity_ids(split, country):
    """(S1 ids, pool ids, pool src) arrays for one country, cached in a light pickle."""
    path = _country_path(split, country, "ids")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    # rebuilt from the raw files (2 columns only); same row order as split_by_country:
    # S1 rows of the country in file order; pool = S2 rows then S3 rows, file order
    parts = []
    for src in config.SOURCES:
        raw = pd.read_csv(config.source_path(split, src), sep="\t", dtype=str, quoting=3,
                          keep_default_na=False, usecols=["entity_id", "country"])
        ids = raw.loc[raw["country"].values == country, "entity_id"].values.astype(object)
        parts.append(ids)
        del raw
    s1_ids = parts[0]
    pool_ids = np.concatenate(parts[1:])
    pool_src = np.concatenate([np.full(len(parts[1]), 2, np.int8), np.full(len(parts[2]), 3, np.int8)])
    out = (s1_ids, pool_ids, pool_src)
    gc.collect()
    with open(path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_table(split, country, side, keep_raw=False):
    """Returns (DataFrame, nums_matrix). Derived columns are computed once and cached."""
    path = _country_path(split, country, side)
    with open(path, "rb") as f:
        tab = pickle.load(f)
    fresh = "dver" not in tab.columns or tab["dver"].iat[0] != DERIVED_VERSION
    tab, nums = add_derived(tab)
    if fresh:
        with open(path, "wb") as f:
            pickle.dump(tab.drop(columns=["nospace"]), f, protocol=pickle.HIGHEST_PROTOCOL)
    if not keep_raw:
        tab = tab.drop(columns=["raw_name", "raw_addr"])
    apply_translit(tab)
    gc.collect()
    return tab, nums


def apply_translit(tab):
    """Replace romanised Indic words by their learned Latin spelling (skeleton kept as is)."""
    import translit
    from normalize import strip_indic_legal
    mapping = translit.load()
    idx = np.flatnonzero(tab["indic"].values == 1)
    if len(idx) == 0:
        return tab
    core = tab["core"].values.copy()
    srt = tab["sorted_core"].values.copy()
    get = mapping.get
    for i in idx:
        toks = [get(t, t) for t in strip_indic_legal(core[i].split())]
        core[i] = " ".join(toks)
        srt[i] = " ".join(sorted(set(toks)))
    tab["core"], tab["sorted_core"] = core, srt
    tab["nospace"] = tab["core"].str.replace(" ", "", regex=False)
    return tab


def parse_nums(nums, width=3):
    """'2442 7 12' -> int64 matrix (n, width) padded with -1; long digit runs keep last 9 digits."""
    out = np.full((len(nums), width), -1, dtype=np.int64)
    for i, s in enumerate(nums):
        if not s:
            continue
        for j, t in enumerate(s.split()[:width]):
            out[i, j] = int(t[-9:])
    return out


DERIVED_VERSION = 2
_RE_DIGIT = re.compile(r"\d")


def _refix(s):
    return " ".join(_ocr_fix(t) for t in s.split()) if s and _RE_DIGIT.search(s) else s


def _derive_chunk(args):
    cores, aliases = args
    cores = [_refix(c) for c in cores]
    aliases = [_refix(a) for a in aliases]
    skel = [skeleton(c) if c else "" for c in cores]
    srt = [" ".join(sorted(set(c.split()))) for c in cores]
    asrt = [" ".join(sorted(set(a.split()))) if a else "" for a in aliases]
    return cores, aliases, skel, srt, asrt


def add_derived(tab):
    if "dver" not in tab.columns or tab["dver"].iat[0] != DERIVED_VERSION:
        from concurrent.futures import ProcessPoolExecutor
        cores, aliases = tab["core"].tolist(), tab["alias"].tolist()
        step = 200_000
        tasks = [(cores[i:i + step], aliases[i:i + step]) for i in range(0, len(cores), step)]
        cols = [[], [], [], [], []]
        with ProcessPoolExecutor(max_workers=min(6, config.N_JOBS)) as ex:
            for res in ex.map(_derive_chunk, tasks):
                for acc, part in zip(cols, res):
                    acc.extend(part)
        for name, vals in zip(["core", "alias", "skel", "sorted_core", "alias_sorted"], cols):
            tab[name] = vals
        tab["dver"] = np.int8(DERIVED_VERSION)
    tab["nospace"] = tab["core"].str.replace(" ", "", regex=False)
    return tab, parse_nums(tab["nums"].values)
