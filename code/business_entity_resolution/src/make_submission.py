"""Final step: tune the decision layer on out-of-fold train predictions, apply it to test,
and write output/matching_results.tsv + output/candidate_pairs.tsv.

    python src/make_submission.py [--score-col p1|p2|p3]
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

import config
from data import entity_ids, split_by_country
from decide import apply_params, macro_f05_vec, tune
from preprocess import load_gt, read_tsv
from stage1_model import cand_path, is_fit_row


def eval_frame(oof, gt):
    """Restrict to honest S1s (not used to fit stage 1) and index them 0..n-1."""
    s1_all = []
    for c in split_by_country("train"):
        ids = entity_ids("train", c)[0]
        s1_all.append(pd.DataFrame({"s1_id": ids, "i1": np.arange(len(ids)), "country": c}))
    s1_all = pd.concat(s1_all, ignore_index=True)
    s1_all = s1_all[~is_fit_row(s1_all["i1"].values)].reset_index(drop=True)
    s1_all["g"] = np.arange(len(s1_all))
    n_true = np.array([len(gt.get(s, ())) for s in s1_all["s1_id"].values], dtype=np.float64)
    d = oof.merge(s1_all[["s1_id", "g"]], on="s1_id", how="inner")
    return d, n_true, len(s1_all), s1_all


def write_lists(path, col, s1_ids, lists):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s in s1_ids:
            ids = list(dict.fromkeys(lists.get(s, ())))
            f.write(f"{s}\t{','.join(ids)}\n")


def apply(d, params, score_col):
    p = d[score_col].values.astype(np.float64)
    gpool = pd.factorize(d["pool_id"])[0]
    g1 = pd.factorize(d["s1_id"])[0]
    return apply_params(g1, gpool, d["src"].values, p, params)


def stage1_frames():
    """(train OOF frame, test frame) built from the stage-1 candidate sets (score = p1)."""
    frames = {}
    for split in ("train", "test"):
        parts = []
        for c in split_by_country(split):
            blob = pickle.load(open(cand_path(split, c), "rb"))
            cd = blob["cands"]
            s1, pool, _ = entity_ids(split, c)
            cols = {"s1_id": s1[cd["i1"].values], "pool_id": pool[cd["ip"].values], "country": c,
                    "i1": cd["i1"].values, "ip": cd["ip"].values, "src": cd["src"].values, "p1": cd["p1"].values}
            if "label" in cd:
                cols["label"] = cd["label"].values
            parts.append(pd.DataFrame(cols))
        frames[split] = pd.concat(parts, ignore_index=True)
    return frames["train"], frames["test"]


def main(score_col="p2"):
    gt = load_gt()
    if score_col == "p1":
        oof, pred_test = stage1_frames()
        pred_test.to_pickle(config.work_path("pred_test_p1.pkl"))
    else:
        oof = pd.read_pickle(config.work_path("oof_train.pkl"))
    d, n_true, n_groups, s1_all = eval_frame(oof, gt)
    g1 = d["g"].values
    gpool = pd.factorize(d["pool_id"])[0]
    best_score, params = tune(g1, gpool, d["src"].values, d[score_col].values.astype(np.float64),
                              d["label"].values.astype(np.float64), n_true, n_groups)
    # per-country breakdown with the chosen params (over ALL honest S1, incl. those without candidates)
    sel = apply_params(g1, gpool, d["src"].values, d[score_col].values.astype(np.float64), params)
    _, per_s1 = macro_f05_vec(g1, sel, d["label"].values.astype(np.float64), n_true, n_groups)
    print(f"[decide] OOF macro-F0.5 = {best_score:.5f} with {params}", flush=True)
    for c in s1_all["country"].unique():
        gs = s1_all.loc[s1_all["country"] == c, "g"].values
        print(f"   {c}: {per_s1[gs].mean():.5f}  (n={len(gs):,})", flush=True)
    sing = n_true == 0
    print(f"   singletons: {per_s1[sing].mean():.4f} (n={sing.sum():,})  "
          f"non-singletons: {per_s1[~sing].mean():.4f}", flush=True)
    json.dump({"score": best_score, "params": params, "score_col": score_col},
              open(config.work_path("decision.json"), "w"), indent=1)

    # ---- test
    pred = pd.read_pickle(config.work_path("pred_test_p1.pkl" if score_col == "p1" else "pred_test.pkl"))
    sel_t = apply(pred, params, score_col)
    matches = pred.loc[sel_t].groupby("s1_id")["pool_id"].apply(list).to_dict()
    cands = {}
    for c in split_by_country("test"):
        blob = pickle.load(open(cand_path("test", c), "rb"))
        cd = blob["cands"]
        s1_ids, pool_ids, _ = entity_ids("test", c)
        tmp = pd.DataFrame({"s1_id": s1_ids[cd["i1"].values], "pool_id": pool_ids[cd["ip"].values]})
        cands.update(tmp.groupby("s1_id")["pool_id"].apply(list).to_dict())
    test_s1 = read_tsv(config.source_path("test", 1))["entity_id"].tolist()
    for s, ms in matches.items():                       # guarantee matches are a subset of candidates
        assert set(ms) <= set(cands.get(s, ())), s
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    write_lists(os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv"), "candidate_entity_ids", test_s1, cands)
    write_lists(os.path.join(config.OUTPUT_DIR, "matching_results.tsv"), "matched_entity_ids", test_s1, matches)
    n_c = np.mean([len(cands.get(s, ())) for s in test_s1])
    n_m = np.mean([len(matches.get(s, ())) for s in test_s1])
    empty = np.mean([len(matches.get(s, ())) == 0 for s in test_s1])
    print(f"[submit] test S1={len(test_s1):,} mean candidates={n_c:.2f} mean matches={n_m:.2f} "
          f"empty rows={empty:.3%}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-col", default="p2")
    main(ap.parse_args().score_col)
