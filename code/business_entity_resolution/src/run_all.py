"""End-to-end pipeline: raw TSVs -> output/candidate_pairs.tsv + output/matching_results.tsv.

    python src/run_all.py                          # every step, in order
    python src/run_all.py --from stage2_fit        # resume from a step (earlier outputs are cached)
    python src/run_all.py --check                  # only check environment + data, run nothing

Each step runs in its own process so memory is released between steps. Blocking
(pairs_*) additionally resumes at chunk level after an interruption.
"""
import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

STEPS = [
    ("preprocess", ["preprocess.py", "train", "test"]),
    ("split_countries", ["-c", "from data import split_by_country as s; s('train'); s('test')"]),
    ("translit", ["translit.py"]),
    ("train_ranker", ["train_ranker.py", "--frac", "0.03"]),
    ("pairs_train", ["stage1_pairs.py", "train"]),
    ("pairs_test", ["stage1_pairs.py", "test"]),
    ("stage1_fit", ["stage1_model.py", "fit"]),
    ("stage1_select_train", ["stage1_model.py", "select", "train"]),
    ("stage1_select_test", ["stage1_model.py", "select", "test"]),
    ("submission_stage1", ["make_submission.py", "--score-col", "p1"]),   # early safety-net output
    ("stage2_features_train", ["stage2.py", "features", "train"]),
    ("stage2_features_test", ["stage2.py", "features", "test"]),
    ("stage2_fit", ["stage2.py", "fit"]),
    ("stage2_predict_test", ["stage2.py", "predict", "test"]),
    ("stage3", ["stage3.py"]),
    ("submission", ["make_submission.py", "--score-col", "p3"]),
    ("validate", None),
]


def check_env():
    import config
    print(f"python      {sys.version.split()[0]}")
    for mod in ("numpy", "pandas", "scipy", "sklearn", "rapidfuzz", "unidecode", "lightgbm", "xgboost"):
        try:
            m = importlib.import_module(mod)
            print(f"{mod:<11} {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"{mod:<11} MISSING" + ("" if mod in ("lightgbm", "xgboost") else "  <-- required"))
    from models import backend
    print(f"backend     {backend()}  (xgb = XGBoost on CUDA GPU, lgb = LightGBM on CPU)")
    print(f"cpu workers {config.N_JOBS}")
    print(f"DATA_DIR    {config.DATA_DIR}")
    print(f"WORK_DIR    {config.WORK_DIR}")
    print(f"OUTPUT_DIR  {config.OUTPUT_DIR}")
    missing = [config.source_path(s, k) for s in config.SPLITS for k in config.SOURCES
               if not os.path.exists(config.source_path(s, k))]
    if not os.path.exists(config.gt_path()):
        missing.append(config.gt_path())
    if missing:
        print("MISSING DATA FILES:\n  " + "\n  ".join(missing))
        return False
    print("data files  OK (7/7)")
    return True


def validate():
    import config
    v = os.environ.get("ER_VALIDATOR") or os.path.join(os.path.dirname(config.DATA_DIR), "utils",
                                                       "validate_submission.py")
    if not os.path.exists(v):
        print(f"validator not found at {v} (set ER_VALIDATOR); skipping")
        return
    subprocess.run([sys.executable, v,
                    "--matching", os.path.join(config.OUTPUT_DIR, "matching_results.tsv"),
                    "--candidate", os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv"),
                    "--test-dir", os.path.join(config.DATA_DIR, "test")], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None, help="step name to start from")
    ap.add_argument("--check", action="store_true", help="only check environment and data")
    args = ap.parse_args()
    ok = check_env()
    if args.check or not ok:
        sys.exit(0 if ok else 1)
    names = [n for n, _ in STEPS]
    first = names.index(args.start) if args.start else 0
    t0 = time.time()
    for name, cmd in STEPS[first:]:
        t = time.time()
        print(f"=== {name}  [{time.strftime('%H:%M:%S')}]", flush=True)
        if cmd is None:
            validate()
        else:
            subprocess.run([sys.executable, *cmd], cwd=SRC, check=True)
        print(f"=== {name} done in {time.time() - t:.0f}s", flush=True)
    print(f"=== ALL DONE in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
