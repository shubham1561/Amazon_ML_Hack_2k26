"""Paths and global settings. Override the data/work/output dirs with environment variables."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # code/business_entity_resolution
_ROOT = os.path.dirname(os.path.dirname(_PKG))     # submission root

DATA_DIR = os.environ.get("ER_DATA_DIR", os.path.join(_ROOT, "student_resource", "dataset"))
WORK_DIR = os.environ.get("ER_WORK_DIR", os.path.join(_ROOT, "work"))
OUTPUT_DIR = os.environ.get("ER_OUTPUT_DIR", os.path.join(_ROOT, "output"))

N_JOBS = int(os.environ.get("ER_N_JOBS", max(1, (os.cpu_count() or 2) - 2)))
SEED = 42

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def source_path(split, src):
    return os.path.join(DATA_DIR, split, f"{split}_source{src}.tsv")


def gt_path():
    return os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")


def work_path(*parts):
    p = os.path.join(WORK_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
