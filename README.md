# Business Entity Resolution: ML Challenge 2026 (private)

An end-to-end solution for the Amazon ML Challenge business entity-resolution task.

**Start here:**
- [`CLAUDE.md`](CLAUDE.md): the complete handoff report (task, what was built, results, how to run, gotchas, next steps).
- [`code/business_entity_resolution/CLUSTER_RUN.md`](code/business_entity_resolution/CLUSTER_RUN.md): cluster setup, the one-command run, timings and troubleshooting.

## On a new machine / cluster

```bash
git clone <this-repo-url> er-entity-resolution
cd er-entity-resolution
# copy the organiser-provided folder next to code/ (NOT in git):
#   student_resource/dataset/{train,test}/*.tsv   and   student_resource/utils/validate_submission.py
cd code/business_entity_resolution
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py --check     # libraries, GPU backend, paths, 7/7 data files
bash run.sh                       # outputs -> ../../output/{matching_results,candidate_pairs}.tsv
```

## Layout

```
code/business_entity_resolution/   pipeline (src/), run.sh, run.ps1, requirements.txt, CLUSTER_RUN.md
docs/                              problem brief + EDA, research notes, methodology write-up
make_package.ps1                   builds <team>_submission.zip in the required structure
student_resource/                  (git-ignored) organiser data + validator, copied in manually
work/  output/                     (git-ignored) caches and results, created by the run
```

The challenge data is not in this repository: it is provided by the organisers and is too large for git (files of 175–509 MB). Keep this repository **private** while the competition is running.
