#!/usr/bin/env bash
# Build <team_name>_submission.zip in the structure required by the organisers.
#   bash make_package.sh <team_name> [output_dir]      (output_dir defaults to ./output)
set -euo pipefail
cd "$(dirname "$0")"
TEAM="${1:?usage: bash make_package.sh <team_name> [output_dir]}"
OUT="${2:-output}"
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution/src"
cp "$OUT/matching_results.tsv" "$OUT/candidate_pairs.tsv" "$STAGE/output/"
cp code/business_entity_resolution/src/*.py "$STAGE/code/business_entity_resolution/src/"
cp code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt \
   "$STAGE/code/business_entity_resolution/"
cp docs/Documentation_template.md "$STAGE/Documentation_template.md"
ZIP="$(pwd)/${TEAM}_submission.zip"
rm -f "$ZIP"
(cd "$STAGE" && zip -qr "$ZIP" .)
rm -rf "$STAGE"
ls -lh "$ZIP"
