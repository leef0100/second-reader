#!/usr/bin/env bash
# Split → cache → train → eval on NIH ChestX-ray14. Resumable: re-run to continue.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/experiments/cxr_nih_baseline.yaml
SPL=datasets/splits/nih-v1
CACHE=data/cache/nih-v1
[ -f $SPL/manifest.sha256 ] || uv run secondreader cxr split datasets/manifests/nih_cxr14.csv --out datasets/splits --version nih-v1 --salt second-reader-nih-v1
uv run secondreader cxr cache --splits $SPL --out $CACHE --config $CFG --tiers train,val,test
uv run secondreader cxr train --config $CFG --splits $SPL --registry models --cache $CACHE --resume --notes "NIH ChestX-ray14 baseline v1 (research use only)"
MID=$(ls -t models | grep -v -E "pending|ACTIVE" | head -1)
uv run secondreader cxr eval "$MID" --tier test --splits $SPL --registry models --out runs --n-boot 200
echo "NIH_BASELINE_DONE model=$MID"
