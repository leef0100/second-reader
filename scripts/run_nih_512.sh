#!/usr/bin/env bash
# 512 px NIH experiment: cache(512) → train → eval(test). Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/experiments/cxr_nih_512.yaml; SPL=datasets/splits/nih-v1; CACHE=data/cache/nih-v1-512
uv run secondreader cxr cache --splits $SPL --out $CACHE --config $CFG --tiers train,val,test
uv run secondreader cxr train --config $CFG --splits $SPL --registry models --cache $CACHE --resume --notes "NIH 512px experiment (research use only)"
MID=$(ls -t models | grep -v -E "pending|ACTIVE" | head -1)
uv run secondreader cxr eval "$MID" --tier test --splits $SPL --registry models --out runs --n-boot 200
echo "NIH_512_DONE model=$MID"
