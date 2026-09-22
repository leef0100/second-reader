#!/usr/bin/env bash
# Download the gate-evaluation sets that need no credentials (see datasets/OOD_EVAL_PLAN.md):
#
#   kermany_pediatric   Mendeley Data "ZhangLabData" v3 (CC BY 4.0), one 8.4 GB zip that also
#                       carries the OCT set; only chest_xray/ is extracted.
#                       datasets/LICENSES/kermany_pediatric.md
#   nlm_tb_shenzhen     NLM/LHNCBC Shenzhen Hospital TB set — 662 PNG + clinical readings,
#   nlm_tb_montgomery   Montgomery County TB set — 138 PNG + readings + lung masks.
#                       Served per file (no zip); each folder has an index.json with Key + Size.
#                       datasets/LICENSES/nlm_tb_shenzhen_montgomery.md
#
# Resumable: re-running skips complete files (size match), resumes partial ones (curl -C -),
# verifies every size against the value the server advertises (index.json / the pinned zip
# size), extracts once and never re-extracts. Safe to Ctrl-C at any point.
#
# Usage:
#   scripts/download_ood_sets.sh [--root DIR] [--only kermany|nlm] [--jobs N] [--keep-zip]
#   nohup scripts/download_ood_sets.sh > data/raw/ood_download.log 2>&1 &
#
# Layout produced (ROOT defaults to data/raw):
#   $ROOT/kermany_pediatric/ZhangLabData.zip            (deleted after extraction unless --keep-zip)
#   $ROOT/kermany_pediatric/chest_xray/{train,test}/{NORMAL,PNEUMONIA}/*.jpeg
#   $ROOT/nlm_tb/Shenzhen-Hospital-CXR-Set/{CXR_png,ClinicalReadings,NLM-ChinaCXRSet-ReadMe.docx,...}
#   $ROOT/nlm_tb/Montgomery-County-CXR-Set/MontgomerySet/{CXR_png,ClinicalReadings,ManualMask,...}
#
# data/ is git-ignored. Never commit images. For research use only. Not for diagnostic use.
set -euo pipefail

ROOT="data/raw"
ONLY=""
JOBS=4
KEEP_ZIP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --keep-zip) KEEP_ZIP=1; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- Kermany paediatric (Mendeley Data, verified 2026-09-21; see LICENSES/kermany_pediatric.md)
KERMANY_URL="https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/810b2ce2-11c3-4424-996e-3bef36600907/file_downloaded"
KERMANY_BYTES=8441154111
KERMANY_EXPECTED_JPEG=5856
KERMANY_LICENCE_API="https://data.mendeley.com/public-api/datasets/rscbjbr9sj?version=3"

# ---- NLM TB sets (per-file HTTPS; verified 2026-09-21; see LICENSES/nlm_tb_shenzhen_montgomery.md)
NLM_BASE="https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"
# Folders to mirror (index.json in each lists files with Key + Size). Top-level set folders
# carry the README (licence terms) and consensus-ROI CSVs; subfolders carry images/readings.
NLM_DIRS=(
  "Shenzhen-Hospital-CXR-Set"
  "Shenzhen-Hospital-CXR-Set/CXR_png"
  "Shenzhen-Hospital-CXR-Set/ClinicalReadings"
  "Montgomery-County-CXR-Set/MontgomerySet"
  "Montgomery-County-CXR-Set/MontgomerySet/CXR_png"
  "Montgomery-County-CXR-Set/MontgomerySet/ClinicalReadings"
  "Montgomery-County-CXR-Set/MontgomerySet/ManualMask/leftMask"
  "Montgomery-County-CXR-Set/MontgomerySet/ManualMask/rightMask"
)
NLM_EXPECTED_PNG_SHENZHEN=662
NLM_EXPECTED_PNG_MONTGOMERY=138

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
file_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }  # GNU first (BSD -f is fs info on Linux)

# fetch_sized URL DEST BYTES → resumable download until the size matches (max 8 attempts)
fetch_sized() {
  local url="$1" dest="$2" bytes="$3" attempt=0 have
  local curl_cmd=(curl -sS -L --fail --retry 8 --retry-delay 10 --retry-all-errors --speed-limit 5000 --speed-time 120)
  mkdir -p "$(dirname "$dest")"
  while :; do
    have=$( [[ -f "$dest" ]] && file_size "$dest" || echo 0 )
    if [[ "$have" == "$bytes" ]]; then return 0; fi
    if [[ "$have" -gt "$bytes" ]]; then log "oversize      $dest ($have > $bytes) — deleting"; rm -f "$dest"; fi
    attempt=$((attempt + 1))
    if [[ $attempt -gt 8 ]]; then log "FAILED        $dest after 8 attempts (have $have/$bytes)"; return 1; fi
    "${curl_cmd[@]}" -C - -o "$dest" "$url" || sleep 15
  done
}
# count_mismatch LIST → number of listed files whose on-disk size differs from the advertised one
count_mismatch() {
  local n=0 url dest bytes
  while IFS=$'\t' read -r url dest bytes; do
    [[ -f "$dest" && "$(file_size "$dest")" == "$bytes" ]] || n=$((n + 1))
  done < "$1"
  echo "$n"
}
export -f fetch_sized log file_size

mkdir -p "$ROOT"
log "OOD/gate sets → $ROOT (only=${ONLY:-all}, jobs=$JOBS)"
log "For research use only. Not for diagnostic use."

rc=0

# =========================== Kermany paediatric ===========================
if [[ -z "$ONLY" || "$ONLY" == "kermany" ]]; then
  KROOT="$ROOT/kermany_pediatric"
  ZIP="$KROOT/ZhangLabData.zip"
  STAMP="$KROOT/.chest_xray.extracted"
  mkdir -p "$KROOT"
  # Re-verify the licence on the day of download (Mendeley public API); saved next to the data.
  if curl -sS --max-time 30 "$KERMANY_LICENCE_API" -o "$KROOT/mendeley_version3.json" 2>/dev/null; then
    lic=$(python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); d=j.get("data_licence") or {}; print(d.get("short_name") or d.get("name") or d.get("url") or d)' "$KROOT/mendeley_version3.json" 2>/dev/null || echo "?")
    log "licence check kermany_pediatric: Mendeley data_licence = $lic (expect CC BY 4.0)"
  else
    log "licence check kermany_pediatric: Mendeley API unreachable — relying on LICENSES/kermany_pediatric.md (2026-09-21)"
  fi
  if [[ -f "$STAMP" ]]; then
    log "done already  kermany_pediatric ($(find "$KROOT/chest_xray" -iname '*.jpeg' -o -iname '*.jpg' | wc -l | tr -d ' ') jpeg)"
  else
    log "downloading   ZhangLabData.zip (have $( [[ -f "$ZIP" ]] && file_size "$ZIP" || echo 0 )/$KERMANY_BYTES bytes)"
    if fetch_sized "$KERMANY_URL" "$ZIP" "$KERMANY_BYTES"; then
      log "size ok       ZhangLabData.zip ($KERMANY_BYTES bytes)"
      if unzip -tq "$ZIP" >/dev/null 2>&1; then
        log "zip ok        ZhangLabData.zip"
      else
        log "zip CORRUPT   ZhangLabData.zip — deleting; re-run to retry"; rm -f "$ZIP"; rc=1
      fi
      if [[ -f "$ZIP" ]]; then
        log "extracting    chest_xray/ only (OCT set is skipped)"
        # zip members look like "CellData/chest_xray/train/NORMAL/x.jpeg" (v3) or "chest_xray/..."
        # (listing goes through a file and `grep -m1` stops by itself: `| head -1` would
        # SIGPIPE the producer, which `set -o pipefail` turns into a silent exit)
        unzip -Z1 "$ZIP" > "$KROOT/.zip_listing.txt"
        prefix=$(grep -m1 'chest_xray/' "$KROOT/.zip_listing.txt" | sed 's#chest_xray/.*#chest_xray/#')
        log "zip prefix    '${prefix}' ($(grep -c 'chest_xray/.*\.jpe\?g$' "$KROOT/.zip_listing.txt") jpeg members)"
        rm -rf "$KROOT/_extract"
        unzip -q -o "$ZIP" "${prefix}*" -d "$KROOT/_extract"
        rm -rf "$KROOT/chest_xray"
        mv "$KROOT/_extract/${prefix}" "$KROOT/chest_xray"
        rm -rf "$KROOT/_extract"
        n=$(find "$KROOT/chest_xray" -iname '*.jpeg' -o -iname '*.jpg' | wc -l | tr -d ' ')
        log "extracted     chest_xray → $n jpeg (expected $KERMANY_EXPECTED_JPEG)"
        if [[ "$n" -ge 5000 ]]; then
          touch "$STAMP"
          [[ $KEEP_ZIP -eq 0 ]] && rm -f "$ZIP" && log "removed       ZhangLabData.zip (use --keep-zip to keep)"
        else
          log "INCOMPLETE    kermany_pediatric: only $n jpeg — zip kept for inspection"; rc=1
        fi
      fi
    else
      rc=1
    fi
  fi
fi

# =========================== NLM TB sets ===========================
if [[ -z "$ONLY" || "$ONLY" == "nlm" ]]; then
  NROOT="$ROOT/nlm_tb"
  mkdir -p "$NROOT"
  LIST="$NROOT/.filelist.tsv"   # url \t dest \t bytes
  : > "$LIST"
  for d in "${NLM_DIRS[@]}"; do
    idx="$NROOT/$d/index.json"
    mkdir -p "$NROOT/$d"
    if ! curl -sS --fail --max-time 60 --retry 5 --retry-delay 10 "$NLM_BASE/$d/index.json" -o "$idx"; then
      log "index FAILED  $d — server unreachable?"; rc=1; continue
    fi
    python3 - "$idx" "$NLM_BASE/$d" "$NROOT/$d" >> "$LIST" <<'PY'
import json, sys
idx, base, dest = sys.argv[1:4]
for c in json.load(open(idx)).get("Contents", []):
    k = c["Key"]
    if k in ("index.json", "index.html") or k.endswith("/"):
        continue
    print(f"{base}/{k}\t{dest}/{k}\t{c['Size']}")
PY
  done
  total=$(wc -l < "$LIST" | tr -d ' ')
  todo=$(count_mismatch "$LIST")
  log "nlm_tb        $total files listed, $todo to fetch (jobs=$JOBS)"
  # each line is independently resumable; xargs -P runs $JOBS curl processes
  tr '\t' ' ' < "$LIST" | xargs -P "$JOBS" -n 3 bash -c 'fetch_sized "$0" "$1" "$2" || echo "FAIL $1"' > "$NROOT/.failed.txt" || rc=1
  if [[ -s "$NROOT/.failed.txt" ]]; then log "nlm_tb        $(wc -l < "$NROOT/.failed.txt" | tr -d ' ') files failed (see $NROOT/.failed.txt) — re-run to retry"; rc=1; fi
  # size audit
  bad=$(count_mismatch "$LIST")
  ns=$(find "$NROOT/Shenzhen-Hospital-CXR-Set/CXR_png" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
  nm=$(find "$NROOT/Montgomery-County-CXR-Set/MontgomerySet/CXR_png" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
  log "nlm_tb        size mismatches: $bad; Shenzhen png $ns/$NLM_EXPECTED_PNG_SHENZHEN; Montgomery png $nm/$NLM_EXPECTED_PNG_MONTGOMERY"
  [[ "$bad" -ne 0 || "$ns" -ne "$NLM_EXPECTED_PNG_SHENZHEN" || "$nm" -ne "$NLM_EXPECTED_PNG_MONTGOMERY" ]] && rc=1
  if [[ -f "$NROOT/Shenzhen-Hospital-CXR-Set/NLM-ChinaCXRSet-ReadMe.docx" && -f "$NROOT/Montgomery-County-CXR-Set/MontgomerySet/NLM-MontgomeryCXRSet-ReadMe.pdf" ]]; then
    log "licence check nlm_tb: both README files (Citation and Use terms) are on disk — compare with LICENSES/nlm_tb_shenzhen_montgomery.md"
  fi
fi

if [[ $rc -eq 0 ]]; then
  log "COMPLETE. Next: uv run python -m secondreader.modalities.cxr.cli_datasets manifest kermany $ROOT/kermany_pediatric ; ... nlm-tb $ROOT/nlm_tb"
else
  log "INCOMPLETE — re-run this script to resume"
fi
exit $rc
