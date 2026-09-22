#!/usr/bin/env bash
# Download + verify + extract NIH ChestX-ray14 (112,120 PNGs, ~45.1 GB compressed).
#
# Source of truth: NIH Clinical Center Box folder "ChestXray-NIHCC"
#   https://nihcc.app.box.com/v/ChestXray-NIHCC
# The 12 image-archive URLs below are copied verbatim from NIH's own
# `images/batch_download_zips.py` in that folder (verified 2026-09-21; each URL
# answered HTTP 206 with the byte size listed here). MD5s are the ones NIH
# publishes in FAQ_CHESTXRAY.pdf, Q07. Licence: datasets/LICENSES/nih_cxr14.md.
#
# Resumable: re-running skips finished archives, resumes partial ones (curl -C - /
# aria2c -c), verifies size (+ MD5 unless --no-md5), extracts once, and never
# re-extracts. Safe to Ctrl-C at any point.
#
# Usage:
#   scripts/download_nih_cxr14.sh [--root DIR] [--no-md5] [--delete-archives] [--jobs N]
#   nohup scripts/download_nih_cxr14.sh > data/raw/nih_cxr14/download.log 2>&1 &
#
# Layout produced:
#   $ROOT/archives/images_0NN.tar.gz     (kept unless --delete-archives)
#   $ROOT/images/*.png                   (112,120 files)
#   $ROOT/Data_Entry_2017_v2020.csv, BBox_List_2017.csv, train_val_list.txt, test_list.txt
#
# The metadata files are NOT fetchable by URL: Box answers "This user is not
# allowed to use direct links" for them. They must be saved once through the Box
# web UI (Download button on each file). This script only verifies their hashes.
#
# data/ is git-ignored. Never commit images. For research use only.
set -euo pipefail

ROOT="data/raw/nih_cxr14"
VERIFY_MD5=1
DELETE_ARCHIVES=0
JOBS=2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --no-md5) VERIFY_MD5=0; shift ;;
    --delete-archives) DELETE_ARCHIVES=1; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# name  url  bytes  md5   (order = images_001 .. images_012, as in batch_download_zips.py)
ARCHIVES=(
  "images_001.tar.gz https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz 2008470987 fe8ed0a6961412fddcbb3603c11b3698"
  "images_002.tar.gz https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz 3952623504 ab07a2d7cbe6f65ddd97b4ed7bde10bf"
  "images_003.tar.gz https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz 3929234850 2301d03bde4c246388bad3876965d574"
  "images_004.tar.gz https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz 3838903983 9f1b7f5aae01b13f4bc8e2c44a4b8ef6"
  "images_005.tar.gz https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz 3935496531 1861f3cd0ef7734df8104f2b0309023b"
  "images_006.tar.gz https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz 3986301172 456b53a8b351afd92a35bc41444c58c8"
  "images_007.tar.gz https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz 4016328426 1075121ea20a137b87f290d6a4a5965e"
  "images_008.tar.gz https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz 4018347353 b61f34cec3aa69f295fbb593cbd9d443"
  "images_009.tar.gz https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz 4111327929 442a3caa61ae9b64e61c561294d1e183"
  "images_010.tar.gz https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz 4181556296 09ec81c4c31e32858ad8cf965c494b74"
  "images_011.tar.gz https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz 4187084020 499aefc67207a5a97692424cf5dbeed5"
  "images_012.tar.gz https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz 2914187733 dc9fda1757c2de0032b63347a7d2895c"
)

# Metadata files saved from the Box UI (sha256 of the copies verified 2026-09-21).
METADATA=(
  "Data_Entry_2017_v2020.csv c69a6daca3549af707ca9cacbdf9f9a7b6a9188e8c61157df653520bb72d8eb1"
  "BBox_List_2017.csv 0bbfea9d4c4e9771481b3023b1bc9f0df9dea924453b12986beb29b0c4d0c95b"
  "train_val_list.txt 61fbe896321c1c1c8b75f3e4f3a08e4fef6486d95ef8a667c31d4d60dca6cb81"
  "test_list.txt 38ca5ef7f756092946f57c1a59faca882ed589a1ab1f72590b45dc06c6d5e1cc"
)

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
file_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }  # GNU first: BSD -f means "filesystem" on Linux
md5_of() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }
sha256_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

mkdir -p "$ROOT/archives" "$ROOT/images"
log "NIH ChestX-ray14 → $ROOT (md5=$VERIFY_MD5, delete_archives=$DELETE_ARCHIVES, jobs=$JOBS)"
log "For research use only. Not for diagnostic use."

# ---- metadata check (cannot be fetched by URL; see header) ----
missing=0
for entry in "${METADATA[@]}"; do
  read -r name sha <<<"$entry"
  if [[ -f "$ROOT/$name" ]]; then
    got=$(sha256_of "$ROOT/$name")
    if [[ "$got" == "$sha" ]]; then log "metadata ok   $name"; else log "metadata CHANGED $name (sha256 $got != $sha) — NIH may have re-released it; re-verify"; fi
  else
    log "metadata MISSING $name — save it from https://nihcc.app.box.com/v/ChestXray-NIHCC via the Download button"
    missing=1
  fi
done
[[ $missing -eq 1 ]] && log "continuing with image archives; manifest build needs the CSV"

# ---- one archive: download (resume) → verify → extract → stamp ----
fetch_one() {
  local name="$1" url="$2" bytes="$3" md5="$4"
  local arc="$ROOT/archives/$name" stamp="$ROOT/archives/$name.extracted"
  if [[ -f "$stamp" ]]; then log "done already  $name"; return 0; fi

  local attempt=0
  while :; do
    attempt=$((attempt + 1))
    if [[ -f "$arc" && "$(file_size "$arc")" == "$bytes" ]]; then
      log "size ok       $name ($bytes bytes)"
    else
      log "downloading   $name (attempt $attempt, have $( [[ -f "$arc" ]] && file_size "$arc" || echo 0 )/$bytes bytes)"
      if command -v aria2c >/dev/null; then
        aria2c -c -x 4 -s 4 --file-allocation=none --console-log-level=warn --summary-interval=60 \
          -d "$ROOT/archives" -o "$name" "$url" || true
      else
        # -sS: no progress meter (progress = file size in $ROOT/archives; errors still printed)
        curl -sS -L -C - --retry 10 --retry-delay 15 --retry-all-errors --fail \
          --speed-limit 10000 --speed-time 120 -o "$arc" "$url" || true
      fi
    fi
    if [[ -f "$arc" && "$(file_size "$arc")" == "$bytes" ]]; then break; fi
    if [[ $attempt -ge 8 ]]; then log "FAILED        $name after $attempt attempts"; return 1; fi
    sleep 30
  done

  if [[ $VERIFY_MD5 -eq 1 ]]; then
    local got; got=$(md5_of "$arc")
    if [[ "$got" != "$md5" ]]; then
      log "MD5 MISMATCH  $name ($got != $md5) — deleting and retrying"
      rm -f "$arc"; return 1
    fi
    log "md5 ok        $name"
  fi

  log "extracting    $name"
  # archives contain images/<file>.png → strip the leading component into $ROOT/images/
  tar -xzf "$arc" -C "$ROOT/images" --strip-components=1
  touch "$stamp"
  [[ $DELETE_ARCHIVES -eq 1 ]] && rm -f "$arc"
  log "extracted     $name → $(find "$ROOT/images" -name '*.png' | wc -l | tr -d ' ') png so far"
}

export -f fetch_one log file_size md5_of
export ROOT VERIFY_MD5 DELETE_ARCHIVES

# Run $JOBS archives concurrently; each is independently resumable.
rc=0
printf '%s\n' "${ARCHIVES[@]}" | xargs -P "$JOBS" -I{} bash -c 'fetch_one {}' || rc=$?
# a second pass picks up anything that failed the first time
if [[ $rc -ne 0 ]]; then
  log "retry pass for failed archives"
  printf '%s\n' "${ARCHIVES[@]}" | xargs -P 1 -I{} bash -c 'fetch_one {}'
fi

n=$(find "$ROOT/images" -name '*.png' | wc -l | tr -d ' ')
log "png count: $n (expected 112120)"
if [[ "$n" == "112120" ]]; then
  log "COMPLETE. Next: uv run secondreader cxr manifest nih $ROOT --out datasets/manifests/nih_cxr14.csv"
else
  log "INCOMPLETE — re-run this script to resume"
  exit 1
fi
