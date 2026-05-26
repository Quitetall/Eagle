#!/usr/bin/env bash
# tools/chain_tueg_benches.sh — wait for the corpus repair sweep to
# finish, then chain the 3 deferred Phase-2 benches.
#
# Launches (in serial):
#   1. gzip baseline on tueg edf/000   (paper §II.C 1.61:1 claim)
#   2. per-subset breakdown on tueg    (paper Appendix A Table A.I)
#   3. EDF reader parity (500 pyedflib + 4000 MNE)
#                                       (paper §IV.B verification)
#
# Usage:
#   nohup bash tools/chain_tueg_benches.sh > /tmp/chain_tueg_benches.log 2>&1 &
#
# Optional env override:
#   REPAIR_PID  — PID to poll (default = pgrep newest repair_lma_all.sh)

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TUEG_TREE="/mnt/4tb/data/Archive/edf/tuh_repair/tueg_v2.0.1"
TUEG_EDF000="$TUEG_TREE/edf/000"
REPAIR_PID="${REPAIR_PID:-$(pgrep -f 'repair_lma_all.sh' | tail -1)}"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

if [[ -z "$REPAIR_PID" ]]; then
  log "no repair_lma_all.sh process found; skipping wait"
else
  log "waiting for repair PID=$REPAIR_PID to exit…"
  while kill -0 "$REPAIR_PID" 2>/dev/null; do
    sleep 120
  done
  log "repair finished; proceeding to benches"
fi

if [[ ! -d "$TUEG_TREE" ]]; then
  log "tueg tree not present at $TUEG_TREE — aborting"
  exit 2
fi

# 1. gzip baseline on tueg edf/000
if [[ -d "$TUEG_EDF000" ]]; then
  log "── 1/3 gzip baseline on $TUEG_EDF000 ──"
  bash "$REPO_ROOT/scripts/repair_lma_per_corpus.sh" >/dev/null 2>&1 || true
  bash "$REPO_ROOT/tools/bench_gzip_baseline.sh" "$TUEG_EDF000" \
    || log "gzip baseline EXIT $?"
else
  log "── 1/3 skipped — $TUEG_EDF000 not present ──"
fi

# 2. per-subset breakdown on tueg
log "── 2/3 per-subset breakdown on tueg ──"
python3 "$REPO_ROOT/tools/bench_tueg_subsets.py" \
  --tree "$TUEG_TREE" --group-by montage \
  || log "per-subset EXIT $?"

# 3. EDF reader parity (500 pyedflib + 4000 MNE)
log "── 3/3 EDF reader parity (500 + 4000 samples) on tueg ──"
python3 "$REPO_ROOT/tools/bench_edf_reader_parity.py" \
  --tree "$TUEG_TREE" \
  --pyedflib-samples 500 --mne-samples 4000 \
  || log "parity EXIT $?"

log "── chain complete ──"
