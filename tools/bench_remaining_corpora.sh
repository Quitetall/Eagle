#!/usr/bin/env bash
# tools/bench_remaining_corpora.sh — populate Appendix A rows for
# the 6 TUH corpora that the chain bench doesn't cover.
#
# Chain (chain_tueg_benches.sh) already benches TUEG v2.0.1
# (1.76 TB, ~1.75 h). This script handles the remaining six:
#   TUSL v2.0.1, TUAR v3.0.1, TUEV v2.0.1, TUSZ v2.0.6,
#   TUAB v3.0.1, TUEP v3.1.0.
#
# Runs sequentially. Each corpus produces its own JSON under
# outputs/paper/per_corpus_<name>.json with input/output bytes +
# CR (group_by=montage gives the per-corpus number by summing
# the groups). Aggregator stitches them into the LaTeX table.
#
# Usage:
#   nohup bash tools/bench_remaining_corpora.sh \
#     > /tmp/bench_remaining.log 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPAIR_ROOT="/mnt/4tb/data/Archive/edf/tuh_repair"
OUT_DIR="$REPO_ROOT/outputs/paper"
mkdir -p "$OUT_DIR"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

# Order: small first so quick wins show up early in the log.
CORPORA=(
  tuar_v3.0.1
  tuev_v2.0.1
  tusl_v2.0.1
  tuab_v3.0.1
  tuep_v3.1.0
  tusz_v2.0.6
)

START=$(date +%s)
DONE=0
FAILED=0

for corpus in "${CORPORA[@]}"; do
  tree="$REPAIR_ROOT/$corpus"

  if [[ ! -d "$tree" ]]; then
    log "skip $corpus — tree absent at $tree"
    continue
  fi

  edf_count=$(find "$tree" -name '*.edf' -type f 2>/dev/null | wc -l)
  if [[ "$edf_count" -eq 0 ]]; then
    log "skip $corpus — 0 EDFs"
    continue
  fi

  log "── $corpus ($edf_count EDFs) ──"

  # bench_tueg_subsets.py writes outputs/paper/tueg_subset_breakdown_montage.json,
  # so we rename per-corpus immediately after the run.
  out_default="$OUT_DIR/tueg_subset_breakdown_montage.json"
  out_renamed="$OUT_DIR/per_corpus_${corpus}.json"

  if python3 "$REPO_ROOT/tools/bench_tueg_subsets.py" \
       --tree "$tree" --group-by montage 2>&1 | tail -5; then
    if [[ -f "$out_default" ]]; then
      if mv "$out_default" "$out_renamed"; then
        log "✓ $corpus → $(basename "$out_renamed")"
        DONE=$((DONE + 1))
      else
        log "✗ $corpus — mv $out_default → $out_renamed failed"
        FAILED=$((FAILED + 1))
      fi
    else
      log "✗ $corpus — bench produced no JSON"
      FAILED=$((FAILED + 1))
    fi
  else
    log "✗ $corpus — bench EXIT $?"
    FAILED=$((FAILED + 1))
  fi
done

ELAPSED=$(( $(date +%s) - START ))
log "── DONE — $DONE ok / $FAILED failed; wall = $((ELAPSED/60))m $((ELAPSED%60))s ──"
exit "$FAILED"
