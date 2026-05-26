#!/usr/bin/env bash
# tools/bench_gzip_baseline.sh — gzip baseline on a TUH EDF subset.
#
# For the LamQuant TBioCAS paper (Phase 2, item 11): establish the
# gzip baseline CR cited in §II.C "Limitations of General-Purpose
# Compression on EEG."
#
# Default subset: TUEG edf/000 (the first 1,000-stem block). Override
# with `--tree <path>` to gzip any EDF subtree.
#
# Output:
#   outputs/paper/gzip_baseline_<subset>.json
#   + stdout report.
#
# Methodology:
#   * Walk every `.edf` under the chosen tree.
#   * For each: `gzip -k -9 -c <edf>` piped to `wc -c` (no on-disk
#     output; we only need the compressed size).
#   * Sum input + output bytes; CR = in / out.
#   * `gzip -9` (maximum compression, default level on most distros
#     for benchmark comparisons).

set -euo pipefail

TREE="${1:-/mnt/4tb/data/Archive/edf/tuh_repair/tueg_v2.0.1/edf/000}"
LABEL="$(basename "$TREE")"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs/paper"
OUT_JSON="$OUT_DIR/gzip_baseline_${LABEL}.json"

mkdir -p "$OUT_DIR"

if [[ ! -d "$TREE" ]]; then
  echo "error: tree not found: $TREE" >&2
  echo "       (tueg corpus repair may not have completed yet; run after tueg lands)" >&2
  exit 2
fi

echo "[bench_gzip] scanning $TREE for .edf files..."
mapfile -t EDFS < <(find "$TREE" -name "*.edf" -not -name "*.seizures" | sort)
N="${#EDFS[@]}"
echo "[bench_gzip] $N EDFs"

if [[ "$N" -eq 0 ]]; then
  echo "error: no .edf files under $TREE" >&2
  exit 3
fi

TOTAL_IN=0
TOTAL_OUT=0
START="$(date +%s)"

for i in "${!EDFS[@]}"; do
  edf="${EDFS[$i]}"
  in_sz="$(stat -c '%s' "$edf")"
  out_sz="$(gzip -9 -c "$edf" | wc -c)"
  TOTAL_IN=$((TOTAL_IN + in_sz))
  TOTAL_OUT=$((TOTAL_OUT + out_sz))
  if (( (i + 1) % 50 == 0 )); then
    elapsed=$(( $(date +%s) - START ))
    cr=$(python3 -c "print(f'{$TOTAL_IN / max($TOTAL_OUT, 1):.4f}')")
    printf "[bench_gzip] %4d/%d  elapsed=%ds  CR=%s\n" \
      "$((i + 1))" "$N" "$elapsed" "$cr"
  fi
done

ELAPSED=$(( $(date +%s) - START ))
CR="$(python3 -c "print(f'{$TOTAL_IN / max($TOTAL_OUT, 1):.4f}')")"

cat > "$OUT_JSON" <<JSON
{
  "tool": "gzip -9",
  "tree": "$TREE",
  "label": "$LABEL",
  "files": $N,
  "input_bytes": $TOTAL_IN,
  "output_bytes": $TOTAL_OUT,
  "cr": $CR,
  "wall_seconds": $ELAPSED
}
JSON

echo
echo "[bench_gzip] done. wrote $OUT_JSON"
echo "  files=$N  in=${TOTAL_IN}  out=${TOTAL_OUT}  CR=${CR}  wall=${ELAPSED}s"
