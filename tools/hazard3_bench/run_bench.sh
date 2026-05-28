#!/usr/bin/env bash
# tools/hazard3_bench/run_bench.sh — drive bench_encode under
# Verilator or CXXRTL on the official Hazard3 RTL.
#
#   bash run_bench.sh verilator      # tb_verilator
#   bash run_bench.sh cxxrtl         # tb_cxxrtl
#
# Assumes external/Hazard3 has been cloned and the chosen testbench
# has been built per tools/bench_rp2350_silicon.md. Auto-builds the
# bench ELF, converts to flat bin, runs the sim, parses output.
set -euo pipefail
BACKEND="${1:-verilator}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$REPO_ROOT/tools/hazard3_bench"
HAZARD3="$REPO_ROOT/external/Hazard3"
TB="$HAZARD3/test/sim/tb_$BACKEND/tb"
ELF="$BENCH_DIR/target/riscv32imac-unknown-none-elf/release/bench_encode"
BIN="$ELF.bin"

if [ ! -x "$TB" ]; then
    echo "tb not built: $TB"
    echo "see tools/bench_rp2350_silicon.md for build steps"
    exit 1
fi

echo "[1/3] cargo build --release..."
( cd "$BENCH_DIR" && cargo build --release ) >/dev/null

echo "[2/3] llvm-objcopy ELF -> flat bin..."
llvm-objcopy -O binary "$ELF" "$BIN"
BIN_SIZE=$(stat -c%s "$BIN")
echo "  bin: $BIN_SIZE bytes"

echo "[3/3] running sim..."
OUT=$(mktemp)
"$TB" --bin "$BIN" --cycles 500000000 --cpuret 2>&1 | tee "$OUT"
EXIT_CODE=$?

echo
echo "=== parsed metrics ==="
CPW=$(grep -A1 "cycles_per_window=" "$OUT" | tail -1 | tr -d ' ')
IPW=$(grep -A1 "instrs_per_window=" "$OUT" | tail -1 | tr -d ' ')
CPI=$(grep -A1 "CPI_x1000=" "$OUT" | tail -1 | tr -d ' ')
WUS=$(grep -A1 "window_us@150MHz=" "$OUT" | tail -1 | tr -d ' ')
MSA=$(grep -A1 "Msa_per_s_x100=" "$OUT" | tail -1 | tr -d ' ')

if [ -n "$CPW" ]; then
    # Hex values from tb_print_u32 — convert to decimal.
    CPW_DEC=$(printf "%d" "0x$CPW")
    IPW_DEC=$(printf "%d" "0x$IPW")
    CPI_DEC=$(printf "%d" "0x$CPI")
    WUS_DEC=$(printf "%d" "0x$WUS")
    MSA_DEC=$(printf "%d" "0x$MSA")
    echo "  cycles_per_window  = $CPW_DEC"
    echo "  instrs_per_window  = $IPW_DEC"
    echo "  CPI                = $(awk "BEGIN{printf \"%.3f\", $CPI_DEC/1000.0}")"
    echo "  window_us@150MHz   = $WUS_DEC us"
    echo "  Msa/s              = $(awk "BEGIN{printf \"%.3f\", $MSA_DEC/100.0}")"
fi

exit $EXIT_CODE
