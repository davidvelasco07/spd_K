#!/usr/bin/env bash
# Staged Balsara NR MHD tests: isolate crashes by escalating scheme complexity.
#
# Order (per problem):
#   1. MUSCL + LLF   (mood_force_level=1, mhd/rsolver=llf)
#   2. MUSCL + HLLD  (mood_force_level=1, mhd/rsolver=hlld)
#   3. SD+FB  + LLF  (normal MOOD cascade, mhd/rsolver=llf)
#   4. SD+FB  + HLLD (normal MOOD cascade, mhd/rsolver=hlld)
#
# Usage:
#   scripts/run_balsara_staged.sh [blast|jet|vortex3d|vortex2d] [out_root]
#   BIN=build/spd_K scripts/run_balsara_staged.sh blast /tmp/balsara_staged
#
# Environment:
#   BIN          path to spd_K executable (default: build/spd_K)
#   STOP_ON_FAIL if set (default 1), exit after first failing stage

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/build/spd_K}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

TEST="${1:-blast}"
OUT_ROOT="${2:-/tmp/balsara_staged}"

case "$TEST" in
  blast)     INPUT="$ROOT/inputs/balsara/blast_smoke_p3_fb.athinput" ;;
  jet)       INPUT="$ROOT/inputs/balsara/jet_smoke_p3_fb.athinput" ;;
  vortex3d)  INPUT="$ROOT/inputs/balsara/vortex_3d_p3_fb.athinput" ;;
  vortex2d)  INPUT="$ROOT/inputs/balsara/vortex_2d_p3_fb.athinput" ;;
  *)
    echo "Unknown test '$TEST' (blast|jet|vortex3d|vortex2d)" >&2
    exit 1
    ;;
esac

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found: $BIN" >&2
  exit 1
fi
if [[ ! -f "$INPUT" ]]; then
  echo "Input not found: $INPUT" >&2
  exit 1
fi

run_stage() {
  local label="$1"
  local rsolver="$2"
  local force="$3"
  local outdir="$OUT_ROOT/${TEST}/${label}"

  rm -rf "$outdir"
  mkdir -p "$outdir"
  local log="$outdir/run.log"

  echo ""
  echo "========== $TEST / $label (rsolver=$rsolver mood_force_level=$force) =========="
  echo "  out: $outdir"

  local t0
  t0=$(date +%s)
  set +e
  SPD_OUTPUT_DIR="$outdir" "$BIN" -i "$INPUT" \
    "mhd/rsolver=$rsolver" \
    "mhd/mood_force_level=$force" \
    > "$log" 2>&1
  local rc=$?
  set -e
  local elapsed=$(( $(date +%s) - t0 ))

  if [[ $rc -ne 0 ]]; then
    echo "  FAILED (exit $rc) after ${elapsed}s"
    tail -20 "$log" | sed 's/^/    /'
    return 1
  fi

  local steps divb
  steps=$(grep -o 'evolution: [0-9]* steps' "$log" | tail -1 | awk '{print $2}')
  divb=$(grep -o 'max|divB| = [0-9.eE+-]*' "$log" | tail -1 | sed 's/max|divB| = //')
  echo "  OK  steps=${steps:-?}  final max|divB|=${divb:-?}  time=${elapsed}s"

  if [[ "$TEST" == "blast" || "$TEST" == "jet" ]]; then
    if [[ -f "$ROOT/scripts/plot_balsara_figures.py" ]]; then
      python3 "$ROOT/scripts/plot_balsara_figures.py" "$outdir" \
        --test "$TEST" --dest "$OUT_ROOT/${TEST}/figures" \
        --index -1 >> "$log" 2>&1 || true
    fi
  fi
  return 0
}

mkdir -p "$OUT_ROOT/$TEST"

STAGES=(
  "muscl_llf:llf:1"
  "muscl_hlld:hlld:1"
  "sd_fb_llf:llf:-1"
  "sd_fb_hlld:hlld:-1"
)

FAILED=""
for entry in "${STAGES[@]}"; do
  IFS=: read -r label rsolver force <<< "$entry"
  if ! run_stage "$label" "$rsolver" "$force"; then
    FAILED="$label"
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      echo ""
      echo "Stopped at failed stage: $TEST / $FAILED"
      exit 1
    fi
  fi
done

if [[ -n "$FAILED" && "$STOP_ON_FAIL" != "1" ]]; then
  echo "Completed with failures (last: $FAILED)"
  exit 1
fi

echo ""
echo "All stages passed for $TEST. Outputs under $OUT_ROOT/$TEST"
