#!/usr/bin/env bash
# Balsara Fig.-11 jet on Apollo A100 front node (LLF only).
#
# Usage on Apollo:
#   cd ~/spd_K && bash scripts/run_balsara_jet_apollo.sh [muscl|sd_fb|both]
#
# Or from local (via tunnel):
#   rsync ... && ssh apollo 'cd ~/spd_K && bash scripts/run_balsara_jet_apollo.sh muscl'
set -euo pipefail

ROOT="${ROOT:-$HOME/spd_K}"
PY="${PY:-python3}"
MODE="${1:-muscl}"
GPU="${GPU:-0}"
OUTROOT="${OUTROOT:-$ROOT/balsara_jet}"

export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/lib:/usr/local/openmpi/cuda-12.4/4.1.6/nvhpc245/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda/bin:${PATH:-/usr/bin:/bin}"

# Pick executable: explicit EXE, else CUDA build, else CPU OpenMP build.
if [[ -n "${EXE:-}" ]]; then
  :
elif [[ -x "$ROOT/build/spd_K" ]]; then
  EXE="$ROOT/build/spd_K"
elif [[ -x "$ROOT/build-omp/spd_K" ]]; then
  EXE="$ROOT/build-omp/spd_K"
  echo "NOTE: using CPU OpenMP build ($EXE)"
else
  EXE="$ROOT/build/spd_K"
fi

run_one() {
  local label="$1"
  local input="$2"
  local out="$OUTROOT/$label"
  mkdir -p "$out"
  echo "=== $label on GPU $GPU -> $out ==="
  export SPD_OUTPUT_DIR="$out"
  export CUDA_VISIBLE_DEVICES="$GPU"
  "$EXE" -i "$ROOT/$input" 2>&1 | tee "$out/run.log"
  "$PY" "$ROOT/scripts/plot_balsara_figures.py" "$out" --test jet \
    --dest "$out/figures" --index -1
  echo "figure: $out/figures/"
  ls -lh "$out/figures/"*.png
}

# Rebuild if requested or binary missing
if [[ "${REBUILD:-0}" == "1" || ! -x "$EXE" ]]; then
  echo "=== building spd_K (CUDA) ==="
  mkdir -p "$ROOT/build"
  cd "$ROOT/build"
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j8 spd_K
  cd "$ROOT"
fi

case "$MODE" in
  muscl) run_one muscl_llf inputs/balsara/jet_muscl_p3_fb.athinput ;;
  sd_fb) run_one sd_fb_llf inputs/balsara/jet_p3_fb.athinput ;;
  both)
    run_one muscl_llf inputs/balsara/jet_muscl_p3_fb.athinput
    run_one sd_fb_llf inputs/balsara/jet_p3_fb.athinput
    ;;
  smoke)
    run_one smoke inputs/balsara/jet_smoke_p3_fb.athinput
    ;;
  *) echo "Usage: $0 [muscl|sd_fb|both|smoke]" >&2; exit 1 ;;
esac

echo "done: $OUTROOT"
