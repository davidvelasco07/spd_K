#!/usr/bin/env bash
# Produce 3-panel OT movies (density, pressure, MOOD revision level) at
# fine-grid resolutions 128, 256, 512 (= N=32/64/128 elements at p=3).
#
# Usage on Apollo (A100 front node):
#   cd ~/spd_K && bash scripts/run_ot_movies_apollo.sh
set -euo pipefail

ROOT="${ROOT:-$HOME/spd_K}"
PY="${PY:-python3}"
EXE="${EXE:-$ROOT/build/spd_K}"
MOVIE_PY="$ROOT/scripts/spd_movie.py"
OUTDIR="${OUTDIR:-$ROOT/plots/ot_movies}"
WORKDIR="${WORKDIR:-/tmp/spd_ot_movies}"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/lib:/usr/local/openmpi/cuda-12.4/4.1.6/nvhpc245/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTDIR" "$WORKDIR"
cd "$ROOT"

# fine resolution -> (N elements, GPU id). Two A100s: run 128 & 256 in parallel,
# then 512 on GPU 0.
declare -a JOBS=(
  "128:32:0"
  "256:64:1"
  "512:128:0"
)

run_one() {
  local fine="$1" N="$2" gpu="$3"
  local movie="$OUTDIR/ot_N${fine}_rho_prs_cascade.mp4"
  local wd="$WORKDIR/N${fine}"
  echo "=== OT movie fine=${fine}^2 (N=${N} p=3) on GPU ${gpu} ==="
  "$PY" "$MOVIE_PY" \
    --input "$ROOT/inputs/compare/ot_spdk_n32_fb.athinput" \
    --exe "$EXE" \
    --gpu "$gpu" \
    --tlim 1.0 \
    --dt-out 0.005 \
    --fps 20 \
    --panels rho,prs,cascade \
    --title "Orszag-Tang" \
    --workdir "$wd" \
    --movie "$movie" \
    --keep-frames \
    "mesh/nx1=${N}" "mesh/nx2=${N}" "mesh/nx3=1" \
    "job/fallback=true" "fallback/max_revs=3" \
    "time/integrator=rk3" "time/cfl=0.3"
  echo "wrote $movie"
}

# Parallel: 128 on GPU0, 256 on GPU1
run_one 128 32 0 &
pid128=$!
run_one 256 64 1 &
pid256=$!
wait $pid128
wait $pid256

# Then 512 on GPU0
run_one 512 128 0

ls -lh "$OUTDIR"/ot_N*_rho_prs_cascade.mp4
echo "done: $OUTDIR"
