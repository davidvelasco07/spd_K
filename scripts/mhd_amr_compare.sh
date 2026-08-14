#!/usr/bin/env bash
# Uniform-mesh vs AMR (2 levels) comparison at matched DOF, for the two
# standard 2D MHD tests. Each family runs a base-resolution uniform mesh, a
# uniform mesh carrying roughly the AMR run's total DOF, a uniform mesh at the
# AMR run's effective (finest-level) resolution, and the AMR run itself.
#
# Both families use the MOOD fallback: pure SD CT is unstable once a coarse-fine
# interface is regridded, so the AMR runs need it and the uniform runs use it
# too to keep the comparison to one scheme.
#
#   usage: scripts/mhd_amr_compare.sh [dest] [family]
set -u

BIN=${BIN:-./build-crit/spd_K}
DEST=${1:-mhd_amr_runs}
FAMILY=${2:-all}
mkdir -p "$DEST"

run() {                       # run <name> <input> <overrides...>
  local name=$1 inp=$2; shift 2
  local out="$DEST/$name"
  rm -rf "$out"; mkdir -p "$out"
  echo "[start] $name"
  ( SPD_OUTPUT_DIR="$out" "$BIN" -i "$inp" "$@" >"$out/log.txt" 2>&1
    echo "[done ] $name (exit $?)" ) &
}

COMMON=(job/fallback=true mesh/nx3=1 time/integrator=rk3)
# 8^2-element blocks on a 32-element base, refined twice: effective 128^2. The
# indicator saturates on grid-scale noise, so a purely absolute derefine cut
# lets the mesh walk to uniform refinement; releasing blocks that fall below
# 0.85 of the current peak score keeps the fine region on the feature and the
# block count flat.
AMR=(meshblock/nx1=8 meshblock/nx2=8 amr/max_level=2 amr/adapt_interval=10
     amr/criterion=bfield amr/derefine_frac=0.85)

# ---- Orszag-Tang: base 32 elements, 2 levels -> effective 128 -------------
OT=inputs/orszag_tang.athinput
OT_COMMON=("${COMMON[@]}" time/tlim=0.5 output/dt=0.1)
if [ "$FAMILY" = all ] || [ "$FAMILY" = ot ]; then
run ot_uni32  $OT "${OT_COMMON[@]}" mesh/nx1=32  mesh/nx2=32
run ot_uni64  $OT "${OT_COMMON[@]}" mesh/nx1=64  mesh/nx2=64
run ot_uni128 $OT "${OT_COMMON[@]}" mesh/nx1=128 mesh/nx2=128
fi
if [ "$FAMILY" = all ] || [ "$FAMILY" = ot ] || [ "$FAMILY" = amr ]; then
run ot_amr    $OT "${OT_COMMON[@]}" mesh/nx1=32 mesh/nx2=32 "${AMR[@]}"
fi

# ---- Field loop: base 32 elements, 2 levels -> effective 128 --------------
# tlim = 1 is exactly one advection period for v = (2,1) on the unit box, so
# the initial condition is the exact solution at the final time. The loop fills
# a disc of radius 0.3 and the indicator tracks its rim, so the AMR mesh settles
# near 75 blocks of 8^2 elements, i.e. the DOF of a uniform 70^2 mesh.
FL=inputs/field_loop.athinput
FL_COMMON=("${COMMON[@]}" time/tlim=1.0 output/dt=0.125)
if [ "$FAMILY" = all ] || [ "$FAMILY" = fl ]; then
run fl_uni32  $FL "${FL_COMMON[@]}" mesh/nx1=32  mesh/nx2=32
run fl_uni64  $FL "${FL_COMMON[@]}" mesh/nx1=64  mesh/nx2=64
run fl_uni128 $FL "${FL_COMMON[@]}" mesh/nx1=128 mesh/nx2=128
fi
if [ "$FAMILY" = all ] || [ "$FAMILY" = fl ] || [ "$FAMILY" = amr ]; then
run fl_amr    $FL "${FL_COMMON[@]}" mesh/nx1=32 mesh/nx2=32 "${AMR[@]}"
fi

wait
echo "all runs finished"
