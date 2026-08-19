#!/bin/bash
# Run one MHD Kelvin-Helmholtz case (Athena++ figure 22 reproduction).
#
# Sibling of kh_case.sh; figure 22 is figure 21 plus a uniform Bx = 0.1, at
# t = 1.5 instead of 1.2.
#
# usage: kh_mhd_case.sh <tag> <gpu> <lane> <p> <nelem> <mb> <maxlev> <adapt> <tlim> <outdt> [extra args...]
#
#   lane    muscl -> every cell pinned at MOOD level 1 (PLM on the sub-cell
#                    mesh, no detection), pair with p=0 and rk2: the closest
#                    spd_K analogue of the paper's VL2+PLM
#           sdfb  -> SD with the MOOD cascade live, pair with p=3 and rk3
#   p       polynomial degree (0 for muscl, 3 for sdfb)
#   nelem   elements per direction on the root grid
#   mb      meshblock size in elements (use nelem for a single block)
#   maxlev  0 for a uniform run, >0 for AMR
#   adapt   adapt_interval in steps (ignored when maxlev = 0)
#
# Degrees of freedom are nelem * (p+1) per direction at the finest level.
set -u
cd "$(dirname "$0")/.."

BIN=${SPD_BIN:-./build/spd_K}

tag=$1; gpu=$2; lane=$3; pp=$4; ne=$5; mb=$6; ml=$7; ai=$8; tl=$9; od=${10}; shift 10

case "$lane" in
  muscl) LANE="time/integrator=rk2 mhd/mood_force_level=1" ;;
  sdfb)  LANE="time/integrator=rk3 mhd/mood_force_level=-1" ;;
  *) echo "unknown lane '$lane' (muscl|sdfb)" >&2; exit 1 ;;
esac

D=${KH_OUT:-$HOME/kh_mhd}/$tag
rm -rf "$D"; mkdir -p "$D"

t0=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=$gpu SPD_OUTPUT_DIR=$D $BIN \
  -i inputs/kelvin_helmholtz_mhd.athinput \
  mesh/p="$pp" mesh/nx1="$ne" mesh/nx2="$ne" \
  meshblock/nx1="$mb" meshblock/nx2="$mb" \
  amr/max_level="$ml" amr/criterion=shear amr/adapt_interval="$ai" \
  amr/refine_threshold=0.01 amr/derefine_threshold=0.005 \
  time/tlim="$tl" output/dt="$od" $LANE "$@" > "$D/log" 2>&1
rc=$?
t1=$(date +%s.%N)

echo "$tag rc=$rc elapsed=$(echo "$t1-$t0" | bc)s steps=$(tr -cd . < "$D/log" | wc -c) $(grep -o 'max|divB| = [0-9.e+-]*' "$D/log" | tail -1)"
