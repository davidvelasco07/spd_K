#!/bin/bash
# Run one Kelvin-Helmholtz case (Athena++ figure 21 reproduction).
#
# usage: kh_case.sh <tag> <gpu> <scheme> <p> <nelem> <mb> <maxlev> <adapt> <tlim> <outdt> [extra args...]
#
#   scheme  sd | vl2 (MUSCL-Hancock) | plm (no predictor)
#   p       polynomial degree (0 for vl2/plm, 3 for SDFB4)
#   nelem   elements per direction on the root grid
#   mb      meshblock size in elements (use nelem for a single block)
#   maxlev  0 for a uniform run, >0 for AMR
#   adapt   adapt_interval in steps (ignored when maxlev = 0)
#
# Degrees of freedom are nelem * (p+1) per direction at the finest level.
set -u
cd "$(dirname "$0")/.."

export PATH=/opt/rh/gcc-toolset-12/root/usr/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:/usr/local/openmpi/4.1.0/gcc/lib64:${LD_LIBRARY_PATH:-}

tag=$1; gpu=$2; sch=$3; pp=$4; ne=$5; mb=$6; ml=$7; ai=$8; tl=$9; od=${10}; shift 10

D=${KH_OUT:-$HOME/kh_out}/$tag
rm -rf "$D"; mkdir -p "$D"

t0=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=$gpu SPD_OUTPUT_DIR=$D ./build-cuda/spd_K \
  -i inputs/kelvin_helmholtz.athinput \
  job/scheme="$sch" mesh/p="$pp" mesh/nx1="$ne" mesh/nx2="$ne" \
  meshblock/nx1="$mb" meshblock/nx2="$mb" \
  amr/max_level="$ml" amr/criterion=shear amr/adapt_interval="$ai" \
  time/tlim="$tl" output/dt="$od" "$@" > "$D/log" 2>&1
rc=$?
t1=$(date +%s.%N)

echo "$tag rc=$rc elapsed=$(echo "$t1-$t0" | bc)s steps=$(tr -cd . < "$D/log" | wc -c) $(grep -o 'forest blocks = [0-9]*' "$D/log" | head -1)"
