#!/bin/bash
# Athena++ figure 21 reproduction at matched degrees of freedom.
#
# Four runs, all reaching 1024^2 degrees of freedom at the finest level:
#
#   MUSCL   uniform  1024^2 cells   (p=0), single block
#   MUSCL   AMR      256^2 root cells, 32^2 blocks, 2 levels
#   SDFB4   uniform  256^2 elements (p=3), single block
#   SDFB4   AMR      64^2 root elements, 8^2 blocks, 2 levels
#
# Both AMR runs use 64 root blocks carrying 1024 DoF each, so the two schemes
# are matched block-for-block as well as DoF-for-DoF.
#
# cfl=0.4 everywhere: a coarse-fine interface roughly halves the stable CFL for
# this solver, and the uniform runs use the same value so that uniform-vs-AMR
# differs only in the mesh.
#
# usage: kh_figure21.sh <gpu-lane>   with lane 0 = MUSCL, lane 1 = SDFB4
set -u
cd "$(dirname "$0")/.."

lane=${1:?usage: kh_figure21.sh <0=muscl|1=sd>}
TL=${TL:-1.2}
OD=${OD:-0.3}
ADAPT=${ADAPT:-50}
CFL=${CFL:-0.4}

if [[ "$lane" == "0" ]]; then
  ./scripts/kh_case.sh kh1024_muscl_uni 0 muscl 0 1024 1024 0 0     "$TL" "$OD" time/cfl=$CFL
  ./scripts/kh_case.sh kh1024_muscl_amr 0 muscl 0  256   32 2 $ADAPT "$TL" "$OD" time/cfl=$CFL
else
  ./scripts/kh_case.sh kh1024_sd_uni    1 sd    3  256  256 0 0     "$TL" "$OD" time/cfl=$CFL
  ./scripts/kh_case.sh kh1024_sd_amr    1 sd    3   64    8 2 $ADAPT "$TL" "$OD" time/cfl=$CFL
fi
echo "lane $lane done"
