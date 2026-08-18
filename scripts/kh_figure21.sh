#!/bin/bash
# Athena++ figure 21 reproduction (Stone et al. 2020, ApJS 249, 4, sec. 3.4.3).
#
# The paper's configuration:
#
#   uniform   2048^2 cells
#   AMR       256^2 root cells, 4 levels (so 256 * 2^3 = 2048 at the finest),
#             MeshBlocks of 8^2 and 16^2
#   criterion g = h * max(d_x v_y, d_y v_x); refine above 0.01, derefine
#             below 0.005
#   t = 1.2, cfl = 0.4, gamma = 1.4, HLLC + PLM in the paper
#
# spd_K carries (p+1)^2 degrees of freedom per element, so an "N cells" grid is
# N/(p+1) elements. The vl2 lane (MUSCL-Hancock) runs at p=0 (one cell per
# element, matching the paper's finite-volume scheme); SDFB4 runs at p=3, where the same DoF count
# needs a quarter of the elements per direction. Blocks are matched by DoF, not
# by element count, so the two schemes are comparable block for block.
#
# Cost note: spd_K launches its kernels per block, so wall time scales with the
# leaf count (a 512^2 run costs 3.6 s in one block and 795 s in 1024). The 8^2
# MeshBlock lane is therefore very expensive here; 16^2 is the practical one,
# and the paper found 16^2 optimal for its own throughput anyway.
#
# usage: kh_figure21.sh <lane>
#   0 = MUSCL 1024^2 (half-scale validation)   1 = SDFB4 1024^2
#   2 = MUSCL 2048^2 (paper scale)             3 = SDFB4 2048^2
set -u
cd "$(dirname "$0")/.."

lane=${1:?usage: kh_figure21.sh <0=vl2-1k|1=sd-1k|2=vl2-2k|3=sd-2k>}
TL=${TL:-1.2}
OD=${OD:-0.3}
ADAPT=${ADAPT:-50}
CFL=${CFL:-0.4}
MB=${MB:-16}          # MeshBlock size in *cells* (paper: 8 or 16)
GPU=${GPU:-0}

# Paper thresholds for the velocity-shear criterion.
CRIT="amr/criterion=shear amr/refine_threshold=0.01 amr/derefine_threshold=0.005"

# kh_case.sh <tag> <gpu> <scheme> <p> <nelem> <mb-elem> <maxlev> <adapt> <tlim> <outdt> [args]
case "$lane" in
  0) # MUSCL, 1024^2 effective: root 128 cells, 4 levels (128 * 2^3 = 1024)
     ./scripts/kh_case.sh f21_muscl_uni_1k  "$GPU" vl2 0 1024 1024 0 0      "$TL" "$OD" time/cfl=$CFL
     ./scripts/kh_case.sh f21_muscl_amr_1k  "$GPU" vl2 0  128  $MB  3 $ADAPT "$TL" "$OD" time/cfl=$CFL $CRIT
     ;;
  1) # SDFB4 p=3, 1024^2 DoF: 256 elements uniform; root 32 elements, 4 levels
     ./scripts/kh_case.sh f21_sd_uni_1k     "$GPU" sd 3 256 256 0 0      "$TL" "$OD" time/cfl=$CFL
     ./scripts/kh_case.sh f21_sd_amr_1k     "$GPU" sd 3  32 $((MB/4)) 3 $ADAPT "$TL" "$OD" time/cfl=$CFL $CRIT
     ;;
  2) # MUSCL, paper scale: 2048^2 uniform; root 256 cells, 4 levels
     ./scripts/kh_case.sh f21_muscl_uni_2k  "$GPU" vl2 0 2048 2048 0 0      "$TL" "$OD" time/cfl=$CFL
     ./scripts/kh_case.sh f21_muscl_amr_2k  "$GPU" vl2 0  256  $MB  3 $ADAPT "$TL" "$OD" time/cfl=$CFL $CRIT
     ;;
  3) # SDFB4 p=3, paper scale: 512 elements uniform; root 64 elements, 4 levels
     ./scripts/kh_case.sh f21_sd_uni_2k     "$GPU" sd 3 512 512 0 0      "$TL" "$OD" time/cfl=$CFL
     ./scripts/kh_case.sh f21_sd_amr_2k     "$GPU" sd 3  64 $((MB/4)) 3 $ADAPT "$TL" "$OD" time/cfl=$CFL $CRIT
     ;;
  *) echo "unknown lane $lane" >&2; exit 1 ;;
esac
echo "lane $lane done"
