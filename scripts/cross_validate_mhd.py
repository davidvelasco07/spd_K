#!/usr/bin/env python3
"""Cross-validate the spd_K MHD module against the Python spd implementation.

Runs the Orszag-Tang vortex in Python spd (SD or SDFB/MOOD, rk3, LLF) and
compares the primitive control-volume averages against an spd_K W_cv output
at the same time.

Unit/domain mapping (exact symmetry of ideal MHD):
  Python spd: domain [-1/2,1/2]^2, rho = gamma^2, p = gamma, B0 = 1
  spd_K     : domain [0,1]^2 (quasi-2D slab), rho = 25/36pi, p = 5/12pi,
              B0 = 1/sqrt(4pi)
  =>  rho_K = rho_py / 4pi,  p_K = p_py / 4pi,  B_K = B_py / sqrt(4pi),
      v identical; cell (j,i) maps to cell (j,i) (the half-box coordinate
      shift in the Python IC cancels the domain offset).

Usage:
  cross_validate_mhd.py <spd_repo> <spdk_outdir> <out_idx> <N> <p> <t_end>
                        [--fallback]
"""
import os
import sys

import numpy as np

spd_path, outdir = sys.argv[1], sys.argv[2]
out_idx, N, p, t_end = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
fallback = "--fallback" in sys.argv
sys.path.insert(0, spd_path)

from spd.spd_simulator import SPD_Simulator  # noqa: E402
from spd.initial_conditions.initial_conditions_2d import orszag_tang, orszag_tang_Az  # noqa: E402

NGH = 1
n = p + 1
FOURPI = 4 * np.pi

# ---- Python spd run -------------------------------------------------------
kw = dict(scheme="SDFB", blending=False) if fallback else dict(scheme="SD")
s = SPD_Simulator(soe="mhd", p=p, N=(N, N), init_fct=orszag_tang,
                  vectorpot_fct=orszag_tang_Az, xlim=(-0.5, 0.5),
                  ylim=(-0.5, 0.5), gamma=5.0 / 3.0, cfl_coeff=0.4,
                  time_integrator="rk3", riemann_solver="llf",
                  use_cupy=False, verbose=False, **kw)
s.perform_time_evolution(t_end)
W_py = np.asarray(s.dm.asnumpy(s.dm.W_cv))   # (nvar, Ny, Nx, ny, nx)
print(f"python spd: {s.n_step} steps to t = {s.time:.6f}")

# ---- spd_K output (quasi-2D slab: take one z plane of cell averages) ------
import glob
import re
fs = glob.glob(os.path.join(outdir, f"W_cv_N{N}p{p}_{out_idx}_0.dat"))
assert fs, f"no spd_K W_cv output index {out_idx} in {outdir}"
raw = np.fromfile(fs[0])
# z extent: infer elements from the file size (nvar=8 fixed); a true-2D run
# (mesh/nx3=1) has a single degenerate z element with one point
nz_tot = raw.size // (8 * (N + 2 * NGH) ** 2 * n * n * n)
if nz_tot >= 1:
    W_k = raw.reshape(1, 8, nz_tot, N + 2 * NGH, N + 2 * NGH, n, n, n)
    W_k = W_k[0, :, nz_tot // 2, NGH:-NGH, NGH:-NGH, n // 2]  # (nvar, Ny,Nx, ny,nx)
else:
    W_k = raw.reshape(8, N + 2 * NGH, N + 2 * NGH, 1, n, n)
    W_k = W_k[:, NGH:-NGH, NGH:-NGH, 0]                       # (nvar, Ny,Nx, ny,nx)

# ---- compare (map python -> spd_K units) ----------------------------------
scale = {0: 1 / FOURPI, 4: 1 / FOURPI, 5: 1 / np.sqrt(FOURPI),
         6: 1 / np.sqrt(FOURPI), 7: 1 / np.sqrt(FOURPI)}
names = ["rho", "vx", "vy", "vz", "P", "Bx", "By", "Bz"]
print(f"\nconfig: N={N} p={p} t_end={t_end} fallback={fallback}")
print(f"{'var':4s} {'max|K-py|':>12s} {'L1|K-py|':>12s} {'rel L1':>12s}")
worst = 0.0
for v in range(8):
    py = W_py[v] * scale.get(v, 1.0)
    d = np.abs(W_k[v] - py)
    ref = np.abs(py).max()
    if ref < 1e-12:            # identically-zero field: report absolute error
        print(f"{names[v]:4s} {d.max():12.3e} {d.mean():12.3e} {'(zero)':>12s}")
        continue
    rel = d.mean() / ref
    worst = max(worst, rel)
    print(f"{names[v]:4s} {d.max():12.3e} {d.mean():12.3e} {rel:12.3e}")
print(f"\nworst relative L1 = {worst:.3e}")
