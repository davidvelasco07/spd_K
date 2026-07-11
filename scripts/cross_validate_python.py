#!/usr/bin/env python3
"""Cross-validate spd_K against the Python spd implementation (modular-gpu).

Runs the same 3D sine-wave advection (rho = 1 + 0.125 sin(2pi(x+y)),
v = (1,1,0), P = 1, gamma = 1.4, CFL = 0.8, ADER-SD, no fallback) in the
Python spd package and compares the primitive control-volume averages
against an spd_K W_cv output at the same time. Both fields are also
compared with the exact cell-averaged analytic solution.

Usage: cross_validate_python.py <spd_repo_path> <spdk_Wcv_file> <spdk_Xfaces_file> <N> <p> <t_end>
"""
import sys
import numpy as np

spd_path, wcv_file, xfaces_file = sys.argv[1], sys.argv[2], sys.argv[3]
N, p, t_end = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
sys.path.insert(0, spd_path)

from spd.sdfb_simulator import SPD_Simulator  # noqa: E402

NGH = 1
n = p + 1


def my_ic(xyz, case):
    x, y = xyz[0], xyz[1]
    if case == 0:
        return 1.0 + 0.125 * np.sin(2 * np.pi * (x + y))
    elif case in (1, 2):  # vx, vy
        return np.ones(x.shape)
    elif case == 4:       # pressure
        return np.ones(x.shape)
    else:                 # vz
        return np.zeros(x.shape)


def analytic_cellavg_rho(xf, yf, t):
    """Exact cell average of 1 + 0.125 sin(2pi(x+y-2t)) on the tensor grid
    defined by face coordinates xf, yf (1d arrays)."""
    a = 2 * np.pi

    def F(x, y):
        return -np.sin(a * (x + y - 2 * t)) / a**2

    X0, Y0 = np.meshgrid(xf[:-1], yf[:-1], indexing="ij")
    X1, Y1 = np.meshgrid(xf[1:], yf[1:], indexing="ij")
    area = (X1 - X0) * (Y1 - Y0)
    return 1.0 + 0.125 * (F(X1, Y1) - F(X0, Y1) - F(X1, Y0) + F(X0, Y0)) / area


# ---- Python spd run -------------------------------------------------------
s = SPD_Simulator(p=p, N=(N, N, N), init_fct=my_ic, cfl_coeff=0.8,
                  use_cupy=False, time_integrator="ader", scheme="SD")
s.perform_time_evolution(t_end)
W_py = np.asarray(s.dm.W_cv)  # (nvar, Nz,Ny,Nx, nz,ny,nx), no ghosts

# ---- spd_K output ---------------------------------------------------------
NVAR_K = 6  # 5 physical + FV bookkeeping slot
Nt = N + 2 * NGH
W_k = np.fromfile(wcv_file).reshape(1, NVAR_K, Nt, Nt, Nt, n, n, n)
W_k = W_k[0, :, NGH:-NGH, NGH:-NGH, NGH:-NGH]  # active region

# spd_K variable order: d, vx, vy, vz, p ; python: d, vx, vy, vz, p
names = ["rho", "vx", "vy", "vz", "P"]
print(f"config: N={N} p={p} t_end={t_end}  python steps dt0={s.dt:.6g}")
print(f"{'var':4s} {'max|K-py|':>12s} {'L1|K-py|':>12s}")
for v in range(5):
    d = np.abs(W_k[v] - W_py[v])
    print(f"{names[v]:4s} {d.max():12.3e} {d.mean():12.3e}")

# ---- analytic comparison (rho) -------------------------------------------
# face grid from spd_K's own output (fv faces incl. 2 ghost cells per side)
nGH = 2
xf_all = np.fromfile(xfaces_file)
xf = xf_all[nGH:len(xf_all) - nGH]  # active faces, spans [0,1]
assert abs(xf[0]) < 1e-14 and abs(xf[-1] - 1) < 1e-14, "unexpected face grid"

# sanity: python cell centers must sit on the same grid
xc_py = np.asarray(s.dm.X_cv)[nGH:-nGH]
xc_k = 0.5 * (xf[1:] + xf[:-1])
grid_diff = np.abs(xc_py - xc_k).max()
print(f"\ncell-center grid difference python vs spd_K: {grid_diff:.3e}")

rho_exact = analytic_cellavg_rho(xf, xf, t_end)  # (Nx*n, Ny*n)
# flatten element/point axes -> global cells: (Nz,Ny,Nx,nz,ny,nx) -> (z,y,x)
def to_cells(A):
    Nz, Ny, Nx, nz, ny, nx = A.shape
    return A.transpose(0, 3, 1, 4, 2, 5).reshape(Nz * nz, Ny * ny, Nx * nx)

rho_k = to_cells(W_k[0])
rho_py = to_cells(W_py[0])
# analytic is (x,y); broadcast over z with proper orientation: cells (z,y,x)
ex = rho_exact.T[None, :, :]  # (1, y, x)
e_k = np.abs(rho_k - ex).mean()
e_py = np.abs(rho_py - ex).mean()
print(f"\nL1 error vs analytic:  spd_K = {e_k:.6e}   python = {e_py:.6e}")
print(f"ratio spd_K/python = {e_k/e_py:.6f}")
