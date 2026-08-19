#!/usr/bin/env python3
"""Figure-22-style view of an MHD Kelvin-Helmholtz run (Stone et al. 2020).

Figure 22 itself is a uniform-vs-AMR comparison, which needs both lanes. This
plots what a single lane actually shows: the density rolls and the winding-up
of the initially uniform Bx, side by side in time. Density and |B| share a
column so the field amplification can be read against the roll that causes it.
"""
import argparse, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NGH = 1

def dump_order(outdir):
    try:
        with open(os.path.join(outdir, "parameters.txt")) as fh:
            for line in fh:
                if line.strip().startswith("layout"):
                    return "C" if "LayoutRight" in line else "F"
    except OSError:
        pass
    return "F"

def load(path, var):
    m = re.search(r"_N(\d+)p(\d+)_", os.path.basename(path))
    N, p = int(m.group(1)), int(m.group(2))
    n = p + 1
    Ne = N + 2 * NGH
    raw = np.fromfile(path)
    per_var = Ne * Ne * n * n
    nvar = raw.size // per_var
    A = raw.reshape((1, nvar, 1, Ne, Ne, 1, n, n), order=dump_order(os.path.dirname(path)))
    U = A[0, var, 0, NGH:-NGH, NGH:-NGH, 0]
    Ny, Nx, ny, nx = U.shape
    return U.transpose(0, 2, 1, 3).reshape(Ny * ny, Nx * nx)

ap = argparse.ArgumentParser()
ap.add_argument("rundir")
ap.add_argument("-o", "--out", default="kh_mhd.png")
ap.add_argument("--dt", type=float, default=0.3, help="output cadence")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.rundir, "W_cv_*.dat")),
               key=lambda f: int(re.search(r"_(\d+)_0\.dat$", f).group(1)))
pick = [0, len(files)//2 - 1, len(files)-2, len(files)-1]
pick = sorted(set(i for i in pick if 0 <= i < len(files)))

fig, axes = plt.subplots(2, len(pick), figsize=(3.1*len(pick), 6.4),
                         constrained_layout=True)
if len(pick) == 1:
    axes = axes.reshape(2, 1)

# shared scales so the panels are comparable across time
rhos = [load(files[i], 0) for i in pick]
bmag = [np.hypot(load(files[i], 5), load(files[i], 6)) for i in pick]
rlo, rhi = min(r.min() for r in rhos), max(r.max() for r in rhos)
blo, bhi = min(b.min() for b in bmag), max(b.max() for b in bmag)

for col, idx in enumerate(pick):
    t = idx * a.dt
    im0 = axes[0, col].imshow(rhos[col], origin="lower", extent=[0, 1, 0, 1],
                              cmap="viridis", vmin=rlo, vmax=rhi)
    axes[0, col].set_title(f"$t = {t:.1f}$", fontsize=11)
    im1 = axes[1, col].imshow(bmag[col], origin="lower", extent=[0, 1, 0, 1],
                              cmap="magma", vmin=blo, vmax=bhi)
    for r in (0, 1):
        axes[r, col].set_xticks([]); axes[r, col].set_yticks([])

axes[0, 0].set_ylabel(r"density  $\rho$", fontsize=11)
axes[1, 0].set_ylabel(r"field strength  $|B|$", fontsize=11)
fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.026, pad=0.01)
fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.026, pad=0.01)

name = os.path.basename(os.path.normpath(a.rundir))
fig.suptitle(f"MHD Kelvin-Helmholtz, {name}   "
             r"(uniform, MUSCL + RK2, $B_x=0.1$)", fontsize=12)
fig.savefig(a.out, dpi=150)
print("wrote", a.out)
