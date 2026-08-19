#!/usr/bin/env python3
"""3x3 mosaic: slope limiter (columns) vs resolution (rows) at a fixed time.

The MHD Kelvin-Helmholtz shear layer of Stone et al. (2020) fig. 22 is only
L = 0.01 wide, so on a unit box it spans ~1.3 cells at 128^2 and ~5 at 512^2.
That makes this problem a direct read on how much of the layer each limiter
destroys before the instability can grow -- which is the same quantity that
sets how much of the fig-21/22 uniform-vs-AMR difference map is real.
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
    nvar = raw.size // (Ne * Ne * n * n)
    A = raw.reshape((1, nvar, 1, Ne, Ne, 1, n, n), order=dump_order(os.path.dirname(path)))
    U = A[0, var, 0, NGH:-NGH, NGH:-NGH, 0]
    Ny, Nx, ny, nx = U.shape
    return U.transpose(0, 2, 1, 3).reshape(Ny * ny, Nx * nx)

def last(rundir):
    fs = glob.glob(os.path.join(rundir, "W_cv_*.dat"))
    if not fs:
        return None
    return max(fs, key=lambda f: int(re.search(r"_(\d+)_0\.dat$", f).group(1)))

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/kh_mhd"))
ap.add_argument("--var", type=int, default=0, help="0 = density")
ap.add_argument("-o", "--out", default="kh_mhd_limiters.png")
ap.add_argument("--res", default="128,256,512", help="comma-separated resolutions (rows)")
ap.add_argument("--lims", default="minmod,vanleer,moncen", help="comma-separated limiters (columns)")
a = ap.parse_args()

LIMS = [x for x in a.lims.split(",") if x]
RES  = [int(x) for x in a.res.split(",") if x]

fig, axes = plt.subplots(len(RES), len(LIMS), figsize=(3.7*len(LIMS), 3.9*len(RES)+0.8),
                         constrained_layout=True)

# One shared colour scale across every panel, or the comparison is meaningless.
data = {}
for r, N in enumerate(RES):
    for c, L in enumerate(LIMS):
        f = last(os.path.join(a.root, f"f22_{L}_{N}"))
        data[(r, c)] = None if f is None else load(f, a.var)
vals = [d for d in data.values() if d is not None]
lo = min(d.min() for d in vals); hi = max(d.max() for d in vals)

for r, N in enumerate(RES):
    for c, L in enumerate(LIMS):
        ax = axes[r, c]
        d = data[(r, c)]
        if d is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="0.5")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        im = ax.imshow(d, origin="lower", extent=[0, 1, 0, 1],
                       cmap="viridis", vmin=lo, vmax=hi)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(L, fontsize=13)
        if c == 0:
            # cells across the L = 0.01 shear layer, the quantity that matters
            ax.set_ylabel(f"${N}^2$\n({0.01*N:.1f} cells across $L$)", fontsize=11)

fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01,
             label=r"density $\rho$")
fig.suptitle("MHD Kelvin-Helmholtz at $t=1.5$ (Stone et al. 2020 fig. 22 setup)\n"
             "uniform, MUSCL + RK2, $B_x=0.1$ -- slope limiter vs resolution",
             fontsize=13)
fig.savefig(a.out, dpi=140)
print("wrote", a.out)
