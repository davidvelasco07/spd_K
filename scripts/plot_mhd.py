#!/usr/bin/env python3
"""Quick-look plots for the MHD (Orszag-Tang) runs.

Reads spd_K binary outputs from one or more output dirs and produces, for each
output index: density, gas pressure, |B|^2 and (when present) the MOOD cascade
level, on the mid-z slice. Usage:

    python3 scripts/plot_mhd.py <outdir> [<outdir2> ...] [--dest plots]
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import spdk_io as io


def load_sd_nvar(grid, i, var, prefix, nvar):
    """spdk_io.load_sd with an explicit variable count (W_cv has 8, B2_cv 4)."""
    fn = os.path.join(grid.outdir, f"{prefix}_N{grid.N[0]}p{grid.p}_{i}_0.dat")
    A = np.fromfile(fn).reshape(grid.sd_shape(nvar))
    v = A[0, var]
    sl = lambda d: slice(io.NGH, -io.NGH) if grid.active(d) else slice(None)
    v = v[sl(2), sl(1), sl(0)]
    Nz, Ny, Nx, nz, ny, nx = v.shape
    return v.transpose(0, 3, 1, 4, 2, 5).reshape(Nz * nz, Ny * ny, Nx * nx)


def load_cascade(grid, i):
    fs = glob.glob(os.path.join(grid.outdir, f"cascade_N*_{i}_0.dat"))
    if not fs:
        return None
    A = np.fromfile(fs[0]).reshape(grid.fv_shape(1))
    sl = lambda d: slice(io.nGH, -io.nGH) if grid.active(d) else slice(0, 1)
    return A[0, sl(2), sl(1), sl(0)]


def midz(a):
    return a[a.shape[0] // 2]


def plot_output(grid, i, dest, tag):
    rho = load_sd_nvar(grid, i, 0, "W_cv", 8)
    prs = load_sd_nvar(grid, i, 4, "W_cv", 8)
    b2 = load_sd_nvar(grid, i, 3, "B2_cv", 4)   # rows: Bx,By,Bz,|B|^2
    casc = load_cascade(grid, i)

    x = grid.faces[0]
    y = grid.faces[1]
    ext = [x[0], x[-1], y[0], y[-1]]
    ncol = 4 if casc is not None else 3
    fig, ax = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.9), constrained_layout=True)
    panels = [(midz(rho), "rho", "viridis", None),
              (midz(prs), "P", "inferno", None),
              (midz(b2), "|B|^2", "magma", None)]
    if casc is not None:
        panels.append((midz(casc), "cascade level", "Reds", (0, 2)))
    for a, (f, title, cmap, clim) in zip(ax, panels):
        im = a.imshow(f, origin="lower", extent=ext, cmap=cmap,
                      vmin=None if clim is None else clim[0],
                      vmax=None if clim is None else clim[1])
        a.set_title(title)
        fig.colorbar(im, ax=a, shrink=0.85)
    fig.suptitle(f"{tag}  output {i}")
    fn = os.path.join(dest, f"{tag}_out{i}.png")
    fig.savefig(fn, dpi=110)
    plt.close(fig)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdirs", nargs="+")
    ap.add_argument("--dest", default="plots")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    for outdir in args.outdirs:
        grid = io.Grid(outdir)
        tag = os.path.basename(os.path.normpath(outdir))
        for i in io.output_indices(outdir, "W_cv"):
            try:
                fn = plot_output(grid, i, args.dest, tag)
                print(fn)
            except Exception as e:  # a run may have crashed mid-output
                print(f"skip {tag} output {i}: {e}")


if __name__ == "__main__":
    main()
