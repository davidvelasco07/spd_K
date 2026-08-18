#!/usr/bin/env python3
"""Paper-style figures for Balsara et al. 2025 ApJ 988 non-relativistic MHD tests.

Fig. 10 style (strong blast): rho, P, |v|^2, |B|^2
Fig. 11 style (magnetized jet): log10 rho, log10 P, |B|^2, cascade level

Usage:
  python3 scripts/plot_balsara_figures.py <outdir> --test blast|jet [--dest plots/balsara]
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


def load_cv_field(grid, i, var, prefix="W_cv", nvar=8):
    fn = os.path.join(grid.outdir, f"{prefix}_N{grid.N[0]}p{grid.p}_{i}_0.dat")
    A = np.fromfile(fn).reshape(grid.sd_shape(nvar))
    v = A[0, var]
    sl = lambda d: slice(io.NGH, -io.NGH) if grid.active(d) else slice(None)
    v = v[sl(2), sl(1), sl(0)]
    Nz, Ny, Nx, nz, ny, nx = v.shape
    return v.transpose(0, 3, 1, 4, 2, 5).reshape(Nz * nz, Ny * ny, Nx * nx)


def cell_average(field, grid):
    """Collapse fine FV subcells to one value per cell (element average)."""
    n = grid.n
    Ny, Nx = field.shape
    my, mx = Ny // n, Nx // n
    return field.reshape(my, n, mx, n).mean(axis=(1, 3))


def load_b2(grid, i):
    fn = glob.glob(os.path.join(grid.outdir, f"B2_cv_N*_{i}_0.dat"))
    if not fn:
        bx = load_cv_field(grid, i, 5, nvar=8)
        by = load_cv_field(grid, i, 6, nvar=8)
        bz = load_cv_field(grid, i, 7, nvar=8)
        return bx * bx + by * by + bz * bz
    A = np.fromfile(fn[0]).reshape(grid.sd_shape(4))
    b2 = A[0, 3]
    sl = lambda d: slice(io.NGH, -io.NGH) if grid.active(d) else slice(None)
    b2 = b2[sl(2), sl(1), sl(0)]
    Nz, Ny, Nx, nz, ny, nx = b2.shape
    return b2.transpose(0, 3, 1, 4, 2, 5).reshape(Nz * nz, Ny * ny, Nx * nx)


def load_cascade(grid, i):
    fs = glob.glob(os.path.join(grid.outdir, f"cascade_N*_{i}_0.dat"))
    if not fs:
        return None
    A = np.fromfile(fs[0]).reshape(grid.fv_shape(1))
    sl = lambda d: slice(io.nGH, -io.nGH) if grid.active(d) else slice(0, 1)
    return A[0, sl(2), sl(1), sl(0)]


def midz(a):
    return a[a.shape[0] // 2] if a.ndim == 3 else a


def plot_blast(grid, i, dest, tag):
    rho = midz(load_cv_field(grid, i, 0))
    prs = midz(load_cv_field(grid, i, 4))
    vx = midz(load_cv_field(grid, i, 1))
    vy = midz(load_cv_field(grid, i, 2))
    vmag2 = vx * vx + vy * vy
    b2 = midz(load_b2(grid, i))

    x = grid.faces[0]
    y = grid.faces[1]
    ext = [x[0], x[-1], y[0], y[-1]]
    panels = [
        (rho, r"$\rho$", "viridis", None),
        (prs, r"$P$", "inferno", None),
        (vmag2, r"$|v|^2$", "cividis", None),
        (b2, r"$|B|^2$", "magma", None),
    ]
    fig, ax = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for a, (f, title, cmap, clim) in zip(ax.ravel(), panels):
        im = a.imshow(f, origin="lower", extent=ext, cmap=cmap, aspect="equal",
                      vmin=None if clim is None else clim[0],
                      vmax=None if clim is None else clim[1])
        a.set_xlabel("x")
        a.set_ylabel("y")
        a.set_title(title)
        fig.colorbar(im, ax=a, shrink=0.82)
    fig.suptitle(f"Strong MHD blast (Fig. 10 style) — {tag}  t index {i}", fontsize=12)
    fn = os.path.join(dest, f"{tag}_blast_out{i}.png")
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    return fn


def plot_jet(grid, i, dest, tag):
    rho = midz(load_cv_field(grid, i, 0))
    prs = midz(load_cv_field(grid, i, 4))
    b2 = midz(load_b2(grid, i))
    casc = load_cascade(grid, i)
    if casc is not None:
        casc = midz(casc)

    x = grid.faces[0]
    y = grid.faces[1]
    xmid = 0.5 * (x[0] + x[-1])
    ext = [x[0] - xmid, x[-1] - xmid, y[0], y[-1]]

    log_eps = 1e-6
    lrho = np.log10(np.maximum(rho, log_eps))
    lprs = np.log10(np.maximum(prs, log_eps))

    panels = [
        (lrho, r"$\log_{10}\rho$", "gist_heat", (-2.0, 1.0)),
        (lprs, r"$\log_{10}P$", "gist_heat", (-2.0, 3.0)),
        (b2, r"$|B|^2$", "inferno", (0.0, None)),
    ]
    fig, ax = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    axes = list(ax.ravel())
    for a, (f, title, cmap, clim) in zip(axes, panels):
        vmin = clim[0] if clim else None
        vmax = clim[1] if clim and clim[1] is not None else None
        im = a.imshow(f, origin="lower", extent=ext, cmap=cmap, aspect="auto",
                      vmin=vmin, vmax=vmax)
        a.set_xlabel("x")
        a.set_ylabel("y")
        a.set_title(title)
        fig.colorbar(im, ax=a, shrink=0.85)
    if casc is not None:
        im = axes[-1].imshow(casc, origin="lower", extent=ext, cmap="Reds",
                             vmin=0, vmax=2, aspect="auto")
        axes[-1].set_title("MOOD cascade")
        axes[-1].set_xlabel("x")
        axes[-1].set_ylabel("y")
        fig.colorbar(im, ax=axes[-1], shrink=0.85)
    else:
        axes[-1].axis("off")
    fig.suptitle(f"Mach-800 MHD jet (Balsara et al. 2025 Fig. 11 style) — {tag}", fontsize=12)
    fn = os.path.join(dest, f"{tag}_jet_out{i}.png")
    fig.savefig(fn, dpi=160)
    plt.close(fig)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--test", choices=["blast", "jet"], required=True)
    ap.add_argument("--dest", default="plots/balsara")
    ap.add_argument("--index", type=int, default=-1, help="output index (-1 = last)")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    grid = io.Grid(args.outdir)
    idx = io.output_indices(args.outdir, "W_cv")
    if not idx:
        raise SystemExit(f"no W_cv outputs in {args.outdir}")
    i = idx[args.index]
    tag = os.path.basename(os.path.normpath(args.outdir))
    if args.test == "blast":
        fn = plot_blast(grid, i, args.dest, tag)
    else:
        fn = plot_jet(grid, i, args.dest, tag)
    print(fn)


if __name__ == "__main__":
    main()
