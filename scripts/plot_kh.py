#!/usr/bin/env python3
"""Figure-21-style Kelvin-Helmholtz comparison: MUSCL vs SD+fallback at equal DoF.

Recreates the layout of figure 21 of Stone et al. (2020, Athena++) -- two
density fields, their fractional difference, and a fourth diagnostic panel --
but the two fields being compared are the two *schemes* rather than a uniform
and an AMR mesh, run at a matched degree-of-freedom count.

Both runs live on a DxD grid of degrees of freedom: MUSCL uses D cells (p=0),
SD uses D/(p+1) elements of (p+1)^2 points each.
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

NGH = 1
NVAR = 6


def dump_order(outdir):
    """numpy reshape order for binaries in outdir, from its parameters.txt."""
    try:
        with open(os.path.join(outdir, "parameters.txt")) as fh:
            for line in fh:
                if line.strip().startswith("layout"):
                    return "C" if "LayoutRight" in line else "F"
    except OSError:
        pass
    return "F"


def load_rho(path, var=0):
    """Load one 2D variable from a W_cv dump as a (D, D) array of cell values."""
    m = re.search(r"_N(\d+)p(\d+)_", os.path.basename(path))
    N, p = int(m.group(1)), int(m.group(2))
    n = p + 1
    Ne, shp = N + 2 * NGH, None
    shp = (1, NVAR, 1, Ne, Ne, 1, n, n)
    raw = np.fromfile(path)
    if raw.size != int(np.prod(shp)):
        raise ValueError(f"{path}: size {raw.size} != expected {np.prod(shp)}")
    A = raw.reshape(shp, order=dump_order(os.path.dirname(path)))
    U = A[0, var, 0, NGH:-NGH, NGH:-NGH, 0]      # (Ny, Nx, ny, nx)
    Ny, Nx, ny, nx = U.shape
    return U.transpose(0, 2, 1, 3).reshape(Ny * ny, Nx * nx)


def last_output(outdir):
    files = glob.glob(os.path.join(outdir, "W_cv_*.dat"))
    if not files:
        raise SystemExit(f"no W_cv output in {outdir}")
    return max(files, key=lambda f: int(re.search(r"_(\d+)_\d+\.dat$", f).group(1)))


def label(outdir, fallback):
    """Scheme label with the element/point decomposition behind the DoF count."""
    path = last_output(outdir)
    m = re.search(r"_N(\d+)p(\d+)_", os.path.basename(path))
    N, p = int(m.group(1)), int(m.group(2))
    D = N * (p + 1)
    if p == 0:
        return f"{fallback}\n{N}$^2$ cells = {D}$^2$ DoF"
    return f"{fallback}\n{N}$^2$ elements $\\times$ {p+1}$^2$ pts = {D}$^2$ DoF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muscl", required=True, help="MUSCL run output dir")
    ap.add_argument("--sdfb", required=True, help="SD+fallback run output dir")
    ap.add_argument("--out", default="apollo_plots/kh_figure21.png")
    ap.add_argument("--tlabel", default="1.2")
    ap.add_argument("--zoom", type=float, nargs=4, default=[-0.16, 0.14, 0.10, 0.40],
                    help="x0 x1 y0 y1 in paper coords for the zoom panel")
    args = ap.parse_args()

    rho_m = load_rho(last_output(args.muscl))
    rho_s = load_rho(last_output(args.sdfb))
    if rho_m.shape != rho_s.shape:
        raise SystemExit(f"DoF mismatch: MUSCL {rho_m.shape} vs SD {rho_s.shape}")

    # Paper domain is [-0.5,0.5]^2; the code box is [0,1]^2.
    ext = [-0.5, 0.5, -0.5, 0.5]
    vmin, vmax = 1.0, 2.0
    frac = (rho_m - rho_s) / rho_s

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.4))

    for ax, rho, name in ((axes[0, 0], rho_m, label(args.muscl, "MUSCL")),
                          (axes[0, 1], rho_s, label(args.sdfb, "SD $p{=}3$ + FV fallback"))):
        im = ax.imshow(rho, origin="lower", extent=ext, cmap="RdBu_r",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(name, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, label=r"$\rho$")

    lim = float(np.nanpercentile(np.abs(frac), 99.5))
    im = axes[1, 0].imshow(frac, origin="lower", extent=ext, cmap="RdBu_r",
                           vmin=-lim, vmax=lim, interpolation="nearest")
    axes[1, 0].set_title(
        f"fractional difference in $\\rho$\nmax |.| = {np.abs(frac).max():.3f}, "
        f"rms = {frac.std():.4f}", fontsize=11)
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046,
                 label=r"$(\rho_{\rm MUSCL}-\rho_{\rm SD})/\rho_{\rm SD}$")

    # Fourth panel: the same roll from both runs, side by side, so the
    # small-scale structure each scheme supports at equal DoF is directly
    # comparable. A one-cell white gutter marks the seam.
    x0, x1, y0, y1 = args.zoom
    D = rho_m.shape[0]
    def idx(v):
        return int(np.clip(round((v + 0.5) * D), 0, D))
    i0, i1, j0, j1 = idx(x0), idx(x1), idx(y0), idx(y1)
    sub_m, sub_s = rho_m[j0:j1, i0:i1], rho_s[j0:j1, i0:i1]
    gutter = np.full((sub_m.shape[0], max(2, sub_m.shape[1] // 60)), np.nan)
    pair = np.hstack([sub_m, gutter, sub_s])
    im = axes[1, 1].imshow(pair, origin="lower", cmap="RdBu_r",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1, 1].text(0.25, 0.98, "MUSCL", transform=axes[1, 1].transAxes,
                    ha="center", va="top", fontsize=10,
                    bbox=dict(fc="w", alpha=0.8, ec="none"))
    axes[1, 1].text(0.75, 0.98, "SD $p{=}3$", transform=axes[1, 1].transAxes,
                    ha="center", va="top", fontsize=10,
                    bbox=dict(fc="w", alpha=0.8, ec="none"))
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_title(
        f"same roll, same DoF: $x\\in[{x0},{x1}]$, $y\\in[{y0},{y1}]$", fontsize=11)
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, label=r"$\rho$")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(
        f"Kelvin-Helmholtz at $t={args.tlabel}$: MUSCL vs SD $p{{=}}3$+fallback "
        f"at matched DoF ({rho_m.shape[0]}$^2$)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    print(f"  grid {rho_m.shape}, rho MUSCL [{rho_m.min():.3f},{rho_m.max():.3f}], "
          f"SD [{rho_s.min():.3f},{rho_s.max():.3f}]")


if __name__ == "__main__":
    main()
