#!/usr/bin/env python3
"""Athena++ figure 22 layout from ATHENAK output: uniform vs AMR at equal finest
resolution.

The same four panels as plot_kh_fig21.py -- uniform density, AMR density, their
fractional difference, and the AMR leaf-block map -- but reading AthenaK's binary
output instead of spd_K's dumps, so that the reference code's own version of the
figure can be put beside spd_K's.

AthenaK writes one record per MeshBlock. `mb_index` is BLOCK-LOCAL; global
placement comes from `mb_logical` = (lx1, lx2, lx3, level), and for the AMR field
`read_binary_as_athdf(..., level=n)` assembles everything onto a uniform grid at
level n, replicating coarse cells (nearest-neighbour prolongation) -- which is
what "AMR at 2048^2 finest" means for a picture.

Needs AthenaK's own reader; point --vis at <athenak>/vis/python.
"""
import argparse
import collections
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import numpy as np

LEVEL_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]


def uniform_field(reader, path, var=0):
    """Assemble a single-level run onto its global grid."""
    d = reader.read_binary(path)
    names = [n if isinstance(n, str) else n.decode() for n in d["var_names"]]
    nx1, nx2 = int(d["nx1_out_mb"]), int(d["nx2_out_mb"])
    out = np.full((d["Nx2"], d["Nx1"]), np.nan)
    for m in range(d["n_mbs"]):
        lx1, lx2, _, lev = d["mb_logical"][m]
        if lev != 0:
            raise SystemExit(f"{path}: expected a uniform grid, found level {lev}")
        out[lx2*nx2:(lx2+1)*nx2, lx1*nx1:(lx1+1)*nx1] = \
            np.asarray(d["mb_data"][names[var]][m])[0]
    if np.isnan(out).any():
        raise SystemExit(f"{path}: part of the grid was never filled")
    return out, d


def amr_field(reader, path, level, var_name):
    """Assemble a mixed-level run onto the uniform grid of `level`."""
    d = reader.read_binary_as_athdf(path, level=level, quantities=[var_name])
    return np.asarray(d[var_name])[0], d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform", required=True, help="AthenaK .bin from the uniform run")
    ap.add_argument("--amr", required=True, help="AthenaK .bin from the AMR run")
    ap.add_argument("--vis", default=os.path.expanduser("~/athenak/vis/python"),
                    help="path to AthenaK's vis/python (for bin_convert)")
    ap.add_argument("--scheme", default="MHD PLM, HLLD + rk2")
    ap.add_argument("--tlabel", default="1.5")
    ap.add_argument("--dmin", type=float, default=1e-6,
                    help="floor of the log difference scale")
    ap.add_argument("--dmax", type=float, default=1.0,
                    help="ceiling of the log difference scale")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.vis)
    import bin_convert as reader

    rho_u, du = uniform_field(reader, args.uniform)
    # The finest level present in the AMR run sets the comparison grid; it has to
    # match the uniform run's, or the two are not the same experiment.
    da_raw = reader.read_binary(args.amr)
    levels = np.array(da_raw["mb_logical"])[:, 3]
    max_level = int(levels.max())
    var_name = da_raw["var_names"][0]
    var_name = var_name if isinstance(var_name, str) else var_name.decode()
    rho_a, _ = amr_field(reader, args.amr, max_level, var_name)
    if rho_a.shape != rho_u.shape:
        raise SystemExit(f"resolution mismatch: uniform {rho_u.shape} vs "
                         f"AMR at level {max_level} {rho_a.shape}")

    ext = [-0.5, 0.5, -0.5, 0.5]
    frac = (rho_a - rho_u) / rho_u
    D = rho_u.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.6))
    for ax, rho, name in ((axes[0, 0], rho_u, f"uniform  {D}$^2$ DoF"),
                          (axes[0, 1], rho_a, f"AMR  {D}$^2$ DoF at finest level")):
        im = ax.imshow(rho, origin="lower", extent=ext, cmap="RdBu_r",
                       vmin=1.0, vmax=2.0, interpolation="nearest")
        ax.set_title(name, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, label=r"$\rho$")

    # Log scale on |difference|, as the paper draws it, with the same fixed
    # decades as plot_kh_fig21.py so the two codes' figures can be read side by
    # side.
    adiff = np.abs(frac)
    im = axes[1, 0].imshow(np.maximum(adiff, args.dmin), origin="lower", extent=ext,
                           cmap="inferno", norm=LogNorm(vmin=args.dmin, vmax=args.dmax),
                           interpolation="nearest")
    axes[1, 0].set_title(
        f"|fractional difference| in $\\rho$ (AMR $-$ uniform)\n"
        f"max = {adiff.max():.3f}, rms = {frac.std():.4f}", fontsize=11)
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046,
                 label=r"$|\rho_{\rm AMR}-\rho_{\rm uni}|/\rho_{\rm uni}$")

    ax = axes[1, 1]
    counts = collections.Counter(levels.tolist())
    for m in range(da_raw["n_mbs"]):
        x0, x1, y0, y1 = da_raw["mb_geometry"][m][:4]
        lev = int(da_raw["mb_logical"][m][3])
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                        lw=0.35, ec=LEVEL_COLORS[lev % len(LEVEL_COLORS)]))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.legend(handles=[mpatches.Patch(color=LEVEL_COLORS[l % len(LEVEL_COLORS)],
                                      label=f"level {l}: {counts[l]}")
                       for l in sorted(counts)], fontsize=9, loc="upper right")
    ax.set_title(f"AMR MeshBlocks: {da_raw['n_mbs']} leaves, max level {max_level}",
                 fontsize=11)

    for a in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
        a.set_xlabel("x")
        a.set_ylabel("y")

    fig.suptitle(f"AthenaK: Kelvin-Helmholtz at $t={args.tlabel}$, {args.scheme}: "
                 f"uniform vs AMR ({D}$^2$ DoF)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}: |frac| max {np.abs(frac).max():.4f}, "
          f"rms {frac.std():.5f}, {da_raw['n_mbs']} leaves")


if __name__ == "__main__":
    main()
