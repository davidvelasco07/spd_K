#!/usr/bin/env python3
"""Athena++ figure 21 layout for one scheme: uniform vs AMR at equal resolution.

Panels follow Stone et al. (2020) figure 21:
  top left      density on the uniform grid
  top right     density with AMR at the same finest-level resolution
  bottom left   fractional difference in density between the two
  bottom right  the leaf MeshBlock distribution of the AMR run

Run it once per scheme (MUSCL, SD p=3); both schemes are set up to carry the
same number of degrees of freedom so the two figures are directly comparable.
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import numpy as np

from plot_kh import load_rho, last_output

LEVEL_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]


def load_blocks(path):
    """Parse amr_blocks_N.txt -> (header dict, list of leaf blocks)."""
    blocks, header = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            if header is None:
                header = dict(nblocks=int(v[0]), max_level=int(v[1]))
                continue
            blocks.append(dict(level=int(v[1]),
                               x0=float(v[5]), x1=float(v[6]),
                               y0=float(v[7]), y1=float(v[8])))
    return header, blocks


def blocks_for(outdir, index):
    path = os.path.join(outdir, f"amr_blocks_{index}.txt")
    return load_blocks(path) if os.path.isfile(path) else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform", required=True)
    ap.add_argument("--amr", required=True)
    ap.add_argument("--scheme", required=True, help="label, e.g. 'MUSCL'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tlabel", default="1.2")
    ap.add_argument("--dmin", type=float, default=1e-6,
                    help="floor of the log difference scale")
    ap.add_argument("--dmax", type=float, default=1.0,
                    help="ceiling of the log difference scale")
    args = ap.parse_args()

    f_uni, f_amr = last_output(args.uniform), last_output(args.amr)
    rho_u, rho_a = load_rho(f_uni), load_rho(f_amr)
    if rho_u.shape != rho_a.shape:
        raise SystemExit(f"resolution mismatch: uniform {rho_u.shape} vs AMR {rho_a.shape}")

    idx = int(re.search(r"_(\d+)_\d+\.dat$", os.path.basename(f_amr)).group(1))
    header, blocks = blocks_for(args.amr, idx)

    # Paper domain is [-0.5,0.5]^2; the code box is [0,1]^2.
    ext = [-0.5, 0.5, -0.5, 0.5]
    # rms is the root mean square of the fractional difference, sqrt(mean(frac^2)),
    # the quantity Tables 4/5 of Paper V and kh_stats.py quote; the label used to
    # print frac.std(), which drops the mean.
    frac = (rho_a - rho_u) / rho_u

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.6))
    D = rho_u.shape[0]

    for ax, rho, name in ((axes[0, 0], rho_u, f"uniform  {D}$^2$ DoF"),
                          (axes[0, 1], rho_a, f"AMR  {D}$^2$ DoF at finest level")):
        im = ax.imshow(rho, origin="lower", extent=ext, cmap="RdBu_r",
                       vmin=1.0, vmax=2.0, interpolation="nearest")
        ax.set_title(name, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, label=r"$\rho$")

    # Log scale on |difference|, as the paper draws this panel: the error spans
    # several decades and lives in thin filaments, so a linear diverging scale
    # shows the roll edges and hides everything else. Fixed decades by default so
    # that two of these figures (e.g. spd_K and AthenaK) are directly comparable;
    # override with --dmin/--dmax.
    adiff = np.abs(frac)
    im = axes[1, 0].imshow(np.maximum(adiff, args.dmin), origin="lower", extent=ext,
                           cmap="inferno", norm=LogNorm(vmin=args.dmin, vmax=args.dmax),
                           interpolation="nearest")
    axes[1, 0].set_title(
        f"|fractional difference| in $\\rho$ (AMR $-$ uniform)\n"
        f"max = {adiff.max():.3f}, rms = {np.sqrt((frac**2).mean()):.4f}", fontsize=11)
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046,
                 label=r"$|\rho_{\rm AMR}-\rho_{\rm uni}|/\rho_{\rm uni}$")

    ax = axes[1, 1]
    if blocks:
        levels = sorted({b["level"] for b in blocks})
        for b in blocks:
            c = LEVEL_COLORS[b["level"] % len(LEVEL_COLORS)]
            ax.add_patch(mpatches.Rectangle(
                (b["x0"] - 0.5, b["y0"] - 0.5), b["x1"] - b["x0"], b["y1"] - b["y0"],
                fill=False, ec=c, lw=0.45))
        counts = {L: sum(1 for b in blocks if b["level"] == L) for L in levels}
        ax.legend(handles=[mpatches.Patch(fc="none", ec=LEVEL_COLORS[L % 5],
                                          label=f"level {L}: {counts[L]}")
                           for L in levels], fontsize=9, loc="upper right")
        ax.set_title(f"AMR MeshBlocks: {header['nblocks']} leaves, "
                     f"max level {header['max_level']}", fontsize=11)
    else:
        ax.text(0.5, 0.5, "no amr_blocks file", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")

    for a in axes.ravel():
        a.set_xlabel("x")
        a.set_ylabel("y")

    fig.suptitle(f"Kelvin-Helmholtz at $t={args.tlabel}$, {args.scheme}: "
                 f"uniform vs AMR ({D}$^2$ DoF)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}: |frac| max {np.abs(frac).max():.4f}, rms {np.sqrt((frac**2).mean()):.5f}")


if __name__ == "__main__":
    main()
