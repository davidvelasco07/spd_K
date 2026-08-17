#!/usr/bin/env python3
"""Scheme mosaic: one row per scheme, one column per resolution, density at t_end.

Companion to plot_kh_fig21.py, which compares uniform against AMR for a single
scheme. This one holds the *discretisation* fixed (uniform) and varies the
scheme, so MUSCL (p=0) and SDFB (p=3) can be read side by side at matched
degrees of freedom -- the comparison the fig-21 setup is built for, since both
schemes carry the same DoF count per direction.

usage:
  mosaic_kh_schemes.py --row "MUSCL (p=0):dir512,dir1024,dir2048" \
                       --row "SDFB (p=3):dir512,dir1024,dir2048" \
                       --cols "512^2,1024^2,2048^2" --out mosaic.png
Optionally --times "23.2,150.9,1101.6;25.7,166.4,1214.6" to annotate wall times.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_kh import load_rho, last_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", required=True,
                    help='"<label>:<dir>,<dir>,..." repeatable, one per scheme')
    ap.add_argument("--cols", required=True, help="comma-separated column labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--times", default=None,
                    help="rows separated by ';', wall seconds per column, for annotation")
    ap.add_argument("--tlabel", default="1.2")
    args = ap.parse_args()

    rows = []
    for spec in args.row:
        label, dirs = spec.split(":", 1)
        rows.append((label, [d for d in dirs.split(",") if d]))
    cols = args.cols.split(",")
    times = [r.split(",") for r in args.times.split(";")] if args.times else None

    nr, nc = len(rows), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(4.1 * nc, 4.3 * nr),
                             squeeze=False, constrained_layout=True)

    im = None
    for i, (label, dirs) in enumerate(rows):
        for j in range(nc):
            ax = axes[i][j]
            rho = load_rho(last_output(dirs[j]))
            im = ax.imshow(rho, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5],
                           cmap="RdBu_r", vmin=1.0, vmax=2.0, interpolation="nearest")
            ax.set_xticks([-0.4, 0.0, 0.4])
            ax.set_yticks([-0.4, 0.0, 0.4])
            if i == 0:
                ax.set_title(cols[j], fontsize=13)
            if j == 0:
                ax.set_ylabel(f"{label}\n\ny", fontsize=12)
            if i == nr - 1:
                ax.set_xlabel("x")
            if times:
                ax.text(0.03, 0.965, f"{float(times[i][j]):.0f} s",
                        transform=ax.transAxes, ha="left", va="top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.78,
                                  ec="none"))

    fig.colorbar(im, ax=axes, location="right", shrink=0.82, label=r"$\rho$")
    fig.suptitle(f"Kelvin-Helmholtz at $t={args.tlabel}$, uniform grids: "
                 "scheme vs resolution at matched DoF", fontsize=14)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
