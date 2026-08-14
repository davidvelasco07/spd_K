#!/usr/bin/env python3
"""max|divB| against time for a field loop run at four mesh settings.

Reads the per-output "max|divB| = ..." lines the solver prints and the block
counts from amr_blocks_*.txt, for run directories produced with a uniform
multiblock mesh, a statically refined mixed-level mesh, and dynamic AMR at one
and two levels. The two static meshes hold the divergence constraint at
round-off for the whole run, so what breaks it is the regrid itself and not the
mesh being mixed-level.

    python3 scripts/plot_amr_divb.py <uniform> <static> <l1> <l2> [--dest .]
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def series(outdir):
    log = open(os.path.join(outdir, "log.txt")).read() if \
        os.path.exists(os.path.join(outdir, "log.txt")) else \
        open(outdir + ".log").read()
    divb = [float(v) for v in re.findall(r"max\|divB\| = ([0-9.eE+-]+)", log)]
    nb = []
    for f in sorted(glob.glob(os.path.join(outdir, "amr_blocks_*.txt")),
                    key=lambda s: int(re.search(r"_(\d+)\.txt", s).group(1))):
        with open(f) as fh:
            nb.append(int([l for l in fh if not l.startswith("#")][0].split()[0]))
    n = min(len(divb), len(nb))
    return np.array(divb[:n]), np.array(nb[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs=4, help="uniform, static-refined, L1, L2")
    ap.add_argument("--dest", default=".")
    ap.add_argument("--dt", type=float, default=0.005)
    args = ap.parse_args()

    labels = ["uniform multiblock, no regrid",
              "static refinement, no regrid",
              "dynamic AMR, 1 level", "dynamic AMR, 2 levels"]
    cols = ["#2ca02c", "#7f7f7f", "#1f77b4", "#d62728"]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for d, lab, c in zip(args.dirs, labels, cols):
        divb, nb = series(d)
        t = np.arange(len(divb)) * args.dt
        ax[0].semilogy(t, np.maximum(divb, 1e-17), "o-", ms=4, color=c, label=lab)
        ax[1].plot(t, nb, "o-", ms=4, color=c, label=lab)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"max $|\nabla\cdot B|$")
    ax[0].set_title("field loop: divergence constraint", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("blocks")
    ax[1].set_title("mesh size", fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(r"Field loop: div$\,B$ survives a mixed-level mesh but not a "
                 "regrid", fontsize=12)
    fn = os.path.join(args.dest, "mhd_amr_divb.png")
    fig.savefig(fn, dpi=130)
    print(fn)


if __name__ == "__main__":
    main()
