"""Figures for the Newtonian tests of Balsara et al. 2025 (arXiv:2506.11181).

    python3 scripts/balsara25_figures.py <run-root> <outdir>

Section 8.1 is the Wu & Shu (2018) strongly magnetized blast, and their figure 10
plots density, pressure, |v|^2 and |B|^2 at t = 0.001. This lays those four
quantities (rows) against the four base-scheme lanes (columns) on shared per-row
colour scales, so the rows compare schemes and the columns compare quantities.

Maps are drawn on REGULAR control volumes (dg.to_regular): the SD sub-cells are
the non-uniform control volumes of the solution points, so plotting them evenly
distorts the geometry -- 3.4x width ratio within an element at p=3, 8.0x at p=7.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
import mdz21_diagnostics as dg  # noqa: E402
import spdk_io as io  # noqa: E402

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "axes.edgecolor": "#c9ced8", "savefig.facecolor": "white"})
LANES = [("muscl_rk2", "MUSCL + RK2"), ("muscl_rk2_2x", "MUSCL RK2 2x"),
         ("sdfb4_rk3", "SDFB4 + RK3"), ("sdfb8_rk3", "SDFB8 + RK3")]


def complete(d):
    log = os.path.join(d, "run.log")
    return (os.path.isdir(d) and os.path.isfile(log)
            and "evolution:" in open(log).read() and bool(io.output_indices(d)))


def quantities(d):
    g = io.Grid(d)
    i = io.output_indices(d)[-1]
    reg = lambda a: dg.to_regular(g, a)[0]
    rho = reg(dg.load(g, i, dg.MRHO))
    prs = reg(dg.load(g, i, dg.MPRS))
    v2 = reg(sum(dg.load(g, i, v) ** 2 for v in (dg.MVX, dg.MVY, dg.MVZ)))
    b2 = reg(dg.load_b2(g, i, dg.B2_MAG2))
    return g, (rho, prs, v2, b2)


def main(root, outdir, case="uct_hlld_dof400"):
    base = os.path.join(root, case)
    lanes = [(n, l) for n, l in LANES if complete(os.path.join(base, n))]
    for n, l in LANES:
        if not complete(os.path.join(base, n)):
            print(f"  INCOMPLETE (skipped): {case}/{n}", file=sys.stderr)
    if not lanes:
        print("no completed lanes"); return 1
    os.makedirs(outdir, exist_ok=True)

    data, grids = {}, {}
    for n, l in lanes:
        g, q = quantities(os.path.join(base, n))
        data[n] = q; grids[n] = g
    ROWS = [(r"density $\rho$", 0, "viridis", False),
            (r"pressure $p$", 1, "inferno", True),
            (r"$|v|^2$", 2, "magma", False),
            (r"$|B|^2$", 3, "cividis", False)]
    nr, nc = len(ROWS), len(lanes)
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 3.15 * nr + 1.0), squeeze=False)
    for r, (rl, k, cmap, log) in enumerate(ROWS):
        allv = np.concatenate([data[n][k].ravel() for n, _ in lanes])
        if log:
            norm = colors.LogNorm(max(np.percentile(allv, 1), 1e-8), allv.max())
        else:
            norm = colors.Normalize(np.percentile(allv, 0.2), np.percentile(allv, 99.8))
        for c, (n, l) in enumerate(lanes):
            ax = axes[r][c]
            g = grids[n]
            gx, gy = dg.regular_faces(g, 0), dg.regular_faces(g, 1)
            im = ax.pcolormesh(gx, gy, data[n][k], cmap=cmap, norm=norm, rasterized=True)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(l, fontsize=11, pad=7)
            if c == 0:
                ax.set_ylabel(rl, fontsize=12, labelpad=9)
        fig.colorbar(im, ax=axes[r].tolist(), fraction=.020, pad=.012)
    g0 = grids[lanes[0][0]]
    P = lambda key, dflt=None: dg.param(os.path.join(base, lanes[0][0]), key, dflt)
    beta = 2.0 * float(P("p0", 0.1)) / float(P("amp", 1.0)) ** 2
    fig.suptitle("Strongly magnetized MHD blast (Wu & Shu 2018) "
                 "— Balsara et al. 2025 section 8.1\n"
                 "%d$^2$ DoF, $t$ = %s, UCT-HLLD" %
                 (g0.N[0] * (g0.p + 1), P("tlim", "?")), fontsize=13, y=.995)
    fig.text(.5, .955,
             r"$\rho_0$ = %s | $p_0$ = %s, $p_1$ = %s inside r < %s | "
             r"$B_x$ = %.4f = 1000/$\sqrt{4\pi}$ | $\beta$ = %.2e | $\gamma$ = %s | "
             "cfl %s (%s) | energy_fix %g" %
             (P("d0", "?"), P("p0", "?"), P("p1", "?"), P("radius", "?"),
              float(P("amp", 0)), beta, P("gamma", "?"), P("cfl", "?"),
              P("cfl_type", "?"), float(P("energy_fix", 0) or 0)),
             ha="center", fontsize=8.5, color="#444")
    fig.subplots_adjust(wspace=.06, hspace=.10, top=.925)
    f = os.path.join(outdir, "balsara25_blast.png")
    fig.savefig(f, dpi=135, bbox_inches="tight")
    print("  wrote", f)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2],
                  sys.argv[3] if len(sys.argv) > 3 else "uct_hlld_dof400"))
