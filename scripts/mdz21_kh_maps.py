"""KH colour maps: Rueda-Ramirez et al. 2022 figure 6.

    python3 scripts/mdz21_kh_maps.py kh_runs <outdir> [dof]

B_p / B_t = sqrt(B1^2 + B2^2) / |B3| at several times, one panel grid per emf,
columns = base scheme, rows = time. That is the field the paper plots, and the
reason it plots a RATIO is that the toroidal B3 starts uniform: the ratio starts
near-constant and every structure in it is growth.

Cropped to |y - 1| < 0.5 around the shear layer -- the full [0,2] box is mostly
undisturbed and squashes the panels to invisibility. One shared LOG colour scale
across every panel of a figure, because the whole point is that some schemes
develop structure and others do not.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import mdz21_diagnostics as dg
import spdk_io as io
from mdz21_report import LANE_LABEL, LANE_ORDER, find_runs

EMF_LABEL = {"hlld": "UCT-HLLD", "hll": "UCT-HLL"}
TIMES = [5.0, 8.0, 12.0, 20.0]
TMAX = 20.0
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"],
                     "axes.edgecolor": "#c9ced8", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def index_for_time(d, t):
    """Output index nearest time t, assuming a fixed output/dt over [0, TMAX]."""
    idx = io.output_indices(d)
    return idx[int(round(t / TMAX * (len(idx) - 1)))]


def make(runs, dof, emf, outdir):
    lanes_avail = runs.get(("uct", emf, dof), {})
    lanes = [l for l in LANE_ORDER if l in lanes_avail]
    if not lanes:
        print("  no lanes for", emf, dof)
        return None
    data = {}
    for lane in lanes:
        d = lanes_avail[lane]
        g = io.Grid(d)
        # poloidal_ratio now returns the field on REGULAR control volumes, so
        # index it with the regular cell centres rather than the SD ones.
        y = 0.5 * (dg.regular_faces(g, 1)[1:] + dg.regular_faces(g, 1)[:-1])
        my = np.abs(y - 1.0) < 0.5
        for t in TIMES:
            r = dg.poloidal_ratio(g, index_for_time(d, t))[0]
            xr = 0.5 * (dg.regular_faces(g, 0)[1:] + dg.regular_faces(g, 0)[:-1])
            data[(lane, t)] = (xr, y[my], r[my, :])

    allv = np.concatenate([v[2].ravel() for v in data.values()])
    vmin = max(np.percentile(allv, 1.0), 1e-4)
    vmax = np.percentile(allv, 99.5)
    norm = colors.LogNorm(vmin, vmax)

    nr, nc = len(TIMES), len(lanes)
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc, 3.0 * nr + 0.8),
                             squeeze=False)
    im = None
    for r, t in enumerate(TIMES):
        for c, lane in enumerate(lanes):
            ax = axes[r][c]
            x, y, f = data[(lane, t)]
            im = ax.pcolormesh(x, y, f, cmap="cividis", norm=norm,
                               shading="nearest", rasterized=True)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(LANE_LABEL.get(lane, lane), fontsize=11, pad=7)
            if c == 0:
                ax.set_ylabel(f"$t = {t:.0f}$", fontsize=12, labelpad=8)
    fig.suptitle(f"Kelvin-Helmholtz, $B_p/B_t$ -- {EMF_LABEL.get(emf, emf)}, "
                 f"{dof}x{2*dof} DoF", fontsize=13, y=.985)
    fig.tight_layout(rect=[0, .06, 1, .96])
    fig.subplots_adjust(hspace=.06, wspace=.04)
    cax = fig.add_axes([0.30, 0.032, 0.40, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(r"$B_p / B_t$", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    f = os.path.join(outdir, f"kh_maps_{emf}_dof{dof}.png")
    fig.savefig(f, dpi=115)
    plt.close(fig)
    print("  wrote", f)
    return f


def main(root, outdir, dof=None):
    runs = find_runs(root)
    if not runs:
        print("no completed runs"); return 1
    dofs = sorted({k[2] for k in runs})
    dof = int(dof) if dof else dofs[-1]
    os.makedirs(outdir, exist_ok=True)
    for emf in ("hlld", "hll"):
        make(runs, dof, emf, outdir)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2],
                  sys.argv[3] if len(sys.argv) > 3 else None))
