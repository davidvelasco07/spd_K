"""KH tolerance figures: energy history and a B_p/B_t mosaic.

    python3 scripts/mdz21_kh_tolerance.py <kh-run-root> <outdir>

Mirrors what the field loop and Orszag-Tang get, so the three benchmarks can be
read side by side: the same six lanes (SDFB4 and SDFB8 at two MOOD tolerances,
MUSCL+RK2 at matched and doubled resolution) as a history plot and as a grid of
colour maps.

READ THE ENERGY THE OTHER WAY ROUND. The quantity is RR22 eq. 100, the
normalised POLOIDAL magnetic energy, and on this problem it GROWS: the shear
layer winds the in-plane field up during the transition to turbulence. A less
dissipative scheme therefore ends HIGHER, the same direction as the field loop
but for the opposite reason -- there the exact answer is 1 and everything decays
away from it.

Maps are drawn on REGULAR control volumes (dg.poloidal_ratio does the rebin):
SD sub-cells are the non-uniform control volumes of the solution points, so an
evenly spaced plot of them distorts the geometry.
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

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"],
                     "axes.edgecolor": "#c9ced8", "savefig.facecolor": "white"})

# (directory relative to root, label, style, colour) -- the energy-plot lanes.
# Mosaic order, read left-to-right then top-to-bottom, matching the field-loop
# and Orszag-Tang mosaics: each row is one polynomial order's two tolerances
# followed by a MUSCL reference.
LANES = [
    ("uct_hlld_dof128/sdfb4_rk3", "SDFB4, tol 1e-5 (default)", "-", "#7b2d8e"),
    ("tol_hlld_dof128/p3_1e-3",  "SDFB4, tol 1e-3", "-",  "#b45309"),
    ("uct_hlld_dof128/muscl_rk2", "MUSCL+RK2, 1x", "--", "#0b7285"),
    ("uct_hlld_dof128/sdfb8_rk3", "SDFB8, tol 1e-5 (default)", "-", "#c026d3"),
    ("tol_hlld_dof128/p7_1e-3",  "SDFB8, tol 1e-3", "-",  "#c2410c"),
    ("uct_hlld_dof128/muscl_rk2_2x", "MUSCL+RK2, 2x res", "--", "#1a7f37"),
]


def complete(d):
    log = os.path.join(d, "run.log")
    return (os.path.isdir(d) and os.path.isfile(log)
            and "evolution:" in open(log).read() and bool(io.output_indices(d)))


def main(root, outdir):
    os.makedirs(outdir, exist_ok=True)
    have = [(os.path.join(root, p), l, st, c) for p, l, st, c in LANES
            if complete(os.path.join(root, p))]
    for p, l, _, _ in LANES:
        if not complete(os.path.join(root, p)):
            print(f"  INCOMPLETE (skipped): {p}", file=sys.stderr)
    if not have:
        print("no completed lanes"); return 1

    # ---- energy history -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    rows = []
    for d, lbl, st, c in have:
        e = dg.poloidal_energy(d)
        t = np.linspace(0, dg.final_time(d) or (len(e) - 1), len(e))
        ax.plot(t, e, st, color=c, lw=2.0, label=lbl)
        rows.append((lbl, e[-1]))
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\langle B_p^2\rangle(t)\,/\,\langle B_p^2\rangle(0)$")
    ax.set_yscale("log"); ax.grid(alpha=.25, lw=.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title("Kelvin-Helmholtz (RR22 sec 5.2): poloidal magnetic energy\n"
                 "128x256 DoF, UCT-HLLD — it GROWS here, so higher = less dissipative",
                 fontsize=12)
    fig.tight_layout()
    f = os.path.join(outdir, "kh_tol_energy.png")
    fig.savefig(f, dpi=140); plt.close(fig); print("  wrote", f)

    # ---- B_p/B_t mosaic at the final time -------------------------------
    nc = 3
    nr = int(np.ceil(len(have) / nc))
    panels = []
    for d, lbl, _, _ in have:
        g = io.Grid(d); i = io.output_indices(d)[-1]
        r = dg.poloidal_ratio(g, i)[0]            # already on regular CVs
        gx = dg.regular_faces(g, 0); gy = dg.regular_faces(g, 1)
        # Crop to the shear layer, |y - 1| < 0.5, as the map grids do: the
        # outer half of the 1x2 box is undisturbed and only shrinks the panels.
        yc = 0.5 * (gy[1:] + gy[:-1])
        m = np.abs(yc - 1.0) < 0.5
        k0, k1 = int(np.argmax(m)), int(len(m) - np.argmax(m[::-1]))
        panels.append((lbl, gx, gy[k0:k1 + 1], r[k0:k1]))
    allv = np.concatenate([p[3].ravel() for p in panels])
    norm = colors.LogNorm(max(np.percentile(allv, 1.0), 1e-4),
                          np.percentile(allv, 99.5))
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.5 * nr), squeeze=False)
    for k, (lbl, gx, gy, r) in enumerate(panels):
        ax = axes[k // nc][k % nc]
        im = ax.pcolormesh(gx, gy, r, cmap="cividis", norm=norm, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lbl, fontsize=10)
    for k in range(len(panels), nr * nc):
        axes[k // nc][k % nc].axis("off")
    fig.suptitle("Kelvin-Helmholtz at $t=20$: $B_p/B_t$ on regular control volumes\n"
                 "128x256 DoF, UCT-HLLD", fontsize=12, y=.99)
    fig.subplots_adjust(wspace=.05, hspace=.12, top=.92)
    cb = fig.colorbar(im, ax=axes, fraction=.02, pad=.02)
    cb.set_label(r"$B_p/B_t$", fontsize=10)
    f = os.path.join(outdir, "kh_tol_mosaic.png")
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig); print("  wrote", f)

    print("\n  final normalised poloidal energy (higher = less dissipative):")
    for lbl, v in sorted(rows, key=lambda r: -r[1]):
        print("    %-30s %.4f" % (lbl, v))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
