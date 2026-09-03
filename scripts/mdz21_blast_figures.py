"""Figure 12 of Mignone & Del Zanna 2021 from an spd_K 3D blast run tree.

    python3 scripts/mdz21_blast_figures.py <run-root> [outdir]

The paper's figure has six panels: four 2D maps in the z = 0 plane (density,
log pressure, specific kinetic energy, magnetic energy) from the UCT-HLLD run,
plus two line plots comparing emf schemes -- magnetic energy along the main
diagonal, and density along the vertical axis.

The DIAGONAL panel is the one that discriminates. A dip forms near x_d ~ 0.2 and
DEEPENS as a scheme's dissipation falls; the paper's ordering by minimum, from
highest (most dissipative) to lowest, is UCT-HLL, UCT-GFORCE, CT-Contact,
CT-Flux, UCT-HLLD. The z-axis density panel is the control: the paper reports
only minor differences there, so a run tree that separates the schemes on the
z-axis as much as on the diagonal is telling you about noise, not dissipation.

spd_K's box is [0,1]^3, so the paper's z = 0 plane is z = 0.5 here.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import mdz21_diagnostics as dg  # noqa: E402
import spdk_io as io  # noqa: E402

# Label by what the run actually is, not by the paper's scheme names: spd_K's
# "2sweep" is its own legacy two-sweep edge average, NOT the paper's CT-Flux.
LABEL = {("uct", "hlld"): "UCT-HLLD",
         ("uct", "hll"): "UCT-HLL",
         ("2sweep", "hlld"): "2-sweep edge (HLLD faces)",
         ("2sweep", "hll"): "2-sweep edge (HLL faces)"}
COLOR = {"UCT-HLLD": "#7b2d8e", "UCT-HLL": "#0b7285",
         "2-sweep edge (HLLD faces)": "#b45309",
         "2-sweep edge (HLL faces)": "#1a7f37"}


def find_runs(root):
    """{label: dir} for every completed run under root, any layout."""
    out = {}
    for dirpath, _, files in os.walk(root):
        if not any(f.startswith("W_cv_") and f.endswith(".dat") for f in files):
            continue
        log = os.path.join(dirpath, "run.log")
        if not os.path.isfile(log) or "evolution:" not in open(log).read():
            print(f"  INCOMPLETE (skipped): {dirpath}", file=sys.stderr)
            continue
        emf = "2sweep" if "2sweep" in dirpath else "uct"
        rs = "hll" if "_hll" in dirpath and "hlld" not in dirpath else "hlld"
        lbl = LABEL.get((emf, rs), f"{emf}/{rs}")
        # The emf pair alone is NOT a unique key: a campaign tree puts several
        # BASE-SCHEME lanes (sdfb4_rk3, sdfb8_rk3, muscl_rk2, ...) under one
        # emf directory, and keying on the pair would silently keep whichever
        # the walk reached last and plot it as "UCT-HLLD". Qualify with the lane
        # whenever it is not the sdfb4 reference.
        lane = os.path.basename(dirpath)
        if lane.startswith(("sdfb8", "muscl")):
            lbl = f"{lbl} / {lane}"
        out[lbl] = dirpath
    return out


def main(root, outdir=None):
    runs = find_runs(root)
    if not runs:
        print(f"no completed runs under {root}")
        return 1
    print(f"3D blast, {len(runs)} run(s): " + ", ".join(sorted(runs)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans"],
                         "savefig.facecolor": "white"})

    ref_label = "UCT-HLLD" if "UCT-HLLD" in runs else sorted(runs)[0]
    ref = runs[ref_label]

    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.35, 1.35],
                          hspace=.32, wspace=.52)

    # --- the four 2D maps, z = centre, from the reference run ---
    grid = io.Grid(ref)
    i = io.output_indices(ref)[-1]
    k = dg._mid_index(grid, 2) if grid.active(2) else 0
    # Average onto REGULAR control volumes before drawing. The SD sub-cells are
    # the (non-uniform) control volumes of the solution points -- 3.4x width
    # ratio at p=3, 8.0x at p=7 -- so an imshow of the raw field stretches the
    # element-edge cells and squeezes the interior ones. See dg.to_regular.
    reg = lambda a: dg.to_regular(grid, a)[k]
    rho = reg(dg.load(grid, i, dg.MRHO))
    prs = reg(dg.load(grid, i, dg.MPRS))
    vx = reg(dg.load(grid, i, dg.MVX))
    vy = reg(dg.load(grid, i, dg.MVY))
    vz = reg(dg.load(grid, i, dg.MVZ))
    b2 = reg(dg.load_b2(grid, i, dg.B2_MAG2))
    ekin = 0.5 * (vx ** 2 + vy ** 2 + vz ** 2)          # SPECIFIC kinetic energy
    # extent from the REGULAR faces, which is what the rebinned field lives on
    gx, gy = dg.regular_faces(grid, 0), dg.regular_faces(grid, 1)
    ext = [gx[0], gx[-1], gy[0], gy[-1]]
    panels = [("density", rho, "viridis", False),
              ("pressure (log)", prs, "inferno", True),
              ("specific kinetic energy", ekin, "magma", False),
              ("magnetic energy $B^2/2$", 0.5 * b2, "cividis", False)]
    for n, (title, dat, cmap, logscale) in enumerate(panels):
        ax = fig.add_subplot(gs[n // 2, n % 2])
        d = np.log10(np.maximum(dat, 1e-30)) if logscale else dat
        im = ax.imshow(d, origin="lower", extent=ext, cmap=cmap, aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    fig.text(.115, .905, f"{ref_label}, $z=0.5$ plane", fontsize=11, weight="bold")

    # --- diagonal magnetic energy: the discriminating panel ---
    ax = fig.add_subplot(gs[0, 2:])
    rows = []
    for lbl in sorted(runs):
        xd, e = dg.blast_diagonal_magnetic_energy(runs[lbl])
        ax.plot(xd, e, lw=1.8, color=COLOR.get(lbl), label=lbl)
        mn, at, c = dg.blast_dip_depth(runs[lbl])
        rows.append((lbl, mn, at, c))
    ax.set_xlabel(r"$x_d = \sqrt{2}\,(x - 1/2)$   along the main diagonal")
    ax.set_ylabel(r"$B^2/2$")
    ax.set_title("magnetic energy along the main diagonal, $z=0.5$", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.25, lw=.7)

    # --- z-axis density: the control panel ---
    ax = fig.add_subplot(gs[1, 2:])
    for lbl in sorted(runs):
        z, rr = dg.blast_axis_density(runs[lbl])
        ax.plot(z, rr, lw=1.8, color=COLOR.get(lbl), label=lbl)
    ax.set_xlabel(r"$z$   (the vertical axis through the centre)")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("density along the vertical axis (the control)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.25, lw=.7)

    # Title read from the DATA. It said "192^3 DoF, t=0.01" as a literal, which
    # is the paper's configuration and not necessarily the run's -- a 32^3 test
    # tree came out captioned as the production figure. Label the runs you have.
    dof = grid.N[0] * (grid.p + 1)
    t = dg.final_time(ref)
    tend = f"$t={t:g}$" if t is not None else "final output"
    fig.suptitle("3D blast wave, Mignone & Del Zanna 2021 section 6.4 "
                 f"\u2014 spd_K at {dof}$^3$ DoF, {tend}", fontsize=13, y=1.005)

    # Put the configuration ON the figure, so a panel that escapes into a talk
    # still says what produced it. Everything here is read from the run's own
    # parameters.txt, never hardcoded -- a caption that can drift from the run is
    # worse than no caption.
    P = lambda k, d=None: dg.param(ref, k, d)
    beta = 2.0 * float(P("p0", 0.1)) / float(P("amp", 1.0)) ** 2
    cfg_bits = [
        "SDFB%d (p=%d, %d$^3$ elements), RK3" % (grid.p + 1, grid.p, grid.N[0]),
        "cfl %s (%s)" % (P("cfl", "?"), P("cfl_type", "?")),
        r"$\gamma$ = %s" % P("gamma", "?"),
        "rsolver %s, emf %s" % (P("rsolver", "?"), P("emf", "?")),
        "fallback tol %s" % P("tolerance", "?"),
        "energy_fix %g" % float(P("energy_fix", 0) or 0),
    ]
    ic_bits = [
        r"$\rho_0$ = %s" % P("d0", "?"),
        "$p_0$ = %s, $p_1$ = %s inside r < %s" % (P("p0", "?"), P("p1", "?"), P("radius", "?")),
        r"$B_0$ = %.6f = 100/$\sqrt{4\pi}$" % float(P("amp", 0)),
        "direction (%g,%g,%g)" % (float(P("v1", 0)), float(P("v2", 0)), float(P("v3", 0))),
        r"$\beta \approx$ %.2e" % beta,
        "outflow on all six faces",
    ]
    fig.text(.5, .955, " | ".join(cfg_bits), ha="center", fontsize=8.5, color="#444")
    fig.text(.5, .932, " | ".join(ic_bits), ha="center", fontsize=8.5, color="#444")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        f = os.path.join(outdir, "blast3d_fig12.png")
        fig.savefig(f, dpi=130, bbox_inches="tight")
        print(f"  wrote {f}")

    print("\n  dip on the diagonal (LOWER minimum = LESS dissipative):")
    print("  %-28s %12s %10s %12s" % ("scheme", "min B^2/2", "at |x_d|", "centre"))
    for lbl, mn, at, c in sorted(rows, key=lambda r: -r[1]):
        print("  %-28s %12.4f %10.3f %12.4f" % (lbl, mn, at, c))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
