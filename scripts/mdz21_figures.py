"""Comparison figures for the MDZ21 report.

    python3 scripts/mdz21_figures.py mdz21_runs <outdir>

Produces, from a completed campaign tree:

  ot_pressure.png   pressure maps at the final time, emf x base scheme.
                    This is the paper's figures 10/11 layout: the panel a reader
                    is meant to compare is the CENTRE, where the magnetic island
                    either forms or does not.
  ot_island.png     the same runs zoomed to the central window mu_p is measured
                    over, which is where the whole UCT-HLLD / UCT-HLL difference
                    lives and is invisible at full-domain scale.
  ot_energy.png     volume-integrated magnetic energy against time.

A shared colour scale across panels is the point: per-panel autoscaling would
make every scheme look alike.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402

import mdz21_diagnostics as dg  # noqa: E402
import spdk_io as io  # noqa: E402
from mdz21_report import LANE_LABEL, LANE_ORDER, find_runs, steps_of  # noqa: E402

EMF_LABEL = {"hlld": "UCT-HLLD", "hll": "UCT-HLL"}
EMF_ORDER = ["hlld", "hll"]

# Type and colour chosen to sit with the report page rather than matplotlib's
# defaults: the same teal accent, a neutral grey-blue ink, no chartjunk.
INK = "#10141b"
MUTED = "#5c6472"
ACCENT = "#0b7285"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#c9ced8", "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def field(d, var, i=-1, source="W_cv"):
    """One 2D slice. source="B2_cv" reads the STAGGERED field's cell averages
    (Bx, By, Bz, |B|^2), which is the divergence-free field itself rather than
    W_cv's cell-centred B rows -- and is what the paper's field-loop figures
    show. It carries 4 rows where W_cv carries 8, so it needs its own loader."""
    g = io.Grid(d)
    idx = io.output_indices(d)
    raw = (dg.load_b2(g, idx[i], var) if source == "B2_cv"
           else io.load_sd(g, idx[i], var))
    # Average onto REGULAR control volumes before returning. The SD sub-cells
    # are the control volumes of the solution points and are NOT equally
    # spaced -- 3.4x width ratio within an element at p=3, 8.0x at p=7 -- so a
    # plot that spaces them evenly distorts the geometry. Drawing on the true
    # centres with shading="nearest" is already close, but it places cell edges
    # at midpoints BETWEEN centres rather than at the real faces; on the regular
    # mesh those coincide, so this makes the panels exact as well as consistent
    # with the mosaics. dg.to_regular is conservative (integral preserved to
    # 1.4e-16), so no panel's integrated quantity moves.
    return g, dg.to_regular(g, raw)[0]


def panels(runs, dof):
    """[(emf, lane, dir)] present at this DoF, in report order."""
    out = []
    for emf in EMF_ORDER:
        key = ("uct", emf, dof)
        if key not in runs:
            continue
        for lane in LANE_ORDER:
            d = runs[key].get(lane)
            if d:
                out.append((emf, lane, d))
    return out


def grid_figure(runs, dof, var, fname, title, zoom=None, cmap="magma",
                log=False, label="", source="W_cv", annotate=None,
                aspect=3.5):
    ps = panels(runs, dof)
    if not ps:
        print(f"  no panels at dof={dof}")
        return None
    emfs = [e for e in EMF_ORDER if any(p[0] == e for p in ps)]
    lanes = [l for l in LANE_ORDER if any(p[1] == l for p in ps)]
    nr, nc = len(emfs), len(lanes)

    # One colour scale for every panel, taken over all of them. Per-panel
    # autoscaling is what makes distinct schemes look identical.
    data = {}
    for emf, lane, d in ps:
        g, f = field(d, var, source=source)
        # centres of the REGULAR cells the field was just rebinned onto
        rf = lambda k: 0.5 * (dg.regular_faces(g, k)[1:] + dg.regular_faces(g, k)[:-1])
        x, y = rf(0), rf(1)
        if zoom:
            # (lo, hi) applies to both axes; ((xlo,xhi),(ylo,yhi)) is per-axis.
            # A single window is wrong the moment the box is not square: the
            # field loop sits at (1, 0.5) in a 2:1 box, and reusing the x window
            # on y clipped the panel to the top half of the loop.
            zx, zy = (zoom if isinstance(zoom[0], (tuple, list))
                      else (zoom, zoom))
            mx = (x > zx[0]) & (x < zx[1])
            my = (y > zy[0]) & (y < zy[1])
            f = f[:, mx][my, :]
            x, y = x[mx], y[my]
        data[(emf, lane)] = (x, y, f)
    # Panel height follows the DOMAIN aspect, so a 2:1 box is not squashed.
    x0, y0, _ = next(iter(data.values()))
    yx = max(0.25, min(2.0, (y0[-1] - y0[0]) / (x0[-1] - x0[0])))
    allv = np.concatenate([v[2].ravel() for v in data.values()])
    vmin, vmax = np.percentile(allv, 0.2), np.percentile(allv, 99.8)
    norm = (colors.LogNorm(max(vmin, 1e-12), vmax) if log
            else colors.Normalize(vmin, vmax))

    fig, axes = plt.subplots(nr, nc, figsize=(aspect * nc, aspect * nr * yx + 0.9),
                             squeeze=False)
    im = None
    for r, emf in enumerate(emfs):
        for c, lane in enumerate(lanes):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if (emf, lane) not in data:
                ax.axis("off")
                continue
            x, y, f = data[(emf, lane)]
            im = ax.pcolormesh(x, y, f, cmap=cmap, norm=norm,
                               shading="nearest", rasterized=True)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(LANE_LABEL.get(lane, lane), fontsize=11, pad=7)
            if c == 0:
                ax.set_ylabel(EMF_LABEL.get(emf, emf), fontsize=11.5,
                              labelpad=9, color=INK)
            # The number the panel is evidence for, on the panel.
            try:
                pdir = next(p[2] for p in ps if p[0] == emf and p[1] == lane)
                txt = (annotate(pdir) if annotate
                       else rf"$\mu_p$ = {dg.pressure_peak_ratio(pdir):.2f}")
                ax.text(.5, .028, txt, transform=ax.transAxes,
                        ha="center", fontsize=10.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.28", fc=(0, 0, 0, .55),
                                  ec="none"))
            except Exception:
                pass
    fig.suptitle(title, fontsize=13, y=.985)
    fig.tight_layout(rect=[0, .13, 1, .95])
    fig.subplots_adjust(hspace=.10, wspace=.06)
    if im is not None:
        cax = fig.add_axes([0.30, 0.055, 0.40, 0.022])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=9)
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  wrote {fname}")
    return fname


def energy_figure(runs, dof, fname, tmax=1.0,
                  ylab=r"$E_B(t)\,/\,E_B(0)$",
                  xlab=r"$t$   (= $2\pi$ in the paper's units at $t=1$)"):
    ps = panels(runs, dof)
    if not ps:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    styles = {"muscl_rk2": ("-", "#0b7285"), "muscl_rk2_2x": ("--", "#0b7285"),
              "sdfb4_rk3": ("-", "#b45309"), "sdfb8_rk3": ("-", "#7b2d8e"),
              "muscl_hancock": ("--", "#146c43")}
    for ax, emf in zip(axes, EMF_ORDER):
        drew = False
        for lane in LANE_ORDER:
            d = next((p[2] for p in ps if p[0] == emf and p[1] == lane), None)
            if not d:
                continue
            eb = dg.magnetic_energy(d)
            t = np.linspace(0, tmax, len(eb))   # fixed output/dt
            ls, col = styles.get(lane, ("-", MUTED))
            ax.plot(t, eb / eb[0], ls, color=col, lw=1.9,
                    label=LANE_LABEL.get(lane, lane))
            drew = True
        ax.set_title(EMF_LABEL.get(emf, emf), fontsize=11.5)
        ax.set_xlabel(xlab)
        ax.grid(alpha=.25, lw=.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if drew:
            ax.legend(fontsize=9, frameon=False)
    axes[0].set_ylabel(ylab)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"  wrote {fname}")
    return fname


def field_loop_figures(runs, dof, outdir):
    """The paper's figures 2/3 (magnetic energy maps) and 4 (energy decay).

    In the pressure-dominated limit B is advected almost as a passive scalar, so
    what the maps show is the scheme's own dissipation: a loop that stays round
    and keeps its peak, or one that smears and rings.
    """
    def retained(d):
        eb = dg.magnetic_energy(d)
        return rf"$E_B/E_B(0)$ = {eb[-1] / eb[0]:.3f}"

    grid_figure(runs, dof, dg.B2_MAG2,
                os.path.join(outdir, "fl_b2.png"),
                f"Field loop, $|B|^2$ after one full crossing (t=2), "
                f"{dof} x {dof // 2} degrees of freedom",
                source="B2_cv", cmap="viridis", label=r"$|B|^2$",
                annotate=retained, aspect=4.6)
    grid_figure(runs, dof, dg.B2_MAG2,
                os.path.join(outdir, "fl_b2_zoom.png"),
                "The loop itself: a round, sharp-edged loop is an undissipated one",
                zoom=((0.58, 1.42), (0.08, 0.92)), source="B2_cv", cmap="viridis",
                label=r"$|B|^2$", annotate=retained, aspect=3.6)
    energy_figure(runs, dof, os.path.join(outdir, "fl_energy.png"),
                  tmax=2.0, ylab=r"$E_B(t)\,/\,E_B(0)$",
                  xlab=r"$t$   (one full diagonal crossing at $t=2$)")


def current_sheet_figures(runs, dof, outdir):
    """MDZ21 figure 8: thermal pressure at t = 30, and the magnetic energy history.

    The paper's reading: with no physical resistivity the UNPERTURBED Harris
    sheet is an exact stationary solution, so everything that happens is the
    scheme's own numerical resistivity. A more dissipative scheme reconnects
    EARLIER and loses magnetic energy sooner, eventually forming a large island
    across the vertical boundaries. So on this test a curve that stays FLAT is
    the good one -- the opposite reading from the KH, and the same as the field
    loop.
    """
    grid_figure(runs, dof, dg.MPRS, os.path.join(outdir, "cs_pressure.png"),
                f"Magnetized current sheet, thermal pressure at $t=30$, "
                f"{dof}$\\times${dof // 2} degrees of freedom",
                cmap="magma", label="pressure")
    # tmax = 30 is the deck's tlim; a FLAT curve is the good one here.
    energy_figure(runs, dof, os.path.join(outdir, "cs_energy.png"), tmax=30.0,
                  xlab=r"$t$")   # the long explanation belongs in the caption


def main(root, outdir, which=None):
    runs = find_runs(root)
    if not runs:
        print(f"no completed runs under {root}")
        return 1
    os.makedirs(outdir, exist_ok=True)
    dofs = sorted({k[2] for k in runs})
    dof = dofs[0]
    if which is None:
        # Read the benchmark off the provenance the campaign recorded, so the
        # figures cannot be labelled as a test the runs are not.
        which = "orszag_tang"
        for case in sorted(os.listdir(root)):
            pf = os.path.join(root, case, "provenance.txt")
            if os.path.isfile(pf):
                for line in open(pf):
                    if line.startswith("deck"):
                        which = os.path.basename(line.split()[1]).replace(".athinput", "")
                break
    print(f"benchmark: {which} -- figures at {dof} DoF "
          f"({len(panels(runs, dof))} panels)")
    if which == "field_loop":
        field_loop_figures(runs, dof, outdir)
        return 0
    if which == "current_sheet":
        current_sheet_figures(runs, dof, outdir)
        return 0
    grid_figure(runs, dof, dg.MPRS, os.path.join(outdir, "ot_pressure.png"),
                f"Orszag-Tang, pressure at $t=1$ ($=2\\pi$ in the paper), "
                f"{dof}$^2$ degrees of freedom",
                cmap="magma", label="pressure")
    grid_figure(runs, dof, dg.MPRS, os.path.join(outdir, "ot_island.png"),
                "The central window $\\mu_p$ is measured over "
                "($0.4 < x,y < 0.6$): the magnetic island, or its absence",
                zoom=(0.4, 0.6), cmap="magma", label="pressure")
    energy_figure(runs, dof, os.path.join(outdir, "ot_energy.png"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2],
                  sys.argv[3] if len(sys.argv) > 3 else None))
