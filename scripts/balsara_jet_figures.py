"""Figures and diagnostics for the Mach-800 magnetized jet.

Balsara, Bhoriya, Singh, Kumar, Kappeli & Gatti 2025 (ApJ 988, 134;
arXiv:2506.11181) section 8.2 / figure 11, with the setup of Wu & Shu (2018):
ambient rho = 0.1*gamma, p = 1, B = (0, sqrt(20000), 0); a nozzle on |x| < 0.05
of the y = 0 face injecting rho = gamma at v_y = 800; t = 0.002; 400x600 zones.

Two entry points:

  panels(outdir, png)      the four fig-11 quantities on the PAPER's own colour
                           ranges, so a comparison is direct rather than by eye
  side_by_side(outdir,png) the published figure above the spd_K run, using the
                           panel images embedded in the paper's PDF
  probe(outdir)            the scalar diagnostics that decided the boundary
                           conditions -- see docs/mhd.md, "Boundaries that
                           carry a field"

SD sub-cells are NOT equally spaced, so every field is area-weight rebinned onto
regular control volumes (mdz21_diagnostics.to_regular) and drawn with pcolormesh
on regular_faces. imshow on an SD field is simply wrong (CLAUDE.md rule 6).
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.dirname(__file__))
import spdk_io as io          # noqa: E402
import mdz21_diagnostics as dg  # noqa: E402

B0 = 141.4213562373095   # sqrt(20000), giving beta_a = 2p/B^2 = 1e-4 at p = 1
D0, D1, P0, VJET = 0.14, 1.4, 1.0, 800.0
RADIUS = 0.05

# Paper ranges, read off figure 11's colourbars.
PAPER = {"rho": (-4.8, 1.8), "p": (0.0, 10.0), "B2": (0.0, 1.9e5)}


def _load(outdir, i=None):
    """Regular-rebinned (rho, p, vx, vy, Bx, By, |B|^2) plus the face arrays."""
    g = io.Grid(outdir)
    idx = io.output_indices(outdir)
    if not idx:
        raise SystemExit(f"CHECK VOID: no dumps in {outdir}")
    i = idx[-1] if i is None else i
    R = lambda f: dg.to_regular(g, f)[0]
    out = dict(
        rho=R(dg.load(g, i, dg.MRHO)), p=R(dg.load(g, i, dg.MPRS)),
        vx=R(dg.load(g, i, dg.MVX)), vy=R(dg.load(g, i, dg.MVY)),
        Bx=R(dg.load_b2(g, i, dg.B2_BX)), By=R(dg.load_b2(g, i, dg.B2_BY)),
        B2=R(dg.load_b2(g, i, dg.B2_MAG2)),
    )
    # The paper's domain is [-0.5,0.5]; spd_K is origin-anchored, so shift x.
    out["xf"] = dg.regular_faces(g, 0) - 0.5
    out["yf"] = dg.regular_faces(g, 1)
    out["xc"] = 0.5 * (out["xf"][:-1] + out["xf"][1:])
    out["i"], out["grid"] = i, g
    return out


def complete(outdir):
    """Did the run finish? An ad-hoc comparison must check this, not just take
    the last dump: the floored/pathology fractions grow monotonically with time,
    and comparing output 7 of one run against output 10 of another once
    manufactured a 35% improvement that did not exist (CLAUDE.md 7c)."""
    log = os.path.join(outdir, "run.log")
    if not os.path.exists(log):
        return None
    return re.search(r"^evolution: \d+ steps", open(log, errors="replace").read(),
                     re.M) is not None


def probe(outdir):
    """The numbers that decided the jet's boundary conditions.

    `base_vy` is the mean v_y on the bottom row clear of the nozzle: material
    entering through what should be an exit. `b2_max` and the row it sits on say
    whether the field peaks at the BOW SHOCK (right) or on the INLET (wrong).
    """
    d = _load(outdir)
    off = np.abs(d["xc"]) > 0.12          # clear of the nozzle and its smearing
    row = int(np.unravel_index(d["B2"].argmax(), d["B2"].shape)[0])
    rec = dict(
        outdir=outdir, output=d["i"], complete=complete(outdir),
        base_vy=float(d["vy"][0][off].mean()),
        base_By_drift=float(np.abs(d["By"][0] - B0).max() / B0),
        b2_max=float(d["B2"].max()), b2_max_y=float(d["yf"][row]),
        rho_max=float(d["rho"].max()), p_max=float(d["p"].max()),
        v_max=float(np.sqrt(d["vx"] ** 2 + d["vy"] ** 2).max()),
    )
    print(f"=== {outdir}  (output {rec['output']}, complete={rec['complete']}) ===")
    print(f"  base row v_y outside nozzle = {rec['base_vy']:9.3f}   "
          f"(exact answer ~0; 435 was the unpinned-inlet pathology)")
    print(f"  base row max|By-B0|/B0      = {rec['base_By_drift']:9.3e}")
    print(f"  max |B|^2 = {rec['b2_max']:.4g} at y = {rec['b2_max_y']:.3f}   "
          f"(paper ~1.9e5, in the bow shock; initial {B0**2:.0f})")
    print(f"  max |v|   = {rec['v_max']:9.3f}   (injected {VJET:.0f})")
    print(f"  rho max = {rec['rho_max']:.4g} "
          f"(strong-shock bound 6*rho_j = {6*D1:.2f}),  p max = {rec['p_max']:.4g} "
          f"(ram pressure {D1*VJET**2:.3g})")
    return rec


def panels(outdir, png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _load(outdir)
    lg = lambda f: np.log10(np.maximum(f, 1e-12))
    sets = [(lg(d["rho"]), r"$\log_{10}\rho$", PAPER["rho"]),
            (lg(d["p"]),   r"$\log_{10}p$",    PAPER["p"]),
            (d["B2"],      r"$|B|^2$",         PAPER["B2"]),
            (d["vy"],      r"$v_y$",           (-VJET, VJET)),
            (d["vx"],      r"$v_x$",           (-VJET, VJET))]
    fig, ax = plt.subplots(1, 5, figsize=(19, 6.6))
    for a, (f, lab, lim) in zip(ax, sets):
        cm = "RdBu_r" if lim[0] < 0 else "jet"
        m = a.pcolormesh(d["xf"], d["yf"], f, cmap=cm, shading="flat",
                         vmin=lim[0], vmax=lim[1])
        a.set_aspect("equal"); a.set_title(lab, fontsize=9); a.set_xlabel("x")
        plt.colorbar(m, ax=a, fraction=0.046)
    ax[0].set_ylabel("y")
    fig.suptitle(title or outdir)
    fig.tight_layout(); fig.savefig(png, dpi=105)
    print("wrote", png)


def side_by_side(outdir, png, pdf_page_images, label="spd_K"):
    """`pdf_page_images` is a directory holding the panel images extracted from
    the paper's PDF page with

        pdfimages -f 59 -l 59 -png <paper>.pdf <dir>/p

    which yields p-000 (fig 11a), p-002 (11b) and p-004 (11c) as the plots.
    Those are the published panels themselves, not a redraw.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    d = _load(outdir)
    lg = lambda f: np.log10(np.maximum(f, 1e-12))
    mine = [(lg(d["rho"]), r"$\log_{10}\rho$", PAPER["rho"]),
            (lg(d["p"]),   r"$\log_{10}p$",    PAPER["p"]),
            (d["B2"],      r"$|B|^2$",         PAPER["B2"])]
    paper = [os.path.join(pdf_page_images, f"p-{n:03d}.png") for n in (0, 2, 4)]
    ttl = ["a)  density", "b)  pressure", r"c)  $|B|^2$"]
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 11.0))
    for c in range(3):
        ax[0, c].imshow(mpimg.imread(paper[c])); ax[0, c].axis("off")
        ax[0, c].set_title(ttl[c], fontsize=11)
        f, lab, lim = mine[c]
        m = ax[1, c].pcolormesh(d["xf"], d["yf"], f, cmap="jet", shading="flat",
                                vmin=lim[0], vmax=lim[1])
        ax[1, c].set_aspect("equal"); ax[1, c].set_xlabel("x")
        ax[1, c].set_title(lab, fontsize=11)
        plt.colorbar(m, ax=ax[1, c], fraction=0.046)
    ax[0, 0].text(-0.08, 0.5, "Balsara et al. 2025\nfig. 11  (6th order)",
                  transform=ax[0, 0].transAxes, rotation=90, va="center",
                  ha="center", fontsize=11)
    ax[1, 0].text(-0.30, 0.5, label, transform=ax[1, 0].transAxes, rotation=90,
                  va="center", ha="center", fontsize=11)
    ax[1, 0].set_ylabel("y")
    fig.suptitle(r"Mach-800 magnetized jet, $\beta_a=10^{-4}$, t = 0.002, "
                 r"400$\times$600 DoF", fontsize=13)
    fig.tight_layout(); fig.savefig(png, dpi=115)
    print("wrote", png)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "panels":
        panels(sys.argv[2], sys.argv[3], *sys.argv[4:])
    elif sys.argv[1] == "side_by_side":
        side_by_side(sys.argv[2], sys.argv[3], sys.argv[4], *sys.argv[5:])
    else:
        for outdir in sys.argv[1:]:
            probe(outdir)
