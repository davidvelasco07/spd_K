"""Mosaic comparing spd_K and AthenaK on the Balsara 2025 section 8.2 MHD jet.

One row per code/scheme, one column per quantity, every panel on the SAME
colour scale so rows can be read against each other.

THE DENSITY SCALE IS ANCHORED ON PHYSICS, NOT ON THE PAPER'S COLOURBAR.
Balsara's fig 11a cannot be read through its own printed ticks: they run
1.6 .. -4.8, and on that scale the undisturbed ambient comes out at
log10 rho = -1.99 (rho = 0.010) where the stated initial condition fixes 0.14.
Anchoring instead on two densities the setup fixes exactly -- the ambient 0.14
and the beam core 1.4 -- gives [-2.30, +1.07], and that is what every row uses.
The far-field ambient anchor is safe: (-0.45,1.42) is 1.49 from the nozzle and
the fast speed is 378, so it is causally disconnected until t = 0.0039 > 0.002.

Fig 11c (|B|^2) IS self-consistent -- its ambient recovers 20350 against the
required 20000, 1.7% -- so that column uses the paper's own bar, anchored on the
ambient. Fig 11b (pressure) gets no such treatment: its scale is as unreadable
as 11a's, so the paper's pressure cell shows the published panel image as-is and
must not be compared quantitatively against the rows below it.

Usage:
    python3 scripts/jet_mosaic.py <out.png> <paper_panel_dir>
where <paper_panel_dir> holds the fig-11 rasters extracted with
    pdfimages -f 59 -l 59 -png <balsara2025>.pdf <dir>/p
(p-000 = 11a density, p-002 = 11b pressure, p-004 = 11c |B|^2).
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import balsara_jet_figures as bj  # noqa: E402

RHO0, B20 = 0.14, 20000.0
T_AMB, T_BEAM = 0.4298, 0.7268        # jet-colormap positions measured in 11a
SPAN = (np.log10(1.4) - np.log10(RHO0)) / (T_BEAM - T_AMB)
LO = np.log10(RHO0) - T_AMB * SPAN
HI = LO + SPAN
RANGE = {"rho": (LO, HI), "p": (0.0, 5.5), "B2": (0.0, 1.9e5)}
LABEL = {"rho": r"$\log_{10}\rho$", "p": r"$\log_{10}p$", "B2": r"$|B|^2$"}
_JET = matplotlib.colormaps["jet"](np.linspace(0, 1, 4001))[:, :3]
_TT = np.linspace(0, 1, 4001)


def _panel(png):
    """The data region of an extracted fig-11 raster, colourbar stripped."""
    img = mpimg.imread(png)[:, :, :3]
    if img.max() > 1.001:
        img = img / 255.0
    ink = (img.max(-1) - img.min(-1) > 0.05) | (img.mean(-1) < 0.9)
    cols = np.where(ink.mean(0) > 0.5)[0]
    rows = np.where(ink.mean(1) > 0.5)[0]
    brk = np.where(np.diff(cols) > 5)[0]
    c1 = cols[brk[0]] if len(brk) else cols[-1]
    return img[rows[0]:rows[-1] + 1, cols[0]:c1 + 1]


def _invert(panel):
    """RGB -> normalised colourmap position, by nearest jet colour."""
    flat = panel.reshape(-1, 3)
    out = np.empty(flat.shape[0])
    for i in range(0, flat.shape[0], 150000):
        d = ((flat[i:i + 150000, None, :] - _JET[None, :, :]) ** 2).sum(-1)
        out[i:i + 150000] = _TT[d.argmin(1)]
    return out.reshape(panel.shape[:2])


def _box(field, x, y, h, ny=1.5):
    Hh, Ww = field.shape
    c0 = int((x + 0.5 - h) * (Ww - 1)); c1 = int((x + 0.5 + h) * (Ww - 1))
    r0 = int((ny - y - h) / ny * (Hh - 1)); r1 = int((ny - y + h) / ny * (Hh - 1))
    return float(np.median(field[max(0, r0):max(1, r1), max(0, c0):max(1, c1)]))


def paper_fields(pdir):
    """log10(rho) and |B|^2 from the published rasters, physics-anchored."""
    rho = LO + _invert(_panel(f"{pdir}/p-000.png")) * SPAN
    t = _invert(_panel(f"{pdir}/p-004.png"))
    b2 = t / _box(t, -0.45, 1.42, 0.04) * B20
    return {"rho": rho, "B2": b2, "p_img": _panel(f"{pdir}/p-002.png")}


def spdk_fields(outdir):
    d = bj._load(outdir)
    lg = lambda f: np.log10(np.maximum(f, 1e-12))
    return {"rho": lg(d["rho"]), "p": lg(d["p"]), "B2": d["B2"],
            "xf": d["xf"], "yf": d["yf"]}


def athenak_fields(rundir):
    sys.path.insert(0, "/private/tmp/claude-502/-Users-velasco-spd-K/"
                       "fa052243-6d82-48cd-b675-a5878876f8db/scratchpad")
    import ak_probe as ak
    g = ak.load(rundir)
    lg = lambda f: np.log10(np.maximum(f, 1e-12))
    return {"rho": lg(g["dens"]), "p": lg(g["p"]), "B2": g["B2"],
            "xf": g["xf"], "yf": g["yf"]}


def mosaic(rows, png, pdir=None):
    """rows: list of (label, fields dict). A dict with no 'xf' is the paper."""
    cols = ("rho", "p", "B2")
    n = len(rows)
    fig, ax = plt.subplots(n, 3, figsize=(11.4, 3.55 * n), squeeze=False)
    for r, (lab, f) in enumerate(rows):
        for c, q in enumerate(cols):
            a = ax[r][c]
            lim = RANGE[q]
            if "xf" not in f:                       # paper row
                if q == "p":
                    a.imshow(f["p_img"], extent=[-0.5, 0.5, 0, 1.5],
                             aspect="equal", interpolation="nearest")
                    a.set_title("(published panel;\nscale not readable)"
                                if r == 0 else "", fontsize=7.5, color="0.35")
                else:
                    a.imshow(f[q], origin="upper", extent=[-0.5, 0.5, 0, 1.5],
                             cmap="jet", vmin=lim[0], vmax=lim[1],
                             aspect="equal", interpolation="nearest")
            else:
                m = a.pcolormesh(f["xf"], f["yf"], f[q], cmap="jet",
                                 shading="flat", vmin=lim[0], vmax=lim[1])
                if r == n - 1:
                    cb = fig.colorbar(m, ax=[ax[k][c] for k in range(n)],
                                      fraction=0.020, pad=0.012)
                    cb.ax.tick_params(labelsize=7)
            a.set_aspect("equal")
            a.tick_params(labelsize=7)
            if r == 0 and not (q == "p" and "xf" not in f):
                a.set_title(LABEL[q], fontsize=11)
            if r != n - 1:
                a.set_xticklabels([])
            else:
                a.set_xlabel("x", fontsize=9)
            if c != 0:
                a.set_yticklabels([])
        ax[r][0].set_ylabel("y", fontsize=9)
        ax[r][0].text(-0.42, 0.5, lab, transform=ax[r][0].transAxes,
                      rotation=90, va="center", ha="center", fontsize=9.5)
    fig.suptitle("Mach-800 magnetized jet, $\\beta_a=10^{-4}$, $t=0.002$, "
                 "400$\\times$600 DoF — spd_K vs AthenaK\n"
                 "density scale anchored on $\\rho_{amb}=0.14$ and "
                 "$\\rho_{beam}=1.4$, not on the paper's colourbar",
                 fontsize=11.5)
    fig.savefig(png, dpi=118, bbox_inches="tight")
    print("wrote", png)


if __name__ == "__main__":
    out, pdir = sys.argv[1], sys.argv[2]
    S = ("/private/tmp/claude-502/-Users-velasco-spd-K/"
         "fa052243-6d82-48cd-b675-a5878876f8db/scratchpad")
    rows = [
        ("Balsara et al. 2025\nfig. 11  (6th order)", paper_fields(pdir)),
        ("spd_K  MUSCL+RK2\nHLLD / UCT",              spdk_fields(f"{S}/muscl400")),
        ("spd_K  SDFB4 (p=3)\nHLLD / UCT",            spdk_fields(f"{S}/paper_sdfb4")),
        ("AthenaK  PLM + FOFC\nHLLD / UCT-HLLD",      athenak_fields(f"{S}/akp_plm_fofc")),
        ("AthenaK  WENO-Z + FOFC\nHLLD / UCT-HLLD",   athenak_fields(f"{S}/akp_wenoz_fofc")),
    ]
    mosaic(rows, out, pdir)
