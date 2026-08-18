#!/usr/bin/env python3
"""Fallback-reconstruction mosaic: plm vs vl2, uniform grids, one column per DoF.

Companion to mosaic_kh_schemes.py (scheme vs resolution) and plot_kh_fig21.py
(uniform vs AMR). This one holds the scheme (SDFB p=3), the discretisation
(uniform) and the integrator (rk3) fixed and varies only what the MOOD cascade
falls back to:

  plm  -- plain PLM, no Hancock half-step. Correct under RK, whose outer
          integrator already supplies the time accuracy (commit b836746 derives
          this from the integrator: ADER->vl2, RK->plm).
  vl2  -- MUSCL-Hancock, predictor on. Under RK this DOUBLE-COUNTS the temporal
          predictor, which is what every SDFB+RK run before b836746 did.

The third row is |rho_plm - rho_vl2|, without which the top two rows look like
the same figure printed twice at the finer resolutions -- which is the result:
the two choices diverge at coarse resolution and converge rapidly.

usage:
  mosaic_fallback.py --plm d512,d1024,d2048 --vl2 d512,d1024,d2048 \
                     --cols "512^2,1024^2,2048^2" --out mosaic_fallback.png
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_kh import load_rho, last_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plm", required=True, help="comma-separated run dirs, one per column")
    ap.add_argument("--vl2", required=True, help="comma-separated run dirs, one per column")
    ap.add_argument("--cols", required=True, help="comma-separated column labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plm-times", default=None, help="wall seconds per column")
    ap.add_argument("--vl2-times", default=None)
    ap.add_argument("--tlabel", default="1.2")
    args = ap.parse_args()

    plm = [d for d in args.plm.split(",") if d]
    vl2 = [d for d in args.vl2.split(",") if d]
    cols = args.cols.split(",")
    nc = len(cols)
    assert len(plm) == len(vl2) == nc, "one run dir per column per row"
    times = [args.plm_times.split(",") if args.plm_times else None,
             args.vl2_times.split(",") if args.vl2_times else None]

    fig, axes = plt.subplots(3, nc, figsize=(4.1 * nc, 4.3 * 3),
                             squeeze=False, constrained_layout=True)

    rows = [("SDFB, plm fallback\n(RK-derived, correct)", plm),
            ("SDFB, vl2 fallback\n(predictor double-counted)", vl2)]
    im = None
    for i, (lab, dirs) in enumerate(rows):
        for j in range(nc):
            ax = axes[i][j]
            im = ax.imshow(load_rho(last_output(dirs[j])), origin="lower",
                           extent=[-0.5, 0.5, -0.5, 0.5], cmap="RdBu_r",
                           vmin=1.0, vmax=2.0, interpolation="nearest")
            ax.set_xticks([-0.4, 0.0, 0.4]); ax.set_yticks([-0.4, 0.0, 0.4])
            if i == 0:
                ax.set_title(cols[j], fontsize=13)
            if j == 0:
                ax.set_ylabel(f"{lab}\n\ny", fontsize=11)
            if times[i]:
                ax.text(0.03, 0.965, f"{float(times[i][j]):.0f} s",
                        transform=ax.transAxes, ha="left", va="top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.78,
                                  ec="none"))
    fig.colorbar(im, ax=[axes[0][j] for j in range(nc)] + [axes[1][j] for j in range(nc)],
                 location="right", shrink=0.62, label=r"$\rho$")

    #Difference row: its own per-column scale, because the magnitude falls by
    #orders of magnitude across the row and a shared scale would show three
    #blank panels.
    for j in range(nc):
        ax = axes[2][j]
        a = load_rho(last_output(plm[j]))
        b = load_rho(last_output(vl2[j]))
        d = np.abs(a - b)
        rms = float(np.sqrt(np.mean((d / np.maximum(np.abs(b), 1e-30)) ** 2)))
        imd = ax.imshow(d, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5],
                        cmap="magma", interpolation="nearest")
        ax.set_xticks([-0.4, 0.0, 0.4]); ax.set_yticks([-0.4, 0.0, 0.4])
        ax.set_xlabel("x")
        if j == 0:
            ax.set_ylabel(r"$|\rho_{\rm plm}-\rho_{\rm vl2}|$" + "\n\ny", fontsize=11)
        ax.text(0.03, 0.965, f"max {d.max():.2e}\nrms|frac| {rms:.2e}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.82, ec="none"))
        fig.colorbar(imd, ax=ax, location="right", shrink=0.82)

    fig.suptitle(f"Kelvin-Helmholtz at $t={args.tlabel}$, uniform SDFB (p=3, rk3, "
                 "cascade): what the fallback reconstructs with", fontsize=14)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
