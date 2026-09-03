"""KH history figures: Rueda-Ramirez et al. 2022 figure 7.

    python3 scripts/mdz21_kh_figures.py kh_runs <outdir> [dof]

Plots the two quantities the paper ranks schemes by, against time:
  <B_p^2>(t)  normalised poloidal magnetic energy (their eq. 100)
  dv_y(t)     growth rate (their eq. 101)
A single final-time number is not the diagnostic: the instability grows,
saturates and then decays into turbulence, so two schemes can agree at t=20
having taken completely different routes.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import mdz21_diagnostics as dg
from mdz21_report import LANE_LABEL, LANE_ORDER, find_runs

EMF_LABEL = {"hlld": "UCT-HLLD", "hll": "UCT-HLL"}
STYLE = {"muscl_rk2": ("-", "#0b7285"), "muscl_rk2_2x": ("--", "#0b7285"),
         "sdfb4_rk3": ("-", "#b45309"), "sdfb8_rk3": ("-", "#7b2d8e")}
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"],
                     "axes.edgecolor": "#c9ced8", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def main(root, outdir, dof=None):
    runs = find_runs(root)
    if not runs:
        print("no completed runs"); return 1
    dofs = sorted({k[2] for k in runs})
    dof = int(dof) if dof else dofs[-1]
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    for col, emf in enumerate(("hlld", "hll")):
        lanes = runs.get(("uct", emf, dof), {})
        for lane in LANE_ORDER:
            d = lanes.get(lane)
            if not d:
                continue
            bp = dg.poloidal_energy(d)
            dv = dg.kh_growth_amplitude(d)
            t = np.linspace(0, 20, len(bp))
            ls, c = STYLE.get(lane, ("-", "#5c6472"))
            axes[0][col].semilogy(t, bp, ls, color=c, lw=1.8,
                                  label=LANE_LABEL.get(lane, lane))
            axes[1][col].semilogy(t, dv, ls, color=c, lw=1.8)
        axes[0][col].set_title(f"{EMF_LABEL.get(emf, emf)}  ({dof}$\\times${2*dof} DoF)",
                               fontsize=12)
        axes[1][col].set_xlabel("$t$")
        for r in (0, 1):
            axes[r][col].grid(alpha=.25, lw=.7)
            for sp in ("top", "right"):
                axes[r][col].spines[sp].set_visible(False)
    axes[0][0].set_ylabel(r"$\langle B_p^2\rangle(t)$")
    axes[1][0].set_ylabel(r"$\Delta v_y(t)$")
    axes[0][0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    f = os.path.join(outdir, f"kh_history_dof{dof}.png")
    fig.savefig(f, dpi=130)
    print("wrote", f)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2],
                  sys.argv[3] if len(sys.argv) > 3 else None))
