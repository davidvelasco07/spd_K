"""Turn an mdz21_campaign.sh run tree into the comparison's tables and figures.

    python3 scripts/mdz21_report.py mdz21_runs [outdir]

Walks <root>/<emf>_<rsolver>_dof<N>/<lane>/ and reports, per benchmark, the
quantity Mignone & Del Zanna 2021 actually publishes for it, so the numbers can
be set against the paper's rather than eyeballed:

  Orszag-Tang   mu_p = p_max/<p> in the central window (their table 1)
  field loop    volume-integrated magnetic energy decay (their figure 4)
  KH            dv_y growth rate from a fit over the linear phase (figure 13)

Lanes are at MATCHED DEGREES OF FREEDOM, not matched element counts or step
counts -- see scripts/mdz21_campaign.sh for what that means and why the step
counts differ.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import mdz21_diagnostics as dg  # noqa: E402
import spdk_io as io  # noqa: E402

LANE_LABEL = {
    "muscl_rk2": "MUSCL + RK2",
    "muscl_rk2_2x": "MUSCL RK2 2x",
    "muscl_hancock": "MUSCL-Hancock",
    "sdfb4_rk3": "SDFB4 + RK3",
    "sdfb8_rk3": "SDFB8 + RK3",
}
LANE_ORDER = ["muscl_rk2", "muscl_rk2_2x", "muscl_hancock",
              "sdfb4_rk3", "sdfb8_rk3"]


def steps_of(d):
    log = os.path.join(d, "run.log")
    if not os.path.isfile(log):
        return None
    for line in open(log):
        if line.startswith("evolution:"):
            return int(line.split()[1])
    return None


def find_runs(root):
    """{(emf, rsolver, dof): {lane: dir}} for every completed run under root."""
    out = {}
    if not os.path.isdir(root):
        return out
    for case in sorted(os.listdir(root)):
        parts = case.split("_")
        if len(parts) < 3 or not parts[-1].startswith("dof"):
            continue
        emf, rsolver, dof = parts[0], parts[1], int(parts[-1][3:])
        cdir = os.path.join(root, case)
        lanes = {}
        for lane in sorted(os.listdir(cdir)):
            d = os.path.join(cdir, lane)
            if not os.path.isdir(d):
                continue
            # A run that produced no dumps is not a result. Reporting over zero
            # files is how a comparison comes back "identical" having compared
            # nothing (CLAUDE.md rule 2).
            if not io.output_indices(d):
                print(f"  CHECK VOID: {case}/{lane} has no dumps", file=sys.stderr)
                continue
            # An UNFINISHED run is worse than no run: its last dump is at some
            # earlier time, so it silently enters the table as if it were the
            # final state and gets compared against lanes that did finish. Seen:
            # a mid-flight UCT-HLL case reporting mu_p = 2.17 against a complete
            # UCT-HLLD 1.08, purely because it was two thirds of the way through.
            # The binary prints "evolution: N steps" only on a clean exit.
            if steps_of(d) is None:
                print(f"  INCOMPLETE (skipped): {case}/{lane} "
                      f"has {len(io.output_indices(d))} dumps but has not finished",
                      file=sys.stderr)
                continue
            lanes[lane] = d
        if lanes:
            out[(emf, rsolver, dof)] = lanes
    return out


def table(runs, fn, title, fmt="{:.3f}", note=""):
    """One row per (emf, rsolver, dof), one column per lane."""
    if not runs:
        print(f"\n{title}\n  (no runs)")
        return
    lanes = [l for l in LANE_ORDER
             if any(l in v for v in runs.values())]
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    head = f"  {'emf':<8}{'rsolver':<9}{'DoF':>6}  " + "".join(
        f"{LANE_LABEL.get(l, l):>17}" for l in lanes)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for (emf, rs, dof) in sorted(runs):
        cells = []
        for l in lanes:
            d = runs[(emf, rs, dof)].get(l)
            if d is None:
                cells.append(f"{'--':>17}")
                continue
            try:
                cells.append(f"{fmt.format(fn(d)):>17}")
            except Exception as e:
                cells.append(f"{'ERR':>17}")
                print(f"    {emf}/{rs}/{dof}/{l}: {e}", file=sys.stderr)
        print(f"  {emf:<8}{rs:<9}{dof:>6}  " + "".join(cells))


def main(root, outdir=None):
    runs = find_runs(root)
    if not runs:
        print(f"no completed runs under {root}")
        return 1
    print(f"Mignone & Del Zanna 2021 comparison -- {len(runs)} case(s) under {root}")

    prov = None
    for case in sorted(os.listdir(root)):
        p = os.path.join(root, case, "provenance.txt")
        if os.path.isfile(p):
            prov = open(p).read()
            break
    if prov:
        print("\nprovenance (first case):")
        for line in prov.strip().splitlines():
            print("  " + line)

    # Which benchmark this tree is, read off the deck the campaign recorded, so
    # the tables cannot be labelled as a test the runs are not.
    which = "orszag_tang"
    if prov:
        for line in prov.splitlines():
            if line.startswith("deck"):
                which = os.path.basename(line.split()[1]).replace(".athinput", "")

    if which.startswith("kh"):
        table(runs, lambda d: dg.poloidal_energy(d)[-1],
              "KH: normalised poloidal magnetic energy <B_p^2> at the final time",
              fmt="{:.4f}",
              note="RR22 eq.100. It GROWS during the transition to turbulence, and "
                   "grows faster for a less dissipative scheme -- so higher is less "
                   "dissipative, the opposite reading from the field loop.")
        table(runs, lambda d: dg.kh_growth_amplitude(d)[-1],
              "KH: growth rate dv_y = (max v_y - min v_y)/2 at the final time",
              fmt="{:.5f}",
              note="RR22 eq.101. Their eq. writes v^2_max - v^2_min, which is a "
                   "rendering of v_y max/min; the MDZ21 form used here is the one "
                   "both papers plot.")
        table(runs, lambda d: steps_of(d) or float("nan"),
              "Steps taken", fmt="{:.0f}")
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        return 0

    table(runs, dg.pressure_peak_ratio,
          "Orszag-Tang: mu_p = p_max/<p> over the central 0.4 < x,y < 0.6, final output",
          note="paper table 1 at 128^2 -- RK2+linear: UCT-HLL 1.21, UCT-HLLD 2.54; "
               "RK3+MP5: UCT-HLL 1.30, UCT-HLLD 3.33. mu_p > ~2 means a central "
               "magnetic island formed. See pressure_peak_ratio on the window.")

    table(runs, lambda d: dg.magnetic_energy(d)[-1] / dg.magnetic_energy(d)[0],
          "Magnetic energy retained, E_B(end)/E_B(0)", fmt="{:.6f}",
          note="the paper's dissipation measure (its figures 4 and 7); "
               "higher = less numerical dissipation.")

    table(runs, lambda d: steps_of(d) or float("nan"),
          "Steps taken (lanes are matched in DoF, NOT in step count)", fmt="{:.0f}",
          note="SDFB8 runs at a smaller CFL: p=7 needs <= 0.40 under cfl_type=sum "
               "where p=3 holds 0.5 (see main.cpp).")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        plot_energy(runs, outdir)
    return 0


def plot_energy(runs, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for (emf, rs, dof), lanes in sorted(runs.items()):
        fig, ax = plt.subplots(figsize=(6, 4))
        any_line = False
        for lane in LANE_ORDER:
            d = lanes.get(lane)
            if d is None:
                continue
            try:
                eb = dg.magnetic_energy(d)
            except Exception:
                continue
            ax.plot(np.arange(len(eb)), eb / eb[0], marker="o", ms=3,
                    label=LANE_LABEL.get(lane, lane))
            any_line = True
        if not any_line:
            plt.close(fig)
            continue
        ax.set_xlabel("output index")
        ax.set_ylabel(r"$E_B(t)\,/\,E_B(0)$")
        ax.set_title(f"{emf} / {rs} / {dof}$^2$ DoF")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout()
        f = os.path.join(outdir, f"energy_{emf}_{rs}_dof{dof}.png")
        fig.savefig(f, dpi=140)
        plt.close(fig)
        print(f"  wrote {f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
