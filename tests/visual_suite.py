#!/usr/bin/env python3
"""Visual verification suite for spd_K.

Runs a battery of initial conditions and knob combinations, renders PNG
panels from the binary outputs, and writes docs/gallery.md for browsing.

Usage:
  tests/visual_suite.py [--build-dir DIR] [--only PATTERN] [--fast] [--skip-run]

Sections:
  ics       — every built-in IC at a modest resolution
  knobs     — integrator, order, fallback, blending, NAD, dimensionality
  induction — magnetic-loop advection
"""
import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Clean, publication-style look: STIX serif (Times-like) for text and math,
# with inward ticks. LaTeX-style field labels via matplotlib mathtext.
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# Display labels (mathtext) for the field names used in the panels.
TEX = {
    "rho": r"$\rho$",
    "p": r"$p$",
    "vx": r"$v_x$",
    "vy": r"$v_y$",
    "v_y": r"$v_y$",
    "By": r"$B_y$",
    "|B|": r"$|B|$",
}


def tex(name):
    return TEX.get(name, name)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spdk_io
from spdk_io import VARS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GALLERY_DIR = os.path.join(ROOT, "docs", "gallery")
GALLERY_MD = os.path.join(ROOT, "docs", "gallery.md")

# Cases that need a different binary than the default --build-dir one.
BUILD_OVERRIDE = {}

# Extra comment lines shown inside the reproduction command block in the gallery.
CMD_NOTE = {}

# Cases rendered as a 2D map (t=0 vs t=end) plus a 1D centerline cut of a
# single field, instead of the default map + trouble panels.
FIELD_CUT = {
    "sine_wave_2d": ("rho", VARS["rho"]),
    "user_2d": ("v_y", VARS["vy"]),
}


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return r


def case_cmd(binary, case):
    return [binary, "-i", os.path.join(ROOT, case["input"])] + case.get("overrides", [])


def gallery_cmd(case, build_dir="build"):
    """Portable command line for docs/gallery.md (relative paths only)."""
    return [os.path.join(build_dir, "spd_K"), "-i", case["input"]] + case.get(
        "overrides", []
    )


# (id, section, title, input, overrides, system)
CASES = [
    # --- initial conditions ---
    ("sine_wave_2d", "ics", "Sine wave (2D advection, one period)", "inputs/sine_wave.athinput",
     ["mesh/nx3=1", "mesh/nx1=32", "mesh/nx2=32", "time/tlim=1.0", "output/dt=1.0"], "hydro"),
    ("square_2d", "ics", "Square signal (2D top-hat)", "inputs/square.athinput", [], "hydro"),
    ("sod_1d", "ics", "Sod shock tube (1D)", "inputs/sod.athinput", [], "hydro"),
    ("shu_osher_1d", "ics", "Shu-Osher (1D)", "inputs/shu_osher.athinput", [], "hydro"),
    ("kelvin_helmholtz_2d", "ics", "Kelvin-Helmholtz (2D)", "inputs/kelvin_helmholtz.athinput", [], "hydro"),
    ("implosion_2d", "ics", "Liska-Wendroff implosion (2D, reflective, 400^2 DOF)", "inputs/implosion.athinput",
     ["mesh/nx1=100", "mesh/nx2=100", "output/dt=2.5"], "hydro"),
    ("user_2d", "ics", "User IC hook (viscous velocity pulse)", "inputs/user.athinput",
     ["mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=1", "time/tlim=1.0", "output/dt=1.0",
      "problem/sigma=0.08", "problem/amp=0.5", "problem/v1=1.0", "hydro/nu=1e-3"], "hydro"),
    ("sedov_3d", "ics", "Sedov-Taylor blast (3D)", "inputs/sedov.athinput",
     ["mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=32", "time/tlim=0.5", "output/dt=0.5"], "hydro"),
    ("spherical_blast_2d", "ics", "Spherical blast (2D, periodic rectangular box)",
     "inputs/spherical_blast.athinput",
     ["mesh/nx1=100", "mesh/nx2=150", "mesh/x2len=1.5", "time/tlim=1.5", "output/dt=1.5"], "hydro"),
    ("rti_2d", "ics", "Rayleigh-Taylor instability (2D, gravity)", "inputs/rti.athinput", [], "hydro"),

    # --- knob variations (sine_wave as the representative smooth flow) ---
    ("knob_ader", "knobs", "Integrator: ADER", "inputs/sine_wave.athinput",
     ["mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=1", "time/integrator=ader",
      "time/tlim=0.2", "output/dt=0.1", "job/fallback=true"], "hydro"),
    ("knob_rk3", "knobs", "Integrator: RK3", "inputs/sine_wave.athinput",
     ["mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=1", "time/integrator=rk3",
      "time/tlim=0.2", "output/dt=0.1", "job/fallback=true"], "hydro"),
    ("knob_p3", "knobs", "Order p=3", "inputs/sine_wave.athinput",
     ["mesh/p=3", "mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=1",
      "time/tlim=0.2", "output/dt=0.1", "job/fallback=true"], "hydro"),
    ("knob_p7", "knobs", "Order p=7", "inputs/sine_wave.athinput",
     ["mesh/p=7", "mesh/nx1=8", "mesh/nx2=8", "mesh/nx3=1",
      "time/tlim=0.2", "output/dt=0.1", "job/fallback=true"], "hydro"),
    ("knob_no_fallback", "knobs", "Fallback off (SD only)", "inputs/sine_wave.athinput",
     ["mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=1", "job/fallback=false",
      "time/tlim=0.2", "output/dt=0.1"], "hydro"),
    ("knob_no_blending", "knobs", "Blending off (binary theta)", "inputs/sine_wave.athinput",
     ["mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=1", "fallback/blending=false",
      "time/tlim=0.2", "output/dt=0.1"], "hydro"),
    ("knob_nad_delta", "knobs", "NAD: delta band", "inputs/sine_wave.athinput",
     ["mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=1", "fallback/NAD=delta",
      "time/tlim=0.2", "output/dt=0.1"], "hydro"),

    # --- induction ---
    ("induction_loop_3d", "induction", "Magnetic loop (3D induction)", "inputs/induction_loop.athinput", [], "induction"),
]

# One-line description of what each panel demonstrates.
CAPTIONS = {
    "sine_wave_2d": "Smooth density sine wave advected diagonally for one full period (t=1). The maps (left) show the initial and final states; the centerline cut (right) overlays them, confirming the profile returns to its initial shape with negligible dissipation.",
    "square_2d": "Top-hat density advected on a periodic box; edges stay sharp with fallback active at the discontinuities.",
    "sod_1d": "Sod shock tube: initial jump (top) develops into rarefaction, contact, and shock (bottom).",
    "shu_osher_1d": "Shu-Osher problem: Mach-3 shock interacting with a sinusoidal density field; fine post-shock oscillations are captured.",
    "kelvin_helmholtz_2d": "Kelvin-Helmholtz shear layer rolling up into vortices; trouble map (right) traces the shear interfaces.",
    "implosion_2d": "Liska-Wendroff implosion at 400^2 DOF (N=100, p=3) run to t=2.5; the shock collapses onto the corner and a symmetric jet forms along the diagonal between the reflective walls.",
    "user_2d": "User IC hook (src/user_ic.hpp): a Gaussian pulse in the transverse velocity v_y that varies only in x, advected one full period by v1=1. With viscosity enabled at runtime (hydro/nu=1e-3) it is an exact advection-diffusion problem: the maps (left) and the centerline cut (right, IC vs t=1) show the pulse stays Gaussian while spreading and losing amplitude, the signature of correct viscous diffusion (variance grows linearly in time).",
    "sedov_3d": "Sedov-Taylor point explosion in 3D (32^3 elements, mid-z slice); an expanding spherical blast wave forms with trouble cells on the shock front.",
    "spherical_blast_2d": "Over-pressured blast (p_in/p_out = 100) in a periodic rectangular box (1 x 1.5, 400x600 DOF, square cells), run to t=1.5. The shock and dense shell expand isotropically, then wrap and interact asymmetrically because of the box aspect ratio (cf. the Athena blast test).",
    "rti_2d": "Single-mode Rayleigh-Taylor instability under constant vertical gravity (hydro/g2=1, ported from the spd reference). Heavy fluid (rho=2) rests below light fluid (rho=1) in hydrostatic balance in a tall reflective box (periodic in x); a cos(8 pi x) velocity perturbation grows into the classic rising mushroom plume, with the trouble map (right) tracking the density interface and roll-ups.",
    "knob_ader": "ADER single-step time integration on the smooth sine wave.",
    "knob_rk3": "SSP-RK3 time integration on the same smooth sine wave for comparison.",
    "knob_p3": "Polynomial order p=3 within each element.",
    "knob_p7": "Polynomial order p=7 on a coarser mesh (same DOF budget).",
    "knob_no_fallback": "Pure spectral-difference update (fallback disabled): no trouble detection or FV correction.",
    "knob_no_blending": "Fallback with binary theta (blending off): trouble cells fully replaced by the FV update.",
    "knob_nad_delta": "NAD trouble band scaled by the local range (delta) instead of the value magnitude.",
    "induction_loop_3d": "Kinematic magnetic-loop advection; the field structure is transported across the domain (mid-plane slice).",
}

FAST_OVERRIDES = {
    "mesh/nx1": "8", "mesh/nx2": "8", "mesh/nx3": "8",
    "time/tlim": "0.05", "output/dt": "0.05",
}
FAST_SKIP_SECTIONS = {"knobs"}


def apply_fast(case):
    if case["section"] in FAST_SKIP_SECTIONS:
        return None
    c = dict(case)
    ov = list(c.get("overrides", []))
    keys = {o.split("/")[0] + "/" + o.split("/")[1].split("=")[0] for o in ov if "=" in o}
    for k, v in FAST_OVERRIDES.items():
        base = k.rsplit("=", 1)[0] if "=" in k else k
        if not any(o.startswith(base + "=") for o in ov):
            ov.append(f"{k}={v}")
    # shrink implosion further for the gallery smoke run
    if "implosion" in c["input"]:
        ov += ["time/tlim=0.15", "output/dt=0.075"]
    c["overrides"] = ov
    return c


def run_case(binary, case, outdir):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    env = dict(os.environ, SPD_OUTPUT_DIR=outdir)
    sh(case_cmd(binary, case), env=env)


def _coords_1d(grid):
    return grid.centers(0)


def _pcolormesh2d(ax, grid, field, **kw):
    """Cell-centered 2d field on the FV sub-grid (faces from grid.faces)."""
    xf, yf = grid.faces[0], grid.faces[1]
    #field: (ny, nx) with x fastest in load_sd
    return ax.pcolormesh(xf, yf, field, shading="flat", **kw)

def plot_hydro_1d(grid, idx, png, title):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    fig.suptitle(title, fontsize=11)
    x = _coords_1d(grid)
    labels = [("rho", VARS["rho"]), ("p", VARS["p"]), ("vx", VARS["vx"])]
    for col, (name, var) in enumerate(labels):
        for row, (label, i) in enumerate([("t=0", idx[0]), ("t=end", idx[-1])]):
            y = load_field_1d(grid, i, var)
            axes[row, col].plot(x, y, lw=1.2)
            axes[row, col].set_ylabel(tex(name))
            if row == 0:
                axes[row, col].set_title(tex(name))
            if row == 1:
                axes[row, col].set_xlabel("$x$")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def load_field_1d(grid, i, var):
    f = spdk_io.load_sd_field(grid, i, var)
    return f[0, 0, :]


def load_field_slice(grid, i, var, system):
    prefix = "B2_cv" if system == "induction" else None
    f = spdk_io.load_sd_field(grid, i, var, prefix=prefix)
    if grid.ndim == 3:
        return f[f.shape[0] // 2]
    if grid.ndim == 2:
        return np.squeeze(f)
    return f[0, 0, :]


def plot_hydro_2d(grid, idx, png, title, system, show_trouble):
    fig, axes = plt.subplots(2, 2 if show_trouble else 1, figsize=(8 if show_trouble else 5, 7),
                             squeeze=False)
    fig.suptitle(title, fontsize=11)
    var = 1 if system == "induction" else VARS["rho"]
    vname = tex("By") if system == "induction" else tex("rho")
    for row, (label, i) in enumerate([("t=0", idx[0]), ("t=end", idx[-1])]):
        field = load_field_slice(grid, i, var, system)
        ax = axes[row, 0]
        im = _pcolormesh2d(ax, grid, field, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(f"{vname} ({label})")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        fig.colorbar(im, ax=ax, fraction=0.046)
        if show_trouble:
            tr = spdk_io.load_troubles(grid, i)
            ax2 = axes[row, 1]
            if tr is not None:
                im2 = _pcolormesh2d(ax2, grid, tr, cmap="magma", vmin=0, vmax=1)
                ax2.set_title(rf"trouble $\theta$ ({label})")
                fig.colorbar(im2, ax=ax2, fraction=0.046)
            else:
                ax2.text(0.5, 0.5, "no troubles\n(SD only)", ha="center", va="center",
                         transform=ax2.transAxes)
            ax2.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def plot_induction(grid, idx, png, title):
    """Mid-plane in-plane magnetic-field magnitude for the induction system.

    The output stores the magnetic field B = curl(A); for the loop setup the
    in-plane components (Bx, By) carry the structure while Bz is zero, so we
    show |B_xy| = sqrt(Bx^2 + By^2), which renders the loop as a ring.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), squeeze=False)
    fig.suptitle(title, fontsize=11)
    for col, (label, i) in enumerate([("t=0", idx[0]), ("t=end", idx[-1])]):
        bx = load_field_slice(grid, i, 0, "induction")
        by = load_field_slice(grid, i, 1, "induction")
        mag = np.sqrt(bx ** 2 + by ** 2)
        ax = axes[0, col]
        im = _pcolormesh2d(ax, grid, mag, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(rf"$|B|$ ({label})")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def plot_hydro_3d(grid, idx, png, title, show_trouble):
    fig, axes = plt.subplots(2, 2 if show_trouble else 1, figsize=(8 if show_trouble else 5, 7),
                             squeeze=False)
    fig.suptitle(title, fontsize=11)
    zmid = grid.centers(2)[len(grid.centers(2)) // 2]
    for row, (label, i) in enumerate([("t=0", idx[0]), ("t=end", idx[-1])]):
        vol = spdk_io.load_sd_field(grid, i, VARS["rho"])
        sl = vol.shape[0] // 2
        field = vol[sl]
        ax = axes[row, 0]
        im = _pcolormesh2d(ax, grid, field, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(rf"$\rho$  $z={zmid:.2f}$ ({label})")
        fig.colorbar(im, ax=ax, fraction=0.046)
        if show_trouble:
            trvol = spdk_io.load_troubles(grid, i)
            ax2 = axes[row, 1]
            if trvol is not None:
                im2 = _pcolormesh2d(ax2, grid, trvol[sl], cmap="magma", vmin=0, vmax=1)
                ax2.set_title(rf"trouble $\theta$ ({label})")
                fig.colorbar(im2, ax=ax2, fraction=0.046)
            ax2.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def plot_field_cut(grid, idx, png, title, var, vname, system="hydro", overlay=None):
    """2D maps of a single field at t=0 and t=end, plus a horizontal
    centerline cut overlaying both times.

    Used for the smooth advection / viscous-diffusion tests where the point is
    to compare the initial condition against the solution rather than to look
    for trouble cells. ``overlay`` is an optional (x, y, label) analytic curve
    drawn on the cut panel.
    """
    f0 = load_field_slice(grid, idx[0], var, system)
    f1 = load_field_slice(grid, idx[-1], var, system)
    vmin = float(min(f0.min(), f1.min()))
    vmax = float(max(f0.max(), f1.max()))
    x = grid.centers(0)
    ycen = grid.centers(1)
    jmid = f0.shape[0] // 2
    vlabel = tex(vname)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25])
    fig.suptitle(title, fontsize=11)
    ax_cut = fig.add_subplot(gs[:, 1])
    for row, (label, field, style) in enumerate(
        [("t=0 (IC)", f0, "--"), ("t=end", f1, "-")]
    ):
        ax = fig.add_subplot(gs[row, 0])
        im = _pcolormesh2d(ax, grid, field, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(f"{vlabel} ({label})")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax_cut.plot(x, field[jmid, :], style, lw=1.5, label=label)
    if overlay is not None:
        ox, oy, olabel = overlay
        ax_cut.plot(ox, oy, ":", color="k", lw=1.6, label=olabel)
    ax_cut.set_title(rf"{vlabel} cut at $y={ycen[jmid]:.2f}$")
    ax_cut.set_xlabel("$x$")
    ax_cut.set_ylabel(vlabel)
    ax_cut.legend()
    ax_cut.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def render_case(case, outdir, png):
    grid = spdk_io.Grid(outdir)
    idx = spdk_io.output_indices(outdir, prefix="B2_cv" if case["system"] == "induction" else "W_cv")
    if len(idx) < 1:
        raise RuntimeError(f"no outputs in {outdir}")
    if case["id"] in FIELD_CUT:
        vname, var = FIELD_CUT[case["id"]]
        plot_field_cut(grid, idx, png, case["title"], var, vname, case["system"])
        return
    show_trouble = case["system"] == "hydro"
    if any(o == "job/fallback=false" for o in case.get("overrides", [])):
        show_trouble = False
    if case["system"] == "induction":
        plot_induction(grid, idx, png, case["title"])
    elif grid.ndim == 1:
        plot_hydro_1d(grid, idx, png, case["title"])
    elif grid.ndim == 2:
        plot_hydro_2d(grid, idx, png, case["title"], case["system"], show_trouble)
    else:
        plot_hydro_3d(grid, idx, png, case["title"], show_trouble)


def write_gallery(entries, binary):
    os.makedirs(os.path.dirname(GALLERY_MD), exist_ok=True)
    sections = [("ics", "Initial conditions"),
                ("knobs", "Knobs (sine-wave representative)"),
                ("induction", "Induction")]
    lines = [
        "# spd_K visual test gallery",
        "",
        "Auto-generated by `tests/visual_suite.py`. Each panel shows the",
        "first and last output snapshot; the command above each figure",
        "reproduces exactly the run that produced it.",
        "",
    ]
    by_sec = {}
    for e in entries:
        by_sec.setdefault(e["section"], []).append(e)
    for sec_id, sec_title in sections:
        if sec_id not in by_sec:
            continue
        lines += [f"## {sec_title}", ""]
        for e in by_sec[sec_id]:
            rel_build = os.path.relpath(os.path.dirname(binary), ROOT)
            cmd = " ".join(gallery_cmd(e["case"], rel_build))
            block = ["```bash"]
            if e["id"] in CMD_NOTE:
                block.append(CMD_NOTE[e["id"]])
            block += [cmd, "```"]
            lines += [
                f"### {e['title']}",
                "",
                *block,
                "",
                f"![{e['id']}](gallery/{e['id']}.png)",
                "",
            ]
            caption = CAPTIONS.get(e["id"])
            if caption:
                lines += [f"*{caption}*", ""]
    with open(GALLERY_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {GALLERY_MD} ({len(entries)} cases)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "build"))
    ap.add_argument("--only", default="*", help="fnmatch pattern on case id")
    ap.add_argument("--fast", action="store_true", help="coarse grid, short runs; skip knob section")
    ap.add_argument("--skip-run", action="store_true", help="re-render from existing output dirs")
    args = ap.parse_args()

    binary = os.path.join(args.build_dir, "spd_K")
    if not args.skip_run and not os.path.isfile(binary):
        print(f"building {binary}")
        sh(["cmake", "--build", args.build_dir, "-j8"])

    os.makedirs(GALLERY_DIR, exist_ok=True)
    entries = []
    failures = 0

    for spec in CASES:
        case_id, section, title, inp, overrides, system = spec
        if not fnmatch.fnmatch(case_id, args.only):
            continue
        case = {"id": case_id, "section": section, "title": title,
                "input": inp, "overrides": overrides, "system": system}
        if args.fast:
            case = apply_fast(case)
            if case is None:
                continue
        outdir = os.path.join(args.build_dir, "visual_out", case_id)
        png = os.path.join(GALLERY_DIR, f"{case_id}.png")
        case_binary = os.path.join(BUILD_OVERRIDE[case_id], "spd_K") \
            if case_id in BUILD_OVERRIDE else binary
        try:
            if not args.skip_run:
                print(f"RUN {case_id}")
                if not os.path.isfile(case_binary):
                    raise RuntimeError(
                        f"missing binary {case_binary} (needed for {case_id})")
                run_case(case_binary, case, outdir)
            print(f"PLOT {case_id}")
            render_case(case, outdir, png)
            entries.append({"id": case_id, "section": section, "title": title, "case": case})
        except Exception as exc:
            print(f"[FAIL] {case_id}: {exc}")
            failures += 1

    write_gallery(entries, binary)
    if failures:
        print(f"{failures} case(s) failed")
    return failures


if __name__ == "__main__":
    sys.exit(min(main(), 99))
