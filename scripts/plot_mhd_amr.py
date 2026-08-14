#!/usr/bin/env python3
"""Uniform-mesh vs AMR figures for the 2D MHD tests.

Builds two figures from the run matrix produced by scripts/mhd_amr_compare.sh:

  mhd_amr_orszag_tang.png  density at t = 0.5 for the base, matched-DOF and
                           effective-resolution uniform meshes next to the
                           2-level AMR run, plus the block map and the
                           classic pressure cut along y = 0.4277.
  mhd_amr_field_loop.png   |B|^2 after one advection period next to the exact
                           (initial) loop, plus the block map, the magnetic
                           energy history and the L1 error at matched DOF.

    python3 scripts/plot_mhd_amr.py [rundir] [--dest .]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import spdk_io as io

NMHD = 8            # W_cv layout: rho, vx, vy, vz, P, Bx, By, Bz
RHO, PRS, BX = 0, 4, 5


# --------------------------------------------------------------------------
# loading

def load(outdir, i, N, p, var, prefix, nvar):
    """A true-2D SD output variable, stitched onto the global FV sub-grid."""
    fn = os.path.join(outdir, f"{prefix}_N{N}p{p}_{i}_0.dat")
    n, Ne = p + 1, N + 2 * io.NGH
    A = np.fromfile(fn).reshape(1, nvar, 1, Ne, Ne, 1, n, n)
    v = A[0, var, 0, io.NGH:-io.NGH, io.NGH:-io.NGH, 0]
    return v.transpose(0, 2, 1, 3).reshape(N * n, N * n)


def read_blocks(outdir, i):
    """(header dict, list of blocks) from amr_blocks_<i>.txt, or None."""
    fn = os.path.join(outdir, f"amr_blocks_{i}.txt")
    if not os.path.exists(fn):
        return None
    with open(fn) as f:
        lines = [l for l in f if not l.startswith("#")]
    h = [int(v) for v in lines[0].split()]
    hdr = dict(zip("nblocks max_level ndim NBx NBy NBz Nfx Nfy Nfz".split(), h))
    blocks = []
    for l in lines[1:]:
        w = l.split()
        blocks.append((int(w[1]), *[float(v) for v in w[5:9]]))
    return hdr, blocks


class Run:
    """One output directory, with the DOF bookkeeping needed for the labels.

    An AMR run writes each output on whatever its finest level is at that
    moment, so the element count N is per output rather than per run and the
    coordinate files of every level visited sit side by side in the directory.
    """

    def __init__(self, rundir, name, label):
        self.outdir = os.path.join(rundir, name)
        self.name, self.label = name, label
        self.N = {}
        for f in glob.glob(os.path.join(self.outdir, "W_cv_N*_0.dat")):
            m = re.match(r"W_cv_N(\d+)p(\d+)_(\d+)_0.dat", os.path.basename(f))
            N, p, j = (int(g) for g in m.groups())
            #skip the output a run still in flight is in the middle of writing
            if os.path.getsize(f) == 8 * NMHD * (N + 2 * io.NGH) ** 2 * (p + 1) ** 2:
                self.N[j], self.p = N, p
        self.idx = sorted(self.N)
        self.amr = read_blocks(self.outdir, self.idx[0]) is not None
        self.faces = {}

    def w(self, i, var):
        return load(self.outdir, i, self.N[i], self.p, var, "W_cv", NMHD)

    def b2(self, i):
        """B^2, from W_cv rather than the B2_cv dump, which the multiblock
        output path does not write."""
        return sum(self.w(i, v) ** 2 for v in (BX, BX + 1, BX + 2))

    def x(self, i):
        """FV sub-grid face positions of the output's grid, ghosts stripped."""
        if i not in self.faces:
            xf = np.fromfile(os.path.join(
                self.outdir, f"X_N{self.N[i]}p{self.p}_0.dat"))
            self.faces[i] = xf[io.nGH:len(xf) - io.nGH]
        return self.faces[i]

    def dof(self, i=None):
        """Solution points carried by the mesh (mean over outputs for AMR)."""
        n2 = (self.p + 1) ** 2
        if not self.amr:
            return self.N[self.idx[0]] ** 2 * n2
        idx = self.idx if i is None else [i]
        vals = [read_blocks(self.outdir, j)[0] for j in idx]
        return float(np.mean([h["nblocks"] * h["NBx"] * h["NBy"] * n2
                              for h in vals]))

    def eff_res(self, i=None):
        """Linear resolution in solution points of the finest grid reached."""
        return max(self.N.values()) * (self.p + 1)

    def centers(self, i):
        xf = self.x(i)
        return 0.5 * (xf[1:] + xf[:-1])

    def extent(self, i):
        xf = self.x(i)
        return [xf[0], xf[-1], xf[0], xf[-1]]

    def energy(self, i):
        """Volume integral of |B|^2 / 2 over the domain."""
        w = np.diff(self.x(i))
        return float(0.5 * (self.b2(i) * w[:, None] * w[None, :]).sum())

    def divb(self, i):
        """max|divB| the solver reported at output i."""
        log = os.path.join(self.outdir, "log.txt")
        if not os.path.exists(log):
            return None
        v = re.findall(r"max\|divB\| = ([0-9.eE+-]+)", open(log).read())
        return float(v[i]) if i < len(v) else None

    def tag(self, i=None):
        """Short descriptor: resolution, DOF and the divergence error."""
        k = self.dof() / 1e3
        if self.amr:
            hdr, _ = read_blocks(self.outdir, self.idx[-1])
            base = max(self.N.values()) >> hdr["max_level"]
            s = (f"AMR {base}$^2$ + {hdr['max_level']} levels\n"
                 f"{self.eff_res()}$^2$ effective, {k:.0f}k DOF")
        else:
            s = (f"uniform {self.N[self.idx[0]]}$^2$ elements\n"
                 f"{self.eff_res()}$^2$ points, {k:.0f}k DOF")
        d = None if i is None else self.divb(i)
        return s if d is None else s + f"\nmax$|\\nabla\\cdot B|$ = {d:.1e}"


def draw_blocks(ax, run, i, lw=0.35):
    got = read_blocks(run.outdir, i)
    if got is None:
        return
    hdr, blocks = got
    cols = ["#2b2b2b", "#1f77b4", "#d62728", "#2ca02c"]
    for lev, x0, x1, y0, y1 in blocks:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec=cols[lev % len(cols)], lw=lw))


def level_map(ax, run, i):
    """Block map coloured by refinement level."""
    hdr, blocks = read_blocks(run.outdir, i)
    cols = ["#dfe7f2", "#8fb6dc", "#2f6ca8"]
    for lev, x0, x1, y0, y1 in blocks:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fc=cols[min(lev, 2)], ec="white", lw=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    n = {l: 0 for l in range(hdr["max_level"] + 1)}
    for b in blocks:
        n[b[0]] += 1
    ax.set_title("AMR block map\n"
                 + ", ".join(f"L{l}: {c}" for l, c in sorted(n.items())),
                 fontsize=9)


def panel(ax, run, i, field, ext, clim, cmap, blocks=False):
    im = ax.imshow(field, origin="lower", extent=ext, cmap=cmap,
                   vmin=clim[0], vmax=clim[1], interpolation="nearest")
    if blocks:
        draw_blocks(ax, run, i)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return im


# --------------------------------------------------------------------------
# figures

def common_index(runs):
    """Latest output index every run has reached (all share output/dt)."""
    return min(r.idx[-1] for r in runs)


def figure_ot(rundir, dest, dt_out=0.1):
    runs = [Run(rundir, "ot_uni32", "base"),
            Run(rundir, "ot_uni64", "matched DOF"),
            Run(rundir, "ot_amr", "AMR"),
            Run(rundir, "ot_uni128", "reference")]
    i = common_index(runs)
    t = i * dt_out
    last = {r.name: i for r in runs}
    ref = runs[-1]
    rho_ref = ref.w(i, RHO)
    clim = (rho_ref.min(), rho_ref.max())

    fig = plt.figure(figsize=(14.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0])
    for c, r in enumerate(runs):
        ax = fig.add_subplot(gs[0, c])
        im = panel(ax, r, i, r.w(i, RHO), r.extent(i), clim, "viridis",
                   blocks=r.amr)
        ax.set_title(f"{r.label}\n{r.tag(i)}", fontsize=9.5)
    fig.colorbar(im, ax=fig.axes[:4], shrink=0.8,
                 label=r"$\rho$" + f"   ($t={t:g}$)")

    ax = fig.add_subplot(gs[1, 0])
    level_map(ax, runs[2], i)

    ax = fig.add_subplot(gs[1, 1:3])
    yc = 0.4277
    for r, st in zip(runs, ["-", "-", "-", "k--"]):
        p = r.w(i, PRS)
        c = r.centers(i)
        j = int(np.argmin(abs(c - yc)))
        ax.plot(c, p[j], st, lw=1.2, label=r.tag().split("\n")[0])
    ax.set_xlabel("x")
    ax.set_ylabel("P")
    ax.set_title(f"pressure cut at y = {yc}, t = {t:g}", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    err_bars(ax, [r.w(i, RHO) for r in runs], runs, ref)

    fig.suptitle("Orszag-Tang vortex: uniform mesh vs 2-level AMR at matched DOF",
                 fontsize=13)
    fn = os.path.join(dest, "mhd_amr_orszag_tang.png")
    fig.savefig(fn, dpi=125)
    plt.close(fig)
    return fn


def downsample(a, m):
    """Block-average a onto an m x m grid (a.shape must be a multiple of m)."""
    k = a.shape[0] // m
    return a.reshape(m, k, m, k).mean(axis=(1, 3))


def err_bars(ax, fields, runs, ref):
    """L1 error against the reference field, on the coarsest common grid."""
    m = min(f.shape[0] for f in fields)
    r0 = downsample(fields[runs.index(ref)], m)
    names, errs, dofs = [], [], []
    for f, r in zip(fields, runs):
        if r is ref:
            continue
        e = downsample(f, m)
        names.append(r.label)
        errs.append(float(np.abs(e - r0).mean() / np.abs(r0).mean()))
        dofs.append(r.dof() / 1e3)
    cols = ["#999999", "#1f77b4", "#d62728"]
    ax.bar(names, errs, color=cols[:len(names)])
    for i, (e, d) in enumerate(zip(errs, dofs)):
        ax.text(i, e, f"{e*100:.2f}%\n{d:.0f}k DOF", ha="center",
                va="bottom", fontsize=8)
    ax.set_ylim(0, max(errs) * 1.45)
    ax.set_ylabel(r"relative $L_1$ error in $\rho$")
    ax.set_title(f"error vs the {ref.eff_res()}$^2$ reference", fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")


def figure_fl(rundir, dest, dt_out=0.125):
    runs = [Run(rundir, "fl_uni32", "base"),
            Run(rundir, "fl_uni64", "matched DOF"),
            Run(rundir, "fl_amr", "AMR"),
            Run(rundir, "fl_uni128", "reference")]
    i = common_index(runs)
    t = i * dt_out
    ref = runs[-1]
    #The loop is advected rigidly at v = (2,1) on a periodic unit box, so the
    #exact solution at any time is the initial condition shifted. Every output
    #time lands on a whole number of grid points, so a roll is exact.
    b0 = ref.b2(ref.idx[0])
    n = b0.shape[0]
    exact = np.roll(b0, (int(round(n * t)) % n, int(round(2 * n * t)) % n),
                    axis=(0, 1))
    clim = (0.0, b0.max())
    fields = [r.b2(i) for r in runs]

    fig = plt.figure(figsize=(15.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.35, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, ref, ref.idx[0], exact, ref.extent(ref.idx[0]), clim, "magma")
    ax.set_title(f"exact\n{ref.eff_res()}$^2$ points", fontsize=9.5)
    for c, (r, f) in enumerate(zip(runs, fields)):
        ax = fig.add_subplot(gs[0, c + 1])
        im = panel(ax, r, i, f, r.extent(i), clim, "magma", blocks=r.amr)
        ax.set_title(f"{r.label}\n{r.tag(i)}", fontsize=9.5)
    fig.colorbar(im, ax=fig.axes[:5], shrink=0.8,
                 label=r"$|B|^2$" + f"   ($t={t:g}$)")

    ax = fig.add_subplot(gs[1, 0])
    level_map(ax, runs[2], i)

    ax = fig.add_subplot(gs[1, 1:3])
    for r in runs:
        idx = [j for j in r.idx if j <= i]
        e = np.array([r.energy(j) for j in idx])
        ax.plot([j * dt_out for j in idx], e / e[0], "o-", ms=3.5, lw=1.2,
                label=r.tag().split("\n")[0])
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$E_{mag}(t)\,/\,E_{mag}(0)$")
    ax.set_title("magnetic energy decay (exact: constant)", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 3:])
    m = min(f.shape[0] for f in fields)
    e0 = downsample(exact, m)
    names, errs, dofs = [], [], []
    for f, r in zip(fields, runs):
        e = downsample(f, m)
        names.append(r.label)
        errs.append(float(np.abs(e - e0).mean() / np.abs(e0).mean()))
        dofs.append(r.dof() / 1e3)
    ax.bar(names, errs, color=["#999999", "#1f77b4", "#d62728", "#2ca02c"])
    for c, (e, d) in enumerate(zip(errs, dofs)):
        ax.text(c, e, f"{e*100:.1f}%\n{d:.0f}k DOF", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylim(0, max(errs) * 1.45)
    ax.set_ylabel(r"relative $L_1$ error in $|B|^2$")
    ax.set_title("error against the exact solution", fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Field loop at t = {t:g}: uniform mesh vs 2-level AMR "
                 "at matched DOF", fontsize=13)
    fn = os.path.join(dest, "mhd_amr_field_loop.png")
    fig.savefig(fn, dpi=125)
    plt.close(fig)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir", nargs="?", default="mhd_amr_runs")
    ap.add_argument("--dest", default="mhd_amr_runs")
    ap.add_argument("--only", choices=["ot", "fl"])
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    if args.only in (None, "ot"):
        print(figure_ot(args.rundir, args.dest))
    if args.only in (None, "fl"):
        print(figure_fl(args.rundir, args.dest))


if __name__ == "__main__":
    main()
