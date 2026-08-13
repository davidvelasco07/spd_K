#!/usr/bin/env python3
"""Visualize AMR runs and build performance / roofline-style plots from Apollo study.

SD_Vector inherits the Kokkos view layout of the build that wrote it: CUDA uses
LayoutLeft (leftmost index fastest, so order='F'), host builds use LayoutRight
(order='C'). Guessing wrong does not fail loudly, it just scrambles the field
into a checkerboard, so the layout is read from the run's parameters.txt.

AMR evidence comes from amr_blocks_*.txt (leaf blocks + levels), overlaid on
the finest-level stitched field.
"""
import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

NGH = 1
NVAR = 6
P_DEFAULT = 3

LEVEL_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]


def _parse_Np(path):
    m = re.search(r"_N(\d+)p(\d+)_", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse N,p from {path}")
    return int(m.group(1)), int(m.group(2))


def _output_index(path):
    m = re.search(r"_(\d+)_\d+\.dat$", os.path.basename(path))
    if m:
        return int(m.group(1))
    m = re.search(r"amr_blocks_(\d+)\.txt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def dump_order(outdir):
    """numpy reshape order for binaries in `outdir`, from its parameters.txt.

    Older runs predate the <build> stanza; fall back to Fortran order, which is
    what the CUDA builds those runs came from used.
    """
    path = os.path.join(outdir, "parameters.txt")
    try:
        with open(path) as fh:
            for line in fh:
                if line.strip().startswith("layout"):
                    return "C" if "LayoutRight" in line else "F"
    except OSError:
        pass
    return "F"


def load_field(path, ndim, var=0, p=None, order=None):
    """Load one field from a W_cv binary dump."""
    if order is None:
        order = dump_order(os.path.dirname(path))
    N, pfile = _parse_Np(path)
    if p is None:
        p = pfile
    n = p + 1
    ax = [True, ndim >= 2, ndim >= 3]
    Ne = [N + 2 * NGH if a else 1 for a in ax]
    np_ = [n if a else 1 for a in ax]
    shp = (1, NVAR, Ne[2], Ne[1], Ne[0], np_[2], np_[1], np_[0])
    raw = np.fromfile(path)
    if raw.size != int(np.prod(shp)):
        raise ValueError(f"{path}: size {raw.size} != expected {shp} ({np.prod(shp)})")
    A = raw.reshape(shp, order=order)
    U = A[0, var]
    sk = slice(NGH, -NGH) if ax[2] else slice(None)
    sj = slice(NGH, -NGH) if ax[1] else slice(None)
    si = slice(NGH, -NGH) if ax[0] else slice(None)
    U = U[sk, sj, si]
    if ndim == 1:
        return U[0, 0, :, 0, 0, :].reshape(-1)
    if ndim == 2:
        U2 = U[0, :, :, 0, :, :]
        Ny, Nx, ny, nx = U2.shape
        return U2.transpose(0, 2, 1, 3).reshape(Ny * ny, Nx * nx)
    Nz, Ny, Nx, nz, ny, nx = U.shape
    return U.transpose(0, 3, 1, 4, 2, 5).reshape(Nz * nz, Ny * ny, Nx * nx)


def load_amr_blocks(path):
    """Parse amr_blocks_N.txt -> dict with header + list of blocks."""
    blocks = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if header is None:
                header = {
                    "nblocks": int(toks[0]),
                    "max_level": int(toks[1]),
                    "ndim": int(toks[2]),
                    "NBx": int(toks[3]),
                    "NBy": int(toks[4]),
                    "NBz": int(toks[5]),
                    "Nfx": int(toks[6]),
                    "Nfy": int(toks[7]),
                    "Nfz": int(toks[8]),
                }
                continue
            blocks.append(
                {
                    "ib": int(toks[0]),
                    "level": int(toks[1]),
                    "lx": int(toks[2]),
                    "ly": int(toks[3]),
                    "lz": int(toks[4]),
                    "x0": float(toks[5]),
                    "x1": float(toks[6]),
                    "y0": float(toks[7]),
                    "y1": float(toks[8]),
                    "z0": float(toks[9]),
                    "z1": float(toks[10]),
                }
            )
    return header, blocks


def level_map_2d(header, blocks, p=P_DEFAULT):
    """Cell-centered level map on the finest SD grid (Ny*n, Nx*n)."""
    n = p + 1
    Nfx, Nfy = header["Nfx"], header["Nfy"]
    M = header["max_level"]
    lev = np.full((Nfy, Nfx), -1, dtype=int)
    for b in blocks:
        sc = 1 << (M - b["level"])
        # logical indices are in elements at this block's level; convert to finest
        i0 = b["lx"] * header["NBx"] * sc
        j0 = b["ly"] * header["NBy"] * sc
        di = header["NBx"] * sc
        dj = header["NBy"] * sc
        lev[j0 : j0 + dj, i0 : i0 + di] = b["level"]
    # expand elements -> SD points
    return np.repeat(np.repeat(lev, n, axis=0), n, axis=1)


def _wcv_files(odir):
    return sorted(glob.glob(os.path.join(odir, "W_cv_*_0.dat")), key=_output_index)


def _blocks_file(odir, n_out):
    path = os.path.join(odir, f"amr_blocks_{n_out}.txt")
    return path if os.path.isfile(path) else None


def _last_finite(files, ndim, var, min_finite=0.5):
    """Latest output with a usable fraction of finite samples.

    Falling back to an earlier output keeps the figure renderable, but it also
    makes the panel disagree with its own title, so say so rather than quietly
    plotting a different time than the label claims.
    """
    for path in reversed(files):
        try:
            f = load_field(path, ndim, var=var)
        except ValueError:
            continue
        finite = np.isfinite(f).mean()
        if finite >= min_finite:
            if path != files[-1]:
                print(f"WARNING: {os.path.basename(files[-1])} is non-finite; "
                      f"plotting {os.path.basename(path)} instead", file=sys.stderr)
            return path, f
        print(f"WARNING: {os.path.basename(path)} is {100*(1-finite):.0f}% "
              f"non-finite, skipping", file=sys.stderr)
    for path in files:
        try:
            return path, load_field(path, ndim, var=var)
        except ValueError:
            continue
    return None, None


def _draw_blocks(ax, blocks, Lx=1.0, Ly=1.0, field_shape=None):
    """Overlay leaf-block rectangles. If field_shape given, use cell index coords."""
    for b in blocks:
        if field_shape is not None:
            ny, nx = field_shape
            x0 = b["x0"] / Lx * nx
            x1 = b["x1"] / Lx * nx
            y0 = b["y0"] / Ly * ny
            y1 = b["y1"] / Ly * ny
        else:
            x0, x1, y0, y1 = b["x0"], b["x1"], b["y0"], b["y1"]
        color = LEVEL_COLORS[b["level"] % len(LEVEL_COLORS)]
        rect = mpatches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=color,
            linewidth=1.6 + 0.4 * b["level"],
        )
        ax.add_patch(rect)


def _level_legend(ax, max_level):
    handles = [
        mpatches.Patch(edgecolor=LEVEL_COLORS[lv % len(LEVEL_COLORS)], facecolor="none",
                       linewidth=2, label=f"level {lv}")
        for lv in range(max_level + 1)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)


def plot_amr_evidence(root, plot_dir):
    """Main AMR evidence figure: field + block outlines, and level maps."""
    os.makedirs(plot_dir, exist_ok=True)
    cases = [
        ("smr_sine_2d", 2, 0, r"$\rho$", "Uniform meshblocks (no refine)"),
        ("smr_patch_2d", 2, 0, r"$\rho$", "SMR: level-1 center patch"),
        ("amr_pulse_2d", 2, 4, r"$p$", "Dynamic AMR pulse"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for col, (tag, ndim, var, clabel, title) in enumerate(cases):
        odir = os.path.join(root, "visuals", tag)
        files = _wcv_files(odir)
        ax_f, ax_l = axes[0, col], axes[1, col]
        if not files:
            ax_f.set_title(f"{title}\n(no output)")
            ax_l.axis("off")
            continue

        path, field = _last_finite(files, ndim, var)
        n_out = _output_index(path)
        bpath = _blocks_file(odir, n_out)
        header, blocks = (None, [])
        if bpath:
            header, blocks = load_amr_blocks(bpath)

        # For dynamic AMR, prefer a regridded snapshot for the level map even
        # if its field blew up — that is the AMR evidence.
        lev_header, lev_blocks, lev_out = header, blocks, n_out
        for bf in sorted(glob.glob(os.path.join(odir, "amr_blocks_*.txt")), key=_output_index):
            h, bl = load_amr_blocks(bf)
            if h and h["max_level"] > (lev_header["max_level"] if lev_header else -1):
                lev_header, lev_blocks, lev_out = h, bl, _output_index(bf)

        data = np.ma.masked_invalid(field)
        im = ax_f.imshow(data, origin="lower", aspect="equal", cmap="viridis")
        plt.colorbar(im, ax=ax_f, fraction=0.046, label=clabel)
        if blocks:
            _draw_blocks(ax_f, blocks, field_shape=field.shape)
            _level_legend(ax_f, header["max_level"] if header else 0)
        nblocks = len(blocks) if blocks else "?"
        mlev = header["max_level"] if header else "?"
        ax_f.set_title(f"{title}\nout {n_out}, {nblocks} blocks, max_level={mlev}")
        ax_f.set_xlabel("x cells")
        ax_f.set_ylabel("y cells")

        if lev_header and lev_blocks:
            lev = level_map_2d(lev_header, lev_blocks)
            vmax = max(lev_header["max_level"], 1)
            cmap = ListedColormap(LEVEL_COLORS[: vmax + 1])
            norm = BoundaryNorm(np.arange(-0.5, vmax + 1.5), cmap.N)
            im2 = ax_l.imshow(lev, origin="lower", aspect="equal", cmap=cmap, norm=norm)
            cbar = plt.colorbar(im2, ax=ax_l, fraction=0.046, ticks=range(vmax + 1))
            cbar.set_label("refinement level")
            _draw_blocks(ax_l, lev_blocks, field_shape=lev.shape)
            n_coarse = sum(1 for b in lev_blocks if b["level"] == 0)
            n_fine = sum(1 for b in lev_blocks if b["level"] > 0)
            ax_l.set_title(
                f"leaf-block level map (out {lev_out})\n{n_coarse} coarse + {n_fine} refined"
            )
        else:
            ax_l.text(
                0.5,
                0.5,
                "no amr_blocks_*.txt\n(re-run with updated binary)",
                ha="center",
                va="center",
                transform=ax_l.transAxes,
            )
            ax_l.set_xticks([])
            ax_l.set_yticks([])
        ax_l.set_xlabel("x cells")
        ax_l.set_ylabel("y cells")

    fig.suptitle(
        "AMR evidence: stitched field with leaf-block outlines (color = level)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "amr_evidence.png"), dpi=150)
    plt.close(fig)


def run_hardware(root):
    """Label the machine the visuals came from, per visuals/manifest.txt.

    The perf plots are always Apollo GPU timings, but the visuals can be
    regenerated on a host build, so the two must not both be labelled A100.
    """
    path = os.path.join(root, "visuals", "manifest.txt")
    try:
        with open(path) as fh:
            if any("gpu=" in line for line in fh):
                return "Apollo A100"
    except OSError:
        return "unknown host"
    return "host build"


def plot_visuals(root, plot_dir):
    """Legacy 2x2 summary; still useful for 3D slice + quick look."""
    os.makedirs(plot_dir, exist_ok=True)
    cases = [
        ("smr_sine_2d", 2, 0, r"$\rho$", "Uniform 4×4 meshblocks"),
        ("smr_patch_2d", 2, 0, r"$\rho$", "SMR patch (level 1 center)"),
        ("amr_pulse_2d", 2, 4, r"$p$", "Dynamic AMR pulse (pressure)"),
        ("mb_uniform_3d", 3, 0, r"$\rho$ (mid-z)", "3D uniform 2×2×2 blocks"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.ravel()
    for ax, (tag, ndim, var, clabel, title) in zip(axes, cases):
        odir = os.path.join(root, "visuals", tag)
        files = _wcv_files(odir)
        if not files:
            ax.set_title(f"{title}\n(no output)")
            continue
        path, field = _last_finite(files, ndim, var)
        plot = field[field.shape[0] // 2] if ndim == 3 else field
        data = np.ma.masked_invalid(plot)
        if data.count() == 0:
            ax.text(0.5, 0.5, "all NaN", ha="center", va="center", transform=ax.transAxes)
        else:
            im = ax.imshow(data, origin="lower", aspect="equal", cmap="viridis")
            plt.colorbar(im, ax=ax, fraction=0.046, label=clabel)
            bpath = _blocks_file(odir, _output_index(path))
            if bpath and ndim == 2:
                header, blocks = load_amr_blocks(bpath)
                _draw_blocks(ax, blocks, field_shape=plot.shape)
        ax.set_title(f"{title}\n(out {_output_index(path)})")
        ax.set_xlabel("x cells")
        ax.set_ylabel("y cells")
    fig.suptitle(f"spd_K AMR on {run_hardware(root)} — final output", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "amr_visuals_2d.png"), dpi=150)
    plt.close(fig)

    odir = os.path.join(root, "visuals", "amr_pulse_2d")
    files = _wcv_files(odir)
    if files:
        fig, axes = plt.subplots(2, len(files), figsize=(4.2 * len(files), 7.5))
        if len(files) == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        for col, wf in enumerate(files):
            ax_f, ax_l = axes[0, col], axes[1, col]
            field = load_field(wf, 2, var=4)
            data = np.ma.masked_invalid(field)
            n_out = _output_index(wf)
            N, _ = _parse_Np(wf)
            bpath = _blocks_file(odir, n_out)
            header, blocks = (None, [])
            if bpath:
                header, blocks = load_amr_blocks(bpath)

            if data.count() == 0:
                ax_f.text(0.5, 0.5, "all NaN\n(solver blow-up)", ha="center", va="center",
                          transform=ax_f.transAxes)
            else:
                im = ax_f.imshow(data, origin="lower", aspect="equal", cmap="inferno")
                plt.colorbar(im, ax=ax_f, fraction=0.046, label=r"$p$")
                if blocks:
                    _draw_blocks(ax_f, blocks, field_shape=field.shape)
            nblocks = len(blocks) if blocks else "?"
            mlev = header["max_level"] if header else "?"
            ax_f.set_title(f"output {n_out} (N={N})\n{nblocks} blocks, max_level={mlev}")

            if header and blocks:
                lev = level_map_2d(header, blocks)
                vmax = max(header["max_level"], 1)
                cmap = ListedColormap(LEVEL_COLORS[: vmax + 1])
                norm = BoundaryNorm(np.arange(-0.5, vmax + 1.5), cmap.N)
                im2 = ax_l.imshow(lev, origin="lower", aspect="equal", cmap=cmap, norm=norm)
                plt.colorbar(im2, ax=ax_l, fraction=0.046, ticks=range(vmax + 1), label="level")
                _draw_blocks(ax_l, blocks, field_shape=lev.shape)
                n_fine = sum(1 for b in blocks if b["level"] > 0)
                ax_l.set_title(f"level map ({n_fine} refined leaves)")
            else:
                ax_l.text(0.5, 0.5, "no block dump", ha="center", va="center",
                          transform=ax_l.transAxes)
            ax_f.set_xlabel("x"); ax_l.set_xlabel("x")
            if col == 0:
                ax_f.set_ylabel("y"); ax_l.set_ylabel("y")
        fig.suptitle("Dynamic AMR pulse: pressure (top) and refinement level (bottom)")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "amr_pulse_evolution.png"), dpi=150)
        plt.close(fig)


def parse_bench(root):
    csv_path = os.path.join(root, "bench", "times.csv")
    rows = []
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path) as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            rows.append((parts[0], float(parts[1])))
    return rows


def estimate_dofs(tag, p=P_DEFAULT):
    nvar = NVAR
    n = p + 1
    m = (
        re.match(r"(\d)d_(?:amr|uniform)_N(\d+)_", tag)
        or re.match(r"(\d)d_N(\d+)_nb(\d+)$", tag)
        or re.match(r"(\d)d_N(\d+)_nb(\d+)x(\d+)$", tag)
        or re.match(r"(\d)d_N(\d+)_nb(\d+)x(\d+)x(\d+)$", tag)
    )
    if not m:
        return None, None
    ndim = int(m.group(1))
    N = int(m.group(2))
    ne = N ** ndim
    dofs = nvar * ne * (n ** ndim)
    return dofs, ndim


def plot_performance(root, plot_dir):
    rows = parse_bench(root)
    os.makedirs(plot_dir, exist_ok=True)
    if not rows:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ndim, ax in zip([1, 2, 3], axes):
        pts_single, pts_multi = [], []
        for tag, sec in rows:
            if not tag.startswith(f"{ndim}d_"):
                continue
            dofs, _ = estimate_dofs(tag)
            if dofs is None or sec <= 0:
                continue
            rate = dofs / sec / 1e6
            if "_nb1" in tag or tag.endswith("_nb1"):
                pts_single.append((dofs, rate, tag))
            else:
                pts_multi.append((dofs, rate, tag))
        if pts_single:
            x, y, _ = zip(*sorted(pts_single))
            ax.loglog(x, y, "o-", label="single block", lw=2)
        if pts_multi:
            x, y, _ = zip(*sorted(pts_multi))
            ax.loglog(x, y, "s--", label="multi-block", lw=2)
        ax.set_xlabel("approx. DOFs")
        ax.set_ylabel("M DOF/s")
        ax.set_title(f"{ndim}D")
        ax.grid(True, which="both", alpha=0.3)
        if pts_single or pts_multi:
            ax.legend(fontsize=8)
    fig.suptitle("Strong scaling: zone throughput vs problem size (RK3, A100)")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "perf_scaling_by_dim.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for ndim in [1, 2, 3]:
        by_N = {}
        for tag, sec in rows:
            m = re.match(rf"{ndim}d_N(\d+)_nb(.+)$", tag)
            if not m:
                continue
            N, nb = int(m.group(1)), m.group(2)
            by_N.setdefault(N, []).append((nb, sec))
        for N, lst in sorted(by_N.items()):
            def nblocks(nb):
                if nb == "1":
                    return 1
                if "x" in nb:
                    toks = nb.split("x")
                    return int(toks[0]) ** len(toks)
                return int(nb)

            lst = sorted(lst, key=lambda t: nblocks(t[0]))
            xs = [nblocks(nb) for nb, _ in lst]
            ys = [1.0 / sec for _, sec in lst]
            ax.plot(xs, ys, "o-", label=f"{ndim}D N={N}")
    ax.set_xlabel("number of meshblocks")
    ax.set_ylabel("1 / runtime (s⁻¹)")
    ax.set_title("Meshblock launch overhead — throughput vs block count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "perf_meshblock_overhead.png"), dpi=150)
    plt.close(fig)

    bw_peak = 1555e9
    flop_peak = 156e12
    fig, ax = plt.subplots(figsize=(7, 5))
    intensities = np.logspace(-1, 2, 100)
    roof = np.minimum(flop_peak, bw_peak * intensities)
    ax.loglog(intensities, roof / 1e9, "k-", lw=2, label="A100 roof (FP32)")
    for tag, sec in rows:
        dofs, _ = estimate_dofs(tag)
        if dofs is None or sec <= 0:
            continue
        flops = dofs * 200
        bytes_moved = dofs * 8 * 4
        oi = flops / bytes_moved
        gflops = flops / sec / 1e9
        ax.loglog(oi, gflops, "o", ms=6)
    ax.set_xlabel("operational intensity (FLOP/byte, estimated)")
    ax.set_ylabel("attainable GFLOP/s")
    ax.set_title("Roofline (estimated): AMR/meshblock configs")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "roofline_estimated.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plot_amr_evidence(args.root, args.out)
    plot_visuals(args.root, args.out)
    plot_performance(args.root, args.out)
    print(f"Wrote plots to {args.out}")


if __name__ == "__main__":
    main()
