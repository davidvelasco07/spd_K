#!/usr/bin/env python3
"""Launch an spd_K MHD simulation and produce a panel movie from it.

Runs the solver, renders each output as soon as it is written, deletes the
raw .dat files to keep disk usage bounded, and encodes the frames into an mp4
when the run ends.

Panel layouts (`--panels`):
  rho,prs,cascade   — 1x3 (default): density, gas pressure, MOOD revision level
  rho,prs,b2,cascade — 2x2: density, pressure, |B|^2, cascade

Any `block/key=value` pair accepted by spd_K can be appended to modify the
run, e.g.:

  scripts/spd_movie.py --movie plots/ot_revs3.mp4 \\
      fallback/max_revs=3 fallback/min_P=1e-4 mesh/nx1=64 mesh/nx2=64

  # encode an existing frame directory only (no simulation):
  scripts/spd_movie.py --encode-only /tmp/spd_ot_frames --movie plots/ot.mp4
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NGH, nGH = 1, 2          # SD element ghosts / FV cell ghosts


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("overrides", nargs="*",
                    help="spd_K parameter overrides (block/key=value)")
    ap.add_argument("--input", default=os.path.join(ROOT, "inputs/orszag_tang.athinput"))
    ap.add_argument("--exe", default=os.path.join(ROOT, "build/spd_K"))
    ap.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES (default 1)")
    ap.add_argument("--tlim", type=float, default=1.0)
    ap.add_argument("--dt-out", type=float, default=0.005,
                    help="output/frame cadence (default 0.005 -> 201 frames)")
    ap.add_argument("--workdir", default=None,
                    help="scratch dir for raw outputs/frames (default /tmp/spd_movie_<pid>)")
    ap.add_argument("--movie", default=os.path.join(ROOT, "plots/spd_movie.mp4"))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--title", default="Orszag-Tang",
                    help="figure title prefix (resolution/time appended)")
    ap.add_argument("--panels", default="rho,prs,cascade",
                    choices=("rho,prs,cascade", "rho,prs,b2,cascade"),
                    help="panel layout (default: density / pressure / cascade)")
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep the rendered .png frames")
    ap.add_argument("--keep-data", action="store_true",
                    help="keep the raw .dat outputs (needs the disk space!)")
    ap.add_argument("--encode-only", metavar="FRAMEDIR", default=None,
                    help="skip the simulation; encode frame_*.png from this dir")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# Output discovery and loading
# ----------------------------------------------------------------------------

def find_layout(outdir, timeout=3600):
    """Wait for the first W_cv output and parse (N, p) from its filename.
    Returns (N, p, nz_tot) with nz_tot = 0 for a true-2D run (mesh/nx3=1)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        fs = glob.glob(os.path.join(outdir, "W_cv_N*p*_0_0.dat"))
        if fs:
            m = re.match(r"W_cv_N(\d+)p(\d+)_", os.path.basename(fs[0]))
            N, p = int(m.group(1)), int(m.group(2))
            n = p + 1
            sz = os.path.getsize(fs[0])
            if sz == 8 * 8 * (N + 2 * NGH) ** 2 * n ** 2:      # true 2D
                return N, p, 0
            nz = sz // (8 * 8 * (N + 2 * NGH) ** 2 * n ** 3)
            if nz > 0 and sz == 8 * 8 * nz * (N + 2 * NGH) ** 2 * n ** 3:
                return N, p, nz
        time.sleep(5)
    raise TimeoutError(f"no first output appeared in {outdir}")


def sd_to_cells(v):
    """(Nz,Ny,Nx,nz,ny,nx) active region -> (z,y,x) global cells."""
    Nz, Ny, Nx, nz, ny, nx = v.shape
    return v.transpose(0, 3, 1, 4, 2, 5).reshape(Nz * nz, Ny * ny, Nx * nx)


def load_frame(outdir, i, N, p, nz_tot):
    """nz_tot = 0 means a true-2D run (single degenerate z slot)."""
    n = p + 1
    zel, zpt = (nz_tot, n) if nz_tot else (1, 1)
    W = np.fromfile(f"{outdir}/W_cv_N{N}p{p}_{i}_0.dat").reshape(
        1, 8, zel, N + 2 * NGH, N + 2 * NGH, zpt, n, n)[0]
    B2 = np.fromfile(f"{outdir}/B2_cv_N{N}p{p}_{i}_0.dat").reshape(
        1, 4, zel, N + 2 * NGH, N + 2 * NGH, zpt, n, n)[0]
    cz = zel * zpt // 2
    rho = sd_to_cells(W[0, :, NGH:-NGH, NGH:-NGH])[cz]
    prs = sd_to_cells(W[4, :, NGH:-NGH, NGH:-NGH])[cz]
    b2 = sd_to_cells(B2[3, :, NGH:-NGH, NGH:-NGH])[cz]
    casc = None
    fs = glob.glob(f"{outdir}/cascade_N*_{i}_0.dat")
    if fs:
        nc = N * n
        A = np.fromfile(fs[0])
        fz = A.size // ((nc + 2 * nGH) ** 2)
        A = A.reshape(fz, nc + 2 * nGH, nc + 2 * nGH)
        casc = A[fz // 2, nGH:-nGH, nGH:-nGH]
    return rho, prs, b2, casc


def frame_complete(outdir, i, N, p, nz_tot):
    """Output i is complete once B2_cv (written last by the MHD module) is full."""
    f = f"{outdir}/B2_cv_N{N}p{p}_{i}_0.dat"
    n = p + 1
    zpts = nz_tot * n if nz_tot else 1
    want = 8 * 4 * zpts * (N + 2 * NGH) ** 2 * n ** 2
    return os.path.isfile(f) and os.path.getsize(f) == want


# ----------------------------------------------------------------------------
# Rendering / encoding
# ----------------------------------------------------------------------------

def render(framedir, i, t, title, rho, prs, b2, casc, panels="rho,prs,cascade"):
    catalog = {
        "rho": (rho, "density", "viridis", None),
        "prs": (prs, "gas pressure", "inferno", None),
        "b2": (b2, r"$|B|^2$", "magma", None),
        "cascade": (casc, "MOOD revision level", "Reds", (0, 2)),
    }
    keys = [k.strip() for k in panels.split(",")]
    n = len(keys)
    if n == 3:
        fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.6), constrained_layout=True)
        axes = list(ax)
    else:
        fig, ax = plt.subplots(2, 2, figsize=(11.2, 10.2), constrained_layout=True)
        axes = list(ax.flat)
    for a, key in zip(axes, keys):
        f, name, cmap, clim = catalog[key]
        if f is None:
            a.text(0.5, 0.5, "fallback off", ha="center", va="center")
            a.set_title(name)
            a.axis("off")
            continue
        im = a.imshow(f, origin="lower", extent=[0, 1, 0, 1], cmap=cmap,
                      vmin=None if clim is None else clim[0],
                      vmax=None if clim is None else clim[1],
                      interpolation="nearest")
        a.set_title(name)
        fig.colorbar(im, ax=a, shrink=0.9)
    fig.suptitle(f"{title}   t = {t:.3f}", fontsize=15)
    fig.savefig(os.path.join(framedir, f"frame_{i:04d}.png"), dpi=100)
    plt.close(fig)


def encode(framedir, movie, fps):
    import imageio_ffmpeg
    frames = sorted(glob.glob(os.path.join(framedir, "frame_*.png")))
    assert frames, f"no frames in {framedir}"
    os.makedirs(os.path.dirname(os.path.abspath(movie)), exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps),
         "-pattern_type", "glob", "-i", os.path.join(framedir, "frame_*.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", movie],
        check=True)
    print(f"movie: {movie} ({len(frames)} frames, {fps} fps, "
          f"{os.path.getsize(movie)/1e6:.1f} MB)")


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.encode_only:
        encode(args.encode_only, args.movie, args.fps)
        return 0

    workdir = args.workdir or f"/tmp/spd_movie_{os.getpid()}"
    outdir, framedir = os.path.join(workdir, "out"), os.path.join(workdir, "frames")
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir)
    os.makedirs(framedir, exist_ok=True)

    n_frames = int(round(args.tlim / args.dt_out)) + 1
    cmd = [args.exe, "-i", args.input,
           f"time/tlim={args.tlim}", f"output/dt={args.dt_out}"] + args.overrides
    env = dict(os.environ, SPD_OUTPUT_DIR=outdir, CUDA_VISIBLE_DEVICES=args.gpu)
    logf = open(os.path.join(workdir, "run.log"), "w")
    print(f"workdir: {workdir}\nlaunch: {' '.join(cmd)}", flush=True)
    sim = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)

    N, p, nz_tot = find_layout(outdir)
    title = f"{args.title}  N={N} p={p} ({N*(p+1)}$^2$ cells)"
    print(f"layout: N={N} p={p} nz_tot={nz_tot}; {n_frames} frames expected",
          flush=True)

    done, last_new = 0, time.time()
    while done < n_frames:
        if frame_complete(outdir, done, N, p, nz_tot):
            time.sleep(1)          # let the last write settle (NFS/page cache)
            try:
                rho, prs, b2, casc = load_frame(outdir, done, N, p, nz_tot)
                render(framedir, done, done * args.dt_out, title, rho, prs, b2, casc,
                       panels=args.panels)
            except Exception as e:
                print(f"frame {done}: render failed: {e}", flush=True)
            if not args.keep_data:
                for f in glob.glob(f"{outdir}/*_N*_{done}_0.dat"):
                    os.remove(f)
            print(f"frame {done} done", flush=True)
            done += 1
            last_new = time.time()
            continue
        if sim.poll() is not None and time.time() - last_new > 60:
            print(f"simulation exited (rc={sim.returncode}) with {done} frames; "
                  "encoding what we have", flush=True)
            break
        time.sleep(5)

    if sim.poll() is None:
        sim.wait()
    encode(framedir, args.movie, args.fps)
    if not args.keep_frames:
        shutil.rmtree(framedir, ignore_errors=True)
    if not args.keep_data:
        shutil.rmtree(outdir, ignore_errors=True)
    return 0 if done == n_frames and sim.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
