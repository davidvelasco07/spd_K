#!/usr/bin/env python3
"""Compare spd_K Orszag-Tang W_cv against an AthenaK mhd_w_bcc binary dump.

Matched setup (see inputs/compare/):
  spd_K   : N elements, p, true 2D, domain [0,1]^2, Gaussian OT units
  AthenaK : nx = N*(p+1), domain [-0.5,0.5]^2, same Gaussian OT units

Coordinate / IC mapping
  x_K = x_A + 0.5
  AthenaK OT pgen uses the opposite signs for (vx, vy, Bx) relative to spd_K
  (and to the common Toth convention). The comparison flips those three
  AthenaK fields before differencing.

Usage:
  compare_ot_athenak.py <spdk_W_cv.dat> <athenak_mhd_w_bcc.bin> <N> <p>
                        [--athenak-vis DIR] [--gamma 1.6666666666667]
                        [--plot OUT.png]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _patch_bin_convert(vis_dir: str):
    sys.path.insert(0, vis_dir)
    import bin_convert as bc  # noqa: WPS433

    def _safe_get(header, block, key):
        in_block = False
        for line in header:
            s = line.strip()
            if s.startswith("<") and s.endswith(">"):
                in_block = s[1:-1] == block
                continue
            if in_block and "=" in s:
                k, v = s.split("=", 1)
                if k.strip() == key:
                    return v.strip().split()[0]
        raise KeyError(key)

    bc.get_from_header = _safe_get
    return bc


def read_spdk_wcv(path: str, N: int, p: int, ngh: int = 1) -> np.ndarray:
    """Return primitives on the fine FV mesh, shape (8, Ny, Nx), ghosts stripped.

    True-2D layout: (nvar, Nz=1, Ny, Nx, nz=1, ny, nx) with Ny=Nx=N+2*ngh
    and ny=nx=p+1. Subcells are laid out into a uniform (N*(p+1))^2 grid.
    """
    n = p + 1
    raw = np.fromfile(path)
    Ng = N + 2 * ngh
    expect = 8 * 1 * Ng * Ng * 1 * n * n
    if raw.size != expect:
        raise ValueError(f"{path}: size {raw.size} != expected {expect} for N={N} p={p}")
    W = raw.reshape(8, 1, Ng, Ng, 1, n, n)
    W = W[:, 0, ngh : ngh + N, ngh : ngh + N, 0]  # (8, N, N, n, n)
    # element (j,i), subcell (jj,ii) -> fine (j*n+jj, i*n+ii)
    fine = np.zeros((8, N * n, N * n), dtype=W.dtype)
    for j in range(N):
        for i in range(N):
            fine[:, j * n : (j + 1) * n, i * n : (i + 1) * n] = W[:, j, i]
    return fine


def read_athenak_w(path: str, vis_dir: str, gamma: float) -> tuple[float, np.ndarray]:
    """Return (time, primitives) with shape (8, Ny, Nx): rho,vx,vy,vz,P,Bx,By,Bz."""
    bc = _patch_bin_convert(vis_dir)
    fd = bc.read_binary(path)
    Nx1, Nx2 = fd["Nx1"], fd["Nx2"]
    x1min, x1max = fd["x1min"], fd["x1max"]
    x2min, x2max = fd["x2min"], fd["x2max"]
    dx1 = (x1max - x1min) / Nx1
    dx2 = (x2max - x2min) / Nx2
    nx, ny = fd["nx1_out_mb"], fd["nx2_out_mb"]
    geo = np.asarray(fd["mb_geometry"])
    md = fd["mb_data"]

    def stitch(key: str) -> np.ndarray:
        src = np.asarray(md[key])
        out = np.zeros((Nx2, Nx1), dtype=np.float64)
        for m in range(fd["n_mbs"]):
            i0 = int(round((geo[m, 0] - x1min) / dx1))
            j0 = int(round((geo[m, 2] - x2min) / dx2))
            out[j0 : j0 + ny, i0 : i0 + nx] = src[m, 0, :, :]
        return out

    dens = stitch("dens")
    velx = stitch("velx")
    vely = stitch("vely")
    velz = stitch("velz")
    eint = stitch("eint")  # internal energy density: P = (gamma-1)*eint
    bcc1 = stitch("bcc1")
    bcc2 = stitch("bcc2")
    bcc3 = stitch("bcc3")
    prs = (gamma - 1.0) * eint

    # Flip AthenaK (vx, vy, Bx) to the spd_K / Toth sign convention, then
    # nothing else — domain shift is handled by the caller via axis alignment
    # (both arrays are indexed j=y, i=x over a periodic unit box).
    W = np.stack([dens, -velx, -vely, velz, prs, -bcc1, bcc2, bcc3], axis=0)
    return float(fd["time"]), W


def rel_l1(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    d = np.abs(a - b)
    ref = np.abs(b).max()
    if ref < 1e-14:
        return float(d.max()), float(d.mean()), float("nan")
    return float(d.max()), float(d.mean()), float(d.mean() / ref)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spdk_wcv")
    ap.add_argument("athenak_bin")
    ap.add_argument("N", type=int)
    ap.add_argument("p", type=int)
    ap.add_argument("--athenak-vis", default=os.path.expanduser("~/athenak/fallback/vis/python"))
    ap.add_argument("--gamma", type=float, default=5.0 / 3.0)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    Wk = read_spdk_wcv(args.spdk_wcv, args.N, args.p)
    t, Wa = read_athenak_w(args.athenak_bin, args.athenak_vis, args.gamma)
    nfine = args.N * (args.p + 1)
    if Wa.shape[-2:] != (nfine, nfine):
        raise SystemExit(
            f"resolution mismatch: AthenaK {Wa.shape[-2:]} vs spd_K fine {nfine}x{nfine}"
        )

    names = ["rho", "vx", "vy", "vz", "P", "Bx", "By", "Bz"]
    print(f"AthenaK t = {t:.6f}")
    print(f"grid      = {nfine} x {nfine}  (spd_K N={args.N} p={args.p})")
    print(f"{'var':4s} {'max|K-A|':>12s} {'L1|K-A|':>12s} {'rel L1':>12s}")
    worst = 0.0
    for v, name in enumerate(names):
        mx, l1, rel = rel_l1(Wk[v], Wa[v])
        if np.isnan(rel):
            print(f"{name:4s} {mx:12.3e} {l1:12.3e} {'(zero)':>12s}")
        else:
            worst = max(worst, rel)
            print(f"{name:4s} {mx:12.3e} {l1:12.3e} {rel:12.3e}")
    print(f"\nworst relative L1 = {worst:.3e}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        panels = [
            (axes[0, 0], Wk[0], "spd_K rho"),
            (axes[0, 1], Wa[0], "AthenaK rho (mapped)"),
            (axes[0, 2], Wk[0] - Wa[0], "delta rho"),
            (axes[1, 0], np.sqrt(Wk[5] ** 2 + Wk[6] ** 2), "spd_K |B|"),
            (axes[1, 1], np.sqrt(Wa[5] ** 2 + Wa[6] ** 2), "AthenaK |B|"),
            (axes[1, 2], np.sqrt(Wk[5] ** 2 + Wk[6] ** 2) - np.sqrt(Wa[5] ** 2 + Wa[6] ** 2), "delta |B|"),
        ]
        for ax, data, title in panels:
            im = ax.imshow(data, origin="lower", extent=[0, 1, 0, 1])
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"OT compare t≈{t:.3f}  N_fine={nfine}")
        fig.savefig(args.plot, dpi=140)
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
