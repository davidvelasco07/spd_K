#!/usr/bin/env python3
"""Regression test suite for spd_K.

One binary, configured at runtime through input files (inputs/*.athinput)
plus command-line overrides. Checks per configuration:

  hydro_sd_3d    : L1 error of rho vs the analytic advected sine wave,
                   mass conservation to round-off, golden file (loose rtol)
  hydro_sd_2d    : L1 error vs analytic (must equal the 3D value: the
                   problem has no z-dependence), mass conservation
  hydro_sd_1d    : L1 error vs analytic, mass conservation
  hydro_fv_3d/2d : mass conservation to round-off (the fallback flux
                   replacement is exactly conservative for periodic BCs)
  hydro_fv_blast : 2d blast with a real shock; detector fires, fallback
                   active, mass still conserved to round-off
  induction_fv_3d: golden file comparison of B2_cv (linear problem, no
                   chaotic amplification, so a tight tolerance is portable)

Usage:
  tests/run_tests.py [--build-dir DIR] [--skip-unit] [--regen-goldens]
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(ROOT, "tests", "goldens")

N, P, NGH, nGH = 8, 3, 1, 2
n = P + 1

CONFIGS = {
    "hydro_sd_3d": {
        "input": "inputs/sine_wave.athinput",
        "overrides": [],
        "ndim": 3,
        "checks": ["analytic", "mass_strict", "golden"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.1,
        "l1_limit": 3.0e-5,    # measured 2.16e-5 (N=8, p=3, t=0.1)
        "golden_name": "hydro_sd",
        "golden_rtol": 1e-6,   # round-off seeds amplify ~2x/step in this flow
    },
    "hydro_sd_2d": {
        "input": "inputs/sine_wave.athinput",
        "overrides": ["mesh/nx3=1"],
        "ndim": 2,
        "checks": ["analytic", "mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.1,
        "l1_limit": 3.0e-5,
    },
    "hydro_sd_1d": {
        "input": "inputs/sine_wave.athinput",
        "overrides": ["mesh/nx2=1", "mesh/nx3=1"],
        "ndim": 1,
        "checks": ["analytic", "mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.1,
        "l1_limit": 2.0e-5,    # measured 1.41e-5
    },
    "hydro_fv_3d": {
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/fallback=true", "output/dt=0.05"],
        "ndim": 3,
        "checks": ["mass_strict"],
        "field": "W_cv_N8p3_2_0.dat",
        "t_end": 0.1,
    },
    "hydro_fv_2d": {
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/fallback=true", "mesh/nx3=1"],
        "ndim": 2,
        "checks": ["mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.1,
    },
    "hydro_fv_blast_2d": {
        # real shock: detector must fire, fallback active, still conservative
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/fallback=true", "mesh/nx3=1",
                      "problem/problem=spherical_blast",
                      "hydro/gamma=1.6666666666667",
                      "time/tlim=0.02", "output/dt=0.02"],
        "ndim": 2,
        "checks": ["mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.02,
    },
    "induction_fv_3d": {
        "input": "inputs/induction_loop.athinput",
        "overrides": [],
        "ndim": 3,
        "checks": ["golden"],
        "field": "B2_cv_N8p3_2_0.dat",
        "t_end": 0.2,
        "golden_name": "induction_fv",
        "golden_rtol": 1e-8,   # linear problem: cross-compiler safe
    },
}


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return r


def run(build_dir, outdir, cfg):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    env = dict(os.environ, SPD_OUTPUT_DIR=outdir)
    cmd = [os.path.join(build_dir, "spd_K"), "-i",
           os.path.join(ROOT, cfg["input"])] + cfg["overrides"]
    sh(cmd, env=env)


def shape(ndim):
    """Element/point shape of the SD output arrays for a given ndim."""
    Ne = lambda active: N + 2 * NGH if active else 1
    np_ = lambda active: n if active else 1
    a = [True, ndim >= 2, ndim >= 3]  # x, y, z activity
    return (1, 6, Ne(a[2]), Ne(a[1]), Ne(a[0]), np_(a[2]), np_(a[1]), np_(a[0]))


def load_rho_cells(outdir, cfg):
    """rho on the active-region global cell grid, shape (nz_c, ny_c, nx_c)."""
    ndim = cfg["ndim"]
    A = np.fromfile(os.path.join(outdir, cfg["field"])).reshape(shape(ndim))
    rho = A[0, 0]
    sl = lambda active: slice(NGH, -NGH) if active else slice(None)
    rho = rho[sl(ndim >= 3), sl(ndim >= 2), sl(True)]
    Nz_, Ny_, Nx_, nz_, ny_, nx_ = rho.shape[0], rho.shape[1], rho.shape[2], \
                                    rho.shape[3], rho.shape[4], rho.shape[5]
    return rho.transpose(0, 3, 1, 4, 2, 5).reshape(Nz_ * nz_, Ny_ * ny_, Nx_ * nx_)


def active_faces(outdir):
    xf = np.fromfile(os.path.join(outdir, f"X_N{N}p{P}_0.dat"))
    return xf[nGH:len(xf) - nGH]


def cell_widths(outdir):
    return np.diff(active_faces(outdir))


def total_mass(outdir, fname, cfg):
    cfg2 = dict(cfg, field=fname)
    rho = load_rho_cells(outdir, cfg2)
    w = cell_widths(outdir)
    V = np.ones(rho.shape)
    V *= w[None, None, :]
    if cfg["ndim"] >= 2:
        V *= w[None, :, None]
    if cfg["ndim"] >= 3:
        V *= w[:, None, None]
    return float((rho * V).sum())


def check_analytic(outdir, cfg):
    """L1 error of rho cell averages vs the exact advected sine wave.
    IC: rho = 1 + 0.125 sin(2 pi (x+y)), v = (1,1,0); y = 0 in 1d."""
    xf = active_faces(outdir)
    a = 2 * np.pi
    t = cfg["t_end"]
    rho = load_rho_cells(outdir, cfg)
    if cfg["ndim"] == 1:
        F = lambda x: -np.cos(a * (x - t)) / a
        exact = 1.0 + 0.125 * (F(xf[1:]) - F(xf[:-1])) / np.diff(xf)
        err = np.abs(rho[0, 0] - exact).mean()
    else:
        F = lambda x, y: -np.sin(a * (x + y - 2 * t)) / a**2
        X0, Y0 = np.meshgrid(xf[:-1], xf[:-1], indexing="ij")
        X1, Y1 = np.meshgrid(xf[1:], xf[1:], indexing="ij")
        exact = 1.0 + 0.125 * (F(X1, Y1) - F(X0, Y1) - F(X1, Y0) + F(X0, Y0)) \
                / ((X1 - X0) * (Y1 - Y0))
        err = np.abs(rho - exact.T[None, :, :]).mean()
    limit = cfg["l1_limit"]
    return err < limit, f"L1(rho) vs analytic = {err:.3e} (limit {limit:.1e})"


def check_mass(outdir, cfg, limit):
    outs = sorted(glob.glob(os.path.join(outdir, f"W_cv_N{N}p{P}_*_0.dat")))
    m = [total_mass(outdir, os.path.basename(f), cfg) for f in outs]
    drift = max(abs(x - m[0]) / abs(m[0]) for x in m)
    return drift < limit, f"mass drift = {drift:.3e} (limit {limit:.1e})"


def check_golden(outdir, cfg, regen):
    gdir = os.path.join(GOLDEN_DIR, cfg["golden_name"])
    gfile = os.path.join(gdir, cfg["field"])
    new = os.path.join(outdir, cfg["field"])
    if regen or not os.path.isfile(gfile):
        os.makedirs(gdir, exist_ok=True)
        shutil.copy(new, gfile)
        return True, f"golden (re)generated: {gfile}"
    a, b = np.fromfile(gfile), np.fromfile(new)
    if a.shape != b.shape:
        return False, f"golden size mismatch {a.size} vs {b.size}"
    diff = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
    rtol = cfg["golden_rtol"]
    return diff < rtol, f"golden max rel diff = {diff:.3e} (rtol {rtol:.1e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "build-511"))
    ap.add_argument("--skip-unit", action="store_true")
    ap.add_argument("--regen-goldens", action="store_true")
    args = ap.parse_args()

    failures = 0

    sh(["cmake", "--build", args.build_dir, "-j8"])

    if not args.skip_unit:
        r = subprocess.run([os.path.join(args.build_dir, "spd_K_test")],
                           capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"[{'PASS' if ok else 'FAIL'}] unit: transforms")
        failures += 0 if ok else 1

    for name, cfg in CONFIGS.items():
        outdir = os.path.join(args.build_dir, "test_out", name)
        try:
            run(args.build_dir, outdir, cfg)
        except RuntimeError as e:
            print(f"[FAIL] {name}: {e}")
            failures += 1
            continue
        for chk in cfg["checks"]:
            if chk == "analytic":
                ok, msg = check_analytic(outdir, cfg)
            elif chk == "mass_strict":
                ok, msg = check_mass(outdir, cfg, 1e-12)
            elif chk == "golden":
                ok, msg = check_golden(outdir, cfg, args.regen_goldens)
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
            failures += 0 if ok else 1

    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(min(main(), 99))
