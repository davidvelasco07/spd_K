"""Readers for spd_K binary outputs, shared by the regression suite
(run_tests.py) and the visual suite (visual_suite.py).

Output formats (see src/output.cpp):
  SD solutions (W_cv, ...):  float64, C-order, shape
      (n_ader=1, nvar, Nz_t, Ny_t, Nx_t, nz, ny, nx)
    with N*_t = N + 2*NGH elements (1 if the direction is inactive) and
    n* = p+1 points per element (1 if inactive).
  FV solutions (troubles, ...): float64, C-order, shape
      (nvar, fz, fy, fx)  with f* = N*(p+1) + 2*nGH (1 if inactive).
  Coordinates (X_*, Y_*, Z_*): fv_faces of each direction, i.e. the
    N*(p+1) + 2*nGH + 1 face positions of the FV sub-grid (Gauss points).

Variable order: rho, vx, vy, vz, p/e, bookkeeping (FV trouble aggregate).
"""
import glob
import os
import re

import numpy as np

NGH = 1   # ghost elements around the SD grid
nGH = 2   # ghost cells around the FV grid

VARS = {"rho": 0, "vx": 1, "vy": 2, "vz": 3, "p": 4}


class Grid:
    """Grid geometry of one run, inferred from the coordinate files."""

    def __init__(self, outdir):
        self.outdir = outdir
        self.faces = {}       # active FV faces per direction
        self.N = {}           # elements per direction
        self.p = None
        self.nvar = 6
        for d, name in enumerate("XYZ"):
            fs = glob.glob(os.path.join(outdir, f"{name}_N*_0.dat"))
            if not fs:
                raise FileNotFoundError(f"no {name} coordinate file in {outdir}")
            m = re.match(rf"{name}_N(\d+)p(\d+)_0.dat", os.path.basename(fs[0]))
            N, p = int(m.group(1)), int(m.group(2))
            xf = np.fromfile(fs[0])
            if N > 1:
                self.N[d] = N
                self.p = p
                self.faces[d] = xf[nGH:len(xf) - nGH]
            else:
                self.N[d] = 1
        self.ndim = len(self.faces)
        self.n = self.p + 1
        #infer nvar from the first SD output file on disk
        for pat in ("W_cv_*_0.dat", "B2_cv_*_0.dat"):
            fs = glob.glob(os.path.join(outdir, pat))
            if fs:
                nbytes = os.path.getsize(fs[0])
                npts = int(np.prod(self.sd_shape(6)[2:]))
                if nbytes % (npts * 8) == 0:
                    self.nvar = nbytes // (npts * 8)
                break

    def active(self, d):
        return self.N[d] > 1

    def centers(self, d):
        xf = self.faces[d]
        return 0.5 * (xf[1:] + xf[:-1])

    def widths(self, d):
        return np.diff(self.faces[d])

    def sd_shape(self, nvar=None):
        if nvar is None:
            nvar = self.nvar
        Ne = lambda d: self.N[d] + 2 * NGH if self.active(d) else 1
        np_ = lambda d: self.n if self.active(d) else 1
        return (1, nvar, Ne(2), Ne(1), Ne(0), np_(2), np_(1), np_(0))

    def fv_shape(self, nvar=None):
        if nvar is None:
            nvar = self.nvar
        f = lambda d: self.N[d] * self.n + 2 * nGH if self.active(d) else 1
        return (nvar, f(2), f(1), f(0))


def output_indices(outdir, prefix="W_cv"):
    """Sorted output indices i of <prefix>_..._{i}_0.dat files."""
    idx = []
    for f in glob.glob(os.path.join(outdir, f"{prefix}_*_0.dat")):
        m = re.match(rf"{prefix}_N\d+(?:p\d+)?_(\d+)_0.dat", os.path.basename(f))
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def _sd_file(grid, prefix, i):
    return os.path.join(grid.outdir, f"{prefix}_N{grid.N[0]}p{grid.p}_{i}_0.dat")


def load_sd(grid, i, var=0, prefix="W_cv"):
    """One variable of an SD output on the global active FV sub-grid,
    shape (nz_c, ny_c, nx_c) with n*_c = N*(p+1) (1 if inactive)."""
    A = np.fromfile(_sd_file(grid, prefix, i)).reshape(grid.sd_shape())
    v = A[0, var]
    sl = lambda d: slice(NGH, -NGH) if grid.active(d) else slice(None)
    v = v[sl(2), sl(1), sl(0)]
    Nz, Ny, Nx, nz, ny, nx = v.shape
    return v.transpose(0, 3, 1, 4, 2, 5).reshape(Nz * nz, Ny * ny, Nx * nx)


def load_sd_field(grid, i, var, prefix=None):
    """Load one SD variable; prefix defaults to W_cv or B2_cv if present."""
    if prefix is None:
        prefix = "B2_cv" if glob.glob(os.path.join(grid.outdir, "B2_cv_*_0.dat")) else "W_cv"
    return load_sd(grid, i, var, prefix)


def load_troubles(grid, i):
    """Trouble flags (aggregate slot) on the active FV grid."""
    fs = glob.glob(os.path.join(grid.outdir, f"troubles_N*_{i}_0.dat"))
    if not fs:
        return None
    nbytes = os.path.getsize(fs[0])
    nvar = 6
    fshape = grid.fv_shape(nvar)
    if nbytes != 8 * np.prod(fshape):
        #troubles carry nvar physical slots + aggregate
        nvar = nbytes // (8 * int(np.prod(fshape[1:])))
        fshape = grid.fv_shape(nvar)
    A = np.fromfile(fs[0]).reshape(fshape)
    sl = lambda d: slice(nGH, -nGH) if grid.active(d) else slice(0, 1)
    tr = A[-1, sl(2), sl(1), sl(0)]
    #drop singleton axes of inactive directions
    return np.squeeze(tr)


def total_mass(grid, i, prefix="W_cv"):
    """Volume integral of rho over the active region."""
    rho = load_sd(grid, i, VARS["rho"], prefix)
    V = np.ones(rho.shape)
    w = lambda d: grid.widths(d)
    if grid.active(0):
        V = V * w(0)[None, None, :]
    if grid.active(1):
        V = V * w(1)[None, :, None]
    if grid.active(2):
        V = V * w(2)[:, None, None]
    return float((rho * V).sum())
