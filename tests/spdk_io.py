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


def leaf_dump_indices(outdir):
    """Sorted output indices of the leaf-wise dumps (output/format = leaves | both)."""
    idx = []
    for f in glob.glob(os.path.join(outdir, "leaves_cv_N*p*_*_0.*")):
        m = re.match(r"leaves_cv_N\d+p\d+_(\d+)_0\.(dat|f32)$", os.path.basename(f))
        if m:
            idx.append(int(m.group(1)))
    return sorted(set(idx))


def load_leaves(outdir, i):
    """Leaf-wise dump i: the active control volumes of every leaf of the forest.

    Returns (head, blocks, A, faces):
      head    dict of the amr_blocks_<i>.txt header: nblocks, max_level, ndim, NB (x,y,z), Nf (x,y,z)
      blocks  float array, one row per leaf: ib level lx ly lz x0 x1 y0 y1 z0 z1
      A       cells of every leaf, shape (nblocks, nvar, NBz*nz, NBy*ny, NBx*nx) (1 along an inactive direction)
      faces   the n+1 sub-cell faces of the unit element (a leaf's cells are not equally wide)
    A leaf's cell edges along x are x0 + (x1-x0)/NBx * (e + faces[q]) for element e and point q; see leaf_edges.
    """
    fs = [f for f in glob.glob(os.path.join(outdir, f"leaves_cv_N*p*_{i}_0.*"))
          if re.search(rf"_{i}_0\.(dat|f32)$", f)]
    if not fs:
        raise FileNotFoundError(f"no leaf dump {i} in {outdir}")
    f = fs[0]
    n = int(re.search(r"p(\d+)_", os.path.basename(f)).group(1)) + 1
    rows = [l.split() for l in open(os.path.join(outdir, f"amr_blocks_{i}.txt"))
            if l.strip() and not l.startswith("#")]
    h = [int(v) for v in rows[0]]
    head = {"nblocks": h[0], "max_level": h[1], "ndim": h[2], "NB": tuple(h[3:6]), "Nf": tuple(h[6:9])}
    blocks = np.array(rows[1:], dtype=float)
    act = [True, head["ndim"] >= 2, head["ndim"] >= 3]                      # x, y, z
    NB = [head["NB"][d] if act[d] else 1 for d in range(3)]
    nn = [n if act[d] else 1 for d in range(3)]
    raw = np.fromfile(f, dtype=np.float32 if f.endswith(".f32") else np.float64)
    per = head["nblocks"] * NB[0] * NB[1] * NB[2] * nn[0] * nn[1] * nn[2]
    if raw.size % per:
        raise ValueError(f"{os.path.basename(f)}: {raw.size} values is not a multiple of {per}")
    A = raw.reshape(head["nblocks"], raw.size // per, NB[2], NB[1], NB[0], nn[2], nn[1], nn[0])
    A = A.transpose(0, 1, 2, 5, 3, 6, 4, 7).reshape(head["nblocks"], -1, NB[2] * nn[2], NB[1] * nn[1], NB[0] * nn[0])
    ff = glob.glob(os.path.join(outdir, f"leaf_faces_p{n-1}_0.dat"))
    faces = np.fromfile(ff[0]) if ff else np.linspace(0.0, 1.0, n + 1)
    return head, blocks, A, faces


def leaf_edges(head, block, faces, d):
    """Cell edges of one leaf along direction d (0 x, 1 y, 2 z); [lo, hi] for an inactive direction."""
    lo, hi = block[5 + 2 * d], block[6 + 2 * d]
    if d >= head["ndim"]:
        return np.array([lo, hi])
    NB = head["NB"][d]
    h = (hi - lo) / NB
    e = (np.arange(NB)[:, None] + faces[None, :-1]).ravel()
    return lo + h * np.append(e, NB)

