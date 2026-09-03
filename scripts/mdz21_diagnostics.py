"""Diagnostics for the Mignone & Del Zanna (2021) Newtonian benchmarks.

arXiv:2004.10542, "Systematic construction of upwind constrained transport
schemes for MHD". Each function here computes one quantity the paper reports,
so a run can be compared against a published number rather than against a
picture.

The paper's units are not spd_K's. Orszag-Tang is the one that matters: MDZ21
uses the 2*pi box with rho = 25/9, p = 5/3 and B = (-sin y, sin 2x), while
spd_K uses the unit box with rho = 25/(36 pi), p = 5/(12 pi), B0 = 1/sqrt(4 pi)
(mhd_ic_orszag_tang in src/mhd.cpp). The two are the same problem rescaled, so
    x_paper = 2 pi x_spdk,      t_paper = 2 pi t_spdk.
The paper's reference time t = 2 pi is therefore t = 1 in spd_K, and its
measurement window 4 pi/10 < x,y < 6 pi/10 is 0.2 < x,y < 0.3 here. Every
function below takes spd_K coordinates and says so.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import spdk_io as io  # noqa: E402

# MHD primitive layout (src/mhd.cpp): rho, vx, vy, vz, P, Bx, By, Bz
MRHO, MVX, MVY, MVZ, MPRS, MBX, MBY, MBZ = range(8)


def cell_volumes(grid):
    """Volume of each active FV sub-cell, broadcast to the field shape."""
    V = np.ones(field_shape(grid))
    if grid.active(0):
        V = V * grid.widths(0)[None, None, :]
    if grid.active(1):
        V = V * grid.widths(1)[None, :, None]
    if grid.active(2):
        V = V * grid.widths(2)[:, None, None]
    return V


def field_shape(grid):
    n = grid.n
    f = lambda d: grid.N[d] * n if grid.active(d) else 1
    return (f(2), f(1), f(0))


def load(grid, i, var):
    return io.load_sd(grid, i, var, prefix="W_cv")


# B2_cv holds (Bx, By, Bz, |B|^2) cell averages of the STAGGERED CT field
# (compute_B2_cv, src/induction.cpp), which is the divergence-free field itself
# rather than W_cv's cell-centred B rows. That is what these diagnostics want.
# It carries 4 variables where W_cv carries 8, so it cannot go through
# io.load_sd, which reshapes with the grid-wide nvar inferred from W_cv.
B2_BX, B2_BY, B2_BZ, B2_MAG2 = 0, 1, 2, 3


def load_b2(grid, i, var):
    f = os.path.join(grid.outdir, f"B2_cv_N{grid.N[0]}p{grid.p}_{i}_0.dat")
    A = np.fromfile(f)
    shp = list(grid.sd_shape())
    npts = int(np.prod(shp[2:]))
    nvar = A.size // npts
    if nvar * npts != A.size:
        raise ValueError(f"B2_cv size {A.size} is not a multiple of {npts}")
    shp[1] = nvar
    v = A.reshape(shp)[0, var]
    sl = lambda d: slice(io.NGH, -io.NGH) if grid.active(d) else slice(None)
    v = v[sl(2), sl(1), sl(0)]
    Nz, Ny, Nx, nz, ny, nx = v.shape
    return v.transpose(0, 3, 1, 4, 2, 5).reshape(Nz * nz, Ny * ny, Nx * nx)


def magnetic_energy(outdir):
    """Volume-integrated B^2/2 at each output, and the times.

    MDZ21 figures 4 and 7 plot this normalised to its initial value: the decay
    rate is the paper's measure of a scheme's numerical dissipation. spd_K
    writes B2_cv (already B^2) alongside W_cv, which is what this uses when it
    is present -- it is formed from the STAGGERED field (compute_B2_cv), so it
    is the CT field's energy and not a cell-centred reconstruction of it.
    """
    grid = io.Grid(outdir)
    V = cell_volumes(grid)
    have_b2 = bool(glob.glob(os.path.join(outdir, "B2_cv_*_0.dat")))
    out = []
    for i in io.output_indices(outdir):
        if have_b2:
            b2 = load_b2(grid, i, B2_MAG2)
        else:
            b2 = sum(load(grid, i, v) ** 2 for v in (MBX, MBY, MBZ))
        out.append(0.5 * float((b2 * V).sum()))
    return np.array(out)


def axial_field_error(outdir, B0):
    """max |B'_z| / B0 per output, for the 3D field loop (MDZ21 figure 7).

    The component along the loop's cylindrical axis is analytically zero, so
    this measures the scheme's truncation error directly.
    """
    grid = io.Grid(outdir)
    return np.array([np.abs(load_b2(grid, i, B2_BZ)).max() / B0
                     for i in io.output_indices(outdir)])


def pressure_peak_ratio(outdir, i=-1, lo=0.4, hi=0.6):
    """mu_p = p_max / <p>, the Orszag-Tang island diagnostic (MDZ21 table 1).

    p_max is taken over the central window and <p> over the whole domain.

    THE WINDOW IS AN INTERPRETATION, and it matters. The paper writes the region
    as "4 pi/10 < x, y < 6 pi/10" and calls it the centre of the domain -- but on
    its own [0, 2 pi] box that interval is 1.257 .. 1.885, i.e. 0.2 .. 0.3 of the
    box, which does NOT contain the centre at pi = 3.142. Since the quantity
    exists to measure a magnetic island that forms AT the centre (the O-point of
    figures 10 and 11), the intended region has to be 0.4 L .. 0.6 L, which does
    straddle pi. Taken literally the window measures quiet off-centre gas and
    mu_p collapses to ~1.1 for every scheme -- measured, which is what exposed
    the misreading.

    So the defaults are 0.4 .. 0.6 of the unit box. Pass lo/hi to check the
    literal reading.

    The paper's reading of the number: mu_p < ~2 means no central island formed;
    above ~2 the island's extent grows roughly with mu_p. Its RK2+linear values
    at 128^2 run 1.21 (UCT-HLL) to 2.54 (UCT-HLLD).
    """
    grid = io.Grid(outdir)
    idx = io.output_indices(outdir)
    p = load(grid, idx[i], MPRS)
    V = cell_volumes(grid)
    mean_p = float((p * V).sum() / V.sum())

    x, y = grid.centers(0), grid.centers(1)
    mx = (x > lo) & (x < hi)
    my = (y > lo) & (y < hi)
    if not mx.any() or not my.any():
        raise ValueError(f"empty central window {lo}..{hi} on this grid")
    window = p[:, :, mx][:, my, :]
    return float(window.max()) / mean_p


def poloidal_energy(outdir):
    """Normalised volume-integrated POLOIDAL magnetic energy per output.

    Rueda-Ramirez et al. 2022 eq. 100 (from Mignone et al.):
        <B_p^2>(t) = int B_p^2(t) / int B_p^2(0),   B_p^2 = B1^2 + B2^2
    Only the in-plane components. Their KH starts with a TOROIDAL B3 = ca sin
    theta as well, so using |B|^2 here would bury the poloidal growth under a
    large constant and the curve would barely move -- which is the whole
    quantity they use to rank schemes' dissipation during the transition to
    turbulence.
    """
    grid = io.Grid(outdir)
    V = cell_volumes(grid)
    out = []
    for i in io.output_indices(outdir):
        bp2 = (load_b2(grid, i, B2_BX) ** 2 + load_b2(grid, i, B2_BY) ** 2)
        out.append(float((bp2 * V).sum()))
    out = np.array(out)
    return out / out[0]


def poloidal_ratio(grid, i, floor=1e-12):
    """B_p / B_t on the grid: the field RR22 figure 6 plots.

    B_t is taken from W_cv, NOT from B2_cv. In true 2D (mesh/nx3 = 1) there is
    no z-staggered face field, so compute_B2_cv accumulates its bz only under
    `if(az)` and B2_cv's Bz row is IDENTICALLY ZERO -- reading it here divided by
    the 1e-30 floor and produced a B_p/B_t colour bar running to 1e29. In true 2D
    the toroidal field is a plain cell-centred conserved variable and lives in
    W_cv row _mbz_; the poloidal components still come from the staggered field,
    which is the divergence-free one.
    """
    bx = load_b2(grid, i, B2_BX)
    by = load_b2(grid, i, B2_BY)
    if grid.active(2):
        bz = load_b2(grid, i, B2_BZ)
    else:
        bz = io.load_sd(grid, i, MBZ, prefix="W_cv")
    # This is a DISPLAY quantity (RR22 figure 6), so it is returned on regular
    # control volumes -- the SD sub-cells are not equally spaced and a plot that
    # plots them evenly distorts the geometry (see to_regular). Rebin the
    # COMPONENTS and divide afterwards, rather than rebinning the ratio: an
    # average of a quotient is not the quotient of the averages, and the
    # components are the quantities the rebin is conservative for.
    bp = to_regular(grid, np.sqrt(bx ** 2 + by ** 2))
    bt = to_regular(grid, np.abs(bz))
    return bp / np.maximum(bt, floor)


def regular_faces(grid, d):
    """N*n equally spaced faces along direction d, spd's `regular_faces`."""
    f = grid.faces[d]
    return np.linspace(f[0], f[-1], len(f))


def _rebin_matrix(src_faces, dst_faces):
    """A[j,i] = fraction of target cell j covered by source cell i.

    Exact overlap of two 1D partitions of the same interval, so A @ v is the
    average of a piecewise-constant field over the target cells. Rows sum to 1
    and the integral is preserved exactly.
    """
    lo = np.maximum(src_faces[None, :-1], dst_faces[:-1, None])
    hi = np.minimum(src_faces[None, 1:], dst_faces[1:, None])
    ov = np.clip(hi - lo, 0.0, None)
    return ov / np.diff(dst_faces)[:, None]


def to_regular(grid, field):
    """Average a CV field onto a REGULAR mesh of the same cell count.

    WHY THIS EXISTS. The FV sub-cells of an SD element are the control volumes
    of the solution points, and those are NOT equally spaced: measured on this
    code, the widths within one element run [0.0070, 0.0242, 0.0242, 0.0070] at
    p=3 (a 3.4x ratio) and span 8.0x at p=7. Handing such a field straight to
    imshow -- which gives every cell the same number of pixels -- stretches the
    narrow edge cells by that factor and squeezes the wide interior ones, which
    is exactly the blocky, mottled look the SDFB panels had. The distortion is
    worst at high p, so it disfigured the SDFB8 panels most.

    This is the CV analogue of spd's interpolate_to_regular_mesh
    (spd/spectral_difference/sd_scheme.py): spd Lagrange-interpolates the
    solution POINTS to the midpoints of regular sub-cells and then draws with
    pcolormesh on regular_faces. These arrays are cell AVERAGES rather than
    point values, so the faithful operation is a conservative rebin -- an exact
    area-weighted average onto regular control volumes -- which needs no
    assumption beyond "this is a cell average" and preserves the integral.
    """
    out = np.asarray(field, dtype=float)
    # field is (nz, ny, nx); axis 0 <-> direction 2, axis 2 <-> direction 0
    for axis, d in ((0, 2), (1, 1), (2, 0)):
        if not grid.active(d) or out.shape[axis] < 2:
            continue
        A = _rebin_matrix(grid.faces[d], regular_faces(grid, d))
        out = np.moveaxis(np.tensordot(A, np.moveaxis(out, axis, 0), axes=(1, 0)),
                          0, axis)
    return out


def _mid_index(grid, d, at=0.5):
    """Index of the cell centre nearest `at` along direction d."""
    return int(np.argmin(np.abs(grid.centers(d) - at)))


def blast_diagonal_magnetic_energy(outdir, i=-1):
    """(x_d, B^2/2) along the main diagonal of the z = centre plane.

    MDZ21 figure 12, top right: the magnetic energy density along the main
    diagonal at z = 0, where "x_d is the distance from a point on the diagonal
    to the coordinate origin", i.e. x_d = sqrt(2) x on their [-1/2,1/2]^3 box.
    A dip forms near x_d ~ 0.2 and DEEPENS as a scheme's dissipation falls, so
    the depth of that dip is the quantity that ranks the emf schemes.

    spd_K's box starts at the origin, so their z = 0 is z = 0.5 here and the
    diagonal is the line x = y through the centre.

    x_d is returned SIGNED, running over both halves of the diagonal. The
    configuration is symmetric under (x,y) -> (-x,-y) about the centre -- B is
    along (1,1,0), which is the diagonal itself -- so the two halves are the
    same curve, and plotting both is a free symmetry control on the run.
    """
    grid = io.Grid(outdir)
    idx = io.output_indices(outdir)
    if not idx:
        raise ValueError(f"no dumps in {outdir}")
    b2 = load_b2(grid, idx[i], B2_MAG2)          # staggered |B|^2, the CT field
    k = _mid_index(grid, 2) if grid.active(2) else 0
    plane = b2[k]                                 # (ny, nx)
    ny, nx = plane.shape
    if nx != ny:
        raise ValueError(f"diagonal needs a square xy grid, got {nx}x{ny}")
    x = grid.centers(0)
    diag = np.array([plane[j, j] for j in range(nx)])
    xd = np.sqrt(2.0) * (x - 0.5)
    return xd, 0.5 * diag


def blast_axis_density(outdir, i=-1):
    """(z, rho) along the vertical axis through the centre.

    MDZ21 figure 12, bottom right. Their z-axis is x = y = 0, which is
    x = y = 0.5 here. The paper reports only minor differences between schemes
    on this cut -- it is the control for the diagonal, not the discriminator.
    """
    grid = io.Grid(outdir)
    idx = io.output_indices(outdir)
    if not idx:
        raise ValueError(f"no dumps in {outdir}")
    rho = load(grid, idx[i], MRHO)
    j = _mid_index(grid, 1)
    ii = _mid_index(grid, 0)
    return grid.centers(2), rho[:, j, ii]


def blast_dip_depth(outdir, i=-1, lo=0.05, hi=0.45):
    """Minimum of B^2/2 on the diagonal over lo < |x_d| < hi, and where it sits.

    The scalar form of MDZ21's ranking: a LOWER minimum means a deeper sag and
    hence less numerical dissipation. The window excludes the origin (where the
    explosion's own centre sits) and the outer shock.

    Returns (min value, |x_d| at the minimum, value at x_d = 0 for scale).
    """
    xd, e = blast_diagonal_magnetic_energy(outdir, i)
    m = (np.abs(xd) > lo) & (np.abs(xd) < hi)
    if not m.any():
        raise ValueError(f"empty diagonal window {lo}..{hi}")
    k = int(np.argmin(np.where(m, e, np.inf)))
    c = int(np.argmin(np.abs(xd)))
    return float(e[k]), float(abs(xd[k])), float(e[c])


def kh_growth_amplitude(outdir):
    """delta v_y = (max(v_y) - min(v_y)) / 2 per output (MDZ21 figure 13).

    The KH perturbation amplitude the paper fits its growth rate to.
    """
    grid = io.Grid(outdir)
    out = []
    for i in io.output_indices(outdir):
        vy = load(grid, i, MVY)
        out.append(0.5 * float(vy.max() - vy.min()))
    return np.array(out)


def fit_growth_rate(t, amp, t0, t1):
    """Im(omega) from a least-squares fit of log(amp) over [t0, t1].

    MDZ21 quotes Im(omega a / c_s) ~ 0.395e-2 from linear theory and measures
    0.31-0.40 for Im(omega) depending on scheme and resolution (their shear
    width is a = 0.01, so the two differ by that factor).
    """
    t, amp = np.asarray(t, float), np.asarray(amp, float)
    m = (t >= t0) & (t <= t1) & (amp > 0)
    if m.sum() < 3:
        raise ValueError(f"only {m.sum()} usable points in [{t0}, {t1}]")
    slope, _ = np.polyfit(t[m], np.log(amp[m]), 1)
    return float(slope)


def param(outdir, key, default=None):
    """One scalar from a run's parameters.txt, which the binary writes verbatim.

    The file is sectioned (<time>, <output>, ...) and keys are NOT unique across
    sections -- `dt` appears under <output> and elsewhere -- so this is only
    safe for keys that are, such as `tlim`. Returns `default` if absent.
    """
    pf = os.path.join(outdir, "parameters.txt")
    if not os.path.isfile(pf):
        return default
    for line in open(pf):
        parts = [t.strip() for t in line.split("=", 1)]
        if len(parts) == 2 and parts[0] == key:
            try:
                return float(parts[1])
            except ValueError:
                return parts[1]
    return default


def final_time(outdir):
    """Physical time of the last output, or None if it cannot be established.

    Use this rather than output_times()[-1] for LABELLING. output_times falls
    back to index numbering when parameters.txt records no explicit times, so a
    run to t = 0.002 with three outputs reports its final time as "2" -- which
    is how a 32^3 test figure came out captioned t=2. Returning None lets the
    caller say "final output" instead of printing a wrong number.
    """
    t = param(outdir, "tlim")
    return t if isinstance(t, float) and t > 0 else None


def output_times(outdir):
    """Output times, read from parameters.txt if it records them.

    Falls back to index numbering, which is only meaningful when the run used a
    fixed output/dt -- say so at the call site rather than assuming it.
    """
    pf = os.path.join(outdir, "parameters.txt")
    if os.path.isfile(pf):
        for line in open(pf):
            if line.lower().startswith("output_times"):
                return np.array([float(v) for v in line.split("=")[1].split()])
    return np.array(io.output_indices(outdir), dtype=float)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: mdz21_diagnostics.py <output-dir> [<output-dir> ...]")
        sys.exit(2)
    for d in sys.argv[1:]:
        try:
            eb = magnetic_energy(d)
            line = (f"{os.path.basename(d.rstrip('/')):28s} "
                    f"E_B/E_B(0) = {eb[-1] / eb[0]:.6f} over {len(eb)} outputs")
            try:
                line += f"   mu_p = {pressure_peak_ratio(d):.3f}"
            except Exception:
                pass
            print(line)
        except Exception as e:
            print(f"{os.path.basename(d.rstrip('/')):28s} FAILED: {e}")
