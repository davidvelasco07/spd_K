# MHD module

`job/system = mhd` selects the ideal-MHD solver: a coupled fluid +
constrained-transport (CT) spectral-difference scheme ported from the Python
[spd](https://github.com/davidvelasco07/spd) reference (`spd/MHD`). The
8-variable cell-centered state

```
rho, vx, vy, vz, P/E, Bx, By, Bz
```

is advanced with high-order SD fluxes and an MHD LLF Riemann solver (fast
magnetosonic speed), while the divergence-free magnetic field lives on the cell
faces and is evolved by CT from edge electromotive forces. The edge EMF couples
the face field (interpolated to edges) with the fluid state (interpolated from
cell-centered primitives to the same edge points) through an upwind
electric-field Riemann solver.

Time integration is SSP-RK (`rk1`–`rk3`); ADER MHD is not ported yet. The
module supports 3D and **true 2D** (`mesh/nx3 = 1`, x-y plane; matching the
Python reference). In 2D the CT machinery degenerates to the single `Ez` edge
family (edges reduce to the x-y corner points): the face fields evolve as
`dBx/dt = -dEz/dy`, `dBy/dt = +dEz/dx`, and `Bz` becomes a plain cell-centered
conserved variable advanced by the fluid fluxes (its row is *not* overwritten
by `B_to_U`, and the MOOD detection tests its fluid-updated candidate).
`div(B) = dBx/dx + dBy/dy` still holds at round-off. A 2D run is ~5-8x cheaper
than the old `nx3 = 2` z-invariant slab workaround (fewer points *and* a larger
time step, since the CFL no longer sums the z fast speed).

## Stage chain

Each SSP-RK stage runs, as tasks registered in the driver's stage list:

1. `CopyCons` — stage state + primitives at solution points.
2. `Advance` — SD fluxes + MHD Riemann solve, edge EMF assembly + edge Riemann
   solve, fluid update, CT update of the face field (or the MOOD cascade, see
   below).
3. `Combine` — SSP convex combination of the stage.
4. `BtoU` — projection of the face field onto the cell-centered B rows.

`div(B)` (evaluated with the same derivative operator that drives the CT
update) stays at round-off for all time; every output prints `max|divB|`.

## Initial conditions

The magnetic field is always initialized from a **vector potential** at cell
edges (`B = curl A` evaluated with the SD derivative), so the face field starts
divergence-free to machine precision. Two problems ship with the module:

| `problem/problem` | Setup |
|---|---|
| `orszag_tang` | Orszag-Tang vortex (Gaussian units: `rho = 25/36π`, `P = 5/12π`, `B0 = 1/√4π`), domain `[0,1]²` |
| `field_loop` | Gardiner & Stone weak field loop (`A0 = 1e-3`, `R = 0.3`) advected by `v = (2,1,0)` |

```bash
./build/spd_K -i inputs/orszag_tang.athinput
./build/spd_K -i inputs/field_loop.athinput
```

## MOOD cascade fallback

With `job/fallback = true` the MHD module uses a **MOOD cascade** (not the
fractional blending of the hydro module): every cell carries a cascade level

- level 0 — high-order SD flux / edge EMF,
- level 1 — MUSCL (minmod) FV flux / four-state corner EMF,
- level 2 — first-order (donor cell) FV flux / four-state corner EMF,

and up to `fallback/max_revs` detection/revision sweeps per stage (default 3;
the loop exits early once no revisable troubled cell remains)
demote still-troubled cells one level at a time. Letting the cascade converge
matters: with a truncated sweep budget, cells demoted in the last sweep — and
cells flagged only because a neighbor's demotion changed their update — commit
a candidate that was never re-verified, and rely on the ctoprim floors to
survive. At high resolution this shows up as spreading floored-pressure
regions in low-β shock interactions (and a collapsing time step, since the
floored cells drive the fast speed up). Faces take the maximum level
of their two adjacent cells and
edges the maximum of the (up to four) cells sharing them, so the assembled flux
and edge EMF stay **single-valued**: conservation and `div(B) = 0` hold exactly
at every cascade level.

Trouble detection runs on control-volume averages of the full candidate state
(the fluid candidate with its B rows replaced by the cell average of the
candidate CT update):

- **NAD** on `rho`, gas `P`, and `|B|` — the field enters through its
  *magnitude*, not its components, which is markedly more robust (flagging on
  components over-triggers on rotations of B that are perfectly fine
  physically);
- **PAD** on density and gas pressure (total energy minus kinetic and magnetic
  energy of the candidate field), with runtime floors `fallback/min_rho` and
  `fallback/min_P` (defaults `1e-10`). The raw internal energy is tested, so
  candidates the ctoprim floors would mask are still flagged; raising
  `min_P` toward the problem's pressure scale demotes degenerating low-β cells
  before the floors have to carry them.

The `<fallback>` knobs (`tolerance`, `NAD`, `NAD_neighbors`, …) apply
unchanged. The per-cell cascade level is written as `cascade_N*_{n}_0.dat`
when outputs are on.

### Cons-to-prim floors

Two floor semantics are available through `hydro/floors`:

- **`ramses`** (default): floors act on the primitive *view* only — the
  conserved state is never modified, matching the Python `spd` reference bit
  for bit. Robust, but a cell whose total energy implies negative internal
  energy carries that energy debt forever: its effective pressure stays at the
  floor, and in low-β shock interactions such regions can spread while the
  fast speed collapses the time step.
- **`athenak`**: after each MOOD stage commit, the committed *cell averages*
  are additionally repaired (FOFC-style): density is floored at `hydro/dfloor`
  and, when the internal energy implied by the committed CT field drops below
  `hydro/pfloor`, the total energy is rebuilt from the floored pressure +
  kinetic + magnetic energy. This amputates the energy debt once, at the cost
  of exact energy conservation in the repaired cells. Set a physically
  sensible `pfloor` for this mode (the derived RAMSES `smallp` default is
  essentially zero, which would repair cells to zero pressure). The repair is
  applied at cell-average granularity only — repairing pointwise at SD
  solution points injects non-smooth perturbations into the element polynomial
  and destabilizes the scheme.

The Orszag-Tang vortex is the canonical stress test: without the fallback the
p=3 run crashes when the shocks form (t ≈ 0.25); with the MOOD cascade it runs
stably with the flagged cells tracking the shock fronts.

## Validation

`tests/run_tests.py` includes (see {doc}`testing`):

- `mhd_orszag_tang_2d` — shocks + MOOD cascade: mass conservation to
  round-off, `max|divB| < 1e-11` over the run, golden file;
- `mhd_field_loop_2d` — smooth weak-field advection, pure high order: same
  checks.

`scripts/cross_validate_mhd.py` runs the same Orszag-Tang problem in Python
spd (`soe="mhd"`, rk3, LLF) and compares the primitive CV averages, mapping
between the code-unit conventions (`rho_K = rho_py/4π`, `B_K = B_py/√4π`).
Fallback-off at t = 0.1 agrees to ~1e-6 relative L1; fallback-on the |B|-based
NAD flags different cells than Python's per-component NAD, so agreement past
shock formation is qualitative (~2% relative L1 at t = 0.25).
