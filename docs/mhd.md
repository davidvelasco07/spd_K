# MHD module

`job/system = mhd` selects the ideal-MHD solver: a coupled fluid +
constrained-transport (CT) spectral-difference scheme ported from the Python
[spd](https://github.com/davidvelasco07/spd) reference (`spd/MHD`). The
8-variable cell-centered state

```
rho, vx, vy, vz, P/E, Bx, By, Bz
```

is advanced with high-order SD fluxes and an MHD Riemann solver (default LLF on
the fast magnetosonic speed; optional Miyoshi–Kusano HLLD via `mhd/rsolver`),
while the divergence-free magnetic field lives on the cell faces and is evolved
by CT from edge electromotive forces. Edge points are classified by how many
element interfaces they lie on:

### Two knobs: the face solver and the electromotive force

`mhd/rsolver` picks the **face** Riemann solver — `llf` (default), `hll`, or
`hlld` — and `mhd/emf` picks how the **edge/corner** electromotive force is
built:

| `mhd/emf` | SD edges | MOOD demoted corners |
|---|---|---|
| `2sweep` (default) | two sequential 1-D edge Riemann sweeps (`v_index` 3 then 4) | four-state LLF bound (`mhd_four_state_E`) |
| `uct` | one UCT composition per edge family (`mhd_uct_edge_E`, MDZ21 eq. 33) | `mhd_uct_corner_E` |

Under `emf = uct` the face solve also emits the five MDZ coefficients
`(aL, dL, dR, vt1, vt2)` on the face flux-point lattice, and the edge EMF is
composed from the two adjacent face fans at once instead of upwinding one
direction after the other. The UCT *flavour* therefore follows `mhd/rsolver`:
`hll` gives **UCT-HLL** (MDZ21 eq. 28/32) and `hlld` gives **UCT-HLLD**
(eq. 44/45). It is refused under `llf`, which is a single-speed bound with no
fan to read the coefficients off.

`SPD_EMF_TRACE=1` prints which EMF path each call actually takes. Use it: this
tree shipped `mhd/rsolver = hlld` while **both** UCT kernels had zero call sites
and an earlier version of this section described them as live. What ran was
HLLD faces plus a second HLLD *edge* sweep — the direction-by-direction
upwinding that UCT exists to replace.

**Current limits of `emf = uct`**, both measured:

- **Single block only.** The face coefficients are not haloed, so a ghost
  element's outer face is never written and the edge composition reads it; the
  answer then depends on the block decomposition (2.3e-04 on Orszag-Tang at
  16² by t=0.03, against exactly 0 for `2sweep`). `main.cpp` refuses a
  meshblock/AMR run rather than emit a layout-dependent number.
Both the SD edges and the demoted corners conserve mass to round-off. Measured
on Orszag-Tang 16² to t=0.15 against a 1e-12 limit (two-sweep reference
1.708e-14):

| lane | UCT-HLLD | UCT-HLL |
|---|---|---|
| level 0 (SD) | 1.733e-14 | 1.720e-14 |
| level 1 (MUSCL, `job/scheme=plm`) | 1.708e-14 | 1.708e-14 |
| level 2 (first order) | 1.683e-14 | 1.683e-14 |
| cascade (live) | 1.720e-14 | 1.720e-14 |
| pure SD (`job/fallback=false`) | 8.036e-15 | 8.036e-15 |

Getting there needed two halos that the corner composition reads and nothing
filled: `mood_halo_face_B` (the face fields' TRANSVERSE ghosts — it existed with
zero callers) and a second `apply_E_boundaries` after the composition, because
UCT cannot reproduce a periodic ghost value on its own the way the symmetric
two-sweep solver can. `SPD_MASS_DBG=1` prints the mass across each stage commit
plus a periodicity residual for every array in the chain
`cascade → E1z/E2z → E0z → Bx_old/By_old → F0x/F0y`; the first nonzero entry is
the source. `SPD_UCT_CORNER_OFF=1` builds demoted corners with the four-state
bound instead, as an A/B.

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
2. `Advance` — SD fluxes + face HLLD/LLF (with CT face `Bn` and UCT coeffs when
   HLLD), edge EMF assembly + UCT or two-sweep edge upwinding, fluid update, CT
   update of the face field (or the MOOD cascade, see below).
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
| `ha_jet` | Ha et al. hypersonic jet (HYDRO, `gamma = 5/3`, no field). Ambient `rho = 0.5`, `p = 0.4127`, at rest; a nozzle on `\|y - cy\| < radius` of the x-min face injects `rho = 5` at `v_x = 800`. Needs `x1_bc = inflow`. The unmagnetized counterpart to `mhd_jet`, and the only jet here with published numbers to check against -- see `inputs/rr23/ha_jet.athinput` |
| `mhd_jet` | Mach-800 magnetized jet, Balsara (2012) with the field of Wu & Shu (2018). Quiescent ambient `rho = 0.1*gamma`, `p = 1`, `B = (0,sqrt(20000),0)`; the nozzle `|x-cx| < radius` on the y-min face injects `rho = gamma` at `v_y = 800`. Needs `x2_bc = inflow`, which is "prescribed state on the nozzle, outflow everywhere else on that face" -- see *Boundaries that carry a field* |

```bash
./build/spd_K -i inputs/orszag_tang.athinput
./build/spd_K -i inputs/field_loop.athinput
```

## Boundaries that carry a field

A boundary condition for MHD is two conditions, and getting the fluid one right
is not enough: with constrained transport the NORMAL face field on a boundary is
advanced by the CT curl like any other, so unless something says otherwise it is
driven by an EMF extrapolated out of the interior. Two rules, both measured on
the Mach-800 jet (`problem = mhd_jet`, `inputs/balsara/jet_wushu.athinput`).

**A reflecting wall is only valid where `B.n = 0` on that wall.** `_reflective_`
flips the normal magnetic row, which is the perfectly conducting condition and
gives `B.n = 0` by antisymmetry. Apply it to a face the field passes THROUGH and
it puts a jump of `2|B.n|` into that face's Riemann problem -- a div-B violation
by construction, and HLLD is not well posed on one. The jet's field is normal to
its base with `|B_y| = 141.42`; closing the base as a wall made the quiescent
ambient -- an exact stationary solution -- go exponentially unstable from
round-off, `max|v|` growing 10x per 1.2e-4 of time on the bottom row until dt
collapsed to 1e-6 of `dt0`. Gated by `mhd_jet_base_wall_sensitive_2d`.

**A prescribed inlet must pin its normal field.** A Dirichlet inlet prescribes
the whole state, `B` included, so the tangential EMF along that face is fixed by
the prescribed state (`mhd_pin_bc_emf_fv`, and `mhd_zero_wall_emf` on the SD
lattice). Left free, the jet's inlet field was destroyed: `B_y` on the base row
went from 141.42 to 1308, `|B|^2` peaked at 2.17e+06 ON the inlet row against
the paper's 1.9e+05 in the bow shock, and the resulting magnetic pressure of
1.1e+06 -- above the jet's own ram pressure of 9.0e+05 -- pushed material back
into the domain at a mean `v_y` of 435. That last number is what a plain
outflow base "sucking material in" actually was; the fix is the field condition,
not the fluid one. With the pin the same number is 0.98 and `|B|^2` peaks in the
bow shock. Gated by `mhd_jet_inlet_field_2d`.

Both are `SPD_NO_BC_EMF_PIN=1` away from their unpinned reference. Note the FV
half was a real gap rather than a refinement: the MOOD cascade assembles its EMF
on the FV node lattice, and `job/scheme=plm` or `vl2` sets `cfg.fv_only` and
never touches the SD edge arrays at all -- so that lane previously had NO
boundary EMF condition of any kind, on walls as well as inlets.

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

- **NAD** on `rho`, gas `P`, and the magnetic field — default
  `mhd/mood_nad_b = mag` (`|B|`). `comps` (`Bx,By,Bz`) matches Python `spd`
  and AthenaK and is the better detector; it becomes the default once the
  global NAD scale is decomposition-invariant and the goldens are regenerated.
  `|B|`-only (`mood_nad_b = mag`) is blind to Alfvénic / transverse
  oscillations. Optional `mhd/mood_nad_v = comps|mag` adds velocity.
  The band width uses `mhd/mood_nad_scale` (default **`relative`**, a purely
  local band; `grange`/`gcfl` take a domain-range reduction that is currently
  computed per block, so they are not decomposition-invariant yet). `gcfl` is the domain range of
  each detection variable, softened by the advective CFL — AthenaK's default).
  Alternatives: `grange` (no CFL factor), `relative`, `delta`. `fallback/atol`
  and `fallback/eps0` floor the band near zero crossings;
- **PAD** on density and gas pressure (total energy minus kinetic and magnetic
  energy of the candidate field), plus an `isfinite` check on the conserved
  candidate (NaN demotion). Runtime floors `fallback/min_rho` and
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
Fallback-off at t = 0.1 agrees to ~1e-6 relative L1; fallback-on both codes
now use per-component B NAD (Python `limiting_variables`, spd_K
`mhd/mood_nad_b=comps`), so cascade agreement past shock formation should
track more closely than the earlier `|B|`-only spd_K path.
