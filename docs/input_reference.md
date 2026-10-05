# Input file reference

spd_K uses an Athena++/AthenaK-style parameter file (`*.athinput`). Blocks are
introduced with `<block_name>`; parameters are `name = value` lines. Comments
start with `#`.

## Example

```ini
<job>
system   = hydro
fallback = true

<mesh>
p     = 3
nx1   = 16
nx2   = 16
nx3   = 1
x1_bc = periodic
x2_bc = periodic

<time>
tlim       = 0.2
cfl        = 0.8
integrator = rk3

<output>
dt = 0.1

<hydro>
gamma = 1.4

<problem>
problem = sine_wave
```

## Blocks

### `<job>`

| Parameter | Default | Description |
|---|---|---|
| `system` | `hydro` | `hydro`, `induction`, or `mhd` (see {doc}`mhd`) |
| `fallback` | `true` | Enable FV sub-grid update and trouble detection (MOOD cascade for `mhd`) |

### `<mesh>`

| Parameter | Default | Description |
|---|---|---|
| `p` | `3` | Polynomial order within each element |
| `nx1`, `nx2`, `nx3` | `8` | Global element counts (set to `1` to deactivate a direction) |
| `x1len`, `x2len`, `x3len` | `1.0` | Box lengths in each direction |
| `x1_bc`, `x2_bc`, `x3_bc` | `periodic` | `periodic`, `gradfree`, or `reflective` |

### `<time>`

| Parameter | Default | Description |
|---|---|---|
| `tlim` | `0.1` | End time |
| `cfl` | `0.8` | CFL number |
| `integrator` | `ader` | `ader`, `rk1`, `rk2`, or `rk3` (`mhd` supports the RK integrators only) |

### `<output>`

| Parameter | Default | Description |
|---|---|---|
| `dt` | (none) | Output interval; **must be present and positive** to enable file I/O |

### `<hydro>`

| Parameter | Default | Description |
|---|---|---|
| `gamma` | `1.4` | Ratio of specific heats |
| `nu` | `0.0` | Kinematic viscosity; `>0` switches on the viscous terms at runtime (`0` = inviscid Euler) |
| `floors` | `ramses` | Cons-to-prim floor semantics for `system=mhd`. `ramses`: primitive view only, the conserved state is never touched (matches the Python `spd` reference). `athenak`: additionally repairs the committed cell averages after each MOOD stage — density is floored at `dfloor` and, when the internal energy implied by the committed CT field falls below `pfloor`, the total energy is rebuilt from the floored pressure (sacrificing conservation in those cells) |
| `dfloor` | `1e-10` | Density floor |
| `pfloor` | (derived) | Pressure floor; defaults to the RAMSES `smallp = dfloor·min_c²/γ` (essentially zero). Set a physically sensible value (e.g. `1e-6`) when using `floors=athenak` |
| `beta` | `-2/3·nu` | Bulk-viscosity coefficient (Stokes hypothesis by default) |
| `g1`, `g2`, `g3` | `0.0` | Constant gravitational acceleration per direction (momentum/energy source term); `0` leaves the homogeneous Euler equations unchanged |
| `nscalars` | `0` | Passive scalars (`0`–`2`, `system=hydro` only). Scalar `n` is one more conserved row `rho*s_n` after the energy, advected with the mass flux on every path (SD flux points, interface solvers, MUSCL and first-order fallback, AMR transfer and flux correction); the primitive row written to `W_cv` is the concentration `s_n`. `0` is the same arithmetic as a build without scalars. Each scalar adds a fifth to the memory and to the cost of the per-variable kernels. Start it with `problem/scalar`; its total is written to `scalar.txt` |

### `<induction>`

| Parameter | Default | Description |
|---|---|---|
| `nu` | (problem) | Magnetic diffusivity |

### `<mhd>`

| Parameter | Default | Description |
|---|---|---|
| `rsolver` | `llf` | MHD Riemann solver: `llf` or `hlld` (Miyoshi–Kusano). Selects the SD face/edge path and the MOOD FV face fluxes. With `hlld`, the low-order corner E uses AthenaK UCT-HLLD composition from face coefficients; with `llf`, the four-state LLF bound. FV face fluxes take `Bn` from the CT face field |
| `mood_nad_b` | `mag` | Candidate CT field in MHD NAD: `comps` (`Bx,By,Bz`) or `mag` (`|B|`) |
| `mood_nad_v` | `off` | Velocity in MHD NAD: `off`, `mag` (`|v|`), or `comps` (`vx,vy,vz`) |
| `mood_nad_scale` | `relative` | NAD tolerance scale (AthenaK): `gcfl` (domain range × advective CFL), `grange` (domain range), `relative` (`rtol·|bound|`), or `delta` (`rtol·local range`) |
| `mood_force_level` | `-1` | Diagnostic: `-1` = normal MOOD detect/demote; `0`/`1`/`2` = force that cascade level everywhere and skip detection (`1` = MUSCL, `2` = first-order CT on the subcell mesh) |

### `<fallback>`

| Parameter | Default | Description |
|---|---|---|
| `tolerance` | `1e-5` | NAD band width (`rtol`) |
| `atol` | `0.0` | Absolute floor on the MHD NAD band |
| `eps0` | `1e-12` | Relative floor `eps0·|bound|` on grange/gcfl bands (AthenaK `mood_eps0`) |
| `NAD` | `relative` | Legacy local band for hydro / when `mood_nad_scale` is unset historically: `relative` or `delta` |
| `NAD_neighbors` | `2nd` | `2nd` (Moore neighborhood) or `1st` (face neighbors) |
| `SED` | `true` | Smooth extrema detection (for `p > 1`) |
| `blending` | `true` | Fractional θ blending of fallback fluxes |
| `max_revs` | `3` | Cap on MOOD detection/revision sweeps per stage (`system=mhd`). The loop exits as soon as no revisable troubled cell remains, so this is a safety cap; truncating it commits candidates that were never re-verified after the last demotion (they then lean on the ctoprim floors) |
| `min_rho` | `1e-10` | PAD density floor for MHD trouble detection |
| `NAD_scalars` | `true` | The passive-scalar rows (`hydro/nscalars`) enter NAD and SED next to density and pressure. On the advected-blob test the concentration leaves `[0,1]` by 1e-4/6e-3 with it and by 9e-2/1.3e-1 without. A flagged scalar demotes the whole cell, so an under-resolved smooth scalar also costs hydro accuracy (density L1 on the smooth sine test at 16² elements, p=3: 7.9e-6 → 1.3e-5) |
| `scalar_tolerance` | `1e-5` | NAD band of a scalar row. **Absolute**, unlike `tolerance`: a concentration has a unit scale and zero is a legitimate value, where a relative band has no width |
| `min_P` | `1e-10` | PAD gas-pressure floor for MHD trouble detection. Raising it toward the problem's pressure scale flags degenerating low-β cells before the primitive floors have to carry them |

### `<amr>` and `<refinementN>`

Block-based refinement on the `<meshblock>` forest. Requires `fallback/style=cascade`
and an RK integrator (`time/integrator=rk2|rk3`); ADER is refused on a mixed-level mesh.

| Key | Default | Meaning |
|---|---|---|
| `max_level` | 0 | refinement levels above the root (0 = uniform) |
| `adapt_interval` | 0 | regrid every N steps (0 = never; static patches only) |
| `criterion` | `lohner` | `lohner` (density second derivative), `pressure` (relative gradient), `shear` (velocity shear, Stone+2020 eq. 27), `trouble` (fraction of demoted cells > 0.01), `bfield` (MHD) |
| `refine_threshold` / `derefine_threshold` | per criterion | threshold pair with hysteresis, read by `pressure` (default 0.03 / 0.0075), `lohner` (0.5 / 0.0125) and `shear` (0.1 / 0.05); `trouble` is a fixed 0.01 fraction. Before 2026-09 only `shear` read them |
| `lohner_vars` | `density` | variables the `lohner` criterion scores: `density`, `pressure`, `scalar` (the first passive scalar, `hydro/nscalars >= 1`; scored by its second difference alone, not divided by the block mean) or a comma list of them, e.g. `density,pressure` (the block's score is the larger of the two, each normalized by its own block mean). `density` alone misses a contact-free shock of small density jump; `pressure` alone misses a contact; the block-interior `pressure` criterion misses both a contact and a jump that sits on a block face (the Sod tube at t=0) |
| `refine_frac` / `derefine_frac` | per criterion | ranking fractions (`lohner`) |
| `initial_refine` | false | iterate tag -> refine -> re-evaluate the IC before step 1 until no block is added (Athena++ `Mesh::Initialize`); use with threshold criteria |
| `prolong_dmp` | false | discrete maximum principle on prolongation (measured harmful, off) |

Static patches: `<refinement1>` .. `<refinementN>` blocks with `level` and `x1min/x1max/x2min/x2max/x3min/x3max`.

Physical boundaries on the block path: `periodic` and `gradfree` for every system;
for hydro also `reflective` (walls, the mirror rule of the single-block path, verified
to 1e-15 against a single block on the implosion) and `doublemach` (the Woodward &
Colella double Mach reflection: post-shock inflow on the left and on the bottom for
x < 1/6, a reflecting wall beyond, the exact moving shock on the top, outflow on the
right; `src/dmr.hpp`; needs a `<meshblock>` block). MHD walls under blocks are still
refused.

New problems: `woodward_colella` (1D interacting blasts, `inputs/woodward_colella.athinput`;
refine on `pressure`, the density is uniform at t=0) and `double_mach`
(`inputs/dmr.athinput`, domain [0,4] x [0,1]).

Runtime A/B switches (environment variables, all default off = the batched or reconstructed path):

| Variable | Restores |
|---|---|
| `SPD_NO_FV_GHOST_LIN=1` | injection of coarse cell values into a fine block's control-volume ghost cells (the limited-linear fill is the default; measured 2.7x / 1.8x lower AMR-vs-uniform rms on the fig-21 MUSCL lane at 1024^2 / 2048^2, no change at p=3) |
| `SPD_NO_FV_PRERESTRICT=1` | skip the fine-to-coarse restriction of every direction that runs BEFORE the per-direction exchange sweep when the limited-linear fill is on (the pre-41e7fef order, which left the coarse block's transverse ghost row one stage stale and leaked 6.4e-10 of mass per run on the 2D MUSCL blast with derefinement live; the fixed order conserves to 1e-15) |
| `SPD_NO_FV_SYMFILL=1` | the per-direction exchange sweep (same, finer, coarser, bc for x, then for y) instead of the symmetric order (all same, finer and bc passes of every direction, a snapshot, then every coarse->fine fill reading the snapshot, then bc and the same-level corner pass again, with the ghost-ring corners injected rather than sloped, since both directions write them). The sweep lets the x fill read rows the y passes have not refreshed and, at a corner spanning three levels, rows that are themselves the other direction's fill: x<->y asymmetry 1e-8 to 9e-5 on the 1024^2 Sedov blast with mirror symmetry at 1e-13; the symmetric order is at 3e-13 (1024^2 MUSCL, 16^2 blocks, four levels: 2.8e-6 before). The switch is bit-identical to the pre-change binary; both exchange paths implement the symmetric order and agree bitwise |
| `SPD_OLD_DEREFINE_APPLY=1` | apply derefinements one group at a time with the admissibility test run just before each (the pre-fix loop): the test reads neighbour tables that the previous group's merge has invalidated, so every group after the first was judged on stale indices. Measured on the Liska-Wendroff implosion (512^2, density Lohner): one group of a transpose pair refused, the mesh asymmetric for good (3e-4 by t=0.5, identical on CPU and GPU). Default: every group is judged on the forest after the refine pass, then all are applied |
| `SPD_LOHNER_INTERIOR=1` | the interior-only Lohner score (commit 6339d79 and before): skip every block-edge element, so a block needs 3 elements per side and a 2-element block scores 0 forever. Default: each block-edge element also takes one stencil across the face, at its face-adjacent sub-point, from the ghost point the SD field exchange writes (refreshed right before tagging, `amr/tag_exchange`); at a level jump the divided second difference with the true point spacing. Blocks then need 2 elements per side |
| `SPD_SYM_CHECK=1` | 2D debug: after every regrid, print the tagged groups (logical keys), the block counts after refine / derefine / balance, the refused count and the number of leaves whose x<->y transpose is not a leaf of the same level; catches a transient asymmetric mesh between two outputs |
| `SPD_STEP_MASS=1` | print the total mass after every step (before the regrid gate), to place a leak at a step rather than at an output |
| `SPD_FV_FLUX_CHECK=1` | 2D hydro: after the coarse-fine correction, sum the density-flux mismatch over every same-level face and every coarse-fine face and count block faces in no group; the same-level sum reproduces the per-step drift when a double-valued face is the leak |
| `SPD_FV_GHOST_TMODE=0|1|2` | transverse slopes of that fill: 2 (default) from the coarse block's neighbours including its own transverse ghost rows; 1 interior rows only; 0 none. 1 and 0 are diagnostics: they leak mass (8e-12 on the p=0 dynamic pulse) |
| `SPD_NO_PACK=1`, `SPD_OLD_XCHG=1` | per-block exchange paths |
| `SPD_NO_SCORE_BATCH=1` | host reference of the refinement scores |
| `SPD_NO_DEREFINE=1` | keeps every refinement (diagnostic) |
| `SPD_ADAPT_MASS=1` | brackets every regrid with the conserved mass |

### `<problem>`

Selects the initial condition and supplies runtime parameters; see
{doc}`initial_conditions` for the full list.

| Parameter | Default | Description |
|---|---|---|
| `scalar` | `zero` | Initial concentration of the passive scalars (`hydro/nscalars`) for a problem that does not define its own: `zero`; `uniform` (1 everywhere, which must stay 1 to round-off); `sine` (`0.5 + 0.25 sin(2π(x+y+z))`, scalar `n` shifted by a quarter period); `blob` (1 inside the sphere of `radius` about `cx, cy, cz`, 0 outside); `density` (`rho/d0`) |

## Command-line overrides

Any `block/parameter=value` pair can be appended after `-i file.athinput`:

```bash
./build/spd_K -i inputs/implosion.athinput mesh/nx1=64 time/tlim=0.5 output/dt=-1
```

## Output files

When outputs are enabled, the run directory (default `output/`, or
`$SPD_OUTPUT_DIR`) contains:

- `W_cv_N{N}p{p}_{n}_0.dat` — CV-averaged primitives (hydro: `rho, vx, vy, vz, p`, then one concentration per passive scalar; mhd: 8 vars)
- `mass.txt` — time and total mass at every output; `scalar.txt` — time and the total `∫ rho s_n dV` of each passive scalar (only with `hydro/nscalars > 0`)
- `B2_cv_N{N}p{p}_{n}_0.dat` — CV-averaged magnetic field (induction, mhd)
- `troubles_N*_{n}_0.dat` — FV trouble flags (when fallback is on)
- `cascade_N*_{n}_0.dat` — per-cell MOOD cascade level (mhd with fallback)
- `X_N{N}p{p}_0.dat`, `Y_*`, `Z_*` — face coordinates
- `parameters.txt` — echo of the effective input (when outputs are on)

Arrays are binary `float64`, C-order (LayoutRight on all backends).
