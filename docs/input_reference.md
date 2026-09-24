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
| `SPD_FV_GHOST_TMODE=0|1|2` | transverse slopes of that fill: 2 (default) from the coarse block's neighbours including its own transverse ghost rows; 1 interior rows only; 0 none. 1 and 0 are diagnostics: they leak mass (8e-12 on the p=0 dynamic pulse) |
| `SPD_NO_PACK=1`, `SPD_OLD_XCHG=1` | per-block exchange paths |
| `SPD_NO_SCORE_BATCH=1` | host reference of the refinement scores |
| `SPD_NO_DEREFINE=1` | keeps every refinement (diagnostic) |
| `SPD_ADAPT_MASS=1` | brackets every regrid with the conserved mass |

### `<problem>`

Selects the initial condition and supplies runtime parameters; see
{doc}`initial_conditions` for the full list.

## Command-line overrides

Any `block/parameter=value` pair can be appended after `-i file.athinput`:

```bash
./build/spd_K -i inputs/implosion.athinput mesh/nx1=64 time/tlim=0.5 output/dt=-1
```

## Output files

When outputs are enabled, the run directory (default `output/`, or
`$SPD_OUTPUT_DIR`) contains:

- `W_cv_N{N}p{p}_{n}_0.dat` — CV-averaged primitives (hydro: 6 vars, mhd: 8 vars)
- `B2_cv_N{N}p{p}_{n}_0.dat` — CV-averaged magnetic field (induction, mhd)
- `troubles_N*_{n}_0.dat` — FV trouble flags (when fallback is on)
- `cascade_N*_{n}_0.dat` — per-cell MOOD cascade level (mhd with fallback)
- `X_N{N}p{p}_0.dat`, `Y_*`, `Z_*` — face coordinates
- `parameters.txt` — echo of the effective input (when outputs are on)

Arrays are binary `float64`, C-order (LayoutRight on all backends).
