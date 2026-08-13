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

### `<fallback>`

| Parameter | Default | Description |
|---|---|---|
| `tolerance` | `1e-5` | NAD band width |
| `NAD` | `relative` | `relative` or `delta` (band scaled by local range) |
| `NAD_neighbors` | `2nd` | `2nd` (Moore neighborhood) or `1st` (face neighbors) |
| `SED` | `true` | Smooth extrema detection (for `p > 1`) |
| `blending` | `true` | Fractional θ blending of fallback fluxes |
| `max_revs` | `3` | Cap on MOOD detection/revision sweeps per stage (`system=mhd`). The loop exits as soon as no revisable troubled cell remains, so this is a safety cap; truncating it commits candidates that were never re-verified after the last demotion (they then lean on the ctoprim floors) |
| `min_rho` | `1e-10` | PAD density floor for MHD trouble detection |
| `min_P` | `1e-10` | PAD gas-pressure floor for MHD trouble detection. Raising it toward the problem's pressure scale flags degenerating low-β cells before the primitive floors have to carry them |

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
