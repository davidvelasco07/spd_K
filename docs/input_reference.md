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
| `system` | `hydro` | `hydro` or `induction` |
| `fallback` | `true` | Enable FV sub-grid update and trouble detection |

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
| `integrator` | `ader` | `ader`, `rk1`, `rk2`, or `rk3` |

### `<output>`

| Parameter | Default | Description |
|---|---|---|
| `dt` | (none) | Output interval; **must be present and positive** to enable file I/O |

### `<hydro>`

| Parameter | Default | Description |
|---|---|---|
| `gamma` | `1.4` | Ratio of specific heats |
| `nu` | `0.0` | Kinematic viscosity; `>0` switches on the viscous terms at runtime (`0` = inviscid Euler) |
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

- `W_cv_N{N}p{p}_{n}_0.dat` — CV-averaged primitives (hydro)
- `B2_cv_N{N}p{p}_{n}_0.dat` — CV-averaged magnetic field (induction)
- `troubles_N*_{n}_0.dat` — FV trouble flags (when fallback is on)
- `X_N{N}p{p}_0.dat`, `Y_*`, `Z_*` — face coordinates
- `parameters.txt` — echo of the effective input (when outputs are on)

Arrays are binary `float64`, C-order (LayoutRight on all backends).
