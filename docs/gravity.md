# Gravity and source terms

spd_K supports a constant gravitational acceleration as a source term on the
hydro equations. It is off by default (`g = 0`, homogeneous Euler); setting any
of `g1`, `g2`, `g3` in the `<hydro>` block switches it on at runtime with no
rebuild.

## Equations

With a constant acceleration $\mathbf{g} = (g_1, g_2, g_3)$, the Euler equations
gain the standard gravitational source terms

$$
\partial_t(\rho \mathbf{v}) \mathrel{+}= \rho\,\mathbf{g},
\qquad
\partial_t E \mathrel{+}= \rho\,\mathbf{v}\cdot\mathbf{g}.
$$

The density equation is unchanged. The source is applied consistently in every
update path: the ADER space–time predictor and corrector (using the predicted
state at each temporal node) and the finite-volume fallback (using the sub-cell
average), so troubled cells at a stratified interface stay conservative and
balanced.

## Usage

```ini
<hydro>
gamma = 1.6666666666666667
g2    = -0.1     # downward gravity (acceleration along -y)
```

Any component can be set independently, and combined with viscosity (`nu`).
A nonzero field is echoed at startup:

```
gravity on: g = (0, -0.1, 0)
```

## Hydrostatic setups

To keep a stratified atmosphere at rest, the initial pressure must satisfy
hydrostatic balance, `∇p = ρ g`. spd_K is not well-balanced to machine
precision, so a discrete equilibrium develops small spurious velocities; for the
{doc}`Rayleigh–Taylor test <initial_conditions>` these stay orders of magnitude
below the instability growth and do not affect the result.

## Rayleigh–Taylor instability

The `rti` problem is the canonical use of gravity: a heavy layer resting below a
light one in hydrostatic balance, seeded with a single-mode perturbation that
grows into a rising mushroom plume. See {doc}`initial_conditions` for the full
setup and `inputs/rti.athinput`.
