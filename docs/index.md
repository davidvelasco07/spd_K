# spd_K

**spd_K** is a Kokkos-based C++ port of the spectral-difference (SD) hydrodynamics,
induction, and MHD solvers from [spd](https://github.com/davidvelasco07/spd).
Problems are configured at runtime through Athena++-style input files; the same
binary supports 1D/2D/3D hydro and induction with optional finite-volume fallback.

![Kelvin-Helmholtz instability](gallery/kelvin_helmholtz_2d.png)

*Kelvin-Helmholtz roll-up computed with spd_K (density, left) and the finite-volume
trouble map (right). See the {doc}`gallery` for the full test suite.*

## Contents

```{toctree}
:maxdepth: 2
:titlesonly:

getting_started
input_reference
initial_conditions
gravity
fallback
user_ic
gallery
testing
```

## Quick start

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
cmake --build build -j
./build/spd_K -i inputs/sine_wave.athinput
```

Enable file output by setting a positive `output/dt` in the input file (or on the
command line). Performance runs can disable I/O with `output/dt=-1`.
