# spd_K

Kokkos C++ port of the spectral-difference hydrodynamics, induction, and
ideal-MHD solvers from [spd](https://github.com/davidvelasco07/spd). MHD uses
constrained transport (div B = 0 to round-off) with a MOOD cascade fallback
(|B|-based NAD). Physics modules register tasks into an athenak-style
tasklist/driver. Configure problems at runtime with input
files; run on CPU or NVIDIA GPU via Kokkos.

**Documentation:** [davidvelasco07.github.io/spd_K](https://davidvelasco07.github.io/spd_K/)

## Quick start

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/spd_K -i inputs/sine_wave.athinput
```

See the [getting started guide](https://davidvelasco07.github.io/spd_K/getting_started.html)
for GPU builds, command-line overrides, and output options.

## Repository layout

| Path | Purpose |
|---|---|
| `src/` | Kokkos kernels and driver |
| `inputs/` | Example `*.athinput` files |
| `tests/` | Unit tests, regression, visual suite, benchmarks |
| `docs/` | Sphinx documentation and visual gallery |

## License

See repository license file.
