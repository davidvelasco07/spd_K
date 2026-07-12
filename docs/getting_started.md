# Getting started

## Requirements

- C++20 compiler (GCC 12+ recommended for CUDA builds)
- CMake 3.20+
- Kokkos 5.x (fetched automatically by CMake)
- Optional: CUDA 12.x, OpenMPI (for multi-rank runs)

On RHEL-style systems with Kokkos + CUDA:

```bash
scl enable gcc-toolset-12 bash
module load cudatoolkit/12.4 openmpi/gcc/4.1.0   # if using MPI
```

## Build

CPU-only:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

GPU (NVIDIA Ampere example):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON
cmake --build build -j
```

Unit tests (transform kernels):

```bash
./build/spd_K_test
```

## Run

```bash
./build/spd_K -i inputs/sine_wave.athinput
```

Override any input parameter from the command line (AthenaK-style):

```bash
./build/spd_K -i inputs/sine_wave.athinput \
  time/integrator=rk3 mesh/nx1=32 mesh/p=3 output/dt=0.1
```

Outputs are **opt-in**: files are written only when `<output> dt` is positive.
Disable I/O for benchmarks with `output/dt=-1`.

Redirect output directory:

```bash
SPD_OUTPUT_DIR=/tmp/my_run ./build/spd_K -i inputs/sod.athinput
```

## Systems

| `job/system` | Description |
|---|---|
| `hydro` | Compressible Euler / ideal gas (default) |
| `induction` | Kinematic induction (magnetic loop, etc.) |

Set dimensionality by mesh size: any direction with `nx*=1` is treated as
inactive (one point, no fluxes in that direction).

## Integrators

| `time/integrator` | Notes |
|---|---|
| `ader` | ADER time integration (default) |
| `rk1`, `rk2`, `rk3` | SSP-RK of the given order |

The evolution timer printed at the end of a run excludes initial-condition setup,
host/device transfers outside the loop, and file I/O time.
