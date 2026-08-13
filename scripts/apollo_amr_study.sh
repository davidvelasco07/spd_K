#!/usr/bin/env bash
# spd_K AMR study on Apollo **front-node A100s** (no Slurm).
#
# Apollo login nodes expose two A100-PCIE-40GB GPUs. Run directly on the login
# node; do NOT use srun/sbatch for these interactive GPUs.
#
# Usage on Apollo:
#   cd ~/spd_K-amr && bash scripts/apollo_amr_study.sh
#
# From local (via SSH tunnel):
#   rsync ... apollo:~/spd_K-amr/ && ssh apollo 'cd ~/spd_K-amr && bash scripts/apollo_amr_study.sh'
#
# Environment (matches other Apollo runners in spd_K-mhd):
#   - gcc-toolset-12 for Kokkos 5.x (system gcc 8.5 is too old)
#   - CUDA_VISIBLE_DEVICES selects GPU 0 or 1
#   - LD_LIBRARY_PATH for NVHPC / CUDA MPI libs
set -euo pipefail

ROOT="${ROOT:-$HOME/spd_K-amr}"
OUT="${OUT:-$ROOT/apollo_amr_study}"
BUILD="${BUILD:-$ROOT/build-cuda}"
EXE="${EXE:-$BUILD/spd_K}"
PLOT="$ROOT/scripts/plot_amr_apollo.py"
PY="${PY:-python3}"

export PATH="/opt/rh/gcc-toolset-12/root/usr/bin:/usr/local/cuda/bin:${PATH:-/usr/bin:/bin}"
export LD_LIBRARY_PATH="/opt/rh/gcc-toolset-12/root/usr/lib64:/usr/local/openmpi/4.1.0/gcc/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/lib:/usr/local/openmpi/cuda-12.4/4.1.6/nvhpc245/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"/{visuals,bench,logs,plots}

echo "=== Host: $(hostname)  GPUs: $(nvidia-smi -L | wc -l) ==="
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv

build_if_needed() {
  if [[ -x "$EXE" && "${REBUILD:-0}" != "1" ]]; then
    echo "Using existing $EXE"
    return
  fi
  echo "=== Configure / build CUDA (Ampere80, gcc-toolset-12) ==="
  cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-12/root/usr/bin/g++ \
    -DCMAKE_C_COMPILER=/opt/rh/gcc-toolset-12/root/usr/bin/gcc \
    -DCMAKE_CXX_FLAGS="-I/usr/local/openmpi/4.1.0/gcc/include" \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/openmpi/4.1.0/gcc/lib64 -Wl,-rpath,/usr/local/openmpi/4.1.0/gcc/lib64" \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ARCH_AMPERE80=ON 2>&1 | tee "$OUT/logs/cmake.log"
  cmake --build "$BUILD" -j"$(nproc)" 2>&1 | tee "$OUT/logs/build.log"
}

run_case() {
  local tag="$1"; shift
  local gpu="${1:?}"; shift
  local odir="$OUT/visuals/$tag"
  rm -rf "$odir"; mkdir -p "$odir"
  echo "--- $tag (GPU $gpu) ---"
  CUDA_VISIBLE_DEVICES="$gpu" SPD_OUTPUT_DIR="$odir" "$EXE" "$@" \
    2>&1 | tee "$OUT/logs/${tag}.log"
  echo "$tag,gpu=$gpu" >> "$OUT/visuals/manifest.txt"
}

bench_one() {
  local tag="$1"; shift
  local gpu="${1:?}"; shift
  local odir="$OUT/bench/$tag"
  rm -rf "$odir"; mkdir -p "$odir"
  local t0 t1 elapsed
  t0=$(date +%s.%N)
  CUDA_VISIBLE_DEVICES="$gpu" SPD_OUTPUT_DIR="$odir" "$EXE" "$@" \
    > "$OUT/logs/bench_${tag}.log" 2>&1
  t1=$(date +%s.%N)
  elapsed=$("$PY" - <<PY
print(float("$t1")-float("$t0"))
PY
)
  echo "$tag,$elapsed,gpu$gpu" >> "$OUT/bench/times.csv"
}

build_if_needed

if [[ "${SKIP_VISUALS:-0}" != "1" ]]; then
echo "=== Visual simulations (GPU 0) ==="
: > "$OUT/visuals/manifest.txt"

run_case smr_sine_2d 0 \
  -i "$ROOT/inputs/sine_wave.athinput" \
  mesh/nx3=1 meshblock/nx1=4 meshblock/nx2=4 \
  time/integrator=rk3 time/tlim=0.1 output/dt=0.05

# Refined runs use cfl=0.4. The coarse-fine interface roughly halves the stable
# CFL: at the uniform-grid default of 0.8 a high-frequency mode grows on element
# boundaries inside the refined patch (2e-2 off a uniform run at the same
# resolution, vs 3e-7 at cfl=0.4), which swamps the figures.
run_case smr_patch_2d 0 \
  -i "$ROOT/inputs/sine_wave.athinput" \
  mesh/nx1=16 mesh/nx2=16 mesh/nx3=1 \
  meshblock/nx1=4 meshblock/nx2=4 \
  time/integrator=rk3 time/cfl=0.4 time/tlim=0.1 output/dt=0.05 \
  amr/max_level=1 \
  refinement1/level=1 refinement1/x1min=0.375 refinement1/x1max=0.625 \
  refinement1/x2min=0.375 refinement1/x2max=0.625

run_case amr_pulse_2d 0 \
  -i "$ROOT/inputs/amr_pulse.athinput" \
  time/cfl=0.4 time/tlim=0.1 output/dt=0.02

run_case mb_uniform_3d 1 \
  -i "$ROOT/inputs/sine_wave.athinput" \
  meshblock/nx1=4 meshblock/nx2=4 meshblock/nx3=4 \
  time/integrator=rk3 time/tlim=0.05 output/dt=0.05

fi

echo "=== Performance sweeps (RK3, fallback off) ==="
: > "$OUT/bench/times.csv"
echo "tag,seconds,gpu" >> "$OUT/bench/times.csv"

# Split 1D/2D benchmarks across the two front-node GPUs.
bench_1d() {
  local gpu="$1"
  for N in 64 128; do
    for NB in $N 32 16; do
      [[ $((N % NB)) -eq 0 ]] || continue
      nbx=$((N / NB))
      bench_one "1d_N${N}_nb${nbx}" "$gpu" \
        -i "$ROOT/inputs/sine_wave.athinput" \
        mesh/nx1=$N mesh/nx2=1 mesh/nx3=1 \
        meshblock/nx1=$NB meshblock/nx2=1 meshblock/nx3=1 \
        time/integrator=rk3 job/fallback=false \
        time/tlim=0.05 output/dt=0.05
    done
  done
}

bench_2d() {
  local gpu="$1"
  for N in 32 64 128; do
    bench_one "2d_N${N}_nb1" "$gpu" \
      -i "$ROOT/inputs/sine_wave.athinput" \
      mesh/nx1=$N mesh/nx2=$N mesh/nx3=1 \
      time/integrator=rk3 job/fallback=false \
      time/tlim=0.05 output/dt=0.05
    for NB in 8 4; do
      nbp=$((N / NB))
      bench_one "2d_N${N}_nb${nbp}x${nbp}" "$gpu" \
        -i "$ROOT/inputs/sine_wave.athinput" \
        mesh/nx1=$N mesh/nx2=$N mesh/nx3=1 \
        meshblock/nx1=$NB meshblock/nx2=$NB meshblock/nx3=1 \
        time/integrator=rk3 job/fallback=false \
        time/tlim=0.05 output/dt=0.05
    done
  done
}

bench_3d() {
  local gpu="$1"
  for N in 16 32; do
    bench_one "3d_N${N}_nb1" "$gpu" \
      -i "$ROOT/inputs/sine_wave.athinput" \
      mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N \
      time/integrator=rk3 job/fallback=false \
      time/tlim=0.05 output/dt=0.05
    for NB in 8 4; do
      nbp=$((N / NB))
      bench_one "3d_N${N}_nb${nbp}x${nbp}x${nbp}" "$gpu" \
        -i "$ROOT/inputs/sine_wave.athinput" \
        mesh/nx1=$N mesh/nx2=$N mesh/nx3=$N \
        meshblock/nx1=$NB meshblock/nx2=$NB meshblock/nx3=$NB \
        time/integrator=rk3 job/fallback=false \
        time/tlim=0.05 output/dt=0.05
    done
  done
}

bench_amr() {
  local gpu="$1"
  for N in 64 128; do
    bench_one "2d_amr_N${N}_pulse" "$gpu" \
      -i "$ROOT/inputs/amr_pulse.athinput" \
      mesh/nx1=$N mesh/nx2=$N mesh/nx3=1 \
      meshblock/nx1=4 meshblock/nx2=4 meshblock/nx3=1 \
      time/tlim=0.05 output/dt=0.05
    bench_one "2d_uniform_N${N}_mb16" "$gpu" \
      -i "$ROOT/inputs/sine_wave.athinput" \
      mesh/nx1=$N mesh/nx2=$N mesh/nx3=1 \
      meshblock/nx1=4 meshblock/nx2=4 meshblock/nx3=1 \
      time/integrator=rk3 job/fallback=false \
      time/tlim=0.05 output/dt=0.05
  done
}

# Split 1D/2D/3D benchmarks across the two front-node GPUs sequentially
# within each dimension to avoid CUDA OOM when both GPUs run large cases.
bench_1d 0
bench_2d 0
bench_3d 1
bench_amr 1

echo "=== Plot ==="
"$PY" "$PLOT" --root "$OUT" --out "$OUT/plots"
echo "Done. Results: $OUT/plots/"
