# Running spd_K on Apollo (IAS)

Apollo login nodes (`apollo-login*.sns.ias.edu`) expose **two NVIDIA A100-PCIE-40GB GPUs** for interactive use. Run directly on the login node — **do not use Slurm** for these front-node GPUs.

## Quick start

```bash
ssh apollo
cd ~/spd_K-amr
export PATH=/opt/rh/gcc-toolset-12/root/usr/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:/usr/local/openmpi/4.1.0/gcc/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/lib:/usr/local/openmpi/cuda-12.4/4.1.6/nvhpc245/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0   # or 1
export SPD_OUTPUT_DIR=/tmp/spd_out
./build-cuda/spd_K -i inputs/sine_wave.athinput mesh/nx3=1 time/tlim=0.05 output/dt=0.05
```

## SSH

```bash
ssh apollo   # ProxyCommand via sns (see ~/.ssh/config)
```

## Environment (required)

System GCC 8.5 is too old for Kokkos 5.x. Match the existing `~/spd_K/build` setup:

```bash
export PATH=/opt/rh/gcc-toolset-12/root/usr/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-12/root/usr/lib64:/usr/local/openmpi/4.1.0/gcc/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/lib:/usr/local/openmpi/cuda-12.4/4.1.6/nvhpc245/lib64:$LD_LIBRARY_PATH
```

## Build (CUDA, A100)

```bash
cd ~/spd_K-amr
rm -rf build-cuda   # if reconfiguring after a failed cmake
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-12/root/usr/bin/g++ \
  -DCMAKE_C_COMPILER=/opt/rh/gcc-toolset-12/root/usr/bin/gcc \
  -DCMAKE_CXX_FLAGS="-I/usr/local/openmpi/4.1.0/gcc/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/openmpi/4.1.0/gcc/lib64 -Wl,-rpath,/usr/local/openmpi/4.1.0/gcc/lib64" \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_AMPERE80=ON
cmake --build build-cuda -j$(nproc)
```

After changing headers such as `structs.hpp`, force a full relink if incremental builds look stale:

```bash
rm -f build-cuda/CMakeFiles/spd_K.dir/src/*.o
cmake --build build-cuda --target spd_K -j$(nproc)
```

Setup arrays (transform matrices, face coordinates) must be filled via **host mirrors + `setup_push`** — never write device views from host code directly (see `.cursor/rules/kokkos-no-uvm.mdc`).

## Select a GPU

```bash
export CUDA_VISIBLE_DEVICES=0   # front-node GPU 0, or 1 for the second A100
export SPD_OUTPUT_DIR=/path/to/output
./build-cuda/spd_K -i inputs/sine_wave.athinput ...
```

Both GPUs can run jobs in parallel (`CUDA_VISIBLE_DEVICES=0` and `=1` in separate shells). Avoid launching several large 3D cases on both GPUs at once — 40 GB per card fills quickly with AMR + multiblock sweeps.

Check utilization:

```bash
nvidia-smi
watch -n1 nvidia-smi
```

## AMR study script

```bash
cd ~/spd_K-amr
bash scripts/apollo_amr_study.sh
```

Produces density-field visuals and performance/roofline-style plots under `apollo_amr_study/plots/`.

Sync from laptop:

```bash
rsync -avz --exclude build-cuda --exclude .git ./ apollo:~/spd_K-amr/
scp apollo:~/spd_K-amr/apollo_amr_study/plots/*.png ./apollo_plots/
```

## Slurm (compute nodes)

Use `srun -p apollo --gres=gpu:1 ...` only for **batch** jobs on compute nodes. The compute-node software stack differs (e.g. default GCC 8.5); load gcc-toolset-12 inside the job script if you use Slurm.

## Do not

- Use `srun` for quick tests on the **login-node** A100s.
- Use `Kokkos::CudaUVMSpace`.
- Read or write `Kokkos::CudaSpace` views from host code without mirrors (`setup_mirror` / `setup_push`).
