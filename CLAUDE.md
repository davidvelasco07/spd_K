# Developing spd_K

Rules with a measured reason behind each one. Where a rule cites a number, that
number came from a run in this repo — check it before overriding the rule.

Build: `cmake --build build -j8`. Suite: `python3 tests/run_tests.py --build-dir build`
(filter with `--only <substr>`, NOT `--filter`).

## 1. The block index is a KERNEL AXIS, never a host loop

**Do not write `for (int b = 0; b < nblocks; b++)` in anything that runs per step
or per stage.** Every block carries the same element count regardless of level, so
a pack of blocks is rectangular and one launch spans all of them. Use the `pv.*`
whole-pack views and a `*_b` kernel (`sd_for_cells_b`, `fv_for_cells_b`,
`sd_min_cells_b`, ...). SD packs fold the block into the leading axis as
`b*n_ader`; FV packs as `b*n_var`.

Why: this is the single largest source of avoidable cost in the code, repeatedly.
- Batching the FV halo exchanges: 7.81M launches / 131.5 s → 606k / 28.5 s (4.6x).
- Batching the MHD advance phases: mixed-level advance 448.9 → 80.2 s (5.6x).
- Batching the RK bookkeeping: MHD AMR lane 92.19 → 34.50 s at 1600 leaves (2.67x).
  Those six tasks were 74% of MHD's fenced time and 13.4x hydro's on *identical*
  work, only because hydro's version was already one launch over the pack.
- Batching MOOD detection: 21.3 s of a 24.5 s fenced total, 87%.

Legitimate per-block loops, and nothing else: setup and regrid
(`build_block_solvers`), host-side scalar bookkeeping with no launches
(`sync_block_dt`), and output-time diagnostics. Say so in a comment when you
write one.

## 2. Every batched kernel needs an A/B switch and a bit-identity check

Keep the per-block path and gate the batched one behind a switch:
`SPD_MHD_BATCH_MASK` bits for MHD (1 begin, 2 after_U_halo, 4 assemble, 8 commit,
16 Fluxes_pre, 32 Riemann_Solver, 64 B_to_U, 128 Compute_E, 256 E_Riemann, 512 RK
bookkeeping, 1024 mood/detect, 2048 cf/correct_cf_emf, 4096 cf/enforce_fv_emf;
default 8191), `SPD_NO_MHD_BATCH`, `SPD_NO_RK_BATCH`, `SPD_NO_SCORE_BATCH` (the
AMR refinement scores), `SPD_OLD_XCHG`, `SPD_NO_PACK`. Then md5 the dumps of both
paths -- and the block maps too, for anything that feeds a refinement decision. Verify on a **mixed-level** mesh, and with the feature that exercises the
code turned **both ON and OFF** — both directions have already bitten:

- ON: the batched detection segfaulted on its first run at `mood_force_level=-1`
  and looked perfect at `=1`, where detection returns immediately.
- OFF: the same batched detection then differed from the per-block path at
  `mood_force_level=1`, because it was missing `mood_detect`'s opening
  `if(cfg.mood_force_level>=0) return 0;` and ran the whole detection where the
  reference did nothing. Every lane checked at the time had detection live.

A batched path must reproduce its reference's EARLY RETURNS, not just its
arithmetic.

**The A/B catches what the suite structurally cannot.** Batching the refinement
scores, a device-code fix captured the ghost counts as `gx/gy/gz` -- which are the
shear indicator's own flattened cell indices, so `NGHz+gz/pz` became `gz+gz/pz`.
Three lanes went non-identical in the A/B (dumps AND block maps) while the whole
41-config suite stayed green, because the AMR configs that use the shear criterion
check mass, divB and finiteness -- all of which a differently-refined mesh
satisfies. Compare the tags, not just the health of the run.

A single mask bit per phase is what lets a mismatch be bisected to one phase in
one run instead of guessed at. Mind the arithmetic when you pick a mask: 1023
leaves bit 512 SET, so `SPD_MHD_BATCH_MASK=1023` is a detection-only A/B, not a
per-block reference. The full per-block reference is `SPD_MHD_BATCH_MASK=511`
plus `SPD_NO_RK_BATCH=1`.

**A comparison over zero files reports IDENTICAL.** Count the dumps and say
"CHECK VOID" when there are none — a label with a space in it once broke the run
directories and both sides hashed nothing.

## 3. If it is not inside a `PHASE()` scope, it does not exist

Seven top-level tasks had no fence, and they were 74% of the mixed-level MHD step.
The profile that guided a whole optimisation round reported `cf/correct_cf_emf` at
"66-72%" — of the 2.5 s that happened to be fenced, not of the 20 s run. Fence any
new phase.

Use `STAGE(name)` — a Kokkos region AND a fenced phase in one scope — and the
`sd/ xchg/ cf/ mood/ rk/ amr/` prefixes both systems now share. The hydro advance
was instrumented with `Region` alone, which is a no-op without a Kokkos tool, so
`SPD_PHASE_TIMES` accounted for 0.246 s of hydro's 1.96 s wall and every
hydro-vs-MHD comparison here was made against a table hydro never filled in.
Nesting is fine and useful: `amr/adapt` contains `amr/tag`, which is how the
regrid cost turned out to be 100% tagging (0.207 s of 0.207 s).

Reading the report:
- `SPD_PHASE_TIMES=1`, printed at each output, **cumulative up to that output** —
  it covers 109 of 200 steps in a 200-step run with two outputs. Normalise before
  comparing against a wall time.
- **Profile the configuration you intend to run.** Two AMR phases only exist when
  `max_level > 0`; a single-block run takes the standalone `MHD_ader` advance and
  reports nothing at all; `mood/detect` reads 0.000 s whenever
  `mhd/mood_force_level >= 0`.

## 4. MHD and hydro must not fork the phase structure

`if constexpr (is_hydro) { <one batched call> } else { <per-block loop> }` in
`Mesh` is how every one of the above regressions got in: the hydro side was
batched and the MHD side was left behind, in the same function, invisibly. When
you touch such a pair, collapse it instead of extending it — one batched call
whose system differences are a variable count and whether a CT/EMF stage exists.
A new `is_hydro`/`else` pair in `Mesh` needs a reason in a comment.

## 5. No `#ifdef KOKKOS_ENABLE_CUDA` host branch

`grep -c KOKKOS_ENABLE_CUDA src/*.cpp` is a defect detector in this codebase. Six
instances were found where a CUDA branch pulled an array to the host, computed
there, and pushed it back, shadowing a correct device kernel in the `#else`. All
six were numerically CORRECT, so no test could catch them, and the suite runs on
CPU where the branch is not even compiled. One of them cost 39x
(`mhd_compute_primitives`/`_conservatives`/`_dt`: MHD/hydro went 53x → 2.9x when
they were deleted).

`amr_criteria.cpp` is the exception the grep will flag: its branches are a
deliberate host REFERENCE, kept behind `SPD_NO_SCORE_BATCH=1`. But note what was
actually wrong there — there was no device kernel to shadow, the `#else` looped on
the host too, and the cost was the `W.copy()` each score opens with: ~1000
synchronous block copies per regrid, i.e. **100% of `amr/adapt`**. Now one launch,
one thread per block (`block_scores_b`), 0.42 → 0.011 s (38x). One thread per
block rather than per cell ON PURPOSE: each thread walks the host loop's order, so
Löhner's order-dependent denominator sum stays bit-identical. Criteria 2 (trouble
fraction) and 4 (|B| Löhner) still take the per-block path. Mirrors are for setup and for output only — see
`.cursor/rules/kokkos-no-uvm.mdc` for the dual-view pattern.

## 6. Ghosts

- **SD volume ghost ELEMENTS are dead storage.** Proven: `SPD_POISON_GHOSTS=1`
  NaNs them at the top of every stage and the dumps stay md5-identical. SD couples
  elements only through the flux-point trace on a shared face, and both the field
  exchange and the fp gather write exactly one point layer — `(0, n-1)` and
  `(N-1, 0)`. Do not write code that depends on a ghost element's interior.
- **Update kernels write ACTIVE elements only** (`sd_for_active_cells`). Advancing
  the ring integrates values nothing ever wrote; it was 21% of the loop in 2D, 30%
  in 3D, 47% at 16x16x4.
- **Goldens compare the active region** (`golden_active`). Ghost content at output
  time is a post-update leftover, is not part of the solution, and is not portable
  across backends: it is the entire content of the field-loop "GPU/CPU divergence"
  that stood for five days (raw 2.2e-02 and 7.9e-02, interior 9.1e-15).
- Any cross-backend golden failure: **split interior vs ghost before anything
  else.** Two of the three long-standing "divergences" evaporated under that split.
- **One ghost point layer is NOT dead, and it bit the EMF batching.** The
  per-block `set_interface_flux` sweeps the FULL transverse range, so it also
  writes the coarse block's own opposite-face EMF into its TRANSVERSE-ghost slots;
  the table-driven kernel only touches the range its transactions cover. Measured
  with `SPD_CF_EMF_CHECK=1`: active region identical on every call (112 calls,
  static and dynamic lanes), all 640 differing entries in ghost elements. It still
  reaches the solution, because `edge_integral` ranges over `N+1` elements — a
  block has one more edge than cells, so the last edge point lives in the ghost
  element's storage. With the FV-lattice correction ON the affected slots are
  overwritten and every lane is bit-identical; with `SPD_NO_FV_EMF=1` they are not,
  and that lane's dumps move by 3%. Split ACTIVE from GHOST before concluding
  anything about an SD-array difference.

## 7. Gates: never weaken one to make it green, and give it a negative control

A gate that cannot fail is worth nothing. Each of these has a companion config
that REQUIRES the bad number, so that a gate going vacuous turns something red
instead of leaving everything green: `mhd_orszag_tang_smr_noemf_2d`
(`cf_flux_sensitive`), `mhd_orszag_tang_smr_nocorner_2d` (`sl_flux_sensitive`).
Add the same when you add a gate.

Instrument-specific traps, all real:
- `divB` is **blind** to a coarse-fine EMF mismatch: it is computed per block and
  any within-block single-valued EMF passes it. Two attempts to gate the EMF
  correction on `divb` passed for the wrong reason. Use the `cf_flux` telescoping
  drift.
- Locate indices for a check from physical coordinates (`dimension::fv_faces`),
  never from the arithmetic of the correction under test.
- **Run an instrument before trusting it.** `SPD_POISON_GHOSTS` had never been
  run; `SPD_POISON_FP` cannot fail (it poisons the fp trace before `Exchange_fp`
  refills that slot); and on a single-block mesh `TaskPoison` never executes at
  all, so the obvious config to try tests nothing. `SPD_POISON_ACTIVE=1` is the
  control that works (aborts).
- A pack view that nobody assigned is a silent empty view. `fv_pack_view` aborts
  on a name missing from the pack, not on a `pv` field you forgot — writing
  through it segfaults (`pv.troubles` for MHD was assigned past the `return` that
  ends the MHD branch).

## 8. Coarse-fine EMF: restrict along the edge, inject across it

An edge EMF is a line integral along the direction the edge runs. Restriction
averages the two fine half-edges **along that direction only**; transverse to it
the coarse node COINCIDES with a fine node and the transfer is **injection**.
AthenaK does exactly this (`src/bvals/flux_correct_fc.cpp`: 2D injects
`flx.x3e(m,0,fj,fi)`, 3D averages `0.5*(x3e(fk) + x3e(fk+1))`), applied on the
coarse side only, in one batched kernel over (component, block, neighbour).

`amr_RF_fp` (`amr.cpp`) used to violate this -- `rf_fp(j,j)=0.5; rf_fp(j,m+j)=0.5`
averaged node j of the two fine halves, two DIFFERENT physical points. It
preserved constants and destroyed the coincident-point identity telescoping
needs, which is why the SD path drifted 2.54e-03 where the cascade path drifts
1e-17. **FIXED in `5717dc6`:** each coarse fp node is now restricted from the fine
half that CONTAINS it (a Lagrange interpolation, exact to degree m-1), which
collapses to injection at every coincident node with no special case. SD drift
1.39e-17, same-level control exactly 0.

Both corrections are BATCHED off the fine->coarse transaction table (`xtfi_`),
which is AthenaK's shape -- one kernel over (component, block, neighbour):
`cf/correct_cf_emf` 1.578 -> 0.015 s (105x) and `cf/enforce_fv_emf` 0.353 ->
0.011 s (32x) on 352 leaves over 200 steps, mask bits 2048 and 4096. The SD one
IS `correct_cf_flux_b`: that kernel takes its point counts off the array and the
matrix as a parameter, so it is generic in the lattice. The note that said
reusing it was wrong ("640 differing entries at max|diff| = 1.0") was measuring
two things at once -- the wrong matrix (`amr_RF` where the reference uses
`restrict_mat_for` -> `amr_RF_fp`), and the ghost ring (see rule 6).

## 9. Remote runs (apollo)

- `rsync` **`inputs/` and `tests/` as well as `src/`** — a stale `inputs/` once
  invalidated a whole GPU campaign via `cfl=0.8`.
- Verify the remote binary's provenance BEFORE launching, and record it: md5 the
  source, check the binary mtime against the run's first dump. A paper-scale
  reference once ran on a binary that was rebuilt mid-run.
- **Never rebuild a binary a production run is using.** Linux refuses to write a
  running executable, so the link fails; use a second tree.
- `nvcc` is not on `PATH` over non-interactive ssh: `export PATH=/usr/local/cuda/bin:$PATH`,
  or cmake fails with a bogus `string sub-command REPLACE` error from Kokkos.
- Incremental builds go stale on `structs.hpp`/`define.hpp`: delete the objects
  (`rm -f build/CMakeFiles/spd_K.dir/src/*.o`), not the `.dir`.
- Dumps go to `$SPD_OUTPUT_DIR`, defaulting to `<cwd>/output/`.

## 10. Kokkos policy

`RangePolicy` or `TeamPolicy` only — never `MDRangePolicy`. Never
`Kokkos::CudaUVMSpace` (see `.cursor/rules/kokkos-no-uvm.mdc`). `nvcc` cannot
generate its stub for a `__device__` lambda inside a member function that also
takes a pointer-to-member parameter: it compiles on the host build and fails only
under CUDA.
