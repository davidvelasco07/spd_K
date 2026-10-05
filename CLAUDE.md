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

- Batching `sync/face_B`: uniform 2048^2 in 16^2 blocks went 395.31 -> 61.88
  ms/step (6.4x). It was 77.8% of that lane's fenced time, from ~32k launches per
  stage at 16384 blocks, and **it hid from every AMR profile in this project
  because Mesh only calls it when `max_level == 0`.** Profile the uniform
  multiblock lane too, not just the AMR one.

**`a8f3da0` said no live per-block host loop was left in the per-step path. It was
wrong: grep for the loop, do not trust a list.** The hydro SD update
(`sd/Update_prediction`, `sd/Update_solution`) stayed per-block until it was
batched behind `SPD_NO_SD_UPDATE_BATCH`: 30% of the fenced time on a uniform
256-block ADER lane, 264k launches (304 per stage) on a 304-block mixed-level
lane. On the Serial CPU backend the batch is a small LOSS (`sd/Update_solution`
3.38 -> 3.95 s, +2% of the fenced step): 264k Serial launches cost ~0.1 s, and
the batched body is slower per point (launched once per block it takes 4.06 s).
On an A100 (apollo, 29 Sep) the same phase went 7.03 -> 0.057 s at 304 blocks and
12.69 -> 0.16 s at 1216, the predictor 4.17 -> 0.06 s at 256 (uniform ADER): the
fenced step 6.4x, 4.4x and 9.6x faster. Judge a batch on the GPU; a local Serial
timing says nothing about launch cost. Three loops are still
live and unswitched, each only under a non-default option: `sd/Viscosity`
(`hydro/nu>0`), MHD's `sd/Update_CT` (MHD without the fallback), and
`mhd_reduce_nad_gscales` (`mhd/mood_nad_scale=grange|gcfl`). Every other
`for(int b=0; b<nblocks; b++)` in `mesh.hpp` is either regrid/setup
(`build_block_solvers`, the table builders, snapshot/transfer/finish_ic), host-side
scalar bookkeeping with no launches (`sync_block_dt`), an output-time diagnostic,
or the `else` branch of a batching switch -- which is the A/B reference and must
stay. Check which of those a new loop is, and say so in a comment.

Legitimate per-block loops, and nothing else: setup and regrid
(`build_block_solvers`), host-side scalar bookkeeping with no launches
(`sync_block_dt`), and output-time diagnostics. Say so in a comment when you
write one.

## 2. Every batched kernel needs an A/B switch and a bit-identity check

Keep the per-block path and gate the batched one behind a switch:
`SPD_MHD_BATCH_MASK` bits for MHD (1 begin, 2 after_U_halo, 4 assemble, 8 commit,
16 Fluxes_pre, 32 Riemann_Solver, 64 B_to_U, 128 Compute_E, 256 E_Riemann, 512 RK
bookkeeping, 1024 mood/detect, 2048 cf/correct_cf_emf, 4096 cf/enforce_fv_emf,
8192 the pinned-level dead-work skip, 16384 sync/face_B; default 32767), `SPD_NO_MHD_BATCH`,
`SPD_NO_RK_BATCH`, `SPD_NO_SCORE_BATCH` (the AMR refinement scores),
`SPD_NO_SD_UPDATE_BATCH` (the per-block hydro SD predictor and pure-SD update,
rule 1), `SPD_OLD_XCHG`, `SPD_NO_PACK`, `SPD_NO_FV_GHOST_LIN` (injection instead of the
limited-linear coarse->fine CV ghost fill), `SPD_NO_FV_PRERESTRICT` (the
sweep order that left a transverse ghost row stale, rule 6b), `SPD_NO_FV_SYMFILL`
(the per-direction sweep instead of the symmetric fill order, rule 6b),
`SPD_OLD_DEREFINE_APPLY` (the stale-index derefine loop, rule 7a3), `SPD_LOHNER_INTERIOR` (the interior-only Lohner score, rule 7a4),
`SPD_FV_ONLY_SD` (hydro `job/scheme=vl2|plm`: `=1` runs the dead SD flux path
and blends it in as `f + theta*(fL - f)`, the pre-skip reference; `=2` runs it
but takes the MUSCL flux outright, which must be bit-identical to the default
skip -- see "a discard that rounds" below), `SPD_FV_ONLY_CASCADE` (hydro
`job/scheme=vl2|plm` under a deck's `fallback/style=cascade`, as `dmr` and
`woodward_colella` set: `=1` keeps the cascade, which starts every cell on the SD
flux -- first-order Rusanov at p=0 -- and lifts it to MUSCL only where detection
flags it; that ran the DMR "MUSCL-Hancock" lanes until 1 Oct 2026, with the
limiter barely read, 2e-6 rms between minmod and van Leer), and `hydro/mood_pad_first_order=false` (a DECK
switch: the old hydro cascade, whose PAD wrote the same flag as NAD, so a cell flagged on two revisions went
to first order from NAD alone; the default now stops NAD at MUSCL as Paper IV's MHD cascade does -- on the
128^2 SDFB4 implosion 2.4-9.6% of cells sat at first order before, 0% after, i.e. every one was NAD-driven).
Then md5 the dumps of both paths -- and the block
maps too, for anything that feeds a refinement decision. Every switch must agree
with every other on one mixed-level lane; ten of them do today, checked together. Verify on a **mixed-level** mesh, and with the feature that exercises the
code turned **both ON and OFF** — both directions have already bitten:

- ON: the batched detection segfaulted on its first run at `mood_force_level=-1`
  and looked perfect at `=1`, where detection returns immediately.
- OFF: the same batched detection then differed from the per-block path at
  `mood_force_level=1`, because it was missing `mood_detect`'s opening
  `if(cfg.mood_force_level>=0) return 0;` and ran the whole detection where the
  reference did nothing. Every lane checked at the time had detection live.

A batched path must reproduce its reference's EARLY RETURNS, not just its
arithmetic. And a SKIPPED path must account for every job the code did, not just
the one its name describes: `fv_update_solution_b` in `mood_begin` looks like the
level-0 candidate and nothing else, and it also seeds `U_old_fv` from `U_cv` --
the base state the halo publishes and the commit updates from. Skipping it gave
"non-finite dt at step 1" on every pinned lane while levels -1 and 0 stayed
bit-identical, i.e. the A/B localised it only because the force level was part of
the sweep.

**A discard that rounds is not a discard.** Under hydro `job/scheme=vl2|plm` the
blend pins theta to 1, so the SD flux path looked dead -- 20.5% of the fenced step
on the 2048^2 KH MUSCL AMR lane at 32^2-DoF blocks (A100). Skipping it could NOT
be bit-identical: the blend is `f + theta*(fL - f)`, which at theta = 1 equals
`fL` only while `f` and `fL` are within a factor of two, rounds on the SD flux
otherwise, and is NaN whenever `f` is. Every MUSCL lane carried the SD flux at
round-off (4e-15 to 1.9e-13 against e8d36e0, block maps unchanged). So the skip
(`Mesh::sd_path_dead`, shared with MHD's) came with a blend that takes `fL`
outright, and the A/B has three sides, not two: `SPD_FV_ONLY_SD=2` (path run,
flux taken outright) must match the default bit for bit -- that is what proves
the path dead -- and `=1` must match the pre-change binary. Both held on five
MUSCL lanes (two mixed-level AMR, uniform multiblock, `plm`, p=3 ADER), and two
SD lanes stayed identical. **Before calling work dead, read how its consumer
combines it: a zero weight multiplies, it does not erase.** A MUSCL dump made
before this change is not bit-comparable with a new default run; to reproduce
one, run with `SPD_FV_ONLY_SD=1`.

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
- **SD sub-cells are NOT equally spaced, so never `imshow` an SD field.** The FV
  sub-cells of an element are the control volumes of the solution points.
  Measured widths within ONE element: `[0.0070, 0.0242, 0.0242, 0.0070]` at p=3
  (3.4x ratio) and an 8.0x span at p=7. `imshow` gives every cell the same pixel
  width, so it stretches the element-edge cells by that factor and squeezes the
  interior ones -- the SDFB panels came out faceted and mottled, worst at p=7,
  and it read as a defect in the solution. Average onto regular control volumes
  first (`mdz21_diagnostics.to_regular`, an exact area-weighted rebin: identity
  on the source faces, rows summing to 1, integral preserved to 1.4e-16) and draw
  with `pcolormesh` on `regular_faces`. This is the cell-average analogue of the
  Python spd's `interpolate_to_regular_mesh`, which Lagrange-interpolates the
  solution POINTS to regular sub-cell midpoints and plots on `regular_faces`.
  `pcolormesh` on the true `centers` is nearly right (it honours the spacing but
  puts edges at midpoints between centres); `imshow` is simply wrong. Scalar
  diagnostics were never affected -- they integrate with `grid.widths()`.
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

## 6b. A fine block's CV-lattice ghosts are RECONSTRUCTED from the coarse neighbour, not injected

`fv_inject_coarser`/`gather_fv_coarser` used to copy the coarse cell value into
every fine ghost cell it covers: first order at every coarse-fine face, every
stage. At p=0 the CV lattice is the only lattice, so MUSCL reconstructed its
slopes next to every level jump from piecewise-constant data; AthenaK's
`ProlongCC` uses a minmod-limited linear fill. That was most of the excess in the
AMR-vs-uniform residual (fig-22 lane: rms 0.84% vs AthenaK's 0.10%). Now
`fv_lin_ghost` (amr_boundary.cpp) evaluates the coarse cell's minmod-limited
profile at the true fine sub-cell centres (`amr_x_fp`), for STATE fields only
(flags, theta and the cascade index keep injection), and the finer pass runs
before the coarser pass so the interface slope reads the freshly restricted
coarse ghost. Measured, fig-21, initial refine, 16^2 blocks, A100 (rms / max):

| lane | injection | limited linear |
|---|---|---|
| MUSCL 1024^2 | 0.0039 / 0.081 | **0.0014 / 0.031** |
| MUSCL 2048^2 | 0.0041 / 0.086 | **0.0022 / 0.064** |
| SDFB4 1024^2 | 0.0029 / 0.13 | 0.0030 / 0.18 (unchanged: the SD lattice is exact) |

`SPD_NO_FV_GHOST_LIN=1` is the A/B and is bit-identical to the pre-change binary
on both exchange paths. The far-field block pattern that remains (few 1e-4 to
1e-3 in level-0/1 regions) is the coarse-vs-fine truncation difference and
AthenaK shows it at the same amplitude; do not chase it as a defect.

Two traps met while extending it to walls (2026-09-24), both caught by the
double-Mach problem, neither by the periodic suite:
- The fill's transverse slope reads the coarse block's own transverse ghost
  rows. Right after a regrid those rows hold ANOTHER block's data (the packs
  reuse their slots), and the first coarse->fine pass consumed it: a
  rarefaction under the top wall wherever a refined block touched it, present
  or absent with the output cadence. `Mesh::Exchange_fv_field` now runs one
  extra full exchange after every rebuild (`fv_ghosts_stale_`). Do NOT fix it
  by restricting the slope to interior rows: `SPD_FV_GHOST_TMODE=1` does that
  and leaks mass, 8e-12 on the p=0 dynamic pulse against 3e-15, because two
  fine blocks meeting at a coarse block face then see different profiles of
  the same coarse row.
- **The fill reads TWO ghost rows of the coarse block, and both must be
  current.** The transverse row is the restriction of the fine block on the
  ADJACENT face; a per-direction sweep (x: same, finer, coarser, bc; then y)
  restricts the y face only after the x fill has read it. The one fine cell
  at a coarse block's corner is filled from both directions, the two values
  differed by that one stage, the corner pass copied one into the diagonal
  block while the other block kept its own, and the same-level face between
  them computed two different fluxes at its end row -- nothing symmetrizes a
  same-level face. +6.4e-10 of mass per run on the 2D MUSCL blast, every
  digit in that row (`SPD_FV_FLUX_CHECK=1`), visible ONLY with derefinement
  live (a derefined coarse corner inside non-uniform flow): derefine off
  -3e-16, injection -7e-16, 1D tubes 1e-16. Fixed in `41e7fef`: `Exchange_fv_field` restricts every
  direction once BEFORE the sweep when the fill is on; `SPD_NO_FV_PRERESTRICT=1`
  is the leaking reference. Localised with `SPD_STEP_MASS=1` (mass after
  every step) and the flux check; a leak that needs two switches ON at once
  is bisected by turning each off alone, then asking what only their
  combination touches.
- **The fill order is symmetric now (Jacobi, not Gauss-Seidel).** The
  per-direction sweep let the x fill read a coarse block's y rows before the
  y same-level/wall passes refreshed them, and -- since 2:1 balance constrains
  faces, not corners, so a corner can span three levels -- rows that were
  themselves the y fill. Measured on the 1024^2 Sedov blast with mirror
  symmetry at 1e-13: x<->y asymmetry 1e-8 (32^2-DoF blocks), 3e-6 (16^2),
  9e-5 (8^2). `Mesh::Exchange_fv_symmetric` now runs every same, finer and
  bc pass of every direction, snapshots the state, runs every coarse->fine
  fill of every direction off the snapshot, then bc and the same-level corner
  pass again. That alone took the 1024^2 MUSCL lane from 2.8e-6 to 6.7e-9,
  and the rest was the ghost-ring CORNERS: both the x and the y fill write a
  fine block's corner ghosts (through the coarse block's transverse ghost row
  and through the y-neighbour's), each with centred slopes along its own
  direction only, so the two values differed and the last writer won. Those
  cells are now INJECTED (the two rows are copies of the same diagonal cell,
  so the passes agree bitwise; the corner pass still overwrites them from a
  same-level diagonal owner): 6.7e-9 -> 3.4e-13, i.e. round-off. SDFB4 lanes
  went 1.2e-6 -> 5.7e-14 (Sedov) and 2e-10 -> 7e-14 (implosion, walls +
  AMR); mass unchanged at 1e-15; `SPD_NO_FV_SYMFILL=1` is the sweep,
  bit-identical to the pre-change binary, and packed vs per-block agree
  bitwise on 7 lanes. **And the SD cascade amplified it to O(0.1):** on the
  A100 the SDFB4 Sedov lanes (L=4, 32/16/8-DoF blocks) had max|rho - rho^T|
  = 0.15 / 0.19 / 0.12 with the old order -- the fill's 1e-8 seeded threshold
  flips (rule 7b) that the flow then grew -- and 2e-13 / 1e-13 / 5e-13 with
  the symmetric one. A symmetry the cascade cannot hold is usually a seed in
  the ghosts, not the cascade. Do not put a Gauss-Seidel dependency between
  directions back into the exchange, and do not give a corner ghost a slope
  that only one direction can compute.
- A refined block's `dimension::L` is the BLOCK length, not the box length
  (0.125 for a level-1 block of 4 elements on a unit box). Anything that needs
  the domain size on the block path reads `g_bc_box`.

**Walls on the block path exist now (hydro only):** `reflective` and `doublemach`
are filled by `apply_domain_bc_fp/fv`, and any run with one is routed through
the forest/table exchange (`Mesh::forest_route()`), whose physical-boundary
blocks are the only place walls are implemented. The gate is
`hydro_implosion_muscl_mb_2d`: 4x4 blocks against the single block at 1e-12
(measured 9.9e-16). The p=3 version carries no golden ON PURPOSE: with the
cascade live, single-block and multiblock differ by threshold flips on this
problem -- 166 cells by step 420 in 1D even with GRADFREE walls -- which is
rule 7b, not a wall bug. A wall gate needs a lane without a detector.

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
- **An unknown check name in `run_tests.py` used to print a PASS for a check that
  never ran.** The `if/elif` dispatch chain had no final `else`, so an
  unrecognised name fell through with `ok, msg` still holding the PREVIOUS
  check's values and printed them a second time under the new name. Adding
  `"mass"` and `"finite"` (neither is a dispatched name; `check_finite` already
  runs unconditionally) made the suite print each real check TWICE, all green.
  A typo in any config would have manufactured a gate that cannot fail. Now an
  unknown name is a hard FAIL that lists the valid ones.
- **`mass_strict` and `divb` are ABSOLUTE gates, so they are only calibrated for
  the problem they were written against.** Both now take a per-config override
  (`mass_limit`, `divb_limit`, matching `cf_limit`). Raise one ONLY with the
  control measured and written into the config:
  - `divb`: the 3D blast has B0 = 28.2 and a sub-cell dx of 0.031, so its
    2.1e-11 is 2.3e-14 RELATIVE -- the same CT quality the 2D configs show at
    1e-11 absolute.
  - `mass_strict`: gradfree boundaries extrapolate and do not conserve. The
    identical IC run with PERIODIC walls drifts 4.7e-15 while the outflow
    version drifts 1.3e-10, which localises the leak to the boundary rather
    than the scheme. Run that control before touching the number.
- A pack view that nobody assigned is a silent empty view. `fv_pack_view` aborts
  on a name missing from the pack, not on a `pv` field you forgot — writing
  through it segfaults (`pv.troubles` for MHD was assigned past the `return` that
  ends the MHD branch).

## 7a. A parameter nobody reads: prove the input moves the output before calibrating

`amr/refine_threshold` and `amr/derefine_threshold` were read by the shear
criterion ONLY; pressure and Lohner hard-coded 0.03/0.0075 and 0.5/0.0125 in
`refine_from_score` and ignored the deck. The tell was cheap and was missed for
weeks: Sod MUSCL AMR with `refine_threshold` 0.03, 0.1, 0.3 and unset gave
md5-identical block maps. Every calibration made on those numbers before
`4c6af95` was a no-op, and every threshold quoted for those runs was wrong
(the Apollo DMR `*p` lanes ran 0.03/0.0075 whatever their log says).

Now every threshold criterion reads the pair, the compiled values are the
per-criterion defaults resolved in `main.cpp`, and `parameters.txt` records the
resolved pair, not a sentinel. **Before calibrating anything, run the deck at
two values of the knob and md5 the block maps: identical maps mean the knob is
not connected.** The same A/B in the other direction (deck sets nothing -> must
be bit-identical to the old binary) is what made the fix safe: 0 of 4 files
differ on three decks.

**The `<mhd>` parameter block in `main.cpp` is UNCONDITIONAL** -- it runs for every system and assigns shared
`cfg` fields (`mood_pad_first_order`, `mood_max_level`, ...). The first build of `hydro/mood_pad_first_order`
parsed it BEFORE that block, which reset it to the MHD default, and the new default came out bit-identical
to the old binary on every lane. The A/B's "new default" side is what caught it: a fix that changes nothing
is a knob that is not connected. A hydro parse of a shared field goes AFTER the MHD block.

## 7a2. A pointwise IC can be invisible to a coarse root, and then nothing refines

The Sedov deposit is evaluated at cell centres / solution points. On a 32^2
root the four central cell centres sit 0.022 from the box centre, so a deposit
of radius 0.02 lands in NO cell: the root is a uniform state, the pressure
criterion tags nothing, `initial_refine` never triggers, and the run finishes
in ONE step (dt = box/c_ambient > tlim) with a clean log. R = 0.025 fixed it;
the gate `hydro_sedov_amr_2d` pins that. Before any AMR sweep whose feature is
smaller than a root cell, check the t = 0 dump of the COARSEST root for the
feature (max p, or the leaf count after initial refinement), not the finest.

## 7a4. The Lohner score is an ELEMENT-lattice quantity: it changes meaning with p, and a 2-element block is blind to it

`lohner_score` / `block_scores_b` take the undivided second difference between
neighbouring ELEMENTS at the same sub-point and skip the block-edge elements
(a ghost element's interior is dead storage, rule 6). Two consequences, both
measured on the Liska-Wendroff implosion (512^2, t=2.5, 32^2-DoF blocks):

- At p=0 an element is a cell and this is the textbook estimator. At p=3 the
  difference is taken at 4x the cell spacing, so a smooth gradient scores up
  to 16x higher and a discontinuity the same: 0.02 tags 6% of the MUSCL blocks
  (39% of the jet blocks, 1% of the rest) and 55% of the SDFB4 blocks, jet and
  rest alike; SDFB4 separates the jet only above 0.1. The SDFB4 sweep at the
  MUSCL pair ended 89% refined. **A Lohner threshold is per-p; calibrate it on
  the uniform field of the scheme you run** (SDFB4: 0.12/0.03 tags the same 6%).
- A block with < 3 elements per side has no element with two interior
  neighbours, scores exactly 0, and is never tagged: the four 8^2-DoF SDFB4
  lanes (2 elements) ran to t=2.5 on their ROOT with a clean log. **Fixed: the
  score now reaches across the block edge.** Each edge element takes ONE extra
  stencil, at its face-adjacent sub-point (n-1 across the low face, 0 across the
  high face), whose outer value is the ghost point the SD field exchange writes
  -- the only valid point of a ghost element (rule 6). `Mesh::exchange_for_tagging`
  refreshes that layer right before every tag: nothing else writes it after the
  last stage, and a freshly built block never had it. At a level jump that ghost
  is the NEIGHBOUR's face-adjacent point, moved along the face but not across it,
  so it sits at the neighbour's spacing: the plain difference would read a smooth
  gradient as curvature (a first-derivative term of 0.07h at p=3, 0.5h at p=0),
  and the stencil uses the divided form with the true spacing there
  (`lohner_edge_ratios`, `lohner_d2`). Blocks now need 2 elements per side;
  `SPD_LOHNER_INTERIOR=1` is the old score and still needs 3. Gate:
  `hydro_implosion_lohner_edge_2d` (2-element SDFB4 blocks must refine; the old
  score refuses that config, so the gate cannot pass without the stencil).

A CV-lattice score is NOT the fix: the SD sub-cells are not equally spaced
(rule 6), an undivided second difference on that lattice is of the order of
the first difference for any smooth gradient, and the divided form with the
true widths still tags 29% of the SDFB4 blocks (the cascade leaves cell-scale
structure the element lattice averages over).

## 7a3. Identical on CPU and GPU means deterministic, not round-off; instrument the regrid

The Liska-Wendroff implosion (512^2, density Lohner, symmetric fill) lost its
diagonal symmetry at 2e-5 by t = 0.12 with block maps that were symmetric at
every output. Threshold flips from round-off were the obvious story and would
have been WRONG: the CPU and the A100 produced the same 5.9e-5 at the same
step, which round-off cannot do. `SPD_SYM_CHECK=1` (a transpose-mismatch count
of the leaf set after EVERY regrid, plus the tagged groups and the per-stage
block counts) found it in one run: step 2220, two transpose-partner groups
tagged and allowed, `derefine_blocks_keys` applied the first and REFUSED the
second -- its admissibility test read neighbour tables that the first merge
had invalidated and that the loop rebuilt only at the end. Fixed by judging
every group on the post-refine forest and then applying all of them
(`SPD_OLD_DEREFINE_APPLY=1` is the old loop). Every AMR lane with more than one
derefinement per regrid could have been hit; on the seven A/B lanes only the
one with a refusal moved, bit-identical elsewhere.

Two lessons. A symmetry test needs an instrument at the REGRID cadence, not
the output cadence: the mesh was symmetric at every dump and asymmetric in
between. And before calling anything "round-off", run it on a second backend:
the same digits on both is a deterministic defect.

It had already been misread once. Two days earlier the 1024^2 Sedov lane with
8^2-DoF MUSCL blocks broke its MIRROR symmetry at 9e-3 (47 of 3154 leaves without
a mirror partner), and it was written up -- in the paper and a report -- as
threshold flips: flat interior pressure, 8-cell blocks scoring on the cuts,
mirror scores differing at round-off. Plausible, detailed, and wrong: the rerun
on the fixed derefinement is 5.9e-13 on the same lane. **A threshold story for a
symmetry break is a hypothesis until `SPD_SYM_CHECK=1` has seen every regrid.**

## 7b. With the cascade live, cell values are NOT reproducible across backends

The MOOD detector is a THRESHOLD. A round-off difference flips one cell's
troubled flag, and a flipped flag changes that cell's scheme by O(1) -- so
round-off does not stay round-off. Measured on the 3D blast (MDZ21 6.4,
beta ~ 2.5e-4, 32^3 DoF, CPU vs A100, active region only):

| configuration | worst rel CPU-GPU diff |
|---|---|
| t = 0 (the IC itself) | 2.6e-14 |
| cascade live, after 12 steps | **2.1e-01** |
| cascade live, after 24 steps | 1.2e-01 |
| pinned `mood_force_level=1` (MUSCL, no decisions) | **2.8e-14** |
| volume-integrated E_B, cascade live | 4.1e-06 |
| total mass, cascade live | 1.1e-10 (both) |

**Pinning the cascade is how you tell flag chaos from a backend bug.** The
pinned lane agreeing to 2.8e-14 is what proved the 21% spread was NOT a GPU
defect -- and this codebase has had real ones (see
`.cursor/rules/kokkos-no-uvm.mdc` and the mirror push-without-pull class), so
the question is not rhetorical. Do that A/B before concluding either way.

Consequences: do not put a golden on a cascade-live config that must run on more
than one backend; compare INTEGRALS (they survive at 1e-6) not cells; and read
any pointwise cross-backend difference as a flag-flip count until pinning says
otherwise. `mood_force_level >= 0` also makes `mood/detect` read 0.000 s and
returns from `mood_detect` immediately (rule 2) -- that is the same switch.

## 7c. The energy correction works, but ONLY at a stage boundary

spd_K replaces a state's B rows with the CT field in three places without moving
the total energy, so `p = (g-1)(E - Ekin - B^2/2)` is then read off a B that E was
never built from: `mhd_B_to_U` (the END OF A STAGE), `mhd_set_candidate_B` and
`mhd_face_B_to_fp` (both MID-UPDATE). At beta ~ 1 that is noise; on MDZ21 6.4
(beta ~ 2.5e-4) the thermal energy is 0.06% of the magnetic and the residual
swamps it -- 15% of the domain ends on the RAMSES smallp, carrying AMBIENT
density and field.

`mhd/energy_fix` is a BITMASK over those sites (1 SD, 2 FV, 4 fp; 0 = off,
bit-identical). 96^3 DoF, UCT-HLLD, five COMPLETED runs at the same output:

| mask | site | floored | steps |
|---|---|---|---|
| 0 | none | 19.94% | 488 |
| **1** | **SD `B_to_U`, end of stage** | **0.02%** | 469 |
| 2 | FV `set_candidate_B`, mid-update | 23.20% | 482 |
| 4 | fp `face_B_to_fp`, mid-update | 19.95% | 487 |
| 7 | all three | 0.00% | 474 |

**Use mask 1.** At a stage boundary the cell-centred B is discarded garbage and
swapping it wholesale is the intended operation. Mid-update it is a working
value, and "correcting" around it ADDS an inconsistency -- the FV site alone is
worse than doing nothing.

**THE SIGN IS THE WHOLE THING.** Keeping the recovered pressure fixed across the
swap needs `E <- E + (Bf^2 - Bc^2)/2`, i.e. strip the OLD magnetic energy before
the swap and add the NEW one after. The first implementation here had it
backwards (`+ (Bc^2 - Bf^2)/2`), which DOUBLES the mismatch: the stage-boundary
site then made the floors worse, 19.94% -> 25.52%, and this rule previously
recorded the correction as a measured FAILURE. Derive the sign from "the recovered
p must not move", not from the paper's formula as remembered.

**A second process failure worth not repeating.** The first production check
reported a 35% improvement that did not exist: it compared
`output_indices(d)[-1]` of both runs while one had not finished -- output 7 of
the fix against output 10 of the baseline -- and the floored fraction grows
monotonically with time. `run_tests.py`/`mdz21_report.py` refuse incomplete runs
for exactly this reason; the ad-hoc script bypassed that gate. **Any ad-hoc
comparison must check for the `evolution:` line, not just take the last dump.**

## 7d. The first-order tier is PAD-driven, not detector noise

SDFB can finish WORSE than the scheme it falls back to: on RR22 KH under
UCT-HLL, SDFB4 reaches <B_p^2> = 1.0004 against plain MUSCL's 1.1290 -- no
growth at all. That is only possible because the cascade goes PAST MUSCL to
first order: 3.68% of cells at level 2 against 4.95% at level 1, i.e. nearly
half of all demotions go the whole way.

The obvious diagnosis -- a twitchy NAD flagging smooth flow -- is WRONG, and
`mhd/mood_pad_first_order` was built to test it: it lets only a PAD failure
(negative rho/p, non-finite) reach first order, while a NAD flag alone stops at
MUSCL. MEASURED, it is a NO-OP on both the current sheet and KH -- identical
energy and identical level-1/level-2 fractions to every digit. **Every level-2
demotion there is already PAD-driven**: the MUSCL candidate is physically
inadmissible in those cells, and NAD never gets a say. (The switch is not dead
code -- it does move the balsara blast, which has NAD-driven level-2 cells.)

`mhd/mood_max_level` caps the cascade (1 = stop at MUSCL). The result is
problem-dependent, which is the tell that first order is load-bearing:

| | current sheet (exact = 1) | KH under UCT-HLL (MUSCL = 1.1290) |
|---|---|---|
| allow first order | **0.9527** / 0.4651 | 1.0004 / 1.0010 |
| cap at MUSCL | 0.5535 / 0.2240 | **2.4268** / **1.6391** |

Capping HELPS on KH and HURTS on the current sheet. Denying those cells the
tier they need makes them get FLOORED instead, which is less destructive to a KH
instability than donor cell but far worse on an under-resolved current sheet.

So the lever is not the detector. Either loosen the tolerance (1e-3 gives
0.9991/0.9993 on the sheet and 5.40/5.76 on KH -- better than any max_level at
the default), or make the level-1 candidate itself positivity-preserving so it
stops failing PAD.

**A domain-averaged demotion rate is the wrong statistic.** Under UCT-HLL at
t = 10, SDFB4 demotes 7.04% of the domain but 27.37% INSIDE the shear layer -- a
3.9x enrichment, 6.3x at p=7. An earlier version of this file ruled the cascade
out on the strength of a 0.01-0.12 mean level. Ask where it fires.

## 7e. Output cadence must never drive the timestep

`driver.hpp` and `hydro_ader.hpp` used to do `dt = t_output - t` unconditionally,
so an `output/dt` SMALLER than the timestep set dt = output/dt EVERY step: the
run then takes t_end/output_dt steps and writes a dump on each one. Measured the
expensive way -- `output/dt=1e-9 tlim=1.0` wrote **740 GB**, twice, filling the
apollo filesystem and breaking an unrelated build (`nvcc fatal: Could not open
output file '/tmp/...'`). Both drivers now (a) anchor the schedule to multiples
of dt_output rather than to the time reached, and (b) warn and write every step
instead of shrinking dt.

`time/nlim = 0` reads as "zero steps" and was NOT: the cap is `cfg.nlim > 0`, so
0 fell through as unlimited. That is half of how the above happened. It is now a
hard error pointing at -1.

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

## 8b. A boundary that carries a field needs TWO conditions, and the FV lattice needs its own

With CT the NORMAL face field on a boundary is advanced by the curl like any
interior face, so unless something pins it, it is driven by an EMF extrapolated
out of the interior. Getting the FLUID condition right is not enough. Both of
these were measured on the Mach-800 jet (Balsara 2025 8.2), which had never once
run, and both were invisible because **there was no jet config in the suite**.

**`_reflective_` is only valid where `B.n = 0` on that wall.** It flips the
normal magnetic row, which is what makes `B.n = 0` by antisymmetry -- so on a
face the field passes THROUGH it imposes a jump of `2|B.n|` in the normal field,
straight into that face's Riemann problem. That is a div-B violation by
construction. The jet's base carries `|B_y| = 141.42` normal to it; closing it as
a wall destroyed an EXACT stationary solution (the quiescent ambient), growing
`max|v|` 10x per 1.2e-4 of time from round-off on the bottom row until dt
collapsed to 1e-6 of `dt0`. This is the trap in "a jet emerges from a nozzle in
a solid surface": true, and irrelevant, because the field does not know that.

**A prescribed inlet must pin its normal field too.** A Dirichlet inlet
prescribes the whole state, `B` included. Left free, the jet inlet's `B_y` went
141.42 -> 1308 on the base row, `|B|^2` peaked at 2.17e+06 ON the inlet row
against the paper's 1.9e+05 in the bow shock, and the resulting magnetic
pressure of 1.1e+06 -- ABOVE the jet's own ram pressure of 9.0e+05 -- pushed
material back in at a mean `v_y` of 435. **That is what "a plain outflow base
sucks material in" actually was.** An earlier round read that 435 as an argument
for closing the base and reached for the wall above, which is far worse. The
re-entry was downstream of a field error: fix the field condition and the same
number is 0.98 with the base still a plain outflow, which is also what Wu & Shu
(2018) specify. **When a boundary misbehaves in MHD, check what it is doing to
B before you change what it does to the fluid.**

**And the two lattices are separate.** `mhd_zero_wall_emf` pinned the SD edge
arrays only; its own comment flagged the FV lattice as a latent gap. The MOOD
cascade assembles its EMF there (`E0z` from `E1z/E2z`), and `job/scheme=plm|vl2`
sets `cfg.fv_only` and never touches the SD edge arrays at all -- so that lane
had NO boundary EMF condition of any kind. `mhd_pin_bc_emf_fv` closes it, for
inlets and walls alike.

Be precise about which half that mattered for, because only one of them was
live. For INLETS it was not latent at all: the jet pathology above was measured
in the plm lane, i.e. entirely on the FV lattice. For WALLS it is STILL latent
-- `mhd_current_sheet_equilibrium_2d` is the only reflecting-wall config and it
runs the SD lane, so the FV wall pin does not move it: 6.31e-13 with the pin
against 6.26e-13 without, both round-off. The FV wall pin is therefore closed
but UNGATED, and will stay that way until some config runs `plm`/`vl2` against
a reflecting wall.

`SPD_NO_BC_EMF_PIN=1` is the A/B for both halves. It is a no-op on every
periodic/outflow face: `inputs/balsara/blast_smoke_p3_fb.athinput` is
bit-identical across it, 9 dumps per side (counted -- a comparison over zero
files reports IDENTICAL, rule 2).

The FV pin is `standalone_` only, and that is a real restriction, not an
oversight: it pins the first/last node of the array it is given, which is the
domain boundary only when the block IS the domain, so under `Mesh` it would pin
an interior block's internal edge. Doing it there needs per-block
touches-the-boundary flags off `xtfi_`, like the cf/ corrections. So a
MULTIBLOCK `plm` run against a reflecting wall still has no FV boundary EMF
condition. `x2_bc=inflow` is standalone-and-jet-only (hard error otherwise), so
the inlet half is fully covered.

Gates, with the negative control each one needs (rule 7):
`mhd_jet_base_equilibrium_2d` / `mhd_jet_base_wall_sensitive_2d` for the wall,
`mhd_jet_inlet_field_2d` / `mhd_jet_inlet_unpinned_sensitive_2d` for the pin.
The equilibrium config CANNOT gate the pin -- on a quiescent base the prescribed
E_z is zero anyway, so pinning it changes nothing (7.3e-11 pinned vs 1.4e-10
free). It needs the real jet, i.e. a config where something actually flows.

Both gates are pinned to MUSCL (`mhd/mood_force_level=1`) ON PURPOSE: the SD
cascade at beta = 1e-4 carries its own 1.15e-02 of spurious velocity on the
quiescent ambient (MUSCL: 1.40e-10; and pure outflow is worse still at
3.58e-02), which would drown the boundary signal. **That SD error is real and
unexplained** -- it is not the boundary, since the inflow face is the BEST of the
three there.

## 8c. A boundary-fed beam is invisible to a dt taken from the interior

`ha_jet` injects only through the x-min face, so at t = 0 the domain is
quiescent and the CFL condition sees the ambient sound speed of 1.17 -- not the
v_x = 800 about to enter. Measured: dt = 9.99e-04 against a tlim of 1e-03, i.e.
the whole run in ONE step. The boundary is part of the problem, so its signal
speed has to bound the step: `Config::inflow_rho/_p/_vx` hold the prescribed
primitive state and `compute_inflow_dt` folds it in with the same CFL form the
interior uses. It is a no-op wherever no inflow is prescribed.

`mhd_jet` never showed this because its ambient carries B = 141.42, so the fast
speed is 378 from t = 0 and the interior dt was already small. The bug was there
all along; only the hydro problem exposed it. Cap BOTH sites -- the per-step
`ComputeDt()` AND the setup `Dt`, which calls `compute_dt` directly.

## 8d. Gate on the number the LITERATURE says is scheme-robust, not the one you like

Rueda-Ramirez et al. 2023 (arXiv:2303.00374) table 5 runs the Ha jet with eight
limiter/CFL combinations at 1024^2 DoF and reports three ranges. They disagree
wildly about how reproducible they are:

| quantity | spread across their 8 runs |
|---|---|
| p_max | **1.3x** |
| rho_max | 1.7x |
| rho_min | **18.7x** |

So `hydro_ha_jet_2d` gates on p_max and says nothing about rho_min. The paper
explains its own spread as a feedback -- less dissipation lowers rho_min, which
raises the sound speed, which cuts dt, which cuts dissipation again. Bolm et al.
2026 (arXiv:2607.06045) remark 8 generalises it: this benchmark family is
"highly sensitive to minor differences in the numerical setup" because the
calculations mix "very large numbers" with "numbers very close to 0", and they
report asymmetric solutions from symmetric ICs. **Read the paper's own scatter
before treating any single published number as a target.**

Measured, PLM+RK2, converging against their MCL cluster at 1.748e+05:
128^2 -> 1.605e+05, 256^2 -> 1.683e+05, 512^2 -> 1.693e+05. That is 3.2% low on
p_max at a QUARTER of their degrees of freedom.

**And do not invent a check name.** `"mass"` is not a dispatched name; adding it
to a config is the exact mistake rule 7 records, and the hard-fail on unknown
names is what caught it. Mass is not conserved on this problem anyway -- it
enters through the nozzle and leaves through the far boundary.

## 8e. SDFB on the hypersonic jets: two defects, two tiers, two fixes

SDFB4/SDFB8 at deck defaults failed both Ha jets (Mach 80 and 2000) where
MUSCL-Hancock runs clean at every resolution. Neither "more revisions", the NAD
tolerance, neighbourhood or SED, nor a smaller CFL saved them. `SPD_POS_TRACE=1`
(hydro_ader.hpp; per revision, the worst candidate cell's level, flag, stage input
and local Courant number on its own sub-cell widths) plus the `cascade_*` dumps
separated two defects that each live in ONE tier:

- **Negative pressure is made BY the first-order tier.** At the nozzle lips a
  v = 800 beam meets gas at rest; HLLC's acoustic star pressure goes negative in
  that rarefaction and is not clipped. `hydro/fo_riemann=llf` (Rusanov) fixed
  Mach 80 for both schemes. What is left sits at level 2 (55 of 56 cells on
  Mach 2000 SDFB4): Rusanov is positivity-preserving only to a sub-cell CFL of
  1/2, and under `sum` the narrowest sub-cell runs at ~0.67 (p=3) and ~1 (p=7).
  `time/cfl_type=squared` ((p+1)^2, the MHD paper's condition) removes them --
  0 cells -- but did NOT save a run, because of the second defect.
- **The near-vacuum is made by the MUSCL tier, and NAD cannot move it.** Strips
  of rho ~ 1e-5..2e-7 with p ~ 1 beside the beam, sound speed ~ 300 (10x the
  beam), all at levels 0/1: they pass PAD (min_rho 1e-10), and with the default
  cap (a89608b) a NAD flag stops at MUSCL. They set dt: 85k steps instead of 2.5k,
  or a stall. `hydro/mood_pad_first_order=false` (NAD may reach first order) took
  rho_min to 2-5e-2 (MUSCL-Hancock: 1-3e-2) and the step count back to
  1.3-2.0x MUSCL-Hancock's, at 0.4-2% of cells on first order.

The paper configuration is therefore Rusanov + NAD-to-first-order, 3 revisions,
`sum` CFL: every SDFB4/SDFB8 lane at 1N, 2N, 4N and under AMR completes both
jets; on Mach 2000 a handful of p < 0 cells remain (1-2 per SDFB4 lane, 9-26
per SDFB8 lane) along the beam edges, at the lip and on the bow shock. Six
revisions take Mach 2000 at 1N to 0 (SDFB4) and 1 (SDFB8, p = -7e-3) at the same
step count: the last revision's demotions are what is committed unchecked. A PAD density floor (`fallback/min_rho=1e-3`)
also rescued 1N and failed at 2N/4N -- it treats the symptom in the wrong tier.
**Ask which tier owns a bad cell before tuning the detector:** the `cascade`
dump at the bad cell answers it in one line.

The jets also run on the block path now: `x1_bc=inflow` (problem = ha_jet) and
`outflow` faces go through the forest/table exchange (`Mesh::wall_bc`,
`jet_face_to_ghost`, mode 4 of `apply_domain_bc_fv`) and the Mesh caps dt with
the inflow state (`Mesh::inflow_dt`, rule 8c). Gate: `hydro_ha_jet_mb_2d`
(4x4 blocks = the single-block golden, 0.0); controls
`hydro_ha_jet_mb_ambient_sensitive_2d` and `hydro_ha_jet_mb_noinflow_sensitive_2d`;
`hydro_ha_jet_amr_2d`; and `hydro_fo_riemann_llf_sensitive_2d` /
`hydro_cfl_squared_sensitive_2d` prove the two knobs are connected.

## 8f. Passive scalars: rows after the energy, a tail loop, and a diagnostic that must read the conserved average

`hydro/nscalars = n` (5 Oct 2026, for the shock-cloud flagship) carries `rho*s` as conserved rows `NVAR..NVAR+n-1`;
the primitive row is the concentration. Three things to keep:

- **The scalar count is a COMPILE-TIME template parameter `NS` of every hydro kernel** (`hydro.cpp`; `NSCAL_DISPATCH`
  picks the instantiation from `cfg.nscal`, up to `NSCAL_MAX` = 2). Arrays are `[NVAR+NS]`, loops run to `NVAR+NS`, so
  the `NS = 0` instantiation is the Euler code character for character. The first version carried a RUNTIME count
  (`[NVAR_MAX]` arrays, a tail loop `NVAR <= var < NVAR+ns`): bit-identical on all 70 non-scalar suite configurations
  on CPU, and on the A100 **35 of 43 hydro lanes differed from 620d9db by round-off and the 3D Sedov lane ran 24 per
  cent slower with no scalar at all** (13.4 -> 16.7 s per 150 steps) -- the runtime-bounded tail pushed the kernels'
  local arrays out of the registers, which also moved the FMA contraction. **A CPU A/B says nothing about GPU code
  generation: run the default-path A/B and a timing on the GPU before calling a kernel change free.** A new hydro
  kernel takes `NS` the same way, or the scalar silently stops being advected on that path.
- **The scalar flux is the mass flux times the upwind concentration on every path** (pointwise `rho s v` at the SD
  flux points, LLF, HLLC on the contact like the transverse momenta, MUSCL reconstructing the concentration). The
  gate is that a uniform concentration stays uniform to round-off on a shocked, mixed-level run (`scalar_uniform`,
  measured 8e-15 with two levels and the cascade live) and that its total equals the mass. It is not a bound on a
  non-uniform scalar: on the blast-with-blob lane the concentration reaches 1.04 behind a 13:1 contact with SDFB4 and
  1.03 with MUSCL-Hancock at p=0 -- `rho s` and `rho` carry different errors there, and their ratio shows it.
- **At the first output `W_cv` is the cell average of the initial PRIMITIVES**, so `rho_cv * s_cv` is not the average
  of `rho s`: the scalar total read 3.7e-4 off at t=0 and looked like a drift. `fv_scalar_mass` integrates the
  conserved cell average recomputed from `U_sp`. Mass never showed this because density is linear.

Detection: the scalar rows join NAD/SED with an ABSOLUTE band (`fallback/scalar_tolerance`), since a relative band has
no width at zero. It is what holds a discontinuous scalar (blob: excursion 1e-4/6e-3 against 9e-2/1.3e-1 without),
and it is not free on smooth flow, because a flagged scalar demotes the whole cell: density L1 on the smooth sine at
16^2, p=3, went 7.9e-6 -> 1.3e-5. `fallback/NAD_scalars=false` takes them out.

## 9. Remote runs (apollo)

- `rsync` **`inputs/` and `tests/` as well as `src/`** — a stale `inputs/` once
  invalidated a whole GPU campaign via `cfl=0.8`.
- Verify the remote binary's provenance BEFORE launching, and record it: md5 the
  source, check the binary mtime against the run's first dump. A paper-scale
  reference once ran on a binary that was rebuilt mid-run.
- **Never rebuild a binary a production run is using** -- but not for the reason
  this rule used to give. "Linux refuses to write a running executable, so the
  link fails" is WRONG for `cmake --build`: `ld` creates a fresh inode
  (unlink + create), so the link succeeds, the running process keeps the old
  image and finishes correctly. Measured: a paper-scale run was rebuilt under it
  mid-flight and its remaining dumps still matched the reference bit for bit.
  ETXTBSY only appears when something writes the path IN PLACE (`cp`, `install`).
  The real hazard is PROVENANCE: after the link, the binary at that path is no
  longer the one the run started with, and this project has already published a
  reference that "ran on a binary rebuilt mid-run". Use a second tree, and record
  the src md5 at launch.
- **A tree copied WITH its build directory builds the ORIGINAL tree's sources.**
  `cp -a ~/spd_K-deref ~/spd_K-edge` then `cmake --build ~/spd_K-edge/build-cuda`
  compiled `~/spd_K-deref/src` -- the copied CMakeCache and Makefiles hold that
  tree's absolute paths (`spd_K_SOURCE_DIR`, `CMAKE_HOME_DIRECTORY`) -- and
  reported success. The source md5 of the new tree matched the laptop; the binary
  did not contain the change, and a GPU smoke test "failed" for a reason that had
  nothing to do with the code. Configure a copied tree fresh (`rm -rf build-cuda`,
  full configure line below) and check `spd_K_SOURCE_DIR` in its cache before
  trusting anything it builds. Kokkos lives at
  `FETCHCONTENT_SOURCE_DIR_KOKKOS=~/spd_K-blast/build-cuda/_deps/kokkos-src`.
- **`rsync -a` keeps the LAPTOP's mtime, so a resync can be OLDER than the remote object and make skips it.**
  1 Oct 2026: `main.cpp` was fixed locally at 16:09:43 while the first remote build was still running; that
  build compiled the OLD file at 16:10:00, the resync landed afterwards with mtime 16:09:43, and the rebuild
  compiled nothing -- same binary md5 before and after, and a whole campaign launched on the unfixed code
  (it showed up as old-vs-new differing by 1e-13). After any resync into a tree that has built, delete the
  objects (`rm -f build-cuda/CMakeFiles/spd_K.dir/src/*.o`) and CHECK that the binary md5 changed; then
  run one A/B on the remote binary that the change must move before launching anything long.
- `nvcc` is not on `PATH` over non-interactive ssh: `export PATH=/usr/local/cuda/bin:$PATH`,
  or cmake fails with a bogus `string sub-command REPLACE` error from Kokkos.
- **The full working apollo configure line** (the default toolchain does not
  work, in two separate ways, and both errors point somewhere unhelpful):

  ```
  export PATH=/usr/local/cuda/bin:$PATH
  cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-12/root/usr/bin/g++ \
        -DCMAKE_CXX_FLAGS=-I/opt/sns/mpich-3.3/include \
        -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON
  ```

  The default `g++` is 8.5 and Kokkos rejects it with "CMake wants to use
  -std=c++2a which is not supported by NVCC", which reads like an nvcc problem
  and is a host-compiler one. `find_package(MPI)` then fails its compile check
  against gcc-toolset-12, but `spd_k.hpp` only needs `mpi.h` (every MPI CALL is
  behind `#ifdef MPI`), so the include path alone is enough -- do not chase the
  MPI detection. Nodes are A100 (AMPERE80); the `h200` partition would need a
  separate HOPPER90 build.
- Apollo has SLURM (`-p apollo --gres=gpu:1`, 8 GPUs/node, 7-day limit) AND two
  idle A100s on the login node. Use the queue for anything long; the login GPUs
  are for probes and short runs.
- Incremental builds go stale on `structs.hpp`/`define.hpp`: delete the objects
  (`rm -f build/CMakeFiles/spd_K.dir/src/*.o`), not the `.dir`.
- Dumps go to `$SPD_OUTPUT_DIR`, defaulting to `<cwd>/output/`.

## 9b. CFL limits depend on p AND on time/cfl_type, and only an exact solution proves one

`time/cfl` defaults to 0.4, which is the p=3 limit **under the default
`cfl_type=sum`**. Both halves of that sentence matter. Measured on the
unperturbed Harris current sheet (`problem=current_sheet`, `problem/p1=0`), which
is an EXACT stationary solution, so any motion is the scheme going unstable
(64 DoF in y, t=0.5, round-off ~6e-13):

| cfl | p=3 min | p=3 sum | p=7 min | p=7 sum |
|---|---|---|---|---|
| 0.50 | 2.6e-01 | 6.3e-13 | collapse | 1.3e-01 |
| 0.40 | 1.3e-02 | 6.3e-13 | collapse | 3.0e-14 |
| 0.30 | 6.3e-13 | 6.3e-13 | collapse | 2.5e-14 |
| 0.25 | 6.3e-13 | 6.3e-13 | 1.3e-01 | 3.0e-14 |
| 0.20 | 6.3e-13 | 6.4e-13 | 1.9e-14 | 4.0e-14 |

p=3 needs <= 0.30 under `min` and holds 0.5 under `sum`; p=7 needs <= 0.20 under
`min` and <= 0.40 under `sum`.

**Use a problem whose exact answer you know.** An earlier probe ran Orszag-Tang
and asked "did dt collapse?", and reported p=7 stable at 0.30 under `min`. That
is wrong by a factor of 1.5: the run was quietly unstable long before the guard
fired. On the equilibrium the same configuration shows 1e-01 of spurious
velocity where the answer is exactly zero. A stability limit measured by
absence-of-catastrophe is not measured.

**And say which cfl_type a limit was measured under.** `min` gives a ~1.79x
larger dt than `sum` at the same nominal cfl (measured on the figure-22 KH lane),
so a limit quoted without the convention is off by that factor. This bit the
MDZ21 decks: they were written with `cfl_type=min, cfl=0.4`, which is past the
p=3 limit, and the current-sheet equilibrium grew 1.3e-02 of velocity out of
nothing until the convention was fixed.

## 10. Kokkos policy

`RangePolicy` or `TeamPolicy` only — never `MDRangePolicy`. Never
`Kokkos::CudaUVMSpace` (see `.cursor/rules/kokkos-no-uvm.mdc`). `nvcc` cannot
generate its stub for a `__device__` lambda inside a member function that also
takes a pointer-to-member parameter: it compiles on the host build and fails only
under CUDA.
