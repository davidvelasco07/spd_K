#!/usr/bin/env python3
"""Regression test suite for spd_K.

One binary, configured at runtime through input files (inputs/*.athinput)
plus command-line overrides. Checks per configuration:

  hydro_sd_3d    : L1 error of rho vs the analytic advected sine wave,
                   mass conservation to round-off, golden file (loose rtol)
  hydro_sd_2d    : L1 error vs analytic (must equal the 3D value: the
                   problem has no z-dependence), mass conservation
  hydro_sd_1d    : L1 error vs analytic, mass conservation
  hydro_fv_3d/2d : mass conservation to round-off (the fallback flux
                   replacement is exactly conservative for periodic BCs)
  hydro_fv_blast : 2d blast with a real shock; detector fires, fallback
                   active, mass still conserved to round-off
  hydro_muscl_2d : job/scheme=vl2 (MUSCL-Hancock; blend pinned to 1 on every
                   face) on a
                   smooth periodic wave; the low-order reference lane for the
                   figure-21 runs, conservative to round-off
  *_rk3          : same checks with the SSP-RK3 integrator (per-stage
                   fallback correction; temporal error below the spatial
                   floor at p=3, so the ADER L1 limit applies unchanged)
  *_mb           : multi-block (meshblock) runs; the per-cell arithmetic is
                   identical to single-block, so the stitched active region
                   must match the single-block golden (ghosts excluded: the
                   stitched output array has no evolved ghost data)
  induction_fv_3d: golden file comparison of B2_cv (linear problem, no
                   chaotic amplification, so a tight tolerance is portable)
  hydro_smr_2d   : static centre patch; the mesh must stay mixed-level
  hydro_amr_2d   : dynamic AMR on a pulse; the mesh must become mixed-level
  hydro_amr_2level_2d : same with two refinement levels
  hydro_amr_muscl_2d : dynamic AMR at p=0; the only cover for the
                   limited-linear prolongation (p>=1 uses the amr_P matrix)
  hydro_implosion_2d : reflective-wall implosion, mass conserved; golden
  hydro_implosion_mb_2d : the same in 4x4 blocks -- reflecting walls on the
                   block path (apply_domain_bc_fp/fv); the stitched active
                   region must match the single-block golden
  hydro_implosion_amr_2d : the same with dynamic AMR (walls + level jumps)
  hydro_wc_blast_1d_mb : Woodward-Colella interacting blasts in 8 blocks with
                   walls, then with 3 levels of AMR; mass conserved to round-off
  hydro_dmr_amr_2d : double Mach reflection, the `doublemach` boundary on the
                   block path with dynamic AMR; finite and mixed-level
  mhd_*          : Orszag-Tang / field-loop MHD goldens + divB checks
  mhd_*_smr_2d   : true-2D MHD static refinement (mixed levels + divB)
  mhd_*_amr_2d   : true-2D MHD dynamic AMR (face-B transfer + mixed levels)
  mhd_*_smr_fb_* : mixed-level MHD with the MOOD cascade live, gated on
                   coarse-fine magnetic-flux telescoping (cf_flux) -- the thing
                   divb cannot see, since divb is per-block and any within-block
                   single-valued EMF passes it
  mhd_kh_*_p0_2d : p=0 MHD Kelvin-Helmholtz (Stone+2020 fig 22) on a static
                   patch and under the paper's dynamic shear criterion -- the
                   low-order PLM lane, which every other mixed-level MHD config
                   misses because they are all p=3
  mhd_orszag_tang_smr_noemf_2d : the same with the coarse-fine edge-EMF
                   correction switched off. It must FAIL to telescope
                   (cf_flux_sensitive), which is what keeps the cf_flux gate
                   above from silently becoming vacuous
  mhd_orszag_tang_smr_nocorner_2d : the same with only the PATCH-CORNER spread
                   switched off (SPD_NO_EMF_CORNER=1). The coarse-fine drift
                   stays clean there and the SAME-LEVEL control goes to 2.2e-04,
                   so it is the negative control for the second half of
                   check_cf_flux the way noemf_2d is for the first

Every configuration is additionally gated on all dumps being finite, before
any tolerance is applied.

Usage:
  tests/run_tests.py [--build-dir DIR] [--skip-unit] [--skip-golden]
                     [--regen-goldens] [--only SUBSTR]
"""
import argparse
import glob
import math
import os
import re
import shutil
import subprocess
import time
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spdk_io  # shared reader module
from spdk_io import NGH, nGH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(ROOT, "tests", "goldens")

N, P = 8, 3
n = P + 1

# Components per hydro solution array: rho, vx, vy, vz, e (NVAR in define.hpp).
# Configurations with a different layout (induction) set "nvar" themselves.
NVAR = 5

CONFIGS = {
    "hydro_sd_3d": {
        "input": "inputs/sine_wave.athinput",
        "overrides": [],
        "ndim": 3,
        # GHOSTS EXCLUDED: update_solution now advances active elements only, so
        # the ghost ring of a pure-SD lane (no FV halo to refill it) holds its
        # initial values instead of integrated garbage. Measured over the whole
        # suite when that landed: 97 W_cv dumps, 6 changed in raw bytes -- every
        # one a pure-SD lane -- and 0 changed in the ACTIVE region, all exactly
        # 0.000e+00. Comparing the active region is what this check was always
        # meant to assert.
        "checks": ["analytic", "mass_strict", "golden_active"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.1,
        "l1_limit": 3.0e-5,    # measured 2.16e-5 (N=8, p=3, t=0.1)
        "golden_name": "hydro_sd",
        "golden_rtol": 1e-6,   # round-off seeds amplify ~2x/step in this flow
    },
    "hydro_sd_2d": {
        "input": "inputs/sine_wave.athinput",
        "overrides": ["mesh/nx3=1"],
        "ndim": 2,
        "checks": ["analytic", "mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.1,
        "l1_limit": 3.0e-5,
    },
    "hydro_sd_1d": {
        "input": "inputs/sine_wave.athinput",
        "overrides": ["mesh/nx2=1", "mesh/nx3=1"],
        "ndim": 1,
        "checks": ["analytic", "mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.1,
        "l1_limit": 2.0e-5,    # measured 1.41e-5
    },
    "hydro_fv_3d": {
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/fallback=true", "output/dt=0.05"],
        "ndim": 3,
        "checks": ["mass_strict"],
        "field": "W_cv_N8p3_2_0.dat",
        "t_end": 0.1,
    },
    "hydro_fv_2d": {
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/fallback=true", "mesh/nx3=1"],
        "ndim": 2,
        "checks": ["mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.1,
    },
    "hydro_fv_blast_2d": {
        # real shock: detector must fire, fallback active, still conservative
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/fallback=true", "mesh/nx3=1",
                      "problem/problem=spherical_blast",
                      "hydro/gamma=1.6666666666667",
                      "time/tlim=0.02", "output/dt=0.02"],
        "ndim": 2,
        "checks": ["mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.02,
    },
    "hydro_cascade_2d": {
        # MOOD cascade on a smooth wave. The detector still fires on ~0.6% of
        # cells here (round-off-level NAD violations that survive SED), and a
        # discrete cascade gives those cells no partial credit: the face takes
        # the MUSCL flux outright, where the fractional blend would weight it.
        # So the L1 error is about twice the blend/SD value (measured 4.36e-05
        # against 2.16e-05) -- a property of the scheme, not a defect. The
        # limit is set to catch a real regression, not to assert SD accuracy.
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/fallback=true", "mesh/nx3=1",
                      "fallback/style=cascade"],
        "ndim": 2,
        "checks": ["analytic", "mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.1,
        "l1_limit": 6.0e-5,
    },
    "hydro_cascade_blast_2d": {
        # Real shock: cells demote to levels 1 and 2, and the assembled
        # single-valued face flux must still conserve mass to round-off.
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/fallback=true", "mesh/nx3=1",
                      "problem/problem=spherical_blast",
                      "hydro/gamma=1.6666666666667",
                      "time/tlim=0.02", "output/dt=0.02",
                      "fallback/style=cascade"],
        "ndim": 2,
        "checks": ["mass_strict", "golden"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.02,
        "golden_name": "hydro_cascade_blast",
        "golden_rtol": 1e-6,
    },
    "hydro_cascade_blast_mb_2d": {
        # The same MOOD blast on a 4x4 uniform forest. A uniform forest is the
        # same discretisation as one block -- every interface is same-level, so
        # nothing about the cascade may depend on where the block edges fall --
        # and it must reproduce the single-block golden.
        #
        # This is the check that pins block-interface handling in the cascade.
        # The cascade selects one flux per face from the pooled levels, and the
        # level is a per-cell decision, so without a single-valued face flux
        # (coarse-fine restriction plus same-level symmetrization) the two sides
        # of a block edge can disagree and the run silently diverges from the
        # single-block answer. spd covers this with
        # test_mood_amr_matches_uniform_forest; spd_K had no equivalent.
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/fallback=true", "mesh/nx3=1",
                      "meshblock/nx1=2", "meshblock/nx2=2",
                      "problem/problem=spherical_blast",
                      "hydro/gamma=1.6666666666667",
                      "time/tlim=0.02", "output/dt=0.02",
                      "fallback/style=cascade"],
        "ndim": 2,
        "checks": ["mass_strict", "golden_active"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.02,
        "golden_name": "hydro_cascade_blast",
        "golden_rtol": 1e-6,
    },
    "hydro_sod_1d": {
        # gradfree outflow reference. Mass is not conserved by construction
        # (it leaves through both ends), so this is a golden comparison only.
        "input": "inputs/sod.athinput",
        "overrides": [],
        "ndim": 1,
        "checks": ["golden"],
        "field": "W_cv_N32p3_2_0.dat",
        "t_end": 0.2,
        "golden_name": "hydro_sod",
        "golden_rtol": 1e-6,
    },
    "hydro_sod_1d_mb_forest": {
        # Four blocks with SPD_NO_PACK=1, which forces the per-block forest
        # exchange instead of the packed fast path. That is the only route
        # that reaches apply_domain_bc_fv, and before it existed the FV ghost
        # slab at the gradfree wall was never filled: this case diverged on
        # step 1. The active region must match the single-block golden.
        "input": "inputs/sod.athinput",
        "overrides": ["meshblock/nx1=8"],
        "env": {"SPD_NO_PACK": "1"},
        "ndim": 1,
        "checks": ["golden_active"],
        "field": "W_cv_N32p3_2_0.dat",
        "t_end": 0.2,
        "golden_name": "hydro_sod",
        "golden_rtol": 1e-6,
    },
    "hydro_muscl_2d": {
        # job/scheme=vl2 (MUSCL-Hancock) pins the blend to 1 on every face, which is
        # a different path from the hydro_fv_* tests: those run SD and only
        # reach a MUSCL flux where the detector fires. This is the low-order
        # reference lane for the figure-21 runs. It drifted ~3e-06 when the
        # lane was first added; it is exact now (0.0 here, 2.2e-16 multiblock),
        # so it is a real check rather than a known failure.
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/scheme=vl2", "mesh/p=0",
                      "mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=1",
                      "meshblock/nx1=32", "meshblock/nx2=32", "meshblock/nx3=1",
                      "output/dt=0.05"],
        "ndim": 2,
        "checks": ["mass_strict"],
        "field": "W_cv_N32p0_1_0.dat",
        "t_end": 0.1,
    },
    "hydro_muscl_2d_mb": {
        # same scheme across meshblock boundaries; the per-face flux must not
        # depend on which block computes it.
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/scheme=vl2", "mesh/p=0",
                      "mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=1",
                      "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=1",
                      "output/dt=0.05"],
        "ndim": 2,
        "checks": ["mass_strict"],
        "field": "W_cv_N32p0_1_0.dat",
        "t_end": 0.1,
    },
    "hydro_sd_3d_rk3": {
        "input": "inputs/sine_wave.athinput",
        "overrides": ["time/integrator=rk3"],
        "ndim": 3,
        "checks": ["analytic", "mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.1,
        "l1_limit": 3.0e-5,    # measured 2.19e-5 (temporal error negligible)
    },
    "hydro_fv_blast_2d_rk3": {
        # shock + fallback + RK: per-stage blended fluxes stay conservative
        "input": "inputs/sine_wave.athinput",
        "overrides": ["time/integrator=rk3", "job/fallback=true",
                      "mesh/nx3=1", "problem/problem=spherical_blast",
                      "hydro/gamma=1.6666666666667",
                      "time/tlim=0.02", "output/dt=0.02"],
        "ndim": 2,
        "checks": ["mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.02,
    },
    "hydro_sd_3d_mb": {
        # 2x2x2 blocks of 4^3 elements: must reproduce the single-block
        # solution (block exchanges feed the same operands to the same
        # kernels, so the active region matches to the golden tolerance)
        "input": "inputs/sine_wave.athinput",
        "overrides": ["meshblock/nx1=4", "meshblock/nx2=4", "meshblock/nx3=4"],
        "ndim": 3,
        "checks": ["analytic", "mass_strict", "golden_active"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.1,
        "l1_limit": 3.0e-5,
        "golden_name": "hydro_sd",
        "golden_rtol": 1e-6,
    },
    "hydro_fv_blast_2d_mb": {
        # shock + fallback across block boundaries (4x2 blocks): the blended
        # fluxes on both sides of every block face are identical, so mass is
        # still conserved to round-off
        "input": "inputs/sine_wave.athinput",
        "overrides": ["job/fallback=true", "mesh/nx3=1",
                      "problem/problem=spherical_blast",
                      "hydro/gamma=1.6666666666667",
                      "time/tlim=0.02", "output/dt=0.02",
                      "meshblock/nx1=2", "meshblock/nx2=4"],
        "ndim": 2,
        "checks": ["mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.02,
    },
    "hydro_fv_blast_2d_rk3_mb": {
        # RK stages + per-stage fallback + block exchanges combined
        "input": "inputs/sine_wave.athinput",
        "overrides": ["time/integrator=rk3", "job/fallback=true",
                      "mesh/nx3=1", "problem/problem=spherical_blast",
                      "hydro/gamma=1.6666666666667",
                      "time/tlim=0.02", "output/dt=0.02",
                      "meshblock/nx1=4", "meshblock/nx2=2"],
        "ndim": 2,
        "checks": ["mass_strict"],
        "field": "W_cv_N8p3_1_0.dat",
        "t_end": 0.02,
    },
    "hydro_implosion_2d": {
        # reflective BCs: the flux-form update must conserve to round-off
        # (zero mass flux through the mirrored walls)
        "input": "inputs/implosion.athinput",
        "overrides": ["time/tlim=0.1", "output/dt=0.05"],
        "ndim": 2,
        "checks": ["mass_strict", "golden"],
        "golden_name": "hydro_implosion",
        "golden_rtol": 1e-6,
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.1,
    },
    "hydro_implosion_mb_2d": {
        # Reflecting walls on the block path. The mesh path used to refuse any
        # non-periodic, non-gradfree boundary (its uniform neighbour tables
        # wrapped a wall around the domain: 56% error while conserving mass).
        # Walls now go through the forest/table exchange, whose physical
        # boundary blocks are filled by apply_domain_bc_fp/fv with the same
        # mirror rule boundary.cpp uses, so 4x4 blocks must reproduce the
        # single-block golden in the active region.
        "input": "inputs/implosion.athinput",
        "overrides": ["time/tlim=0.1", "output/dt=0.05",
                      "meshblock/nx1=8", "meshblock/nx2=8"],
        # No golden here: with the cascade live the single-block and the
        # block path differ by threshold flips on this problem (166 cells by
        # step 420 in 1D even with GRADFREE walls, i.e. it is not the wall),
        # see CLAUDE.md 7b. The exact wall gate is the MUSCL pair below.
        "ndim": 2,
        "checks": ["mass_strict"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.1,
    },
    "hydro_implosion_muscl_2d": {
        # Reference for the wall gate: MUSCL has no detector, so a single
        # block and 4x4 blocks must agree to the last bit.
        "input": "inputs/implosion.athinput",
        "overrides": ["time/tlim=0.1", "output/dt=0.05", "mesh/p=0",
                      "job/scheme=vl2", "time/integrator=rk1"],
        "ndim": 2,
        "checks": ["mass_strict", "golden"],
        "field": "W_cv_N32p0_1_0.dat",
        "t_end": 0.1,
        "golden_name": "hydro_implosion_muscl",
        "golden_rtol": 1e-12,
    },
    "hydro_implosion_muscl_mb_2d": {
        # THE reflecting-wall gate on the block path: bit-identical to the
        # single-block golden (rtol 1e-12) with walls on all four sides.
        "input": "inputs/implosion.athinput",
        "overrides": ["time/tlim=0.1", "output/dt=0.05", "mesh/p=0",
                      "job/scheme=vl2", "time/integrator=rk1",
                      "meshblock/nx1=8", "meshblock/nx2=8"],
        "ndim": 2,
        "checks": ["mass_strict", "golden_active"],
        "field": "W_cv_N32p0_1_0.dat",
        "t_end": 0.1,
        "golden_name": "hydro_implosion_muscl",
        "golden_rtol": 1e-12,
    },
    "hydro_implosion_amr_2d": {
        # Walls plus level jumps: dynamic AMR on the implosion, cascade live.
        "input": "inputs/implosion.athinput",
        "overrides": ["time/tlim=0.1", "output/dt=0.05",
                      "meshblock/nx1=8", "meshblock/nx2=8", "fallback/style=cascade",
                      "time/integrator=rk3",
                      "amr/max_level=1", "amr/adapt_interval=3", "amr/criterion=pressure",
                      "amr/refine_threshold=0.1", "amr/derefine_threshold=0.025",
                      "amr/initial_refine=true"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict"],
        "field": "W_cv_N32p3_2_0.dat",
        "t_end": 0.1,
    },
    "hydro_implosion_lohner_edge_2d": {
        # The Lohner score's block-edge stencil (CLAUDE.md 7a4): SDFB4 blocks of
        # TWO elements per side (8^2 DoF) must refine. The interior-only score
        # skips the block-edge elements, so such a block scored exactly 0 and the
        # 8^2-DoF implosion lanes ran on their root with a clean log. Negative
        # control: SPD_LOHNER_INTERIOR=1 (the old score) refuses this config at
        # start-up, so a regression that drops the edge stencil turns this red.
        "input": "inputs/implosion.athinput",
        "overrides": ["mesh/p=3", "fallback/style=cascade", "time/integrator=rk3",
                      "mesh/nx1=8", "mesh/nx2=8", "meshblock/nx1=2", "meshblock/nx2=2",
                      "time/tlim=0.02", "output/dt=0.01",
                      "amr/max_level=1", "amr/adapt_interval=3", "amr/criterion=lohner",
                      "amr/refine_threshold=0.12", "amr/derefine_threshold=0.03",
                      "amr/initial_refine=true"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict"],
        "field": "W_cv_N8p3_2_0.dat",
        "t_end": 0.02,
    },
    "hydro_wc_blast_1d_mb": {
        # Woodward-Colella interacting blast waves: reflecting walls on a 1D
        # forest of 8 blocks (the first wall test in 1D on the block path).
        "input": "inputs/woodward_colella.athinput",
        "overrides": [],
        "ndim": 1,
        "checks": ["mass_strict"],
        "field": "W_cv_N64p3_1_0.dat",
        "t_end": 0.038,
    },
    "hydro_sedov_amr_2d": {
        # 2D Sedov-Taylor point explosion with the energy given as problem/p0
        # (E = 1 in R = 0.025, ambient p1 = 1e-3; r_s(0.05) = 0.258 for gamma =
        # 5/3), two levels of dynamic AMR above a 32^2 root at p = 0. The deposit
        # radius must exceed the coarsest root's half-diagonal cell distance
        # (0.022 here) or no root cell sees it and the run is a uniform state
        # that finishes in one step -- which is exactly what R = 0.02 did.
        "input": "inputs/sedov.athinput",
        "overrides": ["mesh/p=0", "job/scheme=vl2", "time/integrator=rk1",
                      "mesh/nx1=32", "mesh/nx2=32", "meshblock/nx1=8", "meshblock/nx2=8",
                      "amr/max_level=2", "amr/adapt_interval=20", "amr/criterion=pressure",
                      "amr/refine_threshold=0.03", "amr/derefine_threshold=0.0075",
                      "amr/initial_refine=true", "problem/p0=1", "problem/p1=1e-3",
                      "problem/radius=0.025", "time/tlim=0.05", "output/dt=0.025"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict"],
        "field": "W_cv_N128p0_2_0.dat",
        "t_end": 0.05,
    },
    "hydro_wc_blast_1d_amr": {
        # The same with three levels of dynamic AMR above a 16-element root.
        # The density is uniform at t=0, so the Lohner density indicator cannot
        # tag anything before the first step: refine on the pressure gradient.
        "input": "inputs/woodward_colella.athinput",
        "overrides": ["mesh/nx1=16", "meshblock/nx1=4", "amr/max_level=3",
                      "amr/adapt_interval=3", "amr/criterion=pressure",
                      "amr/refine_threshold=0.1", "amr/derefine_threshold=0.025",
                      "amr/initial_refine=true"],
        "ndim": 1,
        "checks": ["mixed_levels", "mass_strict"],
        "field": "W_cv_N16p3_1_0.dat",
        "t_end": 0.038,
    },
    "hydro_dmr_amr_2d": {
        # Double Mach reflection with the `doublemach` boundary type on the
        # block path (dmr.hpp: post-shock inflow left and on the bottom for
        # x < 1/6, a reflecting wall beyond, the exact moving shock on top,
        # outflow right) under dynamic AMR. Two things this has caught already:
        # the top boundary evaluated the shock at the BLOCK height instead of
        # the domain height on refined blocks (dimension::L is the block
        # length there), and the first exchange after a regrid read another
        # block's stale ghost rows through the limited-linear fill. Finite and
        # mixed-level; no golden (cascade live, CLAUDE.md 7b).
        "input": "inputs/dmr.athinput",
        "overrides": ["mesh/nx1=32", "mesh/nx2=8", "amr/max_level=1",
                      "time/tlim=0.01", "output/dt=0.01"],
        "ndim": 2,
        "checks": ["mixed_levels"],
        "field": "W_cv_N64p3_1_0.dat",
        "t_end": 0.01,
    },
    "induction_fv_3d": {
        "input": "inputs/induction_loop.athinput",
        "overrides": [],
        "ndim": 3,
        "checks": ["golden"],
        "field": "B2_cv_N8p3_2_0.dat",
        "t_end": 0.2,
        "golden_name": "induction_fv",
        "golden_rtol": 1e-8,   # linear problem: cross-compiler safe
    },
    "hydro_smr_2d": {
        # static refinement via <refinement1>. 4x4 base blocks with only the
        # centre tagged, so a coarse rim survives and the run exercises real
        # coarse-fine interfaces (12 coarse + 16 fine blocks).
        "input": "inputs/sine_wave.athinput",
        "overrides": ["mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=1",
                      "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "amr/max_level=1",
                      "refinement1/level=1", "refinement1/x1min=0.375",
                      "refinement1/x1max=0.625", "refinement1/x2min=0.375",
                      "refinement1/x2max=0.625"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.1,
    },
    "hydro_smr_fallback_2d": {
        # static refinement with the FV fallback active. Isolates the
        # coarse-fine handling of the assembled cascade fluxes from regridding:
        # the mesh never changes, so any drift is the interface, not the
        # transfer. AMR requires fallback/style=cascade (see main.cpp).
        "input": "inputs/amr_pulse.athinput",
        "overrides": ["fallback/style=cascade", "amr/adapt_interval=0",
                      "refinement1/level=1", "refinement1/x1min=0.375",
                      "refinement1/x1max=0.625", "refinement1/x2min=0.375",
                      "refinement1/x2max=0.625"],
        # Runs on the table exchange (SPD_NEW_XCHG=1). The cascade does up to
        # max_revs revisions per step, each with its own halo and a full
        # detect, and on the per-block forest path that is a launch explosion:
        # this config alone ran >80 min unfinished. The table path is verified
        # bit-identical to the forest path at 1 and 2 levels on both backends,
        # so this costs no coverage of the physics -- but it does mean the
        # forest path is no longer exercised here.
        "env": {"SPD_NEW_XCHG": "1"},
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.1,
    },
    "hydro_amr_2d": {
        # dynamic AMR on a Gaussian pulse
        "input": "inputs/amr_pulse.athinput",
        "overrides": ["fallback/style=cascade"],
        # Runs on the table exchange (SPD_NEW_XCHG=1). The cascade does up to
        # max_revs revisions per step, each with its own halo and a full
        # detect, and on the per-block forest path that is a launch explosion:
        # this config alone ran >80 min unfinished. The table path is verified
        # bit-identical to the forest path at 1 and 2 levels on both backends,
        # so this costs no coverage of the physics -- but it does mean the
        # forest path is no longer exercised here.
        "env": {"SPD_NEW_XCHG": "1"},
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.1,
    },
    "hydro_amr_muscl_2d": {
        # Dynamic AMR at p = 0 (job/scheme=vl2). This is the only config that
        # exercises prolongate_block_lim: every other AMR test runs at p = 3 and
        # so takes the amr_P matrix path. At p = 0 amr_P degenerates to
        # piecewise-constant injection, and the limited-linear reconstruction
        # that replaces it reads the coarse block's NEIGHBOURS -- a new coupling
        # the matrix path does not have, and one that has to stay exactly
        # conservative through a regrid. mass_strict is the sharp test of that;
        # the minmod limiter also has to leave the pulse admissible.
        "input": "inputs/amr_pulse.athinput",
        "overrides": ["mesh/p=0", "job/scheme=vl2"],
        "env": {"SPD_NEW_XCHG": "1"},
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict"],
        "field": "W_cv_N32p0_1_0.dat",
        "t_end": 0.1,
    },
    "hydro_amr_2level_2d": {
        # Two refinement levels, which is what the figure-21 Kelvin-Helmholtz
        # runs use. This drifted ~1.4e-10 while the FV coarse-fine ghost fill
        # ignored the sub-face index: the two sides of a level jump then read
        # each other at the wrong transverse offset, so they blended the
        # fallback flux differently and the interface correction could not
        # balance it. Round-off since amr_boundary.cpp got the transverse
        # mapping right.
        "input": "inputs/amr_pulse.athinput",
        "overrides": ["fallback/style=cascade", "amr/max_level=2"],
        # Runs on the table exchange (SPD_NEW_XCHG=1). The cascade does up to
        # max_revs revisions per step, each with its own halo and a full
        # detect, and on the per-block forest path that is a launch explosion:
        # this config alone ran >80 min unfinished. The table path is verified
        # bit-identical to the forest path at 1 and 2 levels on both backends,
        # so this costs no coverage of the physics -- but it does mean the
        # forest path is no longer exercised here.
        "env": {"SPD_NEW_XCHG": "1"},
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict"],
        "field": "W_cv_N64p3_1_0.dat",
        "t_end": 0.1,
    },
    "mhd_orszag_tang_2d": {
        # quasi-2D OT vortex with the MOOD cascade active: shocks form by
        # t=0.15, the |B| NAD + face/edge cascade must keep the run stable,
        # conservative, and divergence-free
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=true", "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=4", "time/tlim=0.15", "output/dt=0.15"],
        "ndim": 3,
        "checks": ["mass_strict", "divb", "golden"],
        "field": "W_cv_N16p3_1_0.dat",
        "t_end": 0.15,
        "golden_name": "mhd_ot",
        "golden_rtol": 1e-6,
    },
    "mhd_field_loop_2d": {
        # smooth weak-field loop advection (high order everywhere): tests the
        # SD MHD + CT path without the fallback
        "input": "inputs/field_loop.athinput",
        "overrides": ["mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=4",
                      "time/tlim=0.1", "output/dt=0.1"],
        "ndim": 3,
        "checks": ["mass_strict", "divb", "golden_active"],
        # GHOSTS EXCLUDED, and this is the whole story of the "GPU/CPU
        # divergence" this lane carried since 2026-08-14. Measured 2026-08-19,
        # CPU vs GPU at t=0.1: the raw-file comparison reproduces the reported
        # failure exactly, and ALL of it is in the ghost ring --
        #     true2d  raw 2.205e-02 | interior 9.10e-15 | ghosts 4.46e-02
        #     3D      raw 7.901e-02 | interior 9.55e-15 | ghosts 1.60e-01
        # -- with B agreeing to 1.5e-17 on both. update_solution runs over
        # sd_for_cells, i.e. it time-advances ghost ELEMENTS too, using whatever
        # sits in their flux slots; the pure-SD path (fallback=false) never
        # refills that ring, so it keeps deterministic-but-meaningless values
        # that differ between backends. Deterministic: two GPU runs are
        # md5-identical. The same config with job/fallback=true agrees to
        # 1.8e-14 INCLUDING ghosts, because the FV halo does refill them.
        # Comparing the active region is strictly stronger here: it resolves a
        # 1e-14 change where the raw check was masked by 1e-2 of ghost noise.
        "nvar": 8,
        "field": "W_cv_N16p3_1_0.dat",
        "t_end": 0.1,
        "golden_name": "mhd_loop",
        "golden_rtol": 1e-6,
    },
    "mhd_orszag_tang_true2d": {
        # true 2D (mesh/nx3=1): CT degenerates to the Ez edge family, Bz is a
        # cell-centered conserved variable; MOOD cascade active through the
        # early shocks. Conservation and divB = dBx/dx + dBy/dy at round-off.
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=true", "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "time/tlim=0.15", "output/dt=0.15"],
        "ndim": 2,
        "checks": ["mass_strict", "divb", "golden"],
        "field": "W_cv_N16p3_1_0.dat",
        "t_end": 0.15,
        "golden_name": "mhd_ot_2d",
        "golden_rtol": 1e-6,
    },
    "mhd_blast3d_mdz_3d": {
        # MDZ21 section 6.4 / Balsara & Spicer: the strongly magnetized 3D blast,
        # at 32^3 DoF rather than the paper's 192^3 so it fits a smoke suite.
        #
        # WHAT IT GUARDS. This is the only 3D MHD config in the suite, and the
        # only one exercising the oblique-field blast IC (direction (1,1,0) via
        # problem/v1..v3) and gradfree boundaries on all three axes. beta ~
        # 2.5e-4 in the ambient medium; MDZ21 states plainly that on this test
        # "no scheme preserves energy positivity without energy correction, not
        # even with a minmod limiter", and applies E <- E - (B_c^2 - B_f^2)/2
        # after every step. spd_K has NO such correction -- the MOOD cascade is
        # its robustness mechanism instead -- so the fact that this run finishes
        # at all is the thing being checked. Pin it: if a change to the cascade
        # makes this go non-finite, that is the regression.
        #
        # The deck sets mhd/energy_fix=1 (the stage-boundary correction). That
        # is the point of this config as a guard: without it 15% of the domain
        # at 192^3 ends on the pressure floor, and a change that breaks the
        # correction should turn this red rather than quietly degrade.
        #
        # NOT a golden config. Cell values here are NOT reproducible across
        # backends: with the cascade live, a round-off difference flips a
        # troubled-cell threshold and that flips one cell's scheme by O(1).
        # Measured CPU vs A100 at this resolution: 2.6e-14 at t=0 growing to
        # 2.1e-01 pointwise within 12 steps, while the volume-integrated
        # magnetic energy still agreed to 4.1e-06 and mass to 1.1e-10. Pinning
        # the cascade (mhd/mood_force_level=1) removes the decisions and the two
        # backends then agree to 2.8e-14 -- which is how that spread was shown
        # to be flag chaos and not a GPU defect. So the checks here are
        # conservation and finiteness, which survive it; a dump comparison
        # would not.
        "input": "inputs/mdz21/blast3d.athinput",
        "overrides": ["mesh/nx1=8", "mesh/nx2=8", "mesh/nx3=8",
                      "time/tlim=0.002", "output/dt=0.001"],
        "ndim": 3,
        "nvar": 8,
        # check_finite runs unconditionally for every config, so it is not
        # listed. Do NOT extend this config's tlim without revisiting the mass
        # gate: at tlim=0.002 the fast front has travelled ~0.09 of the 0.5
        # half-width and is nowhere near the wall, but at the paper's t=0.01 it
        # reaches the boundary and mass legitimately leaves the domain.
        "checks": ["mass_strict", "divb"],
        # mass_limit, not the 1e-12 default: the gradfree boundaries leak.
        # MEASURED, by running this exact IC with periodic walls instead --
        # the interior update conserves to 4.7e-15, i.e. round-off, while the
        # outflow version drifts 1.3e-10. So this limit bounds BOUNDARY
        # extrapolation, not the scheme, and 1e-9 still fails on any real
        # conservation defect (those run 1e-3 and up).
        "mass_limit": 1e-9,
        # divb is an ABSOLUTE gate defaulting to 1e-11, which is calibrated for
        # the suite's B ~ O(1) problems. Here B0 = 28.2 and the sub-cell dx is
        # 0.031, so |divB| ~ 2.1e-11 measured is 2.3e-14 RELATIVE to B0/dx --
        # round-off, and the same quality of CT the 2D configs show. 1e-10
        # leaves ~5x headroom on that floor while staying four orders of
        # magnitude below what a real CT defect produces (a broken coarse-fine
        # or wall EMF gives 1e-2 and up), so the gate can still fail.
        "divb_limit": 1e-10,
        "field": "W_cv_N8p3_2_0.dat",
        "t_end": 0.002,
    },
    # ------------------------------------------------------------------
    # Balsara et al. 2025 section 8.2 / Wu & Shu (2018) Mach-800 magnetized
    # jet. Four configs: two gates and the negative control each one needs.
    #
    # There was NO jet configuration in this suite, which is why a test that
    # had never once worked went unnoticed through several rounds of MHD work.
    # Both gates are cheap (a few seconds) because they are pinned to MUSCL:
    # mhd/mood_force_level=1. That is deliberate -- the SD cascade at
    # beta = 1e-4 carries its own 1e-02 of error on the quiescent ambient
    # (measured: 1.15e-02 SD vs 1.40e-10 MUSCL, and pure outflow is worse
    # still at 3.58e-02), so an SD lane would drown the boundary signal these
    # gates are for. The SD error is a real and separate issue; see
    # CLAUDE.md 7c on reading pressure off a swapped B at beta << 1.
    # ------------------------------------------------------------------
    # Ha et al. hypersonic jet -- the PURE HYDRO counterpart to the MHD jet
    # above, and the only jet in this suite that can pass or fail against
    # PUBLISHED NUMBERS. Balsara's MHD jet figure cannot: read through its own
    # printed colourbar, its undisturbed ambient comes out at rho = 0.010 where
    # the stated IC fixes 0.14. Rueda-Ramirez et al. 2023 (arXiv:2303.00374)
    # table 5 gives rho and p ranges instead of only a picture.
    #
    # Being unmagnetized, it also separates "does spd_K do hypersonic jets"
    # from "does spd_K do beta = 1e-4" -- two questions the MHD jet conflates.
    "hydro_ha_jet_2d": {
        "input": "inputs/rr23/ha_jet.athinput",
        # 128^2 DoF and MUSCL keep this at ~9 s. The gate is p_max, which is
        # the scheme-ROBUST number in RR23's own table (1.3x spread across
        # eight limiter/CFL combinations, against 18.7x for rho_min).
        "overrides": ["mesh/nx1=32", "mesh/nx2=32",
                      "job/scheme=plm", "time/integrator=rk2",
                      "output/dt=0.0005"],
        "ndim": 2,
        "nvar": 5,
        # No mass check: mass is NOT conserved here by construction -- it enters
        # through the nozzle and leaves through the far boundary.
        "checks": ["ha_jet"],
        "pmax_lo": 1.3e5,       # measured 1.605e+05 at this resolution
        "pmax_hi": 2.3e5,       # RR23 report 1.726e+05-2.282e+05 at 4x the DoF
        "sym_tol": 0.5,         # measured 0.000; the MHD jet's SDFB4 lane is 21
        "field": "W_cv_N32p3_2_0.dat",
        "t_end": 0.001,
    },
    "hydro_ha_jet_noinflow_sensitive_2d": {
        # Negative control for hydro_ha_jet_2d: the same deck with the nozzle
        # switched off (x1_bc = outflow). Nothing enters, the quiescent gas is
        # an exact stationary solution, and p_max MUST stay at its initial
        # 0.4127 -- measured exactly that, with rho exactly 0.5 everywhere.
        # If this goes red the inflow BC is firing when disabled; if the paired
        # gate ever stops depending on the inflow, this is what notices.
        "input": "inputs/rr23/ha_jet.athinput",
        "overrides": ["mesh/nx1=32", "mesh/nx2=32", "mesh/x1_bc=outflow",
                      "job/scheme=plm", "time/integrator=rk2",
                      "output/dt=0.0005"],
        "ndim": 2,
        "nvar": 5,
        "checks": ["ha_jet_sensitive"],
        "pmax_ceiling": 1.0,    # the IC is 0.4127
        "field": "W_cv_N32p3_2_0.dat",
        "t_end": 0.001,
    },
    "mhd_jet_base_equilibrium_2d": {
        # The jet deck with the injection turned OFF (problem/d1 = d0 and
        # problem/v2 = 0), so the nozzle prescribes the ambient state and the
        # quiescent ambient is an EXACT stationary solution: every bit of
        # velocity is boundary error, with no physics to hide behind.
        #
        # This gates the CHOICE of what the base does outside the nozzle.
        # Wu & Shu specify outflow there; a reflecting wall is catastrophic
        # because B is NORMAL to this base (see the paired control).
        "input": "inputs/balsara/jet_wushu.athinput",
        "overrides": ["problem/d1=0.14", "problem/v2=0.0",
                      "mesh/nx1=8", "mesh/nx2=12",
                      "mhd/mood_force_level=1",
                      "time/tlim=0.004", "output/dt=0.002"],
        "ndim": 2,
        "nvar": 8,
        "checks": ["mass_strict", "divb", "static_equilibrium"],
        "equil_tol": 1e-8,          # measured 1.40e-10
        # divb is an ABSOLUTE gate and the default 1e-11 is calibrated for
        # B0 ~ 1 (CLAUDE.md rule 7). This problem has B0 = 141.42 and a
        # sub-cell dx of 0.03125, so divB scales as B0/dx: the measured
        # 5.14e-10 is 1.14e-13 RELATIVE, against the 3.0e-13 the existing
        # 2D configs encode at 1e-11 with B0 ~ 1, dx ~ 0.03. The CT is
        # BETTER here than in those, not worse -- only the units differ.
        # 3.0e-13 * B0/dx = 1.36e-09, rounded to 2e-09 (3.9x headroom).
        "divb_limit": 2e-9,         # measured 5.14e-10
        "field": "W_cv_N8p3_2_0.dat",
        "t_end": 0.004,
    },
    "mhd_jet_base_wall_sensitive_2d": {
        # Negative control for mhd_jet_base_equilibrium_2d. The same exact
        # equilibrium with the base closed as a reflecting wall, which flips
        # the NORMAL magnetic row -- the conducting-wall condition, valid only
        # where B.n = 0. Here B.n = 141.42 on that face, so the flip puts a
        # jump of 2*141.42 in the normal field into the y-face Riemann problem.
        # The exact solution must then fall apart: measured max |v| = 2.16e+00
        # against 1.40e-10, and at 100x150 DoF dt collapses to 1e-6 of dt0.
        # If this goes GREEN the paired gate is measuring nothing.
        "input": "inputs/balsara/jet_wushu.athinput",
        "overrides": ["problem/d1=0.14", "problem/v2=0.0",
                      "mesh/nx1=8", "mesh/nx2=12",
                      "mhd/mood_force_level=1", "mesh/x2_bc=reflective",
                      "time/tlim=0.004", "output/dt=0.002"],
        "ndim": 2,
        "nvar": 8,
        "checks": ["equilibrium_sensitive"],
        "equil_floor": 1e-3,        # measured 2.16e+00
        "field": "W_cv_N8p3_2_0.dat",
        "t_end": 0.004,
    },
    "mhd_jet_inlet_field_2d": {
        # The REAL jet, run to the paper's t = 0.002 at 48x72 DoF. Gates the
        # inlet EMF pin, which the equilibrium config above cannot see: on a
        # quiescent base the prescribed E_z is zero anyway, so pinning it
        # changes nothing there (measured 7.3e-11 pinned vs 1.4e-10 free).
        # It only bites once there is flow.
        "input": "inputs/balsara/jet_wushu.athinput",
        "overrides": ["mesh/nx1=12", "mesh/nx2=18",
                      "mhd/mood_force_level=1", "output/dt=0.001"],
        "ndim": 2,
        "nvar": 8,
        "checks": ["divb", "inlet_field"],
        # Same B0/dx rescaling as mhd_jet_base_equilibrium_2d, with
        # dx = 0.02083 here: the measured 9.30e-10 is 1.37e-13 RELATIVE.
        "divb_limit": 3e-9,         # measured 9.30e-10
        "inlet_b2_limit": 5e5,      # measured 4.72e+04, at y-row frac 0.806
        "field": "W_cv_N12p3_2_0.dat",
        "t_end": 0.002,
    },
    "mhd_jet_inlet_unpinned_sensitive_2d": {
        # Negative control for mhd_jet_inlet_field_2d: the identical run with
        # SPD_NO_BC_EMF_PIN=1. The inlet field must then be destroyed --
        # measured max |B|^2 = 1.28e+06 ON the inlet row, against 4.72e+04 at
        # the bow shock with the pin, and the base re-entry that follows takes
        # the mean v_y outside the nozzle from 2.44 to 283.77.
        "input": "inputs/balsara/jet_wushu.athinput",
        "overrides": ["mesh/nx1=12", "mesh/nx2=18",
                      "mhd/mood_force_level=1", "output/dt=0.001"],
        "env": {"SPD_NO_BC_EMF_PIN": "1"},
        "ndim": 2,
        "nvar": 8,
        "checks": ["inlet_field_sensitive"],
        "inlet_b2_floor": 3e5,      # measured 1.28e+06
        "field": "W_cv_N12p3_2_0.dat",
        "t_end": 0.002,
    },
    "mhd_current_sheet_equilibrium_2d": {
        # MDZ21 section 6.2 with the perturbation OFF (problem/p1=0), so the
        # Harris sheet is an EXACT stationary solution and any motion is error.
        # sigma=10 widens the sheet so it is fully resolved -- the balance holds
        # for every width, so this isolates the scheme and the walls from the
        # under-resolved-sheet dissipation the paper actually studies.
        #
        # This is the ONLY config exercising MHD reflecting walls (x2_bc), which
        # need the normal B row flipped (boundary.cpp) and the wall-tangential
        # EMF pinned to zero (mhd_zero_wall_emf) or the equilibrium walks off.
        # It is also the sharpest CFL gate in the suite: at cfl_type=min and
        # cfl=0.4 this same setup grows 1.3e-02 of velocity out of nothing.
        "input": "inputs/mdz21/current_sheet.athinput",
        # 64 DoF in y, not 32: at 32 the paired cfl_sensitive control below is
        # still STABLE under min/0.4 and the control goes vacuous. The
        # instability is resolution dependent, so the gate and its control have
        # to sit where it actually bites.
        "overrides": ["problem/p1=0.0", "problem/sigma=10.0",
                      "mesh/nx1=32", "mesh/nx2=16",
                      "time/tlim=0.5", "output/dt=0.25"],
        "ndim": 2,
        "nvar": 8,
        "checks": ["mass_strict", "divb", "static_equilibrium"],
        "equil_tol": 1e-10,
        "field": "W_cv_N32p3_2_0.dat",
        "t_end": 0.5,
    },
    "mhd_current_sheet_cfl_sensitive_2d": {
        # Negative control for mhd_current_sheet_equilibrium_2d. Identical setup
        # at cfl_type=min, cfl=0.4, which is past the p=3 stability limit: the
        # exact equilibrium must then visibly fall apart. If this goes GREEN the
        # paired gate is measuring nothing.
        "input": "inputs/mdz21/current_sheet.athinput",
        "overrides": ["problem/p1=0.0", "problem/sigma=10.0",
                      "mesh/nx1=32", "mesh/nx2=16",
                      "time/cfl_type=min", "time/cfl=0.4",
                      "time/tlim=0.5", "output/dt=0.25"],
        "ndim": 2,
        "nvar": 8,
        "checks": ["equilibrium_sensitive"],
        "equil_floor": 1e-6,
        "field": "W_cv_N32p3_2_0.dat",
        "t_end": 0.5,
    },
    "mhd_orszag_tang_uct_hlld_2d": {
        # UCT-HLLD (Mignone & Del Zanna 2021 eq. 33 + 44/45): the upwind CT emf
        # composed from the face solver's five-wave fan, replacing BOTH 1-D edge
        # sweeps. Same setup as mhd_orszag_tang_true2d so the only difference is
        # mhd/emf, which is what makes the golden_differs control below mean
        # something. Single block: the coefficients are not haloed yet, so
        # main.cpp refuses a meshblock/AMR run rather than emit a
        # decomposition-dependent number.
        "input": "inputs/orszag_tang.athinput",
        # The LIVE cascade, so both the SD edge UCT and the demoted-corner UCT
        # are exercised. mass_strict is the gate that matters here: the corner
        # UCT reads the face fields transversally, and until mood_halo_face_B
        # was wired in it inherited unfilled ghosts and drifted 2.9e-06.
        "overrides": ["job/fallback=true", "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "mhd/rsolver=hlld", "mhd/emf=uct",
                      "time/tlim=0.15", "output/dt=0.15"],
        "ndim": 2,
        "checks": ["mass_strict", "divb", "golden_differs"],
        # The reference is the TWO-SWEEP golden, and this config requires the
        # result to differ from it: see check_golden_differs.
        "golden_name": "mhd_ot_2d",
        "differs_floor": 1e-8,
        "field": "W_cv_N16p3_1_0.dat",
        "t_end": 0.15,
    },
    "mhd_orszag_tang_uct_hll_2d": {
        # UCT-HLL: the two-wave fan (MDZ21 eq. 28/32) feeding the same
        # composition. The paper's most diffusive emf, and the one it reports
        # failing to form the central O-point at low resolution -- so it must be
        # a DIFFERENT answer from UCT-HLLD, not just different from 2sweep.
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=true", "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "mhd/rsolver=hll", "mhd/emf=uct",
                      "time/tlim=0.15", "output/dt=0.15"],
        "ndim": 2,
        "checks": ["mass_strict", "divb", "golden_differs"],
        "golden_name": "mhd_ot_2d",
        "differs_floor": 1e-8,
        "field": "W_cv_N16p3_1_0.dat",
        "t_end": 0.15,
    },
    "mhd_orszag_tang_uct_muscl_2d": {
        # The MUSCL+RK2 lane of the MDZ21 base-scheme comparison: job/scheme=plm
        # pins every cell at cascade level 1, so the edge EMF is built ENTIRELY
        # by mhd_uct_corner_E. This is the configuration the corner-UCT
        # conservation defect was found on (2.869e-06 against a 1e-12 limit),
        # and the one that proves the transverse face-B halo is in place.
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=true", "job/scheme=plm",
                      "time/integrator=rk2", "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "mhd/rsolver=hlld", "mhd/emf=uct",
                      "time/tlim=0.15", "output/dt=0.15"],
        "ndim": 2,
        "checks": ["mass_strict", "divb", "golden_differs"],
        "golden_name": "mhd_ot_2d",
        "differs_floor": 1e-8,
        "field": "W_cv_N16p3_1_0.dat",
        "t_end": 0.15,
    },
    "mhd_orszag_tang_true2d_mb": {
        # 2x2 uniform multiblock OT: same-level exchange of face B, edge EMF,
        # and the MOOD cascade must reproduce the single-block golden in the
        # active region (ghosts may differ) with round-off divB.
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=true", "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "meshblock/nx1=8", "meshblock/nx2=8",
                      "time/tlim=0.15", "output/dt=0.15"],
        "ndim": 2,
        "nvar": 8,
        "checks": ["mass_strict", "divb", "golden_active"],
        "field": "W_cv_N16p3_1_0.dat",
        "t_end": 0.15,
        "golden_name": "mhd_ot_2d",
        "golden_rtol": 1e-6,
    },
    "mhd_field_loop_true2d": {
        # true 2D field-loop advection without the fallback: pure SD + Ez CT
        "input": "inputs/field_loop.athinput",
        "overrides": ["mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=1",
                      "time/tlim=0.1", "output/dt=0.1"],
        "ndim": 2,
        "checks": ["mass_strict", "divb", "golden_active"],
        # GHOSTS EXCLUDED, and this is the whole story of the "GPU/CPU
        # divergence" this lane carried since 2026-08-14. Measured 2026-08-19,
        # CPU vs GPU at t=0.1: the raw-file comparison reproduces the reported
        # failure exactly, and ALL of it is in the ghost ring --
        #     true2d  raw 2.205e-02 | interior 9.10e-15 | ghosts 4.46e-02
        #     3D      raw 7.901e-02 | interior 9.55e-15 | ghosts 1.60e-01
        # -- with B agreeing to 1.5e-17 on both. update_solution runs over
        # sd_for_cells, i.e. it time-advances ghost ELEMENTS too, using whatever
        # sits in their flux slots; the pure-SD path (fallback=false) never
        # refills that ring, so it keeps deterministic-but-meaningless values
        # that differ between backends. Deterministic: two GPU runs are
        # md5-identical. The same config with job/fallback=true agrees to
        # 1.8e-14 INCLUDING ghosts, because the FV halo does refill them.
        # Comparing the active region is strictly stronger here: it resolves a
        # 1e-14 change where the raw check was masked by 1e-2 of ghost noise.
        "nvar": 8,
        "field": "W_cv_N16p3_1_0.dat",
        "t_end": 0.1,
        "golden_name": "mhd_loop_2d",
        "golden_rtol": 1e-6,
    },
    "mhd_field_loop_smr_2d": {
        # Statically refined patch; field loop crosses the coarse-fine boundary.
        # Pure SD CT + EMF correction (short tlim: longer advection still drifts).
        # cf_flux GATED as of the amr_RF_fp fix + the SD corner spread: this lane
        # went 8.5e-04 -> 1.37e-18, same-level control 1.64e-03 -> exactly 0.
        "input": "inputs/field_loop.athinput",
        "overrides": ["job/fallback=false", "mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=1",
                      "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "time/tlim=0.02", "output/dt=0.01",
                      "amr/max_level=1", "amr/adapt_interval=0",
                      "refinement1/level=1", "refinement1/x1min=0.375",
                      "refinement1/x1max=0.625", "refinement1/x2min=0.375",
                      "refinement1/x2max=0.625"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb", "cf_flux"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.02,
    },
    "mhd_orszag_tang_smr_2d": {
        # Short OT with static refinement (no fallback): mass + divb + mixed levels
        # + cf_flux. GATED AS OF THE amr_RF_fp FIX: this SD lane used to drift
        # 2.54e-03 on every coarse-fine interface (same-level 1.63e-03), which is
        # why it was left ungated -- the note in the handoff said quarantining it
        # "would turn 4 green configs red". Restricting each coarse fp node from
        # the fine half that CONTAINS it, plus the SD patch-corner spread, takes it
        # to 1.39e-17 with the same-level control at exactly 0.
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=false", "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "time/tlim=0.01", "output/dt=0.005",
                      "amr/max_level=1", "amr/adapt_interval=0",
                      "refinement1/level=1", "refinement1/x1min=0.375",
                      "refinement1/x1max=0.625", "refinement1/x2min=0.375",
                      "refinement1/x2max=0.625"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb", "cf_flux"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.01,
    },
    "mhd_field_loop_smr_fb_2d": {
        # Same static patch as mhd_field_loop_smr_2d but with the MOOD cascade
        # LIVE. Nothing covered mixed-level MHD + fallback before: every smr/amr
        # MHD config ran job/fallback=false, which is why the coarse-fine FV flux
        # and edge-EMF corrections (steps 4-5) had no test at all.
        "input": "inputs/field_loop.athinput",
        "overrides": ["job/fallback=true", "mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=1",
                      "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "time/tlim=0.02", "output/dt=0.01",
                      "amr/max_level=1", "amr/adapt_interval=0",
                      "refinement1/level=1", "refinement1/x1min=0.375",
                      "refinement1/x1max=0.625", "refinement1/x2min=0.375",
                      "refinement1/x2max=0.625"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb", "cf_flux"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.02,
    },
    "mhd_orszag_tang_smr_fb_2d": {
        # OT on a static patch with the cascade live and detection running.
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=true", "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "time/tlim=0.01", "output/dt=0.005",
                      "amr/max_level=1", "amr/adapt_interval=0",
                      "refinement1/level=1", "refinement1/x1min=0.375",
                      "refinement1/x1max=0.625", "refinement1/x2min=0.375",
                      "refinement1/x2max=0.625"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb", "cf_flux"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.01,
    },
    "mhd_orszag_tang_smr_fb_lvl1_2d": {
        # Same, pinned at MOOD level 1 (pure MUSCL on the sub-cell mesh, no
        # detection). This is the lane that isolates the coarse-fine corrections
        # from the detector: any difference here is a communication or
        # restriction bug, not a detection one.
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=true", "mhd/mood_force_level=1",
                      "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "time/tlim=0.01", "output/dt=0.005",
                      "amr/max_level=1", "amr/adapt_interval=0",
                      "refinement1/level=1", "refinement1/x1min=0.375",
                      "refinement1/x1max=0.625", "refinement1/x2min=0.375",
                      "refinement1/x2max=0.625"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb", "cf_flux"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.01,
    },
    "mhd_orszag_tang_smr_noemf_2d": {
        # THE NEGATIVE CONTROL for the cf_flux gate, and the reason to trust it.
        # Identical to mhd_orszag_tang_smr_fb_lvl1_2d with the coarse-fine
        # edge-EMF correction switched off, which is what the cascade did before
        # step 5. It must FAIL to telescope: cf_flux_sensitive requires the drift
        # to be large, so a change that quietly makes the paired gate measure
        # nothing turns this red instead of leaving both green.
        #
        # Measured 2026-08-19: correction on 1.4e-17, off 2.2e-04. Note divb is
        # 1.58e-12 either way -- it cannot see this, which is why the earlier
        # attempts to gate step 5 with divb passed for the wrong reason.
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=true", "mhd/mood_force_level=1",
                      "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "time/tlim=0.01", "output/dt=0.005",
                      "amr/max_level=1", "amr/adapt_interval=0",
                      "refinement1/level=1", "refinement1/x1min=0.375",
                      "refinement1/x1max=0.625", "refinement1/x2min=0.375",
                      "refinement1/x2max=0.625"],
        "env": {"SPD_NO_FV_EMF": "1"},
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb", "cf_flux_sensitive"],
        "cf_floor": 1e-6,
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.01,
    },
    "mhd_orszag_tang_smr_nocorner_2d": {
        # NEGATIVE CONTROL for the same-level half of check_cf_flux, added with
        # the patch-corner spread. Identical to mhd_orszag_tang_smr_fb_lvl1_2d
        # with ONLY the corner spread off, so the coarse-fine drift stays at
        # 1.4e-17 while the same-level control returns to what it was before the
        # spread existed: the corner point of the patch is multi-valued in the
        # coarse block diagonal to it, and the two coarse-coarse faces that
        # terminate there stop telescoping.
        #
        # Measured 2026-08-19: spread on 1.39e-17, off 2.21e-04, with divb at
        # 1.59e-12 in BOTH -- the same blindness that made divb useless for
        # gating step 5 makes it useless here.
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=true", "mhd/mood_force_level=1",
                      "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "time/tlim=0.01", "output/dt=0.005",
                      "amr/max_level=1", "amr/adapt_interval=0",
                      "refinement1/level=1", "refinement1/x1min=0.375",
                      "refinement1/x1max=0.625", "refinement1/x2min=0.375",
                      "refinement1/x2max=0.625"],
        "env": {"SPD_NO_EMF_CORNER": "1"},
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb", "sl_flux_sensitive"],
        "sl_floor": 1e-6,
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.01,
    },
    "mhd_orszag_tang_amr_fb_2d": {
        # Dynamic AMR with the cascade live: regrid + face-B transfer + the
        # coarse-fine corrections all in one lane.
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=true", "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "time/tlim=0.01", "output/dt=0.005",
                      "amr/max_level=1", "amr/adapt_interval=2",
                      "amr/criterion=bfield", "amr/refine_frac=1.0"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.01,
    },
    "mhd_orszag_tang_amr_2d": {
        # Dynamic AMR on OT. At 4x4 base blocks the B^2 Löhner scores agree to
        # 0.2% across the mesh (OT is symmetric at t=0), so no threshold can
        # separate them; refine_frac=1 tags the peak block instead and 2:1
        # balance grows a compact fine patch, giving the mixed levels this test
        # is about. Face-B prolongate + div-free projection keeps divB at
        # round-off. Very short tlim without the fallback, since pure SD CT plus
        # regrid can diverge once shocks form.
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=false", "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "time/tlim=0.01", "output/dt=0.005",
                      "amr/max_level=1", "amr/adapt_interval=2",
                      "amr/criterion=bfield", "amr/refine_frac=1.0"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.01,
    },
    "mhd_kh_smr_p0_2d": {
        # p=0 mixed-level MHD with the cascade: the lane figure 22 actually uses,
        # and the one every other mixed-level MHD config missed -- they are all
        # p=3. At p=0 one element IS one cell, so the FV halo (2) exceeds the SD
        # ghost supply (1) and the coarse-fine transfer runs on a different code
        # path than at p>=1. Static patch so cf_flux has a stable reference.
        #
        # Measured 2026-08-19: CF flux drift 1.0e-20 and the same-level control
        # EXACTLY 0 over all 136 faces -- the patch-corner residual that p=3
        # carries (2.2e-04) is absent at p=0.
        "input": "inputs/kelvin_helmholtz_mhd.athinput",
        "overrides": ["mesh/nx1=32", "mesh/nx2=32",
                      "meshblock/nx1=8", "meshblock/nx2=8",
                      "time/tlim=0.05", "output/dt=0.025",
                      "amr/max_level=1", "amr/adapt_interval=0",
                      "refinement1/level=1", "refinement1/x2min=0.13",
                      "refinement1/x2max=0.37"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb", "cf_flux"],
        "field": "W_cv_N64p0_1_0.dat",
        "t_end": 0.05,
    },
    "mhd_kh_amr_p0_2d": {
        # The figure-22 lane end to end at test scale: p=0 PLM (cascade pinned at
        # the MUSCL level), rk2, dynamic AMR driven by the paper's shear criterion
        # at its 0.01 / 0.005 cuts. 8 block rows so the y = 0.25 / 0.75 shear
        # layers do NOT land on a row boundary everywhere -- with 4 rows they do,
        # every row gets tagged, and the mesh refines uniformly, which tests
        # nothing about mixed levels.
        #
        # No cf_flux here: a regrid invalidates the t=0 reference by design, so
        # the drift is only meaningful on the static lane above.
        "input": "inputs/kelvin_helmholtz_mhd.athinput",
        "overrides": ["mesh/nx1=32", "mesh/nx2=32",
                      "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/tlim=0.05", "output/dt=0.025",
                      "amr/max_level=1", "amr/adapt_interval=2",
                      "amr/criterion=shear", "amr/refine_threshold=0.01",
                      "amr/derefine_threshold=0.005"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb"],
        "field": "W_cv_N64p0_1_0.dat",
        "t_end": 0.05,
    },
}


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return r


def run(build_dir, outdir, cfg):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    env = dict(os.environ, SPD_OUTPUT_DIR=outdir)
    # "env" in a config selects a code path that no input-file option reaches,
    # e.g. SPD_NO_PACK=1 to force the per-block forest exchange on a uniform
    # mesh (which the packed fast path would otherwise handle).
    env.update(cfg.get("env", {}))
    cmd = [os.path.join(build_dir, "spd_K"), "-i",
           os.path.join(ROOT, cfg["input"])] + cfg["overrides"]
    return sh(cmd, env=env).stdout


def n_from_field(field):
    import re
    m = re.search(r"N(\d+)p", field)
    return int(m.group(1)) if m else N


def shape_for(n_cells, ndim, nvar=NVAR):
    """Element/point shape of the SD output arrays for a given ndim and N."""
    Ne = lambda active, nc: nc + 2 * NGH if active else 1
    np_ = lambda active: n if active else 1
    a = [True, ndim >= 2, ndim >= 3]  # x, y, z activity
    nx_c = n_cells if a[0] else 1
    ny_c = n_cells if a[1] else 1
    nz_c = n_cells if a[2] else 1
    return (1, nvar, Ne(a[2], nz_c), Ne(a[1], ny_c), Ne(a[0], nx_c),
            np_(a[2]), np_(a[1]), np_(a[0]))


def shape(ndim, nvar=NVAR):
    return shape_for(N, ndim, nvar)


def load_rho_cells(outdir, cfg):
    """rho on the active-region global cell grid, shape (nz_c, ny_c, nx_c)."""
    ndim = cfg["ndim"]
    nc = n_from_field(cfg["field"])
    A = np.fromfile(os.path.join(outdir, cfg["field"])).reshape(shape_for(nc, ndim))
    rho = A[0, 0]
    sl = lambda active: slice(NGH, -NGH) if active else slice(None)
    rho = rho[sl(ndim >= 3), sl(ndim >= 2), sl(True)]
    Nz_, Ny_, Nx_, nz_, ny_, nx_ = rho.shape[0], rho.shape[1], rho.shape[2], \
                                    rho.shape[3], rho.shape[4], rho.shape[5]
    return rho.transpose(0, 3, 1, 4, 2, 5).reshape(Nz_ * nz_, Ny_ * ny_, Nx_ * nx_)


def active_faces(outdir):
    xf = np.fromfile(os.path.join(outdir, f"X_N{N}p{P}_0.dat"))
    return xf[nGH:len(xf) - nGH]


def cell_widths(outdir, cfg=None):
    nc_elem = n_from_field(cfg["field"]) if cfg and "field" in cfg else N
    xf_path = os.path.join(outdir, f"X_N{nc_elem}p{P}_0.dat")
    if os.path.isfile(xf_path):
        xf = np.fromfile(xf_path)
        return np.diff(xf[nGH:len(xf) - nGH])
    #Unit-box AMR stitched outputs: N in the filename is the element count.
    ncells = nc_elem * n
    return np.full(ncells, 1.0 / ncells)


def total_mass(outdir, fname, cfg):
    cfg2 = dict(cfg, field=fname)
    rho = load_rho_cells(outdir, cfg2)
    w = cell_widths(outdir, cfg2)
    V = np.ones(rho.shape)
    V *= w[None, None, :]
    if cfg["ndim"] >= 2:
        V *= w[None, :, None]
    if cfg["ndim"] >= 3:
        V *= w[:, None, None]
    return float((rho * V).sum())


def check_analytic(outdir, cfg):
    """L1 error of rho cell averages vs the exact advected sine wave.
    IC: rho = 1 + 0.125 sin(2 pi (x+y)), v = (1,1,0); y = 0 in 1d."""
    xf = active_faces(outdir)
    a = 2 * np.pi
    t = cfg["t_end"]
    rho = load_rho_cells(outdir, cfg)
    if cfg["ndim"] == 1:
        F = lambda x: -np.cos(a * (x - t)) / a
        exact = 1.0 + 0.125 * (F(xf[1:]) - F(xf[:-1])) / np.diff(xf)
        err = np.abs(rho[0, 0] - exact).mean()
    else:
        F = lambda x, y: -np.sin(a * (x + y - 2 * t)) / a**2
        X0, Y0 = np.meshgrid(xf[:-1], xf[:-1], indexing="ij")
        X1, Y1 = np.meshgrid(xf[1:], xf[1:], indexing="ij")
        exact = 1.0 + 0.125 * (F(X1, Y1) - F(X0, Y1) - F(X1, Y0) + F(X0, Y0)) \
                / ((X1 - X0) * (Y1 - Y0))
        err = np.abs(rho - exact.T[None, :, :]).mean()
    limit = cfg["l1_limit"]
    return err < limit, f"L1(rho) vs analytic = {err:.3e} (limit {limit:.1e})"


def output_index(path):
    m = re.search(r"_(\d+)_\d+\.dat$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def sorted_outputs(outdir):
    """W_cv dumps in output order. Under AMR the element count in the file
    name changes with the mesh, so lexicographic order is not time order."""
    return sorted(glob.glob(os.path.join(outdir, "W_cv_N*p*_0.dat")),
                  key=output_index)


def field_dumps(outdir):
    """Every cell-average field dump, whichever variable a config writes."""
    return sorted(glob.glob(os.path.join(outdir, "*_cv_N*p*_*.dat")),
                  key=lambda f: (os.path.basename(f).split("_cv_")[0],
                                 output_index(f)))


def check_finite(outdir, cfg):
    """No NaN/Inf anywhere in any dump. Runs unconditionally for every config:
    NaN silently satisfies the comparisons the other checks are built on
    (max() keeps the finite operand, `nan < tol` is False), so a run that
    produces NaN must be rejected before any tolerance is consulted."""
    outs = field_dumps(outdir)
    if not outs:
        return False, "no field output files found"
    bad = []
    for f in outs:
        A = np.fromfile(f)
        n_bad = int((~np.isfinite(A)).sum())
        if n_bad:
            bad.append(f"{os.path.basename(f)} {100.0 * n_bad / A.size:.1f}%")
    if bad:
        return False, "non-finite output: " + ", ".join(bad)
    return True, f"{len(outs)} outputs finite"


def check_mass(outdir, cfg, limit):
    """Mass drift: prefer mass.txt (AMR/mesh exact integral); else field dumps."""
    path = os.path.join(outdir, "mass.txt")
    if os.path.exists(path):
        m = [float(tok[1]) for tok in
             (line.split() for line in open(path)) if len(tok) == 2]
        if len(m) < 2:
            return False, f"mass.txt has {len(m)} entries, need at least 2"
        if not all(math.isfinite(x) for x in m):
            return False, f"non-finite mass: {m}"
        if m[0] == 0.0:
            return False, "reference mass is zero"
    else:
        grid = spdk_io.Grid(outdir)
        m = [spdk_io.total_mass(grid, i) for i in spdk_io.output_indices(outdir)]
        if len(m) < 2:
            return False, f"need >=2 outputs for mass check, got {len(m)}"
    drift = max(abs(x - m[0]) / abs(m[0]) for x in m)
    if not math.isfinite(drift):
        return False, f"non-finite mass drift (masses {m})"
    return drift < limit, f"mass drift = {drift:.3e} (limit {limit:.1e})"


def check_divb(stdout, limit=1e-11):
    """max|divB| diagnostics printed by the MHD module at every output."""
    vals = [float(v) for v in re.findall(r"max\|divB\| = ([-\d.e+]+(?:inf)?)",
                                         stdout)]
    if not vals:
        return False, "no divB diagnostics in run output"
    worst = max(vals)
    ok = np.isfinite(worst) and worst < limit
    return ok, f"max|divB| over run = {worst:.3e} (limit {limit:.1e})"


def check_cf_flux(stdout, limit):
    """Coarse-fine magnetic-flux telescoping drift, which the divb check cannot
    see: mhd_max_divB is computed per block, and a CT update is locally
    divergence-free for ANY within-block single-valued EMF, including one that
    disagrees with the neighbour across a level jump. The mesh prints, at every
    output, the drift of the flux mismatch between the two sides of every
    coarse-fine face -- exactly zero for a telescoping scheme -- plus the same
    measurement over SAME-level faces as a control on the instrument itself.

    The same-level number is GATED too, as of the patch-corner spread. It used
    to be reported and not gated, because the corner point of a refined patch
    was left multi-valued in the one coarse block that touches it without
    sharing a face with any fine block -- 2.21e-04 on exactly the 16 of 88 OT
    pairs that terminate on a patch corner, with the coarse-fine correction
    itself working (1.4e-17). spread_fv_emf_corners_b hands that block the value
    its two neighbours already took from the fine side, which brings the control
    to 1.39e-17, i.e. to the same round-off as the coarse-fine number. Gating it
    is what keeps that from silently coming back: SPD_NO_EMF_CORNER=1 restores
    the old behaviour and must turn this red.

    Note the negative control (SPD_NO_FV_EMF=1) has a CLEAN same-level number --
    with nothing injected, nothing becomes multi-valued -- so the two halves of
    this check fail in different configurations, not together.
    """
    vals = [float(v) for v in re.findall(
        r"CF flux drift = ([-\d.e+]+(?:inf)?)", stdout)]
    same = [float(v) for v in re.findall(
        r"same-level control = ([-\d.e+]+(?:inf)?)", stdout)]
    bad = [int(v) for v in re.findall(r"interfaces, (\d+) bad", stdout)]
    if not vals:
        return False, "no CF flux diagnostics in run output"
    if bad and max(bad) > 0:
        return False, (f"{max(bad)} coarse-fine pair(s) failed the geometry "
                       "check: the diagnostic is not measuring the faces it "
                       "claims to")
    # -1 marks "no reference yet" (first output) or an interface set that
    # changed under a regrid; both are not measurements.
    worst = max([v for v in vals if v >= 0], default=-1.0)
    if worst < 0:
        return False, "no CF flux measurement with a t=0 reference"
    worst_same = max([v for v in same if v >= 0], default=-1.0)
    ok = np.isfinite(worst) and worst < limit
    # The control is only gated where it was measured: a regrid resets its
    # reference exactly as it resets the coarse-fine one, and -1 means "no
    # measurement", not "no drift".
    if ok and worst_same >= 0 and not (np.isfinite(worst_same) and worst_same < limit):
        return False, (f"same-level control = {worst_same:.3e} exceeds the limit "
                       f"{limit:.1e} while the coarse-fine drift is clean "
                       f"({worst:.3e}): the patch-corner EMF value is "
                       "multi-valued again")
    return ok, (f"CF flux drift = {worst:.3e} (limit {limit:.1e}); "
                f"same-level control = {worst_same:.3e}")


def check_cf_flux_sensitive(stdout, floor):
    """The negative control, kept in the suite on purpose.

    A gate that cannot fail is worth nothing, and this one has a specific way of
    going quiet: if the coarse-fine EMF correction ever stops being what holds
    the telescoping together, check_cf_flux above would keep passing while
    measuring nothing. This config disables the correction (SPD_NO_FV_EMF=1) and
    requires the drift to be LARGE. It failing means the paired gate has become
    vacuous, not that the code regressed.
    """
    vals = [float(v) for v in re.findall(
        r"CF flux drift = ([-\d.e+]+(?:inf)?)", stdout)]
    worst = max([v for v in vals if v >= 0], default=-1.0)
    if worst < 0:
        return False, "no CF flux measurement with a t=0 reference"
    ok = worst > floor
    return ok, (f"correction OFF -> CF flux drift = {worst:.3e} "
                f"(must exceed {floor:.1e}, else the paired gate is vacuous)")


def check_sl_flux_sensitive(stdout, floor):
    """Negative control for the same-level half of check_cf_flux.

    The patch-corner spread is what brought the same-level control from 2.21e-04
    to round-off, and check_cf_flux now gates it. That gate has the same way of
    going quiet as the coarse-fine one: if the corner point stopped being
    measured at all, the control would read clean and prove nothing. This config
    switches the spread off (SPD_NO_EMF_CORNER=1) and requires the control to be
    LARGE. It failing means the same-level gate has become vacuous, not that the
    code regressed.
    """
    same = [float(v) for v in re.findall(
        r"same-level control = ([-\d.e+]+(?:inf)?)", stdout)]
    worst = max([v for v in same if v >= 0], default=-1.0)
    if worst < 0:
        return False, "no same-level control with a t=0 reference"
    ok = worst > floor
    return ok, (f"corner spread OFF -> same-level control = {worst:.3e} "
                f"(must exceed {floor:.1e}, else the same-level gate is vacuous)")


def check_mixed_levels(outdir, cfg):
    """The mesh must really carry a coarse-fine interface at some point in the
    run. Without this a refinement region that happens to cover the whole
    domain degenerates to a uniform grid and tests nothing about AMR."""
    files = sorted(glob.glob(os.path.join(outdir, "amr_blocks_*.txt")),
                   key=lambda f: int(re.search(r"_(\d+)\.txt$", f).group(1)))
    if not files:
        return False, "no amr_blocks_*.txt written"
    seen = []
    for path in files:
        levels = []
        with open(path) as fh:
            header = None
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                tok = line.split()
                if header is None:
                    header = tok
                    continue
                levels.append(int(tok[1]))
        seen.append(sorted(set(levels)))
        if len(seen[-1]) > 1:
            return True, (f"{os.path.basename(path)}: levels {seen[-1]} over "
                          f"{len(levels)} blocks (coarse-fine present)")
    return False, f"mesh never mixed-level (levels per output: {seen})"


def check_equilibrium_sensitive(outdir, cfg, floor):
    """The negative control for check_static_equilibrium, kept on purpose.

    A gate that cannot fail is worth nothing, and this one has a specific way of
    going quiet: if the current-sheet IC ever degenerated to a trivially uniform
    state, or the run stopped evolving at all, the paired gate would keep passing
    while measuring nothing. This config runs the SAME equilibrium at
    cfl_type=min, cfl=0.4 -- measured unstable -- and requires the spurious
    velocity to be LARGE. It failing means the paired gate has become vacuous,
    not that the code regressed.
    """
    grid = spdk_io.Grid(outdir)
    worst = 0.0
    for i in spdk_io.output_indices(outdir):
        v2 = sum(spdk_io.load_sd(grid, i, c) ** 2 for c in (1, 2, 3))
        worst = max(worst, float(np.sqrt(v2).max()))
    return worst > floor, (f"cfl_type=min cfl=0.4 -> max |v| = {worst:.3e} "
                           f"(must exceed {floor:.1e}, else the paired gate is vacuous)")


def check_static_equilibrium(outdir, cfg, tol):
    """max |v| on a problem whose exact solution is 'nothing moves'.

    The unperturbed Harris current sheet is an exact stationary solution of ideal
    MHD for ANY sheet width (the balance is d/dy(p + Bx^2/2) = 0), so every bit
    of velocity is the scheme's own error -- with no physics to hide behind. That
    makes it far sharper than asking whether a run survived: a configuration that
    merely fails to blow up can still be growing 1e-01 of spurious velocity where
    the answer is exactly zero, which is how a CFL limit came to be quoted 1.5x
    too high from an Orszag-Tang 'did dt collapse?' probe.

    It is also the test that exercises the MHD REFLECTING walls, which nothing
    else in the suite does.
    """
    grid = spdk_io.Grid(outdir)
    idx = spdk_io.output_indices(outdir)
    if len(idx) < 2:
        return False, f"only {len(idx)} outputs; nothing to compare"
    worst = 0.0
    for i in idx:
        v2 = sum(spdk_io.load_sd(grid, i, c) ** 2 for c in (1, 2, 3))
        worst = max(worst, float(np.sqrt(v2).max()))
    return worst < tol, (f"max |v| on the exact equilibrium = {worst:.3e} "
                         f"(limit {tol:.1e}; exact answer is 0)")


def check_inlet_field(outdir, cfg, limit):
    """Where does |B|^2 peak, and how big is it?

    On the Mach-800 jet the answer is a physics statement: the field is
    compressed at the BOW SHOCK, and the inlet plane at y = 0 -- where the state
    is prescribed -- must keep the uniform field it was given. Balsara et al.
    2025 figure 11c shows exactly that: |B|^2 peaks around 1.9e+05 in a thin arc
    along the bow shock, and the base stays at its initial 2.0e+04.

    spd_K used to fail this badly, and invisibly: the inlet's normal face field
    is advanced by the CT curl like any other, so with no boundary EMF condition
    it is driven by an EMF extrapolated out of the interior. B_y on the base row
    went from 141.42 to 1308, |B|^2 peaked at 2.17e+06 ON the inlet row, and the
    resulting magnetic pressure of 1.1e+06 -- above the jet's own ram pressure
    of 9.0e+05 -- pushed material back into the domain at a mean v_y of 435.
    That was read, for a while, as a reason to close the base with a reflecting
    wall, which is far worse (see mhd_jet_base_wall_sensitive_2d).

    So this checks BOTH the magnitude and the LOCATION. The location is the
    robust half: a peak on the inlet row is wrong at any resolution, whereas the
    magnitude of a shock peak depends on how well the shock is resolved.
    """
    grid = spdk_io.Grid(outdir)
    idx = spdk_io.output_indices(outdir)
    if len(idx) < 2:
        return False, f"only {len(idx)} outputs; nothing to compare"
    worst, row, ny = 0.0, None, None
    for i in idx:
        B2 = sum(spdk_io.load_sd(grid, i, c) ** 2 for c in (5, 6, 7))
        m = float(B2.max())
        if m > worst:
            worst, row = m, int(np.unravel_index(B2.argmax(), B2.shape)[1])
        ny = B2.shape[1]
    frac = row / ny
    at_inlet = frac < 0.05
    msg = (f"max |B|^2 = {worst:.3e} at y-row {row}/{ny} (frac {frac:.3f}); "
           f"limit {limit:.1e}, and the peak must NOT be on the inlet row")
    return (worst < limit and not at_inlet), msg


def check_inlet_field_sensitive(outdir, cfg, floor):
    """Negative control for check_inlet_field, kept on purpose.

    Same jet with SPD_NO_BC_EMF_PIN=1, i.e. the inlet EMF left free. The inlet
    field MUST then be destroyed: this requires |B|^2 to exceed `floor` AND to
    peak on the inlet row. If this config goes green, the paired gate is no
    longer measuring the boundary EMF condition -- for instance because the pin
    was made unconditional and the switch stopped reaching it.
    """
    grid = spdk_io.Grid(outdir)
    worst, row, ny = 0.0, None, None
    for i in spdk_io.output_indices(outdir):
        B2 = sum(spdk_io.load_sd(grid, i, c) ** 2 for c in (5, 6, 7))
        m = float(B2.max())
        if m > worst:
            worst, row = m, int(np.unravel_index(B2.argmax(), B2.shape)[1])
        ny = B2.shape[1]
    frac = row / ny
    return (worst > floor and frac < 0.05), (
        f"unpinned inlet -> max |B|^2 = {worst:.3e} at y-row frac {frac:.3f} "
        f"(must exceed {floor:.1e} AND sit on the inlet row, else the paired "
        f"gate is vacuous)")


def check_ha_jet(outdir, cfg, unused=None):
    """Peak pressure and left-right symmetry on the Ha et al. hypersonic jet.

    THE GATE IS ON p_max, and that choice is measured rather than assumed.
    Rueda-Ramirez et al. 2023 (arXiv:2303.00374) table 5 runs the same problem
    with eight limiter/CFL combinations at 1024^2 DoF, and the three quantities
    they report do NOT agree with each other about how scheme-sensitive they are:

        p_max     1.726e+05 .. 2.282e+05     spread  1.3x
        rho_max   23.85 .. 40.67             spread  1.7x
        rho_min   7.11e-04 .. 1.33e-02       spread 18.7x

    So p_max is the one number worth gating on and rho_min is the one to avoid:
    the paper explains its own spread as a dissipation feedback -- less
    dissipation lowers rho_min, which raises the sound speed, which cuts dt,
    which cuts dissipation again. Bolm et al. 2026 (arXiv:2607.06045) remark 8
    says the same thing about this benchmark family outright: results are
    "highly sensitive to minor differences in the numerical setup" because the
    calculations mix "very large numbers" with "numbers very close to 0".

    Measured here with PLM+RK2, converging the right way against their MCL
    cluster at 1.748e+05-1.753e+05:
        128^2 DoF   p_max 1.605e+05   rho_max 12.72
        256^2 DoF   p_max 1.683e+05   rho_max 17.11
        512^2 DoF   p_max 1.693e+05   rho_max 22.10
    i.e. 3.2% low on p_max at a QUARTER of their degrees of freedom.

    Symmetry is free and worth having: the setup and the mesh are exactly
    symmetric about the nozzle axis, so any asymmetry is scheme noise. spd_K
    measures 0.000 here. It is not a vacuous check -- on the Balsara MHD jet the
    same measure reads 18-22 for the SDFB4 and AthenaK lanes.
    """
    grid = spdk_io.Grid(outdir)
    idx = spdk_io.output_indices(outdir)
    if len(idx) < 2:
        return False, f"only {len(idx)} outputs; nothing to compare"
    i = idx[-1]
    # hydro layout is rho, vx, vy, vz, p (spdk_io.VARS)
    rho = spdk_io.load_sd(grid, i, spdk_io.VARS["rho"])
    prs = spdk_io.load_sd(grid, i, spdk_io.VARS["p"])
    pmax = float(prs.max())
    lo = cfg.get("pmax_lo", 1.3e5)
    hi = cfg.get("pmax_hi", 2.3e5)
    # the jet runs along x and the nozzle is centred in y, so mirror in y
    r2 = np.squeeze(rho)
    asym = float(np.abs(r2 - r2[::-1, :]).max() / r2.mean())
    atol = cfg.get("sym_tol", 0.5)
    ok = (lo < pmax < hi) and (asym < atol)
    return ok, (f"p_max = {pmax:.4g} (want {lo:.2e} < p < {hi:.2e}; RR23 report "
                f"1.73e+05-2.28e+05 at 4x the DoF), asymmetry = {asym:.3f} "
                f"(limit {atol})")


def check_ha_jet_sensitive(outdir, cfg, ceiling):
    """Negative control for check_ha_jet, kept on purpose.

    The same deck with x1_bc = outflow, i.e. the nozzle switched off. Nothing
    then enters the domain and the quiescent gas is an exact stationary
    solution, so p_max MUST stay at its initial 0.4127. If this ever rises, the
    inflow boundary is injecting when it should not; if the paired gate ever
    stops depending on the inflow, this is what catches it. Measured: p_max
    exactly 0.4127 and rho exactly 0.5 everywhere.
    """
    grid = spdk_io.Grid(outdir)
    worst = 0.0
    for i in spdk_io.output_indices(outdir):
        worst = max(worst,
                    float(spdk_io.load_sd(grid, i, spdk_io.VARS["p"]).max()))
    return worst < ceiling, (
        f"nozzle off -> p_max = {worst:.6g} (must stay below {ceiling:.3g}, the "
        f"initial 0.4127; a jet here means the inflow BC fires when disabled)")


def check_golden_differs(outdir, cfg, floor):
    """The anti-dead-code gate for the UCT electromotive force.

    UCT sat in this tree with ZERO call sites for a release while docs/mhd.md
    described it as live: every face solve passed a default-empty coefficient
    view, and the corner path was gated on a member hardwired to false. Nothing
    went red, because a scheme that silently falls back to the old one still
    conserves mass, still holds div(B) at round-off, and still matches the
    golden -- it matches it EXACTLY, which is the tell.

    So this config runs mhd/emf=uct against the golden generated by the
    two-sweep path and requires them to DIFFER. It failing means UCT stopped
    running, not that the physics regressed.
    """
    gfile = os.path.join(GOLDEN_DIR, cfg["golden_name"], cfg["field"])
    new = os.path.join(outdir, cfg["field"])
    if not os.path.isfile(gfile):
        return False, f"reference golden missing: {gfile}"
    a, b = np.fromfile(gfile), np.fromfile(new)
    if a.shape != b.shape:
        return False, f"size mismatch {a.size} vs {b.size}"
    diff = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
    ok = diff > floor
    return ok, (f"emf=uct vs 2sweep golden: rel diff = {diff:.3e} "
                f"(must exceed {floor:.1e}, else UCT is not running)")


def check_golden(outdir, cfg, regen, active_only=False):
    gdir = os.path.join(GOLDEN_DIR, cfg["golden_name"])
    gfile = os.path.join(gdir, cfg["field"])
    new = os.path.join(outdir, cfg["field"])
    if regen or not os.path.isfile(gfile):
        if active_only:
            return False, "golden missing (active-only checks never regenerate)"
        os.makedirs(gdir, exist_ok=True)
        shutil.copy(new, gfile)
        return True, f"golden (re)generated: {gfile}"
    a, b = np.fromfile(gfile), np.fromfile(new)
    if a.shape != b.shape:
        return False, f"golden size mismatch {a.size} vs {b.size}"
    if active_only:
        nc = n_from_field(cfg["field"])
        nvar = cfg.get("nvar", NVAR)
        ndim = cfg["ndim"]
        # Element grid: nc elements per active transverse dim, but the z extent
        # is taken from the FILE SIZE rather than assumed equal to nc -- the
        # field-loop 3D lane is nx1=nx2=16 with nx3=4, and shape_for's cubic
        # assumption reshapes it to 18 in z and fails.
        # Sub-cells per element come from the dump name (N32p0 -> p=0 -> 1),
        # not from the module default n = p+1 at p=3: the MUSCL wall gate
        # (hydro_implosion_muscl_mb_2d) is a p=0 active-region comparison.
        mp = re.search(r"_N\d+p(\d+)_", cfg["field"])
        nsp = int(mp.group(1)) + 1 if mp else n
        npx, npy, npz = nsp, (nsp if ndim >= 2 else 1), (nsp if ndim >= 3 else 1)
        Nx = nc + 2 * NGH
        Ny = nc + 2 * NGH if ndim >= 2 else 1
        per_z = nvar * Ny * Nx * npz * npy * npx
        Nz = a.size // per_z if per_z else 1
        shp = (1, nvar, Nz, Ny, Nx, npz, npy, npx)
        if a.size != int(np.prod(shp)):
            return False, (f"golden active-region reshape failed: {a.size} "
                           f"elements do not fit {shp}")
        sl = tuple(slice(NGH, -NGH) if e > 1 else slice(None) for e in (Nz, Ny, Nx))
        idx = (slice(None),) * 2 + sl
        a, b = a.reshape(shp)[idx], b.reshape(shp)[idx]
    diff = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
    rtol = cfg["golden_rtol"]
    return diff < rtol, f"golden max rel diff = {diff:.3e} (rtol {rtol:.1e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "build-511"))
    ap.add_argument("--skip-unit", action="store_true")
    ap.add_argument("--skip-golden", action="store_true",
                    help="skip golden bit-comparison checks (machine/compiler "
                         "specific; recommended in CI on a different toolchain)")
    ap.add_argument("--regen-goldens", action="store_true")
    ap.add_argument("--only", default=None,
                    help="substring filter: run only matching config names")
    args = ap.parse_args()

    failures = 0

    sh(["cmake", "--build", args.build_dir, "-j8"])

    if not args.skip_unit:
        r = subprocess.run([os.path.join(args.build_dir, "spd_K_test")],
                           capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"[{'PASS' if ok else 'FAIL'}] unit: transforms")
        failures += 0 if ok else 1

    timings = []
    for name, cfg in CONFIGS.items():
        if args.only and args.only not in name:
            continue
        outdir = os.path.join(args.build_dir, "test_out", name)
        t_case = time.time()
        try:
            stdout = run(args.build_dir, outdir, cfg)
        except RuntimeError as e:
            print(f"[FAIL] {name}: {e}")
            failures += 1
            continue
        finally:
            timings.append((time.time() - t_case, name))
        ok, msg = check_finite(outdir, cfg)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        failures += 0 if ok else 1

        for chk in cfg["checks"]:
            if chk == "analytic":
                ok, msg = check_analytic(outdir, cfg)
            elif chk == "mass_strict":
                # Per-config limit, same pattern as cf_limit/divb_limit. 1e-12
                # is right for CLOSED (periodic or reflecting) domains, where
                # mass is exactly conserved. OUTFLOW boundaries extrapolate and
                # are not conservative by construction, so a config with
                # gradfree walls has a real leakage floor -- raise it only with
                # the closed-domain control measured (see mhd_blast3d_mdz_3d).
                ok, msg = check_mass(outdir, cfg, cfg.get("mass_limit", 1e-12))
            elif chk == "mixed_levels":
                ok, msg = check_mixed_levels(outdir, cfg)
            elif chk == "divb":
                # Per-config limit, same pattern as cf_limit. The gate is an
                # ABSOLUTE |divB|, so it scales with the field strength and the
                # inverse cell size; a config with B0 = 28 and a finer sub-cell
                # lattice than the 2D norm has a higher round-off floor for the
                # same quality of CT. Raise it only with the relative number
                # written down (see mhd_blast3d_mdz_3d).
                ok, msg = check_divb(stdout, cfg.get("divb_limit", 1e-11))
            elif chk == "cf_flux":
                ok, msg = check_cf_flux(stdout, cfg.get("cf_limit", 1e-14))
            elif chk == "cf_flux_sensitive":
                ok, msg = check_cf_flux_sensitive(stdout,
                                                  cfg.get("cf_floor", 1e-6))
            elif chk == "sl_flux_sensitive":
                ok, msg = check_sl_flux_sensitive(stdout,
                                                 cfg.get("sl_floor", 1e-6))
            elif chk == "golden":
                if args.skip_golden and not args.regen_goldens:
                    print(f"[SKIP] {name}: golden (skipped)")
                    continue
                ok, msg = check_golden(outdir, cfg, args.regen_goldens)
            elif chk == "golden_active":
                ok, msg = check_golden(outdir, cfg, False, active_only=True)
            elif chk == "equilibrium_sensitive":
                ok, msg = check_equilibrium_sensitive(outdir, cfg,
                                                      cfg.get("equil_floor", 1e-6))
            elif chk == "ha_jet":
                ok, msg = check_ha_jet(outdir, cfg)
            elif chk == "ha_jet_sensitive":
                ok, msg = check_ha_jet_sensitive(outdir, cfg,
                                            cfg.get("pmax_ceiling", 1.0))
            elif chk == "inlet_field":
                ok, msg = check_inlet_field(outdir, cfg,
                                            cfg.get("inlet_b2_limit", 5e5))
            elif chk == "inlet_field_sensitive":
                ok, msg = check_inlet_field_sensitive(outdir, cfg,
                                            cfg.get("inlet_b2_floor", 3e5))
            elif chk == "static_equilibrium":
                ok, msg = check_static_equilibrium(outdir, cfg,
                                                   cfg.get("equil_tol", 1e-10))
            elif chk == "golden_differs":
                ok, msg = check_golden_differs(outdir, cfg,
                                               cfg.get("differs_floor", 1e-8))
            else:
                # An unknown check name used to fall through this chain with
                # `ok, msg` still holding the PREVIOUS check's values, so it
                # printed a second PASS line attributed to a check that never
                # ran. That is the worst kind of gate: one that cannot fail and
                # looks green. Caught when "mass" and "finite" -- neither of
                # them dispatched names -- were added to a new config and the
                # suite duly printed each of its real checks twice.
                ok, msg = False, (f"unknown check '{chk}' (known: "
                                  f"analytic, mass_strict, mixed_levels, divb, "
                                  f"cf_flux, cf_flux_sensitive, sl_flux_sensitive, "
                                  f"golden, golden_active, golden_differs, "
                                  f"equilibrium_sensitive, static_equilibrium, inlet_field, inlet_field_sensitive, ha_jet, ha_jet_sensitive)")
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
            failures += 0 if ok else 1

    #Where the wall time goes. A suite nobody will sit through is a suite that
    #stops being run, so the slowest configs are worth seeing every time.
    if timings:
        total = sum(t for t, _ in timings)
        print(f"\ntiming: {total:.0f} s over {len(timings)} configs")
        for t, name in sorted(timings, reverse=True)[:8]:
            print(f"  {t:7.1f} s  {100*t/total:4.1f}%  {name}")
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    return failures


if __name__ == "__main__":
    sys.exit(min(main(), 99))
