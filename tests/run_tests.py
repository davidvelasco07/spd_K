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
  hydro_implosion_2d : reflective-wall implosion, mass conserved
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
        "checks": ["analytic", "mass_strict", "golden"],
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
        "checks": ["mass_strict"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.1,
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
        "checks": ["mass_strict", "divb", "golden"],
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
        "checks": ["mass_strict", "divb", "golden"],
        "field": "W_cv_N16p3_1_0.dat",
        "t_end": 0.1,
        "golden_name": "mhd_loop_2d",
        "golden_rtol": 1e-6,
    },
    "mhd_field_loop_smr_2d": {
        # Statically refined patch; field loop crosses the coarse-fine boundary.
        # Pure SD CT + EMF correction (short tlim: longer advection still drifts).
        "input": "inputs/field_loop.athinput",
        "overrides": ["job/fallback=false", "mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=1",
                      "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "time/tlim=0.02", "output/dt=0.01",
                      "amr/max_level=1", "amr/adapt_interval=0",
                      "refinement1/level=1", "refinement1/x1min=0.375",
                      "refinement1/x1max=0.625", "refinement1/x2min=0.375",
                      "refinement1/x2max=0.625"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb"],
        "field": "W_cv_N32p3_1_0.dat",
        "t_end": 0.02,
    },
    "mhd_orszag_tang_smr_2d": {
        # Short OT with static refinement (no fallback): mass + divb + mixed levels
        "input": "inputs/orszag_tang.athinput",
        "overrides": ["job/fallback=false", "mesh/nx1=16", "mesh/nx2=16",
                      "mesh/nx3=1", "meshblock/nx1=4", "meshblock/nx2=4",
                      "time/integrator=rk3", "time/tlim=0.01", "output/dt=0.005",
                      "amr/max_level=1", "amr/adapt_interval=0",
                      "refinement1/level=1", "refinement1/x1min=0.375",
                      "refinement1/x1max=0.625", "refinement1/x2min=0.375",
                      "refinement1/x2max=0.625"],
        "ndim": 2,
        "checks": ["mixed_levels", "mass_strict", "divb"],
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
        shp = shape_for(nc, cfg["ndim"], nvar)
        sl = tuple(slice(NGH, -NGH) if s > 1 else slice(None) for s in shp[2:5])
        s = (slice(None),) * 2 + sl
        a, b = a.reshape(shp)[s], b.reshape(shp)[s]
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
                ok, msg = check_mass(outdir, cfg, 1e-12)
            elif chk == "mixed_levels":
                ok, msg = check_mixed_levels(outdir, cfg)
            elif chk == "divb":
                ok, msg = check_divb(stdout)
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
