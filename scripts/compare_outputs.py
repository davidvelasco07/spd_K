#!/usr/bin/env python3
"""Compare two directories of spd_K binary .dat outputs (float64 arrays).

Usage: compare_outputs.py <baseline_dir> <new_dir> [rtol]
Exits nonzero if any common file differs beyond tolerance or file sets differ.
"""
import sys
import numpy as np
from pathlib import Path

base, new = Path(sys.argv[1]), Path(sys.argv[2])
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-10

base_files = {f.name for f in base.glob("*.dat")}
new_files = {f.name for f in new.glob("*.dat")}
status = 0
if base_files != new_files:
    print(f"file sets differ: only-baseline={sorted(base_files-new_files)} "
          f"only-new={sorted(new_files-base_files)}")
    status = 1

for name in sorted(base_files & new_files):
    a = np.fromfile(base / name, dtype=np.float64)
    b = np.fromfile(new / name, dtype=np.float64)
    if a.shape != b.shape:
        print(f"FAIL {name}: size {a.size} vs {b.size}")
        status = 1
        continue
    scale = max(np.abs(a).max(), 1e-300)
    diff = np.abs(a - b).max() / scale
    ok = diff < tol
    print(f"{'PASS' if ok else 'FAIL'} {name}: max rel diff = {diff:.3e}")
    if not ok:
        status = 1

sys.exit(status)
