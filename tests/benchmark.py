#!/usr/bin/env python3
"""Throughput benchmark for spd_K, matching the spd (Python) zcps protocol:
3D hydro advected sine wave, RK3, p in {3,7}, R = N*(p+1) cells/dim.

Relies on the built-in evolution timer, which excludes initial conditions,
setup, host<->device transfers and file outputs; outputs are additionally
disabled via output/dt=-1 (they are opt-in, athenak-style).

Prints one JSON line per configuration, compatible with spd's
zcps_results.jsonl rows.

Usage: tests/benchmark.py [--binary PATH] [--repo-tag TAG] [--integrator rk3]
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_case(binary, overrides, timeout):
    cmd = [binary, "-i", os.path.join(ROOT, "inputs/sine_wave.athinput")] + overrides
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    m = re.search(r"evolution: (\d+) steps, ([\d.eE+-]+) s \(([\d.eE+-]+) ms/step\), "
                  r"([\d.eE+-]+) zone-cycles/s", r.stdout)
    if not m:
        raise RuntimeError(f"no evolution line in output:\n{r.stdout[-800:]}")
    return int(m.group(1)), float(m.group(3)), float(m.group(4))


def bench(binary, sim, p, R, integrator, repo_tag, timeout=3600):
    N = R // (p + 1)
    fallback = "true" if sim == "sdfb" else "false"
    tlim = 1.4 / R  # ~10 steps at CFL 0.8
    over = [f"mesh/p={p}", f"mesh/nx1={N}", f"mesh/nx2={N}", f"mesh/nx3={N}",
            f"job/fallback={fallback}", f"time/integrator={integrator}",
            f"time/tlim={tlim}", "output/dt=-1"]
    steps, ms_per_step, zcps = run_case(binary, over, timeout)
    result = {
        "repo": repo_tag,
        "sim": sim,
        "p": p,
        "R": R,
        "steps": steps,
        "ms_per_step": ms_per_step,
        "zcps": zcps,
    }
    print("RESULT " + json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.path.join(ROOT, "build/spd_K"))
    ap.add_argument("--repo-tag", default="spd_K-amr")
    ap.add_argument("--integrator", default="rk3")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench_results.jsonl"))
    args = ap.parse_args()

    rows = []
    for sim in ("sd", "sdfb"):
        for p in (3, 7):
            for R in (32, 64, 128, 256):
                try:
                    rows.append(bench(args.binary, sim, p, R,
                                      args.integrator, args.repo_tag))
                except Exception as e:
                    print(f"FAILED {sim} p={p} R={R}: {e}", flush=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
