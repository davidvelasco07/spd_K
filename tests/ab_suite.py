#!/usr/bin/env python3
"""A/B of two spd_K binaries over the suite's own configurations: every config is run with each binary into its own
directory and every dump and block map is compared by md5. Usage: ab_suite.py REPO REF_BIN NEW_BIN OUTROOT [substr]"""
import sys, os, subprocess, hashlib, glob, shutil, time
repo, ref, new, out = sys.argv[1:5]; only = sys.argv[5] if len(sys.argv) > 5 else ""
sys.path.insert(0, os.path.join(repo, "tests")); import run_tests
def run(binary, name, cfg, tag):
    d = os.path.join(out, tag + f"_{shard}", name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    env = dict(os.environ, SPD_OUTPUT_DIR=d); env.update(cfg.get("env", {}))
    #AB_EXTRA: overrides appended for BOTH binaries, e.g. a boundary word on an inactive face the reference cannot parse
    r = subprocess.run([binary, "-i", os.path.join(repo, cfg["input"])] + list(cfg.get("overrides", []))
                       + os.environ.get("AB_EXTRA", "").split(),
                       capture_output=True, text=True, env=env, cwd=repo)
    return d, r.returncode
def sums(d):
    return {os.path.basename(f): hashlib.md5(open(f, "rb").read()).hexdigest()
            for f in sorted(glob.glob(os.path.join(d, "*.dat")) + glob.glob(os.path.join(d, "amr_blocks_*.txt"))
                            + glob.glob(os.path.join(d, "mass.txt")))}
bad = void = same = 0; t0 = time.time()
shard, nsh = (int(x) for x in os.environ.get("AB_SHARD", "0/1").split("/"))
for num, (name, cfg) in enumerate(run_tests.CONFIGS.items()):
    if only and only not in name: continue
    if num % nsh != shard: continue
    #the reference predates the passive scalars: its binary ignores hydro/nscalars and writes five rows
    if name.startswith("hydro_scalar"): continue
    da, ra = run(ref, name, cfg, "ref"); db, rb = run(new, name, cfg, "new")
    a, b = sums(da), sums(db)
    nd = len([k for k in a if k.endswith(".dat")])
    if nd == 0 or set(a) != set(b):
        void += 1; print(f"CHECK VOID  {name}: {len(a)} vs {len(b)} files, rc {ra}/{rb}", flush=True); continue
    diff = [k for k in a if a[k] != b[k]]
    if diff or ra != rb:
        bad += 1; print(f"DIFFERENT   {name}: {len(diff)} of {len(a)} files ({', '.join(diff[:4])}) rc {ra}/{rb}", flush=True)
    else:
        same += 1; print(f"identical   {name}: {len(a)} files ({nd} dumps) rc {ra}", flush=True)
    shutil.rmtree(da, ignore_errors=True); shutil.rmtree(db, ignore_errors=True)
print(f"AB_DONE identical {same} different {bad} void {void} in {time.time()-t0:.0f} s")
