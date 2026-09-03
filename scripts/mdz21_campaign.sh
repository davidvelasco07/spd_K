#!/usr/bin/env bash
# Run one MDZ21 benchmark across the four base schemes at MATCHED degrees of
# freedom, for a given emf/rsolver pair.
#
#   scripts/mdz21_campaign.sh <deck> <dof> <rsolver> <emf> [outroot]
#   scripts/mdz21_campaign.sh inputs/mdz21/orszag_tang.athinput 128 hlld uct
#
# Matched DoF: the FV sub-grid carries p+1 sub-cells per element, so N elements
# at degree p is N*(p+1) degrees of freedom per direction. The four lanes are
#
#   muscl_rk2   p=3, N=dof/4, job/scheme=plm, rk2   -- cascade level 1 = PLM
#   muscl_hancock  NOT AVAILABLE: mhd_fv_fluxes_t has no predictor and
#                  main.cpp refuses job/scheme=vl2 under system=mhd rather than
#                  run plain PLM under that name. Left out of the loop below
#                  deliberately -- adding it means writing the predictor, not
#                  changing this script.
#   sdfb4_rk3   p=3, N=dof/4, rk3
#   sdfb8_rk3   p=7, N=dof/8, rk3
#
# Runs are launched in parallel (one core each; the build is Kokkos Serial).
#
# SPDK_SEQUENTIAL=1 runs one lane at a time instead. Set it for GPU builds:
# the lanes then share ONE device, and a single 192^3 DoF 3D MHD lane measured
# 22 GB on an A100, so three in parallel is 66 GB on a 40 GB card and they OOM
# each other. On CPU the default (parallel, one core per lane) is still right.
set -u

deck=${1:?usage: mdz21_campaign.sh <deck> <dof> <rsolver> <emf> [outroot]}
dof=${2:?}
rsolver=${3:?}
emf=${4:?}
outroot=${5:-mdz21_runs}

root=$(cd "$(dirname "$0")/.." && pwd)
bin=${SPDK_BIN:-$root/build/spd_K}
[ -x "$bin" ] || { echo "no binary at $bin"; exit 1; }

# Provenance, per CLAUDE.md rule 9: record what this campaign actually ran on.
stamp=$(date -u +%Y%m%dT%H%M%SZ)
# Absolute outroot stays absolute; a relative one is taken from the repo root.
case "$outroot" in /*) outbase="$outroot" ;; *) outbase="$root/$outroot" ;; esac
base="$outbase/${emf}_${rsolver}_dof${dof}"
mkdir -p "$base"
{
  echo "date       $stamp"
  echo "binary     $bin"
  echo "binary_md5 $(if command -v md5 >/dev/null 2>&1; then md5 -q "$bin"; else md5sum "$bin" | cut -d" " -f1; fi)"
  # BSD and GNU date disagree on -r; try both rather than print nothing.
  echo "binary_mtime $(date -u -r "$bin" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "@$(stat -c %Y "$bin")" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
  if (cd "$root" && git rev-parse HEAD >/dev/null 2>&1); then
    echo "git        $(cd "$root" && git rev-parse HEAD) $(cd "$root" && git diff --quiet || echo '(dirty)')"
  else
    echo "git        (not a git checkout -- rsynced tree; trust src_md5)"
  fi
  # LC_ALL=C so the glob order is the same on every platform: macOS collates
  # amr.cpp BEFORE amr_boundary.cpp and Linux after, which changes the
  # concatenation and makes the same tree hash differently on the two machines.
  # Measured -- every per-file md5 matched while this combined one did not.
  # Pick the hasher by TESTING for it. A `md5 -q || md5sum` fallback after a
  # pipe is broken: on Linux the missing `md5` still consumes the pipe and
  # md5sum hashes nothing, so the field comes out EMPTY -- which is worse than
  # absent, because it looks recorded. Seen on the first Vista campaign.
  if command -v md5 >/dev/null 2>&1; then HASH="md5 -q"; else HASH="md5sum"; fi
  echo "src_md5    $(cd "$root" && LC_ALL=C ls src/*.cpp src/*.hpp | LC_ALL=C sort \
                     | xargs cat | $HASH | cut -d" " -f1)"
  echo "deck       $deck"
  echo "dof        $dof"
  echo "rsolver    $rsolver"
  echo "emf        $emf"
} > "$base/provenance.txt"
cat "$base/provenance.txt"

# The deck's OWN nx1:nx2 sets the aspect ratio; `dof` is the x-direction
# resolution and y follows. Forcing nx2 = nx1 would have quietly run the
# field-loop and current-sheet decks -- both 2:1 boxes -- on a square mesh, at a
# different physical setup from the one their comments describe.
case "$deck" in /*) deckpath="$deck" ;; *) deckpath="$root/$deck" ;; esac
deck_nx1=$(sed -n 's/^[[:space:]]*nx1[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' "$deckpath" | head -1)
deck_nx2=$(sed -n 's/^[[:space:]]*nx2[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' "$deckpath" | head -1)
deck_nx3=$(sed -n 's/^[[:space:]]*nx3[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/p' "$deckpath" | head -1)
[ -z "$deck_nx3" ] && deck_nx3=1
if [ -z "$deck_nx1" ] || [ -z "$deck_nx2" ] || [ "$deck_nx1" -eq 0 ]; then
  echo "ERROR: could not read nx1/nx2 from $deckpath"; exit 1
fi

launch() {  # name, p, Nx, extra overrides...
  local name=$1 p=$2 Nx=$3; shift 3
  # Preserve the deck's aspect ratio exactly; refuse rather than round it away.
  local Ny=$(( Nx * deck_nx2 / deck_nx1 ))
  if [ $(( Ny * deck_nx1 )) -ne $(( Nx * deck_nx2 )) ] || [ "$Ny" -lt 1 ]; then
    echo "  SKIP $name: Nx=$Nx does not preserve the deck's ${deck_nx1}:${deck_nx2} aspect"
    return
  fi
  # 3D decks: scale nx3 too. Without this a 3D deck keeps its FILE nx3 while
  # nx1/nx2 are overridden, so every lane silently runs a different z resolution
  # from its x/y -- and the "matched DoF" claim would be false in one direction.
  # nx3 = 1 (a 2D deck) is left alone.
  local nz_args=()
  if [ "$deck_nx3" -gt 1 ]; then
    local Nz=$(( Nx * deck_nx3 / deck_nx1 ))
    if [ $(( Nz * deck_nx1 )) -ne $(( Nx * deck_nx3 )) ] || [ "$Nz" -lt 1 ]; then
      echo "  SKIP $name: Nx=$Nx does not preserve the deck's ${deck_nx1}:${deck_nx3} z aspect"
      return
    fi
    nz_args=(mesh/nx3=$Nz)
  fi
  local out="$base/$name"
  # Resume: a lane that already exited cleanly is left alone, so a campaign can
  # be re-run to add a lane without discarding hours of finished work.
  if [ -f "$out/run.log" ] && grep -q '^evolution:' "$out/run.log" 2>/dev/null; then
    echo "  keeping $name (already complete)"
    return
  fi
  rm -rf "$out"; mkdir -p "$out"
  echo "  launching $name  (p=$p, ${Nx}x${Ny}${nz_args:+x$Nz} elements = $((Nx*(p+1)))x$((Ny*(p+1)))${nz_args:+x$((Nz*(p+1)))} DoF)"
  ( SPD_OUTPUT_DIR="$out" "$bin" -i "$deckpath" \
      mesh/p=$p mesh/nx1=$Nx mesh/nx2=$Ny "${nz_args[@]}" \
      mhd/rsolver=$rsolver mhd/emf=$emf "$@" \
      > "$out/run.log" 2>&1
    echo "$?" > "$out/exit_code" ) &
  [ "${SPDK_SEQUENTIAL:-0}" = 1 ] && wait
}

p3n=$((dof / 4))
p7n=$((dof / 8))
if [ $((p3n * 4)) -ne "$dof" ] || [ $((p7n * 8)) -ne "$dof" ]; then
  echo "ERROR: dof=$dof is not divisible by both 4 (p=3) and 8 (p=7);"
  echo "       the lanes would not be at matched degrees of freedom."
  exit 1
fi

launch muscl_rk2 3 "$p3n" job/scheme=plm time/integrator=rk2
# The honest cost question: is high order worth it, or is the cheap scheme on a
# finer mesh just as good? This lane is MUSCL at DOUBLE the resolution of every
# other lane -- 4x the cells in 2D and ~2x the steps, so ~8x the work of the
# matched-DoF MUSCL lane. It is deliberately NOT at matched DoF; it is the
# control that says whether the SDFB lanes are buying anything a mesh refinement
# would not.
launch muscl_rk2_2x 3 "$((p3n * 2))" job/scheme=plm time/integrator=rk2
launch sdfb4_rk3 3 "$p3n" time/integrator=rk3
# SDFB8 needs a smaller CFL than the p=3 lanes. Under the decks' cfl_type=sum,
# p=3 is stable to 0.5 and p=7 to 0.4 on the exact current-sheet equilibrium
# (see the table in main.cpp); 0.30 leaves margin on the harder shocked lanes.
# The lanes stay matched in DEGREES OF FREEDOM, which is what the comparison is
# about. They are NOT matched in step count -- the tighter stability limit of the
# higher-order scheme is a result to report, not something to hide by lowering
# every lane to match.
launch sdfb8_rk3 7 "$p7n" time/integrator=rk3 time/cfl=0.30
wait

echo
fail=0
for d in "$base"/*/; do
  n=$(basename "$d")
  [ -f "$d/exit_code" ] || continue
  rc=$(cat "$d/exit_code")
  ndump=$(ls "$d"/W_cv_*.dat 2>/dev/null | wc -l | tr -d ' ')
  if [ "$rc" != 0 ]; then
    echo "  FAILED  $n  (exit $rc) -- $(tail -1 "$d/run.log")"
    fail=1
  elif [ "$ndump" -eq 0 ]; then
    # CLAUDE.md rule 2: a comparison over zero files reports IDENTICAL.
    echo "  CHECK VOID  $n  (exit 0 but no dumps)"
    fail=1
  else
    echo "  ok      $n  ($ndump dumps)"
  fi
done
exit $fail
