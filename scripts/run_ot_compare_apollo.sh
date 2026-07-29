#!/usr/bin/env bash
# Launch matched Orszag-Tang comparison runs on Apollo (A100 front node).
# Usage (on Apollo):
#   cd ~/spd_K && git checkout mhd && bash scripts/run_ot_compare_apollo.sh
set -euo pipefail

ROOT_SPDK="${ROOT_SPDK:-$HOME/spd_K}"
ROOT_ATH="${ROOT_ATH:-$HOME/athenak/fallback}"
ATHENA="${ATHENA:-$ROOT_ATH/build_ot/src/athena}"
SPDK="${SPDK:-$ROOT_SPDK/build/spd_K}"
OUT="${OUT:-$HOME/ot_compare_$(date +%Y%m%d)}"
export LD_LIBRARY_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/lib:/usr/local/openmpi/cuda-12.4/4.1.6/nvhpc245/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT"/{spdk,athenak_llf_ct,athenak_hlld_uct}
echo "OUT=$OUT"

[[ -x "$SPDK" ]]  || { echo "ERROR: missing $SPDK"; exit 1; }
[[ -x "$ATHENA" ]] || { echo "ERROR: missing $ATHENA"; exit 1; }
[[ -f "$ROOT_SPDK/inputs/compare/ot_spdk_n32_fb.athinput" ]] || {
  echo "ERROR: compare inputs not present in $ROOT_SPDK — pull/sync the mhd branch first"
  exit 1
}

echo "=== spd_K OT N=32 p=3 fallback ==="
(
  cd "$OUT/spdk"
  cp "$ROOT_SPDK/inputs/compare/ot_spdk_n32_fb.athinput" .
  SPD_OUTPUT_DIR="$OUT/spdk" "$SPDK" -i ot_spdk_n32_fb.athinput | tee run.log
)

echo "=== AthenaK OT 128 LLF+ct_contact+MOOD ==="
(
  cd "$OUT/athenak_llf_ct"
  cp "$ROOT_SPDK/inputs/compare/ot_athenak_llf_ct_n128.athinput" .
  "$ATHENA" -i ot_athenak_llf_ct_n128.athinput | tee run.log
)

echo "=== AthenaK OT 128 HLLD+uct_hlld+MOOD ==="
(
  cd "$OUT/athenak_hlld_uct"
  cp "$ROOT_SPDK/inputs/compare/ot_athenak_hlld_uct_n128.athinput" .
  "$ATHENA" -i ot_athenak_hlld_uct_n128.athinput | tee run.log
)

echo "=== compare at t=0.5 and t=1.0 ==="
PY="$ROOT_SPDK/scripts/compare_ot_athenak.py"
find_bin() {
  local dir="$1" idx="$2"
  local f
  f=$(ls "$dir"/bin/*mhd_w_bcc.$(printf '%05d' "$idx").bin 2>/dev/null | head -1 || true)
  if [[ -z "$f" ]]; then
    f=$(ls "$dir"/*mhd_w_bcc.$(printf '%05d' "$idx").bin 2>/dev/null | head -1 || true)
  fi
  echo "$f"
}

for pair in "0.5:5:5" "1.0:10:10"; do
  IFS=: read -r T IDX AIDX <<<"$pair"
  WCV="$OUT/spdk/W_cv_N32p3_${IDX}_0.dat"
  ALLF=$(find_bin "$OUT/athenak_llf_ct" "$AIDX")
  AHLL=$(find_bin "$OUT/athenak_hlld_uct" "$AIDX")
  echo "---- t≈$T ----"
  if [[ -f "$WCV" && -n "$ALLF" ]]; then
    python3 "$PY" "$WCV" "$ALLF" 32 3 --plot "$OUT/compare_llf_ct_t${T}.png" \
      | tee "$OUT/compare_llf_ct_t${T}.txt"
  else
    echo "missing files for LLF compare (WCV=$WCV ALLF=$ALLF)"
  fi
  if [[ -f "$WCV" && -n "$AHLL" ]]; then
    python3 "$PY" "$WCV" "$AHLL" 32 3 --plot "$OUT/compare_hlld_uct_t${T}.png" \
      | tee "$OUT/compare_hlld_uct_t${T}.txt"
  else
    echo "missing files for HLLD compare (WCV=$WCV AHLL=$AHLL)"
  fi
done

echo "done. results in $OUT"
