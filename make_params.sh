#!/usr/bin/env bash
set -euo pipefail

# ========== Parameters to vary ==========
SIGMA_LIST=(0.01 0.1 0.5 1 2 5)
MU_LIST=(0.01 0.1 0.2 0.3 0.4 0.5)
FEATURE_LIST=("None" "connecting_node" "average_community")

# choose which min_community blocks to include
DO_MINCOMM_60=0
DO_MINCOMM_30=1
DO_MINCOMM_10=0

# Bundle all sigma in one row (set to 0 to emit one row per sigma)
COMBINE_SIGMA=1

# Uncomment this block if you want ALL p-values in one file:
# OUT_ALL=params30_all.tsv
# : > "$OUT_ALL"


# Split p into two batches
OUT_A=params30_pA.tsv   # p = 0.5 0.65
OUT_B=params30_pB.tsv   # p = 0.8 0.95
: > "$OUT_A"; : > "$OUT_B"

PVALS_A="0.5 0.65"; PTAG_A="0.5-0.65"
PVALS_B="0.8 0.95"; PTAG_B="0.8-0.95"

emit_line () {
  # args: out mu sigma_values sigma_tag minc seed feature pvals ptag
  local out="$1" mu="$2" sigma_vals="$3" sigma_tag="$4" minc="$5" seed="$6" feature="$7" pvals="$8" ptag="$9"
  local base="dmon_fcomdice_${mu}_mincomm_${minc}_${sigma_tag}_p${ptag}_feat${feature}.csv"
  echo "mu=$mu sigma_c=\"$sigma_vals\" minc=$minc seed=$seed pvals=\"$pvals\" feature=$feature outfile=$base" >> "$out"
}

# Helper to emit for a given min_community
# args: out minc seed_for_mu_0p01 seed_for_other_mu pvals ptag
emit_block () {
  local out="$1" minc="$2" seed_m001="$3" seed_other="$4" pvals="$5" ptag="$6"

  if [[ "$COMBINE_SIGMA" -eq 1 ]]; then
    local SIGMA_VALS="${SIGMA_LIST[*]}"    # "0.01 0.1 0.5 1 2 5"
    local SIGMA_TAG="sigmaAll"

    for mu in "${MU_LIST[@]}"; do
      local seed="$seed_other"
      [[ "$mu" == "0.01" ]] && seed="$seed_m001"

      for feat in "${FEATURE_LIST[@]}"; do
        emit_line "$out" "$mu" "$SIGMA_VALS" "$SIGMA_TAG" "$minc" "$seed" "$feat" "$pvals" "$ptag"
      done
    done

  else
    # Per-sigma rows
    for sigma in "${SIGMA_LIST[@]}"; do
      local SIGMA_VALS="$sigma"
      local SIGMA_TAG="sigma${sigma}"

      for mu in "${MU_LIST[@]}"; do
        local seed="$seed_other"
        [[ "$mu" == "0.01" ]] && seed="$seed_m001"

        for feat in "${FEATURE_LIST[@]}"; do
          emit_line "$out" "$mu" "$SIGMA_VALS" "$SIGMA_TAG" "$minc" "$seed" "$feat" "$pvals" "$ptag"
        done
      done
    done
  fi
}

# ---------- min_community blocks ----------
if [[ "$DO_MINCOMM_60" == "1" ]]; then
  emit_block "$OUT_A" 60 7 10 "$PVALS_A" "$PTAG_A"
  emit_block "$OUT_B" 60 7 10 "$PVALS_B" "$PTAG_B"
fi

if [[ "$DO_MINCOMM_30" == "1" ]]; then
  emit_block "$OUT_A" 30 25 25 "$PVALS_A" "$PTAG_A"
  emit_block "$OUT_B" 30 25 25 "$PVALS_B" "$PTAG_B"
fi

if [[ "$DO_MINCOMM_10" == "1" ]]; then
  emit_block "$OUT_A" 10 42 42 "$PVALS_A" "$PTAG_A"
  emit_block "$OUT_B" 10 42 42 "$PVALS_B" "$PTAG_B"
fi

echo "Wrote $(wc -l < "$OUT_A") lines to $OUT_A"
echo "Wrote $(wc -l < "$OUT_B") lines to $OUT_B"

# # ---------- min_community blocks (all p in one) ----------
# if [[ "$DO_MINCOMM_60" == "1" ]]; then
#   emit_block "$OUT_ALL" 60 7 10 "0.5 0.65 0.8 0.95" "0.5-0.95"
# fi

# if [[ "$DO_MINCOMM_30" == "1" ]]; then
#   emit_block "$OUT_ALL" 30 25 25 "0.5 0.65 0.8 0.95" "0.5-0.95"
# fi

# if [[ "$DO_MINCOMM_10" == "1" ]]; then
#   emit_block "$OUT_ALL" 10 42 42 "0.5 0.65 0.8 0.95" "0.5-0.95"
# fi

# echo "Wrote $(wc -l < "$OUT_ALL") lines to $OUT_ALL"
