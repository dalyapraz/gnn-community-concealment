#!/usr/bin/env bash
set -euo pipefail

SIGMA_LIST=(0.01 0.1 0.5 1 2 5)
MU_LIST=(0.01 0.1 0.2 0.3 0.4 0.5)
FEATURE_LIST=("None" "connecting_node" "average_community")

# choose which min_community blocks to include
DO_MINCOMM_60=0
DO_MINCOMM_30=0
DO_MINCOMM_10=1

OUT_A=params10_pA.tsv   # p = 0.5 0.65
OUT_B=params10_pB.tsv   # p = 0.8 0.95
: > "$OUT_A"; : > "$OUT_B"

emit_line () {
  local out="$1" mu="$2" sigma="$3" minc="$4" seed="$5" feature="$6" pvals="$7" ptag="$8"
  local base="dmon_dice_${mu}_mincomm_${minc}_sigma${sigma}_p${ptag}_feat${feature}.csv"
  echo "mu=$mu sigma_c=$sigma minc=$minc seed=$seed pvals=\"$pvals\" feature=$feature outfile=$base" >> "$out"
}


# ---------- min_community = 60 ----------
if [[ "$DO_MINCOMM_60" == "1" ]]; then
  for sigma in "${SIGMA_LIST[@]}"; do
    for mu in "${MU_LIST[@]}"; do
      if [[ "$mu" == "0.01" ]]; then seed=7; else seed=10; fi
      for feat in "${FEATURE_LIST[@]}"; do
        emit_line "$mu" "$sigma" 60 "$seed" "$feat"
      done
    done
  done
fi

# ---------- min_community = 30 ----------
if [[ "$DO_MINCOMM_30" == "1" ]]; then
  for sigma in "${SIGMA_LIST[@]}"; do
    for mu in "${MU_LIST[@]}"; do
      for feat in "${FEATURE_LIST[@]}"; do
        emit_line "$mu" "$sigma" 30 25 "$feat"
      done
    done
  done
fi


# ---------- min_community = 10 ----------
if [[ "$DO_MINCOMM_10" == "1" ]]; then
  for sigma in "${SIGMA_LIST[@]}"; do
    for mu in "${MU_LIST[@]}"; do
      for feat in "${FEATURE_LIST[@]}"; do
        emit_line "$OUT_A" "$mu" "$sigma" 10 42 "$feat" "0.5 0.65" "0.5-0.65"
        emit_line "$OUT_B" "$mu" "$sigma" 10 42 "$feat" "0.8 0.95" "0.8-0.95"
      done
    done
  done
fi

echo "Wrote $(wc -l < "$OUT_A") lines to $OUT_A"
echo "Wrote $(wc -l < "$OUT_B") lines to $OUT_B"

# # ---------- min_community = 10 ----------
# if [[ "$DO_MINCOMM_10" == "1" ]]; then
#   for sigma in "${SIGMA_LIST[@]}"; do
#     for mu in "${MU_LIST[@]}"; do
#       for feat in "${FEATURE_LIST[@]}"; do
#         emit_line "$mu" "$sigma" 10 42 "$feat"
#       done
#     done
#   done
# fi

# echo "Wrote $(wc -l < "$OUT") lines to $OUT"
