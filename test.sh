#!/usr/bin/env bash
set -euo pipefail

CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH=/path/to/best.pt}"
POLICY_PATH="${POLICY_PATH:?Set POLICY_PATH=/path/to/policy.json}"
CHANNEL="${CHANNEL:-awgn}"
MODEL="${MODEL:-gauss}"
OUT_DIR="${OUT_DIR:-./test_output}"
KODAK_ROOT="${KODAK_ROOT:-${RESUME_KODAK_ROOT:-}}"

EXTRA=()
if [[ -n "$KODAK_ROOT" ]]; then EXTRA+=(--kodak_root "$KODAK_ROOT"); fi
if [[ "${NORM:-1}" == "1" ]]; then EXTRA+=(--norm); fi

python test.py \
  --ckpt "$CKPT_PATH" \
  --policy "$POLICY_PATH" \
  --model "$MODEL" \
  --stages 4 \
  --bits 12 \
  --batch "${BATCH:-24}" \
  --channel "$CHANNEL" \
  --snrs "${SNRS:--5,0,5,10,15,20}" \
  --cbrs "${CBRS:-0.00521,0.0104167,0.015625}" \
  --out_dir "$OUT_DIR" \
  "${EXTRA[@]}"
