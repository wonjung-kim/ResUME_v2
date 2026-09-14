#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=shell_common.sh
source "$SCRIPT_DIR/shell_common.sh"
require_kodak

CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH=/path/to/best.pt}"
POLICY_PATH="${POLICY_PATH:?Set POLICY_PATH=/path/to/policy.json}"
CHANNEL="${CHANNEL:-awgn}"
MODEL="${MODEL:-gauss}"
STAGES="${STAGES:-4}"
BITS="${BITS:-12}"
OUT_DIR="${OUT_DIR:-./test_output}"

EXTRA=(--kodak_root "$KODAK_ROOT")
if [[ "${NORM:-1}" == "1" ]]; then EXTRA+=(--norm); fi

python "$SCRIPT_DIR/test.py" \
  --ckpt "$CKPT_PATH" \
  --policy "$POLICY_PATH" \
  --model "$MODEL" \
  --stages "$STAGES" \
  --bits "$BITS" \
  --batch "${BATCH:-24}" \
  --channel "$CHANNEL" \
  --snrs "${SNRS:--5,0,5,10,15,20}" \
  --cbrs "${CBRS:-0.00521,0.0104167,0.015625}" \
  --out_dir "$OUT_DIR" \
  "${EXTRA[@]}"
