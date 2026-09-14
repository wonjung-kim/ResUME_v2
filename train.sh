#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=shell_common.sh
source "$SCRIPT_DIR/shell_common.sh"
require_imagenet

# Examples:
#   1) Edit dataset_paths.sh once, then: bash train.sh
#   2) Or override per run: IMAGENET_ROOT=/data/ImageNet PACKET_MODE=group bash train.sh

PACKET_MODE="${PACKET_MODE:-stage}"
GROUP_H="${GROUP_H:-4}"
GROUP_W="${GROUP_W:-4}"
MODEL="${MODEL:-gauss}"
STAGES="${STAGES:-4}"
BITS="${BITS:-12}"
BATCH="${BATCH:-36}"
EPOCHS="${EPOCHS:-200}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
SNR_MIN="${SNR_MIN:-0}"
SNR_MAX="${SNR_MAX:-10}"
OUT_DIR="${OUT_DIR:-./output_${PACKET_MODE}}"

EXTRA=(--imagenet_root "$IMAGENET_ROOT")
if [[ "${FADING:-0}" == "1" ]]; then EXTRA+=(--fading); fi
if [[ "${NORM:-1}" == "1" ]]; then EXTRA+=(--norm); fi

python "$SCRIPT_DIR/train.py" \
  --model "$MODEL" \
  --stages "$STAGES" \
  --bits "$BITS" \
  --batch "$BATCH" \
  --epochs "$EPOCHS" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --snr_min "$SNR_MIN" \
  --snr_max "$SNR_MAX" \
  --packet_mode "$PACKET_MODE" \
  --group_h "$GROUP_H" \
  --group_w "$GROUP_W" \
  --out_dir "$OUT_DIR" \
  "${EXTRA[@]}"
