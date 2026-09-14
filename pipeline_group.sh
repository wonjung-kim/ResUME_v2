#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=shell_common.sh
source "$SCRIPT_DIR/shell_common.sh"
require_imagenet
require_kodak

ROOT="${ROOT:-./output_group_4x4}"
CHANNEL="${CHANNEL:-awgn}"
GROUP_H="${GROUP_H:-4}"
GROUP_W="${GROUP_W:-4}"
STAGES="${STAGES:-4}"
BITS="${BITS:-12}"
FADING_ARGS=()
if [[ "$CHANNEL" == "rayleigh" ]]; then FADING_ARGS+=(--fading); fi

mkdir -p "$ROOT"
python "$SCRIPT_DIR/train.py" --packet_mode group --group_h "$GROUP_H" --group_w "$GROUP_W" --model gauss --stages "$STAGES" --bits "$BITS" \
  --batch "${BATCH:-36}" --epochs "${EPOCHS:-200}" --warmup_epochs "${WARMUP_EPOCHS:-20}" \
  --snr_min "${SNR_MIN:-0}" --snr_max "${SNR_MAX:-10}" --norm --out_dir "$ROOT/train" \
  --imagenet_root "$IMAGENET_ROOT" "${FADING_ARGS[@]}"

python "$SCRIPT_DIR/estimate_information.py" --ckpt "$ROOT/train/best.pt" --packet_mode group --group_h "$GROUP_H" --group_w "$GROUP_W" \
  --model gauss --stages "$STAGES" --bits "$BITS" --batch "${INFO_BATCH:-8}" --epochs "${INFO_EPOCHS:-10}" --norm \
  --out "$ROOT/information.json" --imagenet_root "$IMAGENET_ROOT"

python "$SCRIPT_DIR/build_policy.py" --info "$ROOT/information.json" --channel "$CHANNEL" \
  --snrs="${SNRS:--5,0,5,10,15,20}" --cbrs="${CBRS:-0.00521,0.0104167,0.015625}" --out "$ROOT/policy.json"

python "$SCRIPT_DIR/test.py" --ckpt "$ROOT/train/best.pt" --policy "$ROOT/policy.json" --channel "$CHANNEL" --model gauss \
  --stages "$STAGES" --bits "$BITS" --batch "${TEST_BATCH:-24}" --norm \
  --snrs="${SNRS:--5,0,5,10,15,20}" --cbrs="${CBRS:-0.00521,0.0104167,0.015625}" \
  --out_dir "$ROOT/test" --kodak_root "$KODAK_ROOT"
