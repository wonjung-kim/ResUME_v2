#!/usr/bin/env bash
set -euo pipefail

# Complete stage-level theorem pipeline (G=1).
# Required: ImageNet under IMAGENET_ROOT/RESUME_IMAGENET_ROOT and Kodak under KODAK_ROOT/RESUME_KODAK_ROOT.
ROOT="${ROOT:-./output_stage}"
CHANNEL="${CHANNEL:-awgn}"
FADING_ARGS=()
if [[ "$CHANNEL" == "rayleigh" ]]; then FADING_ARGS+=(--fading); fi
DATA_ARGS=()
if [[ -n "${IMAGENET_ROOT:-${RESUME_IMAGENET_ROOT:-}}" ]]; then DATA_ARGS+=(--imagenet_root "${IMAGENET_ROOT:-${RESUME_IMAGENET_ROOT}}"); fi
KODAK_ARGS=()
if [[ -n "${KODAK_ROOT:-${RESUME_KODAK_ROOT:-}}" ]]; then KODAK_ARGS+=(--kodak_root "${KODAK_ROOT:-${RESUME_KODAK_ROOT}}"); fi

mkdir -p "$ROOT"
python train.py --packet_mode stage --model gauss --stages 4 --bits 12 --batch "${BATCH:-36}" \
  --epochs "${EPOCHS:-200}" --warmup_epochs "${WARMUP_EPOCHS:-20}" --snr_min 0 --snr_max 10 \
  --norm --out_dir "$ROOT/train" "${FADING_ARGS[@]}" "${DATA_ARGS[@]}"

python estimate_information.py --ckpt "$ROOT/train/best.pt" --packet_mode stage --model gauss --stages 4 --bits 12 \
  --batch "${INFO_BATCH:-8}" --epochs "${INFO_EPOCHS:-10}" --norm --out "$ROOT/information.json" "${DATA_ARGS[@]}"

python build_policy.py --info "$ROOT/information.json" --channel "$CHANNEL" \
  --snrs="${SNRS:--5,0,5,10,15,20}" --cbrs="${CBRS:-0.00521,0.0104167,0.015625}" --out "$ROOT/policy.json"

python test.py --ckpt "$ROOT/train/best.pt" --policy "$ROOT/policy.json" --channel "$CHANNEL" --model gauss \
  --stages 4 --bits 12 --batch "${TEST_BATCH:-24}" --norm \
  --snrs="${SNRS:--5,0,5,10,15,20}" --cbrs="${CBRS:-0.00521,0.0104167,0.015625}" \
  --out_dir "$ROOT/test" "${KODAK_ARGS[@]}"
