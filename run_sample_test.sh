#!/usr/bin/env bash
set -euo pipefail

CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH=/path/to/best.pt}"
SAMPLE_DIR="${SAMPLE_DIR:?Set SAMPLE_DIR=/path/to/images}"
PACKET_MODE="${PACKET_MODE:-stage}"
GROUP_H="${GROUP_H:-4}"
GROUP_W="${GROUP_W:-4}"
MOD_ORDERS="${MOD_ORDERS:-2,4,16,64}"
SNR_LIST="${SNR_LIST:-0,5,10,15,20}"
OUT_DIR="${OUT_DIR:-./sample_test_out}"

python sample_test.py \
  --ckpt "$CKPT_PATH" \
  --sample_dir "$SAMPLE_DIR" \
  --mod_orders "$MOD_ORDERS" \
  --stages 4 \
  --bits 12 \
  --active_stages 4 \
  --packet_mode "$PACKET_MODE" \
  --group_h "$GROUP_H" \
  --group_w "$GROUP_W" \
  --img_size 128 \
  --snrs "$SNR_LIST" \
  --out_dir "$OUT_DIR"
