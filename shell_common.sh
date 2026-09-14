#!/usr/bin/env bash
# Shared helper sourced by the launch scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_CONFIG="${DATA_CONFIG:-$SCRIPT_DIR/dataset_paths.sh}"
if [[ -f "$DATA_CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$DATA_CONFIG"
fi

IMAGENET_ROOT="${IMAGENET_ROOT:-${RESUME_IMAGENET_ROOT:-}}"
KODAK_ROOT="${KODAK_ROOT:-${RESUME_KODAK_ROOT:-}}"

require_imagenet() {
  if [[ -z "${IMAGENET_ROOT:-}" ]]; then
    echo "[ERROR] ImageNet path is not set." >&2
    echo "Edit dataset_paths.sh or run: IMAGENET_ROOT=/path/to/ImageNet bash $0" >&2
    exit 2
  fi
  if [[ ! -d "$IMAGENET_ROOT/train" || ! -d "$IMAGENET_ROOT/val" ]]; then
    echo "[ERROR] IMAGENET_ROOT must contain train/ and val/: $IMAGENET_ROOT" >&2
    exit 2
  fi
}

require_kodak() {
  if [[ -z "${KODAK_ROOT:-}" ]]; then
    echo "[ERROR] Kodak path is not set." >&2
    echo "Edit dataset_paths.sh or run: KODAK_ROOT=/path/to/Kodak bash $0" >&2
    exit 2
  fi
  if [[ ! -d "$KODAK_ROOT" ]]; then
    echo "[ERROR] KODAK_ROOT does not exist: $KODAK_ROOT" >&2
    exit 2
  fi
}
