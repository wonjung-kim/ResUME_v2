#!/usr/bin/env bash
# Central dataset-path configuration for ResUME.
# Edit only this file (or export the same variables before running a launcher).
# Paths must be absolute or valid relative paths on the machine where training/testing runs.

# ImageNet root MUST contain:
#   ${IMAGENET_ROOT}/train/
#   ${IMAGENET_ROOT}/val/
export IMAGENET_ROOT="${IMAGENET_ROOT:-}"

# Kodak root MUST directly contain kodim01.png, kodim02.png, ...
export KODAK_ROOT="${KODAK_ROOT:-}"

# Optional legacy/extra dataset root.
export DIV2K_ROOT="${DIV2K_ROOT:-}"

# Mirror the names expected by the Python loaders.
export RESUME_IMAGENET_ROOT="${RESUME_IMAGENET_ROOT:-$IMAGENET_ROOT}"
export RESUME_KODAK_ROOT="${RESUME_KODAK_ROOT:-$KODAK_ROOT}"
export RESUME_DIV2K_ROOT="${RESUME_DIV2K_ROOT:-$DIV2K_ROOT}"
