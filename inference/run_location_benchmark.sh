#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT=${DATASET_ROOT:-${ROOT_DIR}/../datasets/test}
OUT_ROOT=${OUT_ROOT:-${ROOT_DIR}/../outputs/location_eval}
GPU_ID=${GPU_ID:-0}
# Localization uses the representation-driven expert. The released file name remains representation_driven_expert.pt.
REPRESENTATION_CKPT=${REPRESENTATION_CKPT:-${CKPT:-${ROOT_DIR}/../checkpoints/representation_driven_expert.pt}}
BATCH_SIZE=${BATCH_SIZE:-32}

mkdir -p "$OUT_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" python "$ROOT_DIR/moe_location.py" \
  --data_root "$DATASET_ROOT" \
  --out_dir "$OUT_ROOT" \
  --ckpt "$REPRESENTATION_CKPT" \
  --batch_size "$BATCH_SIZE" \
  --amp \
  "$@"
