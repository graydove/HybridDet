#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT=${DATASET_ROOT:-${ROOT_DIR}/../datasets/test}
CALIBRATION_ROOT=${CALIBRATION_ROOT:-${ROOT_DIR}/../datasets/val}
OUT_ROOT=${OUT_ROOT:-${ROOT_DIR}/../outputs/moe_eval}
GPU_IDS=${GPU_IDS:-0}
# Paper terms are artifact-driven / representation-driven. The released file names are artifact_driven_expert.pt and representation_driven_expert.pt.
ARTIFACT_CKPT=${ARTIFACT_CKPT:-${ROOT_DIR}/../checkpoints/artifact_driven_expert.pt}
REPRESENTATION_CKPT=${REPRESENTATION_CKPT:-${ROOT_DIR}/../checkpoints/representation_driven_expert.pt}
ARTIFACT_THR=${ARTIFACT_THR:-0.992167}
BATCH_SIZE=${BATCH_SIZE:-64}

mkdir -p "$OUT_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_IDS" python "$ROOT_DIR/moe_gate.py" \
  --data_root "$DATASET_ROOT" \
  --out_dir "$OUT_ROOT" \
  --artifact_ckpt "$ARTIFACT_CKPT" \
  --representation_ckpt "$REPRESENTATION_CKPT" \
  --calibration_root "$CALIBRATION_ROOT" \
  --routing_score_mode calibrated \
  --artifact_thr "$ARTIFACT_THR" \
  --batch_size "$BATCH_SIZE" \
  --amp \
  "$@"
