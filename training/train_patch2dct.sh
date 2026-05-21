#!/usr/bin/env bash
set -euo pipefail

# Artifact-driven expert training. Override any variable from the command line.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON=${PYTHON:-python}
MASTER_PORT=${MASTER_PORT:-29500}
TRAIN_ROOT=${TRAIN_ROOT:-../datasets/train}
VAL_ROOT=${VAL_ROOT:-../datasets/val}
MODEL_ID=${MODEL_ID:-../pretrained/dinov3-vits16-pretrain-lvd1689m}
INPUT_MODE=${INPUT_MODE:-jpeg_dct}
DCT_SCALES=${DCT_SCALES:-1,2}
DCT_BANDS=${DCT_BANDS:-1-2,3-4,5-6,7-8,9-10,11-14}
TRAIN_JPEG_PROB=${TRAIN_JPEG_PROB:-0.3}
TRAIN_JPEG_MIN=${TRAIN_JPEG_MIN:-85}
TRAIN_JPEG_MAX=${TRAIN_JPEG_MAX:-95}
CROP_SIZE=${CROP_SIZE:-224}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-5}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
SAVE_DIR=${SAVE_DIR:-../checkpoints/artifact_driven_experts}
SAVE_BEST_PREFIX=${SAVE_BEST_PREFIX:-pixel_expert}
RESUME=${RESUME:-}
AMP=${AMP:-1}

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NPROC=${NPROC:-${#GPU_ARRAY[@]}}

args=(
  --train_root "$TRAIN_ROOT" --val_root "$VAL_ROOT"
  --model_id "$MODEL_ID"
  --input_mode "$INPUT_MODE" --use_input_adapter --align_crop
  --dct_scales "$DCT_SCALES" --dct_bands "$DCT_BANDS"
  --train_jpeg_prob "$TRAIN_JPEG_PROB" --train_jpeg_min "$TRAIN_JPEG_MIN" --train_jpeg_max "$TRAIN_JPEG_MAX"
  --crop_size "$CROP_SIZE" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR"
  --num_workers "$NUM_WORKERS" --seed "$SEED"
  --save_dir "$SAVE_DIR" --save_best_prefix "$SAVE_BEST_PREFIX"
)
[ -n "$RESUME" ] && args+=(--resume "$RESUME")
[ "$AMP" = "1" ] && args+=(--amp)

if [ "$NPROC" -ge 2 ]; then
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train_artifact_driven_patch2dct.py "${args[@]}"
else
  "$PYTHON" train_artifact_driven_patch2dct.py "${args[@]}"
fi
