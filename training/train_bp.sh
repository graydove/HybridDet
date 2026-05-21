#!/usr/bin/env bash
set -euo pipefail

# Artifact-driven adapter expert training. Override any variable from the command line.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON=${PYTHON:-python}
MASTER_PORT=${MASTER_PORT:-29500}
TRAIN_ROOT=${TRAIN_ROOT:-../datasets/train}
VAL_ROOT=${VAL_ROOT:-../datasets/val}
MODEL_ID=${MODEL_ID:-../pretrained/dinov3-vits16-pretrain-lvd1689m}
INPUT_MODE=${INPUT_MODE:-jpeg_dct}
CROP_SIZE=${CROP_SIZE:-336}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-5}
WD=${WD:-0.0}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-42}
SAVE_DIR=${SAVE_DIR:-../checkpoints/artifact_driven_experts}
SAVE_BEST_PREFIX=${SAVE_BEST_PREFIX:-pixel_adapter_expert}
RESUME=${RESUME:-}
AMP=${AMP:-1}

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NPROC=${NPROC:-${#GPU_ARRAY[@]}}

args=(
  --train_root "$TRAIN_ROOT" --val_root "$VAL_ROOT"
  --model_id "$MODEL_ID"
  --input_mode "$INPUT_MODE" --crop_size "$CROP_SIZE"
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" --weight_decay "$WD"
  --num_workers "$NUM_WORKERS" --seed "$SEED" --align_crop --unfreeze_last_n 2
  --save_dir "$SAVE_DIR" --save_best_prefix "$SAVE_BEST_PREFIX"
)
[ -n "$RESUME" ] && args+=(--resume "$RESUME")
[ "$AMP" = "1" ] && args+=(--amp)

if [ "$NPROC" -ge 2 ]; then
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train_artifact_driven_adapter.py "${args[@]}"
else
  "$PYTHON" train_artifact_driven_adapter.py "${args[@]}"
fi
