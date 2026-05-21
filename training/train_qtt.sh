#!/usr/bin/env bash
set -euo pipefail

# Representation-driven expert training. Override any variable from the command line.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON=${PYTHON:-python}
MASTER_PORT=${MASTER_PORT:-29501}
TRAIN_ROOT=${TRAIN_ROOT:-../datasets/train}
VAL_ROOT=${VAL_ROOT:-../datasets/val}
MODEL_ID=${MODEL_ID:-../pretrained/dinov3-vitl16-pretrain-lvd1689m}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-1e-5}
BACKBONE_LR=${BACKBONE_LR:-5e-5}
WD=${WD:-0}
BACKBONE_WD=${BACKBONE_WD:-0.01}
CROP_SIZE=${CROP_SIZE:-336}
DROPOUT=${DROPOUT:-0.5}
HFLIP_PROB=${HFLIP_PROB:-0.5}
PAD_MODE=${PAD_MODE:-reflect}
LOSS_TYPE=${LOSS_TYPE:-ce}
LABEL_SMOOTH=${LABEL_SMOOTH:-0}
OHEM_RATIO=${OHEM_RATIO:-0.5}
OHEM_TARGET=${OHEM_TARGET:-fake}
OHEM_REAL_W=${OHEM_REAL_W:-0.2}
CONSISTENCY_W=${CONSISTENCY_W:-0.1}
DYNAMIC_QTT=${DYNAMIC_QTT:-1}
QTT_ALPHA=${QTT_ALPHA:-0.5}
TRAIN_BACKBONE=${TRAIN_BACKBONE:-frozen}
UNFREEZE_LAST_N=${UNFREEZE_LAST_N:-0}
USE_LORA=${USE_LORA:-1}
SAVE_DIR=${SAVE_DIR:-../checkpoints/robust_experts}
RESUME=${RESUME:-}
AMP=${AMP:-1}

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NPROC=${NPROC:-${#GPU_ARRAY[@]}}
LORA_FLAG=()
[ "$USE_LORA" = "1" ] && LORA_FLAG=(--use_lora)

args=(
  --train_root "$TRAIN_ROOT" --val_root "$VAL_ROOT" --model_id "$MODEL_ID"
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" --backbone_lr "$BACKBONE_LR"
  --weight_decay "$WD" --backbone_weight_decay "$BACKBONE_WD"
  --crop_size "$CROP_SIZE" --dropout "$DROPOUT" --hflip_prob "$HFLIP_PROB" --pad_mode "$PAD_MODE"
  --loss_type "$LOSS_TYPE" --label_smoothing "$LABEL_SMOOTH"
  --ohem_ratio "$OHEM_RATIO" --ohem_target "$OHEM_TARGET" --ohem_real_weight "$OHEM_REAL_W"
  --consistency_weight "$CONSISTENCY_W"
  --dynamic_qtt "$DYNAMIC_QTT" --qtt_alpha "$QTT_ALPHA"
  --train_backbone "$TRAIN_BACKBONE" --unfreeze_last_n "$UNFREEZE_LAST_N"
  "${LORA_FLAG[@]}"
  --save_dir "$SAVE_DIR"
)
[ -n "$RESUME" ] && args+=(--resume "$RESUME")
[ "$AMP" = "1" ] && args+=(--amp)

if [ "$NPROC" -ge 2 ]; then
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train_linear_fixedval_dynamic_qtt.py "${args[@]}"
else
  "$PYTHON" train_linear_fixedval_dynamic_qtt.py "${args[@]}"
fi
