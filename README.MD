# HybridDet Anonymous Release

This repository contains the anonymized training and inference code for the CIKM 2026 submission. Datasets, DINOv3 pretrained backbone folders, logs, and experimental outputs are **not** included. The released `.pt` expert checkpoints are included under `checkpoints/`.

## Repository layout

```text
HybridDet-Anonymous/
├── training/                  # expert training code
│   ├── train_patch2dct.sh           # artifact-driven expert training
│   ├── train_bp.sh            # artifact-driven adapter expert training
│   ├── train_qtt.sh           # representation-driven expert training
│   └── train_linear_*.py
├── inference/                 # MoE inference and localization code
│   ├── moe_gate.py            # two-expert MoE routing inference
│   ├── moe_location.py        # sliding-window localization inference
│   ├── run_moe_benchmark.sh
│   └── run_location_benchmark.sh
├── datasets/
│   ├── train/                 # put training set here
│   ├── val/                   # put validation / calibration set here
│   └── test/                  # put test set here
├── pretrained/                # put DINOv3 / CLIP pretrained model folders here
├── checkpoints/               # released .pt expert checkpoints
├── outputs/                   # inference outputs
└── requirements.txt
```

## Environment

Install a PyTorch build matching your CUDA version first, then install the remaining dependencies:

```bash
cd HybridDet-Anonymous
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # adjust if needed
pip install -r requirements.txt
```

## Data and model placeholders

The scripts expect ImageFolder-style binary classification data. Common class-folder names such as `0_real`/`1_fake`, `real`/`fake`, or `nature`/`ai` are supported by the loaders.

```text
datasets/train/
  0_real/ ...
  1_fake/ ...

datasets/val/
  0_real/ ...
  1_fake/ ...

datasets/test/
  0_real/ ...
  1_fake/ ...
```

DINOv3 pretrained backbone weights are **not** released in this anonymous package. If you have access to them, place the local backbone folders under `pretrained/`, for example:

```text
pretrained/dinov3-vits16-pretrain-lvd1689m/
pretrained/dinov3-vitl16-pretrain-lvd1689m/
pretrained/dinov3-vith16plus-pretrain-lvd1689m/
```

## Training

Run training from the `training/` directory. All wrapper scripts use relative placeholder paths and can be overridden by environment variables.

### 1. Train the artifact-driven expert

```bash
cd training
CUDA_VISIBLE_DEVICES=0 \
TRAIN_ROOT=../datasets/train \
VAL_ROOT=../datasets/val \
MODEL_ID=../pretrained/dinov3-vits16-pretrain-lvd1689m \
SAVE_DIR=../checkpoints/artifact_driven_experts \
bash train_patch2dct.sh
```

### 2. Train the artifact-driven adapter expert

```bash
cd training
CUDA_VISIBLE_DEVICES=0 \
TRAIN_ROOT=../datasets/train \
VAL_ROOT=../datasets/val \
MODEL_ID=../pretrained/dinov3-vits16-pretrain-lvd1689m \
SAVE_DIR=../checkpoints/artifact_driven_experts \
bash train_bp.sh
```

### 3. Train the representation-driven expert

```bash
cd training
CUDA_VISIBLE_DEVICES=0 \
TRAIN_ROOT=../datasets/train \
VAL_ROOT=../datasets/val \
MODEL_ID=../pretrained/dinov3-vitl16-pretrain-lvd1689m \
SAVE_DIR=../checkpoints/robust_experts \
bash train_qtt.sh
```

For multi-GPU training, set `CUDA_VISIBLE_DEVICES=0,1,...`; the wrappers will launch `torchrun` automatically.

## Inference

MoE inference combines the artifact-driven expert and the representation-driven expert. This package includes exactly two released expert checkpoints named with the paper terminology:

```text
checkpoints/artifact_driven_expert.pt
checkpoints/representation_driven_expert.pt
```

Weight naming follows the paper terminology: `artifact_driven_expert.pt` is the artifact-driven expert, and `representation_driven_expert.pt` is the representation-driven expert.

DINOv3 pretrained backbone folders are still required separately under `pretrained/` or via `--model_id`/checkpoint metadata resolution, but those pretrained weights are not included.

Run MoE evaluation:

```bash
cd HybridDet-Anonymous
GPU_IDS=0 \
DATASET_ROOT=./datasets/test \
CALIBRATION_ROOT=./datasets/val \
ARTIFACT_CKPT=./checkpoints/artifact_driven_expert.pt \
REPRESENTATION_CKPT=./checkpoints/representation_driven_expert.pt \
bash inference/run_moe_benchmark.sh
```

The default outputs are written to `outputs/moe_eval/`. You can also call the Python entry directly:

```bash
python inference/moe_gate.py \
  --data_root ./datasets/test \
  --calibration_root ./datasets/val \
  --artifact_ckpt ./checkpoints/artifact_driven_expert.pt \
  --representation_ckpt ./checkpoints/representation_driven_expert.pt \
  --out_dir ./outputs/moe_eval \
  --routing_score_mode calibrated \
  --batch_size 64 \
  --amp
```

## Localization inference

For sliding-window localization on fake images:

```bash
cd HybridDet-Anonymous
GPU_ID=0 \
DATASET_ROOT=./datasets/test \
REPRESENTATION_CKPT=./checkpoints/representation_driven_expert.pt \
bash inference/run_location_benchmark.sh
```

or directly:

```bash
python inference/moe_location.py \
  --data_root ./datasets/test \
  --representation_ckpt ./checkpoints/representation_driven_expert.pt \
  --out_dir ./outputs/location_eval \
  --batch_size 32 \
  --amp
```

## Notes for anonymous release

- No datasets, DINOv3 pretrained backbone weights, logs, paper/manuscript files, or experimental outputs are included.
- Released `.pt` expert checkpoints are included in `checkpoints/`.
- Empty dataset/pretrained/output placeholder folders are tracked with `.gitkeep` files.
- Default paths use repository-relative locations; override any path with command-line arguments or environment variables shown above.
