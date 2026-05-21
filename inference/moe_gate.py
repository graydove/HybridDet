import argparse
import csv
import os
import re
import sys
import logging

import math
import random
from pathlib import Path
from typing import List, Union
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection


def _safe_np_pad(arr: np.ndarray, pads, mode: str):
    try:
        if mode == "constant":
            return np.pad(arr, pads, mode="constant", constant_values=0)
        if mode == "edge":
            return np.pad(arr, pads, mode="edge")
        if mode == "reflect":
            return np.pad(arr, pads, mode="reflect")
        if mode == "symmetric":
            return np.pad(arr, pads, mode="symmetric")
        return np.pad(arr, pads, mode="edge")
    except Exception:
        return np.pad(arr, pads, mode="edge")


class CropNoResizeTransform:
    """Pad to >= crop_size then center crop; avoids any resize interpolation."""

    def __init__(
        self,
        crop_size: int,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        pad_mode: str = "edge",
    ):
        self.crop_size = int(crop_size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.pad_mode = pad_mode

    def _pad_if_needed(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        pad_w = max(0, self.crop_size - w)
        pad_h = max(0, self.crop_size - h)
        if pad_w == 0 and pad_h == 0:
            return img

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        if self.pad_mode == "constant":
            return ImageOps.expand(img, border=(left, top, right, bottom), fill=0)

        arr = np.array(img)  # HWC, uint8
        pads = ((top, bottom), (left, right), (0, 0))
        arr2 = _safe_np_pad(arr, pads, mode=self.pad_mode)
        return Image.fromarray(arr2)

    def _center_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        th = tw = self.crop_size
        j = int(round((w - tw) / 2.0))
        i = int(round((h - th) / 2.0))
        j = int(max(0, min(j, w - tw)))
        i = int(max(0, min(i, h - th)))
        return img.crop((j, i, j + tw, i + th))

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self._pad_if_needed(img)
        crop = self._center_crop(img)
        arr = np.array(crop).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - self.mean) / self.std
        return t


class DCTBlockHF9CenterCropTransform:
    """Deterministic center-crop DCT transform (matches infer_linear_fixedval_vits)."""

    def __init__(self, crop_size: int, block: int = 8, masks=(3, 5, 7)):
        self.crop_size = int(crop_size)
        self.block = int(block)
        self.C = self._build_dct_matrix(self.block)
        self.Ct = self.C.t().contiguous()
        self.masks = [self._hf_mask(self.block, th) for th in masks]

    def _build_dct_matrix(self, N):
        C = torch.zeros(N, N)
        for k in range(N):
            a = math.sqrt(1 / N) if k == 0 else math.sqrt(2 / N)
            for n in range(N):
                C[k, n] = a * math.cos(math.pi * (2 * n + 1) * k / (2 * N))
        return C

    def _hf_mask(self, N, th):
        m = torch.zeros(N, N, dtype=torch.bool)
        for u in range(N):
            for v in range(N):
                if u + v >= th and not (u == 0 and v == 0):
                    m[u, v] = True
        return m

    def _center_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        th = tw = self.crop_size
        j = int(round((w - tw) / 2.0))
        i = int(round((h - th) / 2.0))
        j = int(max(0, min(j, w - tw)))
        i = int(max(0, min(i, h - th)))
        return img.crop((j, i, j + tw, i + th))

    def _rgb_to_ycbcr(self, t):
        r, g, b = t
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
        return (y * 255 - 128), (cb * 255 - 128), (cr * 255 - 128)

    def _dct_energy(self, x, mask):
        B = self.block
        h, w = x.shape
        if h % B != 0 or w % B != 0:
            raise ValueError(f"DCT input size must be divisible by {B}, got {h}x{w}.")
        blocks = x.unfold(0, B, B).unfold(1, B, B)
        C = self.C.to(x.device)
        Ct = self.Ct.to(x.device)
        coef = torch.matmul(torch.matmul(C, blocks), Ct)
        mag = coef.abs()
        e = (mag * mask.to(x.device)).sum(dim=(-1, -2))
        return e.repeat_interleave(B, 0).repeat_interleave(B, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        img = img.convert("RGB")
        img = self._center_crop(img)
        t = TF.to_tensor(img)
        y, cb, cr = self._rgb_to_ycbcr(t)
        chans = []
        for comp in (y, cb, cr):
            for m in self.masks:
                e = self._dct_energy(comp, m)
                chans.append(torch.log1p(e))
        return torch.stack(chans, dim=0)


def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def adapt_patch_embed_channels(backbone, in_channels: int, device: str):
    patch_embed = getattr(backbone, "embeddings", None)
    patch_embed = getattr(patch_embed, "patch_embeddings", None)
    proj = getattr(patch_embed, "projection", None) if patch_embed is not None else None
    conv = proj if isinstance(proj, nn.Conv2d) else patch_embed if isinstance(patch_embed, nn.Conv2d) else None
    if conv is None or conv.in_channels == in_channels:
        return
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
    ).to(device)
    with torch.no_grad():
        repeat = int(math.ceil(in_channels / conv.in_channels))
        w = conv.weight.data
        w_rep = w.repeat(1, repeat, 1, 1)[:, :in_channels, :, :].clone()
        w_rep *= conv.in_channels / float(in_channels)
        new_conv.weight.data.copy_(w_rep)
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.data.copy_(conv.bias.data)
    if proj is not None:
        patch_embed.projection = new_conv
    else:
        backbone.embeddings.patch_embeddings = new_conv


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LNDropLinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.ln(x)))


class HFExpert(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        return extract_feats(outputs)


def extract_feats(outputs):
    if getattr(outputs, "image_embeds", None) is not None:
        return outputs.image_embeds
    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output
    if getattr(outputs, "vision_model_output", None) is not None:
        vout = outputs.vision_model_output
        if getattr(vout, "pooler_output", None) is not None:
            return vout.pooler_output
        if getattr(vout, "last_hidden_state", None) is not None:
            return vout.last_hidden_state[:, 0, :]
    if getattr(outputs, "last_hidden_state", None) is not None:
        return outputs.last_hidden_state[:, 0, :]
    raise RuntimeError("Backbone outputs missing usable features (image_embeds/pooler_output/last_hidden_state).")


def create_backbone(model_id: str):
    try:
        cfg = AutoConfig.from_pretrained(model_id)
        model_type = getattr(cfg, "model_type", "")
    except Exception:
        model_type = ""
    if model_type == "clip":
        return CLIPVisionModelWithProjection.from_pretrained(model_id)
    return AutoModel.from_pretrained(model_id)


def get_mean_std(model_id: str):
    try:
        proc = AutoImageProcessor.from_pretrained(model_id)
        mean = tuple(getattr(proc, "image_mean", (0.485, 0.456, 0.406)))
        std = tuple(getattr(proc, "image_std", (0.229, 0.224, 0.225)))
        return mean, std
    except Exception:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _parse_lora_target_modules(raw: str) -> Union[str, List[str]]:
    text = (raw or "").strip()
    if not text or text.lower() in ("auto", "none"):
        return []
    normalized = text.lower().replace("_", "-")
    if normalized in ("all-linear", "all"):
        return "all-linear"
    return [t for t in (s.strip() for s in text.split(",")) if t]


def _infer_lora_target_modules(backbone: nn.Module) -> List[str]:
    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "query",
        "key",
        "value",
        "qkv",
        "to_q",
        "to_k",
        "to_v",
        "to_qkv",
    }
    found = set()
    for name, module in backbone.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        tail = name.rsplit(".", 1)[-1]
        if tail in candidates:
            found.add(tail)
    return sorted(found)


def maybe_apply_lora_from_ckpt(backbone: nn.Module, ckpt: dict, logger: logging.Logger):
    if not ckpt.get("use_lora", False):
        return backbone
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise RuntimeError("LoRA requires peft; please install/upgrade peft.") from exc

    lora_r = int(ckpt.get("lora_r", 8))
    lora_alpha = int(ckpt.get("lora_alpha", 16))
    lora_dropout = float(ckpt.get("lora_dropout", 0.0))
    lora_target_modules = ckpt.get("lora_target_modules", "auto")
    lora_bias = str(ckpt.get("lora_bias", "none"))

    target_modules = _parse_lora_target_modules(lora_target_modules)
    if not target_modules:
        target_modules = _infer_lora_target_modules(backbone)
    if not target_modules:
        raise RuntimeError(
            "LoRA enabled but no target modules matched. "
            "Please set lora_target_modules in the checkpoint (e.g., 'q_proj,v_proj' or 'query,value')."
        )

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=lora_bias,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    backbone = get_peft_model(backbone, lora_cfg)
    logger.info(
        f"[LoRA] enabled: r={lora_r} alpha={lora_alpha} "
        f"dropout={lora_dropout} bias={lora_bias} targets={target_modules}"
    )
    return backbone


def get_ckpt_patch_in_channels(state_dict: dict):
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            continue
        k = key.lower()
        if "patch" in k and ("proj" in k or "projection" in k or "patch_embeddings" in k):
            return int(tensor.shape[1])
    for _key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            return int(tensor.shape[1])
    return None


def remap_backbone_state_dict_keys(state_dict: dict, target_keys, logger: logging.Logger, ckpt_path: str):
    target_keys = set(target_keys)
    has_target_model_prefix = any(k.startswith("model.layer.") for k in target_keys)
    has_ckpt_model_prefix = any(k.startswith("model.layer.") for k in state_dict.keys())
    has_target_plain_layer = any(k.startswith("layer.") for k in target_keys)
    has_ckpt_plain_layer = any(k.startswith("layer.") for k in state_dict.keys())

    if has_target_model_prefix and has_ckpt_plain_layer and not has_ckpt_model_prefix:
        logger.info(f"[Compat] remapped backbone keys for {ckpt_path}: layer.* -> model.layer.*")
        return {("model." + k) if k.startswith("layer.") else k: v for k, v in state_dict.items()}

    if has_target_plain_layer and has_ckpt_model_prefix and not has_ckpt_plain_layer:
        logger.info(f"[Compat] remapped backbone keys for {ckpt_path}: model.layer.* -> layer.*")
        return {(k[len("model."):] if k.startswith("model.layer.") else k): v for k, v in state_dict.items()}

    return state_dict


def load_expert(ckpt_path: str, device: torch.device, logger: logging.Logger):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_id = ckpt.get("model_id", "")
    if not model_id:
        raise ValueError(f"Checkpoint missing model_id: {ckpt_path}")
    mean = tuple(ckpt.get("mean", get_mean_std(model_id)[0]))
    std = tuple(ckpt.get("std", get_mean_std(model_id)[1]))
    crop_size = int(ckpt.get("crop_size", 336)) if ckpt.get("crop_size", None) is not None else None

    backbone = create_backbone(model_id).to(device)
    backbone = maybe_apply_lora_from_ckpt(backbone, ckpt, logger)

    backbone_state = remap_backbone_state_dict_keys(ckpt["backbone"], backbone.state_dict().keys(), logger, ckpt_path)
    ckpt_in_channels = get_ckpt_patch_in_channels(backbone_state)
    if ckpt_in_channels is not None:
        adapt_patch_embed_channels(backbone, in_channels=ckpt_in_channels, device=device)

    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if unexpected:
        logger.warning(f"[Warn] {ckpt_path} backbone unexpected={len(unexpected)}")
    if missing:
        logger.warning(f"[Warn] {ckpt_path} backbone missing={len(missing)} unexpected={len(unexpected)}")

    feat_dim = int(ckpt.get("feat_dim", 0))
    head_state = ckpt["head"]
    if any(k.startswith("ln.") for k in head_state.keys()):
        head = LNDropLinearHead(in_dim=feat_dim, num_classes=2).to(device)
    else:
        head = LinearHead(in_dim=feat_dim, num_classes=2).to(device)
    missing, unexpected = head.load_state_dict(head_state, strict=False)
    if missing or unexpected:
        logger.warning(f"[Warn] {ckpt_path} head missing={len(missing)} unexpected={len(unexpected)}")

    expert = HFExpert(backbone, head)
    expert.eval()
    for p in expert.parameters():
        p.requires_grad = False
    return expert, model_id, feat_dim, ckpt_in_channels, mean, std, crop_size


class GroupedClassDataset(Dataset):
    def __init__(
        self,
        items,
        transform=None,
        route_transform=None,
        pixel_transform=None,
        semantic_transform=None,
    ):
        self.items = items
        self.transform = transform
        self.route_transform = route_transform or transform
        self.pixel_transform = pixel_transform or transform
        self.semantic_transform = semantic_transform or transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, group, subgroup = self.items[idx]
        img = Image.open(path).convert("RGB")
        base = img
        if self.transform:
            base = self.transform(img)
        route_img = self.route_transform(img) if self.route_transform else base
        pixel_img = self.pixel_transform(img) if self.pixel_transform else base
        semantic_img = self.semantic_transform(img) if self.semantic_transform else base
        return pixel_img, semantic_img, route_img, int(label), group, subgroup, path


def collect_grouped_class_paths(data_root, class_dir, label, group_name=None, exts=None):
    if not data_root:
        return []
    if not os.path.isdir(data_root):
        return []
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".jfif", ".tif", ".bmp", ".webp"}
    root = Path(data_root)
    items = []
    direct_cls = root / class_dir
    group_name = group_name or root.name
    if direct_cls.is_dir():
        for path in direct_cls.rglob("*"):
            if path.suffix.lower() in exts:
                items.append((str(path), label, group_name, group_name))
        return items
    for subgroup_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls_dir = subgroup_dir / class_dir
        if not cls_dir.is_dir():
            continue
        for path in cls_dir.rglob("*"):
            if path.suffix.lower() in exts:
                items.append((str(path), label, group_name, subgroup_dir.name))
    return items


def build_grouped_cls_loader(
    real_root,
    fake_root,
    transform,
    route_transform,
    pixel_transform,
    semantic_transform,
    group_name,
    batch_size,
    num_workers,
    device,
    max_images,
    seed,
):
    items = []
    items.extend(collect_grouped_class_paths(real_root, "0_real", 0, group_name=group_name))
    items.extend(collect_grouped_class_paths(fake_root, "1_fake", 1, group_name=group_name))
    if not items:
        return None, None
    if max_images and max_images > 0 and len(items) > max_images:
        rng = random.Random(seed)
        rng.shuffle(items)
        items = items[:max_images]
    ds = GroupedClassDataset(
        items,
        transform=transform,
        route_transform=route_transform,
        pixel_transform=pixel_transform,
        semantic_transform=semantic_transform,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, len(ds)


def collect_binary_calibration_paths(data_root, real_names=("0_real", "nature", "real"), fake_names=("1_fake", "ai", "fake"), exts=None):
    """Collect a flat binary calibration set from common real/fake directory names."""
    if not data_root or not os.path.isdir(data_root):
        return []
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".jfif", ".tif", ".bmp", ".webp"}

    root = Path(data_root)
    items = []

    def add_from_dir(dir_path: Path, label: int, tag: str):
        if not dir_path.is_dir():
            return
        for path in dir_path.rglob("*"):
            if path.suffix.lower() in exts:
                items.append((str(path), label, "CALIB", tag))

    lowered = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    for name in real_names:
        add_from_dir(lowered.get(name.lower(), root / name), 0, name)
    for name in fake_names:
        add_from_dir(lowered.get(name.lower(), root / name), 1, name)

    if items:
        return items

    # Fallback for nested validation roots.
    for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lowered = {p.name.lower(): p for p in group_dir.iterdir() if p.is_dir()}
        for name in real_names:
            add_from_dir(lowered.get(name.lower(), group_dir / name), 0, f"{group_dir.name}/{name}")
        for name in fake_names:
            add_from_dir(lowered.get(name.lower(), group_dir / name), 1, f"{group_dir.name}/{name}")
    return items


def build_calibration_loader(
    data_root,
    pixel_transform,
    semantic_transform,
    batch_size,
    num_workers,
    device,
    max_images,
    seed,
):
    items = collect_binary_calibration_paths(data_root)
    if not items:
        return None, None
    if max_images and max_images > 0 and len(items) > max_images:
        rng = random.Random(seed)
        rng.shuffle(items)
        items = items[:max_images]
    ds = GroupedClassDataset(
        items,
        transform=pixel_transform,
        route_transform=pixel_transform,
        pixel_transform=pixel_transform,
        semantic_transform=semantic_transform,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, len(ds)


def _binary_fake_prob(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    return (logits.float() / temperature).softmax(dim=1)[:, 1]


def calibrate_probability_threshold(raw_prob_threshold: float, temperature: float) -> float:
    """Map a raw binary-softmax threshold to its temperature-scaled probability value."""
    eps = 1e-12
    p = min(max(float(raw_prob_threshold), eps), 1.0 - eps)
    temperature = max(float(temperature), 1e-6)
    logit = math.log(p / (1.0 - p))
    return float(1.0 / (1.0 + math.exp(-logit / temperature)))


def fit_temperature_from_logits(logits: torch.Tensor, labels: torch.Tensor, logger: logging.Logger, name: str) -> float:
    """Fit a scalar temperature by minimizing NLL on held-out logits."""
    logits = logits.detach().float()
    labels = labels.detach().long()
    if logits.numel() == 0 or labels.numel() == 0:
        logger.warning(f"[Calibration] {name}: empty logits; using T=1.0")
        return 1.0

    log_t = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100, line_search_fn="strong_wolfe")
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_t).clamp(min=1e-3, max=1e3)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_t).clamp(min=1e-3, max=1e3).item())
    raw_nll = float(criterion(logits, labels).item())
    cal_nll = float(criterion(logits / temperature, labels).item())
    logger.info(f"[Calibration] {name}: T={temperature:.6f} raw_nll={raw_nll:.6f} cal_nll={cal_nll:.6f}")
    return temperature


@torch.no_grad()
def fit_expert_temperatures(
    pixel_expert,
    semantic_expert,
    loader,
    device,
    use_amp: bool,
    logger: logging.Logger,
):
    pixel_expert.eval()
    semantic_expert.eval()
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pixel_logits = []
    semantic_logits = []
    labels_all = []

    for pixel_imgs, semantic_imgs, _route_imgs, labels, _groups, _subgroups, _paths in tqdm(
        loader, desc="calibrate", total=len(loader)
    ):
        pixel_imgs = pixel_imgs.to(device, non_blocking=True)
        semantic_imgs = semantic_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")):
            feat_png = pixel_expert.forward_features(pixel_imgs)
            feat_jpeg = semantic_expert.forward_features(semantic_imgs)
            logits_png = pixel_expert.head(feat_png)
            logits_jpeg = semantic_expert.head(feat_jpeg)

        pixel_logits.append(logits_png.detach().float().cpu())
        semantic_logits.append(logits_jpeg.detach().float().cpu())
        labels_all.append(labels.detach().cpu())

    pixel_logits = torch.cat(pixel_logits, dim=0)
    semantic_logits = torch.cat(semantic_logits, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    pixel_t = fit_temperature_from_logits(pixel_logits, labels_all, logger, "artifact-driven")
    semantic_t = fit_temperature_from_logits(semantic_logits, labels_all, logger, "representation-driven")
    return pixel_t, semantic_t


def setup_logger(log_file: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    logger = logging.getLogger("router_rule_eval")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # 防止重复输出

    # 清理旧 handler（避免 notebook/多次运行重复写）
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def safe_name(name: str) -> str:
    name = name.strip().replace(os.sep, "_")
    name = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", name)
    return name or "group"


def _has_direct_class_dirs(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "0_real")) and os.path.isdir(os.path.join(path, "1_fake"))


def _has_nested_class_dirs(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isdir(p) and _has_direct_class_dirs(p):
            return True
    return False


def find_group_roots(data_root: str):
    """
    返回 [(group_name, group_root_path), ...]
    兼容两种结构：
      1) data_root/0_real, data_root/1_fake  -> 单组
      2) data_root/<group>/0_real, data_root/<group>/1_fake -> 多组
    """
    if _has_direct_class_dirs(data_root):
        return [("ALL", data_root)]

    groups = []
    if not os.path.isdir(data_root):
        return groups

    for name in sorted(os.listdir(data_root)):
        p = os.path.join(data_root, name)
        if not os.path.isdir(p):
            continue
        if _has_direct_class_dirs(p) or _has_nested_class_dirs(p):
            groups.append((name, p))
    return groups


def plot_prob_distributions(probs_png, probs_jpeg, out_path: str, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    probs_png = np.asarray(probs_png, dtype=np.float32)
    probs_jpeg = np.asarray(probs_jpeg, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, 51)

    plt.figure(figsize=(7, 5))
    plt.hist(probs_png, bins=bins, alpha=0.5, label="artifact-driven P(fake)", color="tab:blue", density=True)
    plt.hist(probs_jpeg, bins=bins, alpha=0.5, label="representation-driven P(fake)", color="tab:orange", density=True)
    plt.xlabel("P(fake)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def evaluate_router_rule(
    pixel_expert,
    semantic_expert,
    loader,
    device,
    use_amp: bool,
    artifact_thr: float,
    qtt_low: float,
    decision_threshold: float,
    pixel_temperature: float = 1.0,
    semantic_temperature: float = 1.0,
    routing_score_mode: str = "calibrated",
    logits_csv_writer=None,
):
    pixel_expert.eval()
    semantic_expert.eval()

    total = 0
    correct = 0
    route_png_total = 0
    real_total = 0
    fake_total = 0
    real_correct = 0
    fake_correct = 0

    subgroup_stats = {}

    probs_png_all = []
    probs_jpeg_all = []
    probs_png_real = []
    probs_jpeg_real = []
    probs_png_fake = []
    probs_jpeg_fake = []

    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32

    for pixel_imgs, semantic_imgs, _route_imgs, labels, groups, subgroups, paths in tqdm(loader, desc="eval", total=len(loader)):
        pixel_imgs = pixel_imgs.to(device, non_blocking=True)
        semantic_imgs = semantic_imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")):
            feat_png = pixel_expert.forward_features(pixel_imgs)
            feat_jpeg = semantic_expert.forward_features(semantic_imgs)
            logits_png = pixel_expert.head(feat_png)
            logits_jpeg = semantic_expert.head(feat_jpeg)

        prob_png_raw = _binary_fake_prob(logits_png, temperature=1.0)
        prob_jpeg_raw = _binary_fake_prob(logits_jpeg, temperature=1.0)
        prob_png_cal = _binary_fake_prob(logits_png, temperature=pixel_temperature)
        prob_jpeg_cal = _binary_fake_prob(logits_jpeg, temperature=semantic_temperature)

        if routing_score_mode == "raw":
            prob_png = prob_png_raw
            prob_jpeg = prob_jpeg_raw
        elif routing_score_mode == "calibrated":
            prob_png = prob_png_cal
            prob_jpeg = prob_jpeg_cal
        else:
            raise ValueError(f"Unsupported routing_score_mode={routing_score_mode!r}")

        probs_png_all.append(prob_png.detach().cpu())
        probs_jpeg_all.append(prob_jpeg.detach().cpu())

        labels_cpu = labels.detach().cpu()
        prob_png_cpu = prob_png.detach().cpu()
        prob_jpeg_cpu = prob_jpeg.detach().cpu()
        prob_png_raw_cpu = prob_png_raw.detach().cpu()
        prob_jpeg_raw_cpu = prob_jpeg_raw.detach().cpu()
        prob_png_cal_cpu = prob_png_cal.detach().cpu()
        prob_jpeg_cal_cpu = prob_jpeg_cal.detach().cpu()
        logits_png_cpu = logits_png.detach().float().cpu()
        logits_jpeg_cpu = logits_jpeg.detach().float().cpu()

        real_mask = labels_cpu == 0
        fake_mask = labels_cpu == 1
        if real_mask.any():
            probs_png_real.append(prob_png_cpu[real_mask])
            probs_jpeg_real.append(prob_jpeg_cpu[real_mask])
        if fake_mask.any():
            probs_png_fake.append(prob_png_cpu[fake_mask])
            probs_jpeg_fake.append(prob_jpeg_cpu[fake_mask])

        use_png = (prob_png >= artifact_thr) & (prob_jpeg < qtt_low)
        p_out = torch.where(use_png, prob_png, prob_jpeg)
        pred = (p_out >= decision_threshold).long()
        pred_cpu = pred.detach().cpu()
        use_png_cpu = use_png.detach().cpu()
        p_out_cpu = p_out.detach().cpu()

        batch_correct = (pred == labels).sum().item()
        correct += batch_correct
        total += labels.size(0)
        real_mask_tensor = labels == 0
        fake_mask_tensor = labels == 1
        real_total += real_mask_tensor.sum().item()
        fake_total += fake_mask_tensor.sum().item()
        real_correct += ((pred == labels) & real_mask_tensor).sum().item()
        fake_correct += ((pred == labels) & fake_mask_tensor).sum().item()

        batch_route_png = use_png.sum().item()
        route_png_total += batch_route_png

        for i, group in enumerate(groups):
            gname = str(group)
            sname = str(subgroups[i])
            stats = subgroup_stats.setdefault(
                (gname, sname),
                {
                    "total": 0,
                    "correct": 0,
                    "fake_total": 0,
                    "fake_correct": 0,
                    "real_total": 0,
                    "real_correct": 0,
                    "route_png_total": 0,
                },
            )
            label = int(labels_cpu[i].item())
            pred_i = int(pred_cpu[i].item())
            is_correct = int(pred_i == label)
            use_png_i = bool(use_png_cpu[i].item())

            stats["total"] += 1
            stats["correct"] += is_correct
            stats["route_png_total"] += int(use_png_i)

            if label == 1:
                stats["fake_total"] += 1
                stats["fake_correct"] += is_correct
            else:
                stats["real_total"] += 1
                stats["real_correct"] += is_correct

            if logits_csv_writer is not None:
                logits_csv_writer.writerow(
                    [
                        gname,
                        sname,
                        str(paths[i]),
                        label,
                        float(logits_png_cpu[i, 0].item()),
                        float(logits_png_cpu[i, 1].item()),
                        float(prob_png_raw_cpu[i].item()),
                        float(prob_png_cal_cpu[i].item()),
                        float(logits_jpeg_cpu[i, 0].item()),
                        float(logits_jpeg_cpu[i, 1].item()),
                        float(prob_jpeg_raw_cpu[i].item()),
                        float(prob_jpeg_cal_cpu[i].item()),
                        routing_score_mode,
                        int(use_png_i),
                        "artifact-driven" if use_png_i else "representation-driven",
                        float(p_out_cpu[i].item()),
                        pred_i,
                        is_correct,
                    ]
                )

    acc = correct / max(1, total)
    route_png_ratio = route_png_total / max(1, total)

    probs_png = torch.cat(probs_png_all).numpy() if probs_png_all else []
    probs_jpeg = torch.cat(probs_jpeg_all).numpy() if probs_jpeg_all else []
    probs_png_real = torch.cat(probs_png_real).numpy() if probs_png_real else []
    probs_jpeg_real = torch.cat(probs_jpeg_real).numpy() if probs_jpeg_real else []
    probs_png_fake = torch.cat(probs_png_fake).numpy() if probs_png_fake else []
    probs_jpeg_fake = torch.cat(probs_jpeg_fake).numpy() if probs_jpeg_fake else []

    return {
        "acc": acc,
        "total": total,
        "correct": correct,
        "real_total": real_total,
        "fake_total": fake_total,
        "real_correct": real_correct,
        "fake_correct": fake_correct,
        "route_png_ratio": route_png_ratio,
        "route_png_total": route_png_total,
        "subgroup_stats": subgroup_stats,
        "probs_png": probs_png,
        "probs_jpeg": probs_jpeg,
        "probs_png_real": probs_png_real,
        "probs_jpeg_real": probs_jpeg_real,
        "probs_png_fake": probs_png_fake,
        "probs_jpeg_fake": probs_jpeg_fake,
    }


def log_group_stats(logger: logging.Logger, subgroup_stats: dict):
    group_map = {}
    for (gname, sname), stats in subgroup_stats.items():
        fake_total = stats["fake_total"]
        real_total = stats["real_total"]
        all_total = stats["total"]

        fake_correct = stats["fake_correct"]
        real_correct = stats["real_correct"]
        all_correct = stats["correct"]

        fake_acc = fake_correct / max(1, fake_total)
        real_acc = real_correct / max(1, real_total)
        all_acc = all_correct / max(1, all_total)
        route_ratio = stats["route_png_total"] / max(1, all_total)

        group_map.setdefault(gname, []).append(
            {
                "subgroup": sname,
                "acc": all_acc,
                "route_ratio": route_ratio,
                "fake_acc": fake_acc,
                "real_acc": real_acc,
                "counts": (all_correct, all_total),
            }
        )

    summary = {}
    for gname in sorted(group_map):
        entries = group_map[gname]
        mean_acc = sum(e["acc"] for e in entries) / max(1, len(entries))
        mean_route = sum(e["route_ratio"] for e in entries) / max(1, len(entries))

        summary[gname] = {
            "mean_acc": mean_acc,
            "mean_route_ratio": mean_route,
            "subgroups": len(entries),
        }

        logger.info(
            f"Method {gname}: "
            f"mean acc={mean_acc:.4f}, "
            f"route_artifact_ratio={mean_route:.4f}, "
            f"subgroups={len(entries)}"
        )
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="MoE gate with router rule between artifact-driven and representation-driven experts."
    )
    ap.add_argument(
        "--data_root",
        type=str,
        default="../datasets/test",
        help="Dataset root containing per-group 0_real/1_fake subfolders.",
    )
    ap.add_argument(
        "--artifact_ckpt",
        type=str,
        metavar="ARTIFACT_CKPT",
        default="../checkpoints/artifact_driven_expert.pt",
        help="Artifact-driven expert checkpoint. The released file name is artifact_driven_expert.pt.",
    )
    ap.add_argument(
        "--representation_ckpt",
        type=str,
        metavar="REPRESENTATION_CKPT",
        default="../checkpoints/representation_driven_expert.pt",
        help="Representation-driven expert checkpoint. The released file name is representation_driven_expert.pt.",
    )
    ap.add_argument(
        "--artifact_input_mode",
        type=str,
        default="dct9",
        choices=["rgb", "dct9"],
    )
    ap.add_argument(
        "--representation_input_mode",
        type=str,
        default="rgb",
        choices=["rgb", "dct9"],
    )
    ap.add_argument("--crop_size", type=int, default=336)
    ap.add_argument("--pad_mode", type=str, default="reflect", choices=["constant", "edge", "reflect", "symmetric"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--artifact_thr", type=float, default=0.992167)
    ap.add_argument("--representation_low", type=float, default=0.5)
    ap.add_argument("--decision_threshold", type=float, default=0.5)
    ap.add_argument(
        "--routing_score_mode",
        type=str,
        default="calibrated",
        choices=["calibrated", "raw"],
        help="Use calibrated probabilities or raw softmax probabilities for CAR routing.",
    )
    ap.add_argument(
        "--artifact_temperature",
        type=float,
        default=1.0,
        help="Artifact-driven expert temperature for calibrated routing.",
    )
    ap.add_argument(
        "--representation_temperature",
        type=float,
        default=1.0,
        help="Representation-driven expert temperature for calibrated routing.",
    )
    ap.add_argument(
        "--calibration_root",
        type=str,
        default="",
        help="Optional held-out calibration root with 0_real/1_fake or nature/ai folders.",
    )
    ap.add_argument(
        "--max_calibration_images",
        type=int,
        default=0,
        help="Optional cap for fitting temperatures; 0 uses the full calibration set.",
    )
    ap.add_argument(
        "--calibrate_thresholds_from_raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When using calibrated routing, interpret --artifact_thr/--representation_low/--decision_threshold "
            "as legacy raw-softmax thresholds and map them through the fitted temperatures."
        ),
    )
    ap.add_argument(
        "--prob_plot_real",
        type=str,
        default="prob_dist_real.png",
        help="Save expert probability distribution plot for 0_real (set empty to disable).",
    )
    ap.add_argument(
        "--prob_plot_fake",
        type=str,
        default="prob_dist_fake.png",
        help="Save expert probability distribution plot for 1_fake (set empty to disable).",
    )
    ap.add_argument("--max_images", type=int, default=0)

    # 新增：日志参数
    ap.add_argument("--out_dir", type=str, default="../outputs/router_rule", help="Output directory for logs and plots.")
    ap.add_argument("--log_file", type=str, default="router_rule_eval.log", help="Log file name or path.")
    ap.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--save_logits_csv", action="store_true", help="Save per-image expert logits/probs into CSV.")
    ap.add_argument(
        "--logits_csv",
        type=str,
        default="sample_logits.csv",
        help="CSV file name or path for per-image logits (used when --save_logits_csv is set).",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_file = args.log_file
    if not os.path.isabs(log_file):
        log_file = os.path.join(args.out_dir, log_file)
    logger = setup_logger(log_file, level=args.log_level)
    logger.info("==== Router Rule Evaluation Started ====")
    logger.info(f"cmdline: {' '.join(sys.argv)}")
    logger.info(f"data_root={args.data_root}")
    logger.info(f"artifact_ckpt={args.artifact_ckpt}")
    logger.info(f"representation_ckpt={args.representation_ckpt}")
    logger.info(f"routing_score_mode={args.routing_score_mode}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    pixel_expert, _pixel_model_id, _pixel_feat_dim, pixel_in_channels, pixel_mean, pixel_std, _pixel_crop_size = load_expert(
        args.artifact_ckpt, device=device, logger=logger
    )
    semantic_expert, _semantic_model_id, _semantic_feat_dim, semantic_in_channels, semantic_mean, semantic_std, _semantic_crop_size = load_expert(
        args.representation_ckpt, device=device, logger=logger
    )

    pixel_input_channels = 9 if args.artifact_input_mode == "dct9" else 3
    semantic_input_channels = 9 if args.representation_input_mode == "dct9" else 3

    if pixel_in_channels and pixel_in_channels != pixel_input_channels:
        logger.warning(
            f"[Warn] artifact-driven expert expects in_channels={pixel_in_channels}, "
            f"but artifact_input_mode={args.artifact_input_mode} uses {pixel_input_channels}"
        )
    if semantic_in_channels and semantic_in_channels != semantic_input_channels:
        logger.warning(
            f"[Warn] representation-driven expert expects in_channels={semantic_in_channels}, "
            f"but representation_input_mode={args.representation_input_mode} uses {semantic_input_channels}"
        )

    adapt_patch_embed_channels(pixel_expert.backbone, in_channels=pixel_input_channels, device=device)
    adapt_patch_embed_channels(semantic_expert.backbone, in_channels=semantic_input_channels, device=device)

    def resolve_pixel_crop(arg_crop: int):
        if arg_crop and int(arg_crop) > 0:
            return int(arg_crop)
        return 336

    def resolve_semantic_crop(arg_crop: int, ckpt_crop):
        if arg_crop and int(arg_crop) > 0:
            return int(arg_crop)
        if ckpt_crop and int(ckpt_crop) > 0:
            return int(ckpt_crop)
        return 336

    pixel_crop = resolve_pixel_crop(args.crop_size)
    semantic_crop = resolve_semantic_crop(args.crop_size, _semantic_crop_size)

    if args.artifact_input_mode == "dct9":
        pixel_tfm = DCTBlockHF9CenterCropTransform(crop_size=pixel_crop)
    else:
        pixel_mean = pixel_mean or (0.485, 0.456, 0.406)
        pixel_std = pixel_std or (0.229, 0.224, 0.225)
        pixel_tfm = CropNoResizeTransform(
            crop_size=pixel_crop,
            mean=pixel_mean,
            std=pixel_std,
            pad_mode=args.pad_mode,
        )

    if args.representation_input_mode == "dct9":
        semantic_tfm = DCTBlockHF9CenterCropTransform(crop_size=semantic_crop)
    else:
        semantic_mean = semantic_mean or (0.485, 0.456, 0.406)
        semantic_std = semantic_std or (0.229, 0.224, 0.225)
        semantic_tfm = CropNoResizeTransform(
            crop_size=semantic_crop,
            mean=semantic_mean,
            std=semantic_std,
            pad_mode=args.pad_mode,
        )

    logger.info(
        "preprocess: "
        f"artifact_input={args.artifact_input_mode} crop={pixel_crop} | "
        f"representation_input={args.representation_input_mode} crop={semantic_crop} | "
        f"pad_mode={args.pad_mode}"
    )

    pixel_temperature = float(args.artifact_temperature)
    semantic_temperature = float(args.representation_temperature)
    if args.routing_score_mode == "calibrated" and args.calibration_root:
        logger.info(f"[Calibration] fitting temperatures on {args.calibration_root}")
        calibration_loader, calibration_len = build_calibration_loader(
            data_root=args.calibration_root,
            pixel_transform=pixel_tfm,
            semantic_transform=semantic_tfm,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            max_images=args.max_calibration_images,
            seed=args.seed,
        )
        if calibration_loader is None or calibration_len is None:
            raise SystemExit(f"No calibration images found under: {args.calibration_root}")
        logger.info(f"[Calibration] samples={calibration_len}")
        pixel_temperature, semantic_temperature = fit_expert_temperatures(
            pixel_expert,
            semantic_expert,
            calibration_loader,
            device=device,
            use_amp=args.amp,
            logger=logger,
        )
    logger.info(
        "[Calibration] active routing temperatures: "
        f"artifact-driven T={pixel_temperature:.6f}, representation-driven T={semantic_temperature:.6f}"
    )
    active_artifact_thr = float(args.artifact_thr)
    active_qtt_low = float(args.representation_low)
    active_decision_threshold = float(args.decision_threshold)
    if args.routing_score_mode == "calibrated" and args.calibrate_thresholds_from_raw:
        active_artifact_thr = calibrate_probability_threshold(args.artifact_thr, pixel_temperature)
        active_qtt_low = calibrate_probability_threshold(args.representation_low, semantic_temperature)
        # The selected output can come from either expert; 0.5 remains 0.5 under temperature scaling.
        active_decision_threshold = calibrate_probability_threshold(args.decision_threshold, 1.0)
        logger.info(
            "[Calibration] mapped legacy raw thresholds to calibrated thresholds: "
            f"artifact_thr {args.artifact_thr:.6f}->{active_artifact_thr:.6f}, "
            f"representation_low {args.representation_low:.6f}->{active_qtt_low:.6f}, "
            f"decision {args.decision_threshold:.6f}->{active_decision_threshold:.6f}"
        )

    logits_csv_fp = None
    logits_csv_writer = None
    if args.save_logits_csv:
        logits_csv_path = args.logits_csv
        if not os.path.isabs(logits_csv_path):
            logits_csv_path = os.path.join(args.out_dir, logits_csv_path)
        os.makedirs(os.path.dirname(logits_csv_path) or ".", exist_ok=True)
        logits_csv_fp = open(logits_csv_path, mode="w", newline="", encoding="utf-8")
        logits_csv_writer = csv.writer(logits_csv_fp)
        logits_csv_writer.writerow(
            [
                "group",
                "subgroup",
                "image_path",
                "label",
                "artifact_logit_real",
                "artifact_logit_fake",
                "artifact_prob_fake_raw",
                "artifact_prob_fake_calibrated",
                "representation_logit_real",
                "representation_logit_fake",
                "representation_prob_fake_raw",
                "representation_prob_fake_calibrated",
                "routing_score_mode",
                "use_artifact",
                "selected_expert",
                "output_prob_fake",
                "pred",
                "is_correct",
            ]
        )
        logger.info(f"logits_csv={os.path.abspath(logits_csv_path)}")

    # 关键修改点 1：按每个子文件夹(group)逐个推理并立刻输出/写日志
    group_roots = find_group_roots(args.data_root)
    if not group_roots:
        raise SystemExit(f"No valid groups found under: {args.data_root}")

    global_total = 0
    global_correct = 0
    global_real_total = 0
    global_fake_total = 0
    global_real_correct = 0
    global_fake_correct = 0
    global_route_png_total = 0
    group_mean_accs = []
    weighted_balanced_acc_sum = 0.0

    try:
        for group_name, group_root in group_roots:
            logger.info("------------------------------------------------------------")
            logger.info(f"[Group Start] {group_name} | root={group_root}")

            loader, total_len = build_grouped_cls_loader(
                real_root=group_root,
                fake_root=group_root,
                transform=pixel_tfm,
                route_transform=pixel_tfm,
                pixel_transform=pixel_tfm,
                semantic_transform=semantic_tfm,
                group_name=group_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                max_images=args.max_images,
                seed=args.seed,
            )

            if loader is None or total_len is None:
                logger.warning(f"[Group Skip] {group_name}: No images found.")
                continue

            metrics = evaluate_router_rule(
                pixel_expert,
                semantic_expert,
                loader,
                device=device,
                use_amp=args.amp,
                artifact_thr=active_artifact_thr,
                qtt_low=active_qtt_low,
                decision_threshold=active_decision_threshold,
                pixel_temperature=pixel_temperature,
                semantic_temperature=semantic_temperature,
                routing_score_mode=args.routing_score_mode,
                logits_csv_writer=logits_csv_writer,
            )
            if logits_csv_fp is not None:
                logits_csv_fp.flush()

            logger.info(
                f"[Group Done] {group_name}: "
                f"weighted_acc={metrics['acc']:.6f} "
                f"real_correct={metrics['real_correct']}/{metrics['real_total']} "
                f"fake_correct={metrics['fake_correct']}/{metrics['fake_total']} "
                f"total={metrics['total']} "
                f"route_artifact_ratio={metrics['route_png_ratio']:.6f} "
                f"routing_score_mode={args.routing_score_mode} "
                f"artifact_thr={active_artifact_thr:.6f} representation_low={active_qtt_low:.6f} "
                f"decision_thr={active_decision_threshold:.6f}"
            )

            # 记录该 group 下更细的 subgroup_stats，并以 group 级别 mean acc 输出
            group_summary = log_group_stats(logger, metrics["subgroup_stats"])
            if group_name in group_summary:
                group_mean_accs.append(group_summary[group_name]["mean_acc"])

            group_real_total = metrics["real_total"]
            group_fake_total = metrics["fake_total"]
            group_real_correct = metrics["real_correct"]
            group_fake_correct = metrics["fake_correct"]
            group_balanced_acc = 0.5 * (
                group_real_correct / max(1, group_real_total) + group_fake_correct / max(1, group_fake_total)
            )
            weighted_balanced_acc_sum += group_balanced_acc * metrics["total"]

            # 概率分布图：按 group 单独保存（文件名加后缀，避免覆盖）
            suffix = safe_name(group_name)
            if args.prob_plot_real:
                base, ext = os.path.splitext(args.prob_plot_real)
                plot_dir = os.path.join(args.out_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                out_path = os.path.join(plot_dir, f"{base}_{suffix}{ext or '.png'}")
                plot_prob_distributions(
                    metrics["probs_png_real"],
                    metrics["probs_jpeg_real"],
                    out_path,
                    f"Expert Probability Distribution (0_real) - {group_name}",
                )
                logger.info(f"saved prob distribution plot to {out_path}")

            if args.prob_plot_fake:
                base, ext = os.path.splitext(args.prob_plot_fake)
                plot_dir = os.path.join(args.out_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                out_path = os.path.join(plot_dir, f"{base}_{suffix}{ext or '.png'}")
                plot_prob_distributions(
                    metrics["probs_png_fake"],
                    metrics["probs_jpeg_fake"],
                    out_path,
                    f"Expert Probability Distribution (1_fake) - {group_name}",
                )
                logger.info(f"saved prob distribution plot to {out_path}")

            global_total += metrics["total"]
            global_correct += metrics["correct"]
            global_real_total += metrics["real_total"]
            global_fake_total += metrics["fake_total"]
            global_real_correct += metrics["real_correct"]
            global_fake_correct += metrics["fake_correct"]
            global_route_png_total += metrics["route_png_total"]

        # 全局汇总
        logger.info("============================================================")
        g_acc = global_correct / max(1, global_total)
        g_route_ratio = global_route_png_total / max(1, global_total)
        mean_group_acc = sum(group_mean_accs) / max(1, len(group_mean_accs))
        weighted_balanced_acc = weighted_balanced_acc_sum / max(1, global_total)
        logger.info(
            f"[All Groups Summary] weighted_acc={g_acc:.6f} "
            f"mean_acc={mean_group_acc:.6f} balanced_acc={weighted_balanced_acc:.6f} "
            f"real_correct={global_real_correct}/{global_real_total} "
            f"fake_correct={global_fake_correct}/{global_fake_total} "
            f"total={global_total} route_artifact_ratio={g_route_ratio:.6f}"
        )
        logger.info("==== Router Rule Evaluation Finished ====")
        logger.info(f"log_file={os.path.abspath(log_file)}")
    finally:
        if logits_csv_fp is not None:
            logits_csv_fp.close()


if __name__ == "__main__":
    main()
