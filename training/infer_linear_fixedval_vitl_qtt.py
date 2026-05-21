import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import List, Union

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import imageio.v3 as iio

from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x


# ---------------- Utils ----------------
def _cuda_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        major, _minor = torch.cuda.get_device_capability()
        return major >= 8  # Ampere+


@dataclass
class Meter:
    correct: int = 0
    total: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, logits: torch.Tensor, y: torch.Tensor):
        pred = logits.argmax(dim=1)
        self.correct += (pred == y).sum().item()
        self.total += y.numel()
        self.tp += ((pred == 1) & (y == 1)).sum().item()
        self.tn += ((pred == 0) & (y == 0)).sum().item()
        self.fp += ((pred == 1) & (y == 0)).sum().item()
        self.fn += ((pred == 0) & (y == 1)).sum().item()

    @property
    def acc(self):
        return self.correct / max(1, self.total)

    @property
    def precision(self):
        return self.tp / max(1, (self.tp + self.fp))

    @property
    def recall(self):
        return self.tp / max(1, (self.tp + self.fn))

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / max(1e-12, (p + r))


def standardize_binary_labels(ds: datasets.ImageFolder):
    new_samples, new_targets = [], []
    for path, _idx in ds.samples:
        cname = os.path.basename(os.path.dirname(path)).lower()
        if ("fake" in cname) or cname.startswith("1_") or cname == "1":
            y = 1
        elif ("real" in cname) or cname.startswith("0_") or cname == "0":
            y = 0
        else:
            raise ValueError(f"无法从文件夹名推断标签：'{cname}'。请使用包含'real/fake'或'0_/1_'的目录名。")
        new_samples.append((path, y))
        new_targets.append(y)
    ds.samples = new_samples
    ds.targets = new_targets
    ds.classes = ["real", "fake"]
    ds.class_to_idx = {"real": 0, "fake": 1}
    return ds


# ---------------- Preprocess: Pad -> 5-Crop -> Normalize (No resize) ----------------
# ---------------- Preprocess: Pad -> CenterCrop -> Normalize (No resize) ----------------
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


def _load_image_any(path: str) -> Image.Image:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")
    except Exception:
        try:
            arr = iio.imread(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {path}") from e

        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] >= 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1) * 255 if arr.max() <= 1 else np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")


class CropNoResizeTransform:
    """
    Pad to >= crop_size then take a center crop; avoids any resize interpolation.
    """
    def __init__(
        self,
        crop_size: int,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        pad_mode: str = "edge",  # constant/edge/reflect/symmetric
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

        arr = np.array(crop).astype(np.float32) / 255.0  # HWC
        t = torch.from_numpy(arr).permute(2, 0, 1)       # CHW
        t = (t - self.mean) / self.std
        return t


# ---------------- Backbone helpers ----------------
class LNDropLinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.ln(x)))


def create_backbone(model_id: str):
    try:
        cfg = AutoConfig.from_pretrained(model_id)
        model_type = getattr(cfg, "model_type", "")
    except Exception:
        model_type = ""
    if model_type == "clip":
        return CLIPVisionModelWithProjection.from_pretrained(model_id)
    return AutoModel.from_pretrained(model_id)


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
        "q_proj", "k_proj", "v_proj",
        "query", "key", "value",
        "qkv", "to_q", "to_k", "to_v", "to_qkv",
    }
    found = set()
    for name, module in backbone.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        tail = name.rsplit(".", 1)[-1]
        if tail in candidates:
            found.add(tail)
    return sorted(found)


def maybe_apply_lora_from_ckpt(backbone: nn.Module, ckpt: dict, log):
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
    log(f"[LoRA] enabled: r={lora_r} alpha={lora_alpha} dropout={lora_dropout} bias={lora_bias} targets={target_modules}")
    return backbone


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


def infer_feat_dim(backbone, device, crop_size: int):
    backbone.eval()
    dummy = torch.zeros(1, 3, crop_size, crop_size, device=device)
    with torch.no_grad():
        feats = extract_feats(backbone(pixel_values=dummy))
    return int(feats.shape[-1])


def get_mean_std(model_id: str):
    try:
        proc = AutoImageProcessor.from_pretrained(model_id)
        mean = tuple(getattr(proc, "image_mean", (0.485, 0.456, 0.406)))
        std = tuple(getattr(proc, "image_std", (0.229, 0.224, 0.225)))
        return mean, std
    except Exception:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------- Eval ----------------
@torch.no_grad()
def evaluate(backbone, head, loader, use_amp: bool, device: torch.device):
    backbone.eval()
    head.eval()
    meter = Meter()

    has_cuda = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (has_cuda and _cuda_supports_bf16()) else (torch.float16 if has_cuda else torch.float32)
    amp_device_type = "cuda" if has_cuda else "cpu"

    # collect file paths in dataloader order (shuffle=False)
    all_paths = [p for p, _ in getattr(loader.dataset, "samples", [])]
    path_cursor = 0
    per_image = []  # list of (filename, prob_fake)

    for pixel_values, y in tqdm(loader, total=len(loader), desc="Eval", leave=False):
        pixel_values = pixel_values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if pixel_values.dim() == 5:
            bsz, n_crops, c, h, w = pixel_values.shape
            pixel_values_flat = pixel_values.reshape(bsz * n_crops, c, h, w)
        elif pixel_values.dim() == 4:
            bsz = pixel_values.shape[0]
            n_crops = 1
            pixel_values_flat = pixel_values
        else:
            raise ValueError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")

        with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(use_amp and has_cuda)):
            feats = extract_feats(backbone(pixel_values=pixel_values_flat))
            logits_flat = head(feats)
            logits = logits_flat.reshape(bsz, n_crops, -1).mean(dim=1) if n_crops > 1 else logits_flat
        meter.update(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
        bsz = len(probs)
        for i in range(bsz):
            if path_cursor + i < len(all_paths):
                fname = os.path.basename(all_paths[path_cursor + i])
            else:
                fname = f"sample_{path_cursor+i}"
            per_image.append((fname, float(probs[i])))
        path_cursor += bsz

    return meter, per_image


# ---------------- Dataset helpers ----------------
def resolve_dataset_root(root: str) -> str:
    for sub in ["test", "val", "validation", "eval", "Eval", "Validation", "Test"]:
        cand = os.path.join(root, sub)
        if os.path.isdir(cand):
            return cand
    return root


def list_immediate_subdirs(root: str):
    try:
        return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    except FileNotFoundError:
        return []


def _is_real_dir_name(name: str) -> bool:
    lname = name.lower()
    if lname in {"real", "0"}:
        return True
    if lname.startswith(("0_", "0-")):
        return True
    return re.search(r"(?:^|[^a-z0-9])real(?:$|[^a-z0-9])", lname) is not None


def _is_fake_dir_name(name: str) -> bool:
    lname = name.lower()
    if lname in {"fake", "1"}:
        return True
    if lname.startswith(("1_", "1-")):
        return True
    return re.search(r"(?:^|[^a-z0-9])fake(?:$|[^a-z0-9])", lname) is not None


def _infer_label_from_path(path: str, root: str):
    rel_dir = os.path.relpath(os.path.dirname(path), root)
    if rel_dir == ".":
        return None
    parts = rel_dir.split(os.sep)
    for part in reversed(parts):
        if _is_real_dir_name(part):
            return 0
        if _is_fake_dir_name(part):
            return 1
    return None


def _iter_all_files(root: str):
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        filenames.sort()
        for fname in filenames:
            if fname.startswith("."):
                continue
            yield os.path.join(dirpath, fname)


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jfif", ".tiff", ".bmp"}


def _has_allowed_extension(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


class AnyImageFolder(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.targets = []

        for path in _iter_all_files(root):
            if not _has_allowed_extension(path):
                continue
            label = _infer_label_from_path(path, root)
            if label is None:
                continue
            self.samples.append((path, label))
            self.targets.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = _load_image_any(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _dir_has_label_children(path: str):
    has_real = False
    has_fake = False
    try:
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if not os.path.isdir(full):
                continue
            if _is_real_dir_name(name):
                has_real = True
            if _is_fake_dir_name(name):
                has_fake = True
    except FileNotFoundError:
        pass
    return has_real, has_fake


def _dir_has_label_grandchildren(path: str):
    has_real = False
    has_fake = False
    try:
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if not os.path.isdir(full):
                continue
            r, f = _dir_has_label_children(full)
            has_real = has_real or r
            has_fake = has_fake or f
            if has_real and has_fake:
                break
    except FileNotFoundError:
        pass
    return has_real, has_fake


def infer_eval_roots(root: str, split_mode: str = "auto"):
    if split_mode == "root":
        return [root]

    subdirs = list_immediate_subdirs(root)
    if split_mode == "subdir":
        return [os.path.join(root, s) for s in subdirs] if subdirs else [root]

    root_has_real, root_has_fake = _dir_has_label_children(root)
    if root_has_real or root_has_fake:
        return [root]

    if not subdirs:
        return [root]

    subdir_flags = []
    for sub in subdirs:
        sub_path = os.path.join(root, sub)
        has_real, has_fake = _dir_has_label_children(sub_path)
        deep_real, deep_fake = _dir_has_label_grandchildren(sub_path)
        deep_only = (deep_real or deep_fake) and not (has_real or has_fake)
        subdir_flags.append((sub, has_real, has_fake, deep_only))

    if any(d for _, _, _, d in subdir_flags):
        return [os.path.join(root, s) for s in subdirs]

    return [os.path.join(root, s) for s in subdirs]


def expand_data_roots(data_roots):
    expanded = []
    for item in data_roots:
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")] if "," in item else [item.strip()]
        for part in parts:
            if not part:
                continue
            if os.path.isfile(part):
                with open(part, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        expanded.append(line)
            else:
                expanded.append(part)
    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_roots",
        nargs="+",
        default=[
            "../datasets/test/",
        ],
        help="一个或多个测试集根目录；可直接指向包含 0_real/1_fake 的目录，也可指向其父目录。",
    )
    parser.add_argument("--ckpt", type=str, default="checkpoints/dinov3_detector_best.pt")
    parser.add_argument("--model_id", type=str, default="", help="若留空则从 ckpt 的 model_id 字段读取。")
    parser.add_argument("--crop_size", type=int, default=336, help="推理时的 pad+crop 尺寸（0 表示使用 ckpt 里记录的尺寸）。")
    parser.add_argument("--pad_mode", type=str, default="edge", choices=["constant", "edge", "reflect", "symmetric"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--split_mode", type=str, default="auto", choices=["auto", "subdir", "root"],
                        help="数据集拆分策略：auto 自动判断；subdir 逐子文件夹评估；root 将每个 data_root 视为单个数据集")
    parser.add_argument("--csv", type=str, default="eval_results_qtt.csv", help="保存数据集汇总结果的 CSV 路径")
    parser.add_argument("--pred_csv", type=str, default="pred_results_qtt.csv", help="保存每张图片预测概率的 CSV 路径")
    parser.add_argument("--log_txt", type=str, default="infer_linear_fixedval_vitl_qtt_log.txt",
                        help="保存控制台输出的日志路径（.txt）")
    args = parser.parse_args()

    log_fp = open(args.log_txt, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_fp.write(f"{msg}\n")
        log_fp.flush()

    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        alt_paths = [
            os.path.join("checkpoints", os.path.basename(ckpt_path)),
            os.path.join("checkpoint", os.path.basename(ckpt_path)),
        ]
        for alt in alt_paths:
            if os.path.isfile(alt):
                ckpt_path = alt
                break
    if not os.path.isfile(ckpt_path):
        log_fp.close()
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_id = args.model_id or ckpt.get("model_id", "dinov3-vitl16-pretrain-lvd1689m")
    mean = tuple(ckpt.get("mean", get_mean_std(model_id)[0]))
    std = tuple(ckpt.get("std", get_mean_std(model_id)[1]))
    crop_size = int(args.crop_size or ckpt.get("crop_size", 224))
    feat_dim = ckpt.get("feat_dim", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(f"[Backbone] init from pretrained: {model_id}")
    backbone = create_backbone(model_id).to(device)
    backbone = maybe_apply_lora_from_ckpt(backbone, ckpt, log)
    for p in backbone.parameters():
        p.requires_grad = False
    if "backbone" in ckpt:
        ckpt_bb = ckpt["backbone"]
        model_state = backbone.state_dict()
        # only load keys that exist and match shape; keep pretrained weights for the rest
        matched = {k: v for k, v in ckpt_bb.items() if k in model_state and model_state[k].shape == v.shape}
        model_state.update(matched)
        backbone.load_state_dict(model_state, strict=True)
        log(f"[Backbone] applied {len(matched)}/{len(ckpt_bb)} keys from checkpoint (backbone params: {len(model_state)}).")
    else:
        log("[Warn] 'backbone' weights not found in checkpoint; using pretrained backbone weights only.")

    if feat_dim is None:
        feat_dim = infer_feat_dim(backbone, device, crop_size)
    head = LNDropLinearHead(in_dim=int(feat_dim), num_classes=2, dropout=0.0).to(device)
    if "head" in ckpt:
        missing_h, unexpected_h = head.load_state_dict(ckpt["head"], strict=False)
        log(f"[Head] loaded; missing={list(missing_h)} unexpected={list(unexpected_h)}")
    else:
        log("[Warn] 'head' weights not found in checkpoint; using randomly initialized head.")

    tf = CropNoResizeTransform(
        crop_size=crop_size,
        mean=mean,
        std=std,
        pad_mode=args.pad_mode,
    )

    pin_mem = device.type == "cuda"
    results = []
    pred_rows = []

    data_roots = expand_data_roots(args.data_roots)
    for root in data_roots:
        log(f"\n[Dataset Root] {root}")
        eval_roots = infer_eval_roots(root, split_mode=args.split_mode)
        if not eval_roots:
            log(f"[Warn] no valid dataset roots found in: {root}")
            continue
        root_results = []
        for eval_root in eval_roots:
            sub_root = resolve_dataset_root(eval_root)
            if not os.path.isdir(sub_root):
                log(f"[Skip] path not found: {sub_root}")
                continue
            ds = AnyImageFolder(sub_root, transform=tf)
            if len(ds) == 0:
                log(f"[Skip] empty dataset at {sub_root}")
                continue

            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_mem,
            )

            meter, per_image = evaluate(backbone, head, loader, use_amp=args.amp, device=device)
            display_name = os.path.basename(os.path.normpath(eval_root))
            if os.path.normpath(eval_root) != os.path.normpath(root):
                display_name = f"{os.path.basename(os.path.normpath(root))}/{display_name}"
            log(f"  {display_name}: Acc {meter.acc*100:.2f}% (N={meter.total})")
            results.append((display_name, meter.acc, meter.precision, meter.recall, meter.f1, meter.total))
            root_results.append((meter.acc, meter.total))
            pred_rows.extend([(display_name, fname, prob) for fname, prob in per_image])
        if root_results:
            all_acc = sum(acc for acc, _ in root_results) / max(1, len(root_results))
            total_n = sum(n for _, n in root_results)
            balanced_acc = sum(acc * n for acc, n in root_results) / max(1, total_n)
            log(f"  All Acc {all_acc*100:.2f}%")
            log(f"  Balanced Acc {balanced_acc*100:.2f}%")

    if results:
        log("\nSummary:")
        for name, acc, p, r, f1, N in results:
            log(f"{name}: Acc {acc*100:.2f}% | P {p*100:.2f}% | R {r*100:.2f}% | F1 {f1*100:.2f}%")

        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "acc(%)", "precision(%)", "recall(%)", "f1(%)", "N"])
            for name, acc, p, r, f1, N in results:
                writer.writerow([name, f"{acc*100:.4f}", f"{p*100:.4f}", f"{r*100:.4f}", f"{f1*100:.4f}", N])
        log(f"[Saved] Summary CSV: {args.csv}")

    if pred_rows:
        with open(args.pred_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "filename", "prob_fake"])
            for name, fname, prob in pred_rows:
                writer.writerow([name, fname, f"{prob:.6f}"])
        log(f"[Saved] Per-image predictions CSV: {args.pred_csv}")
    log_fp.close()


if __name__ == "__main__":
    main()
