import argparse
import io
import os
import random
import time
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets  # 你的原脚本已依赖 torchvision.datasets

from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x


# ---------------- Utils ----------------
def set_seed(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    """
    Ensure per-worker RNG differs, so QTT sampling is randomized at iteration-level.
    torch DataLoader seeds each worker with base_seed + worker_id; we mirror it to python/random & numpy.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_mode(args):
    env_rank = int(os.environ.get("RANK", -1))
    env_world_size = int(os.environ.get("WORLD_SIZE", -1))
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))

    args.rank = env_rank if env_rank >= 0 else 0
    args.world_size = env_world_size if env_world_size > 0 else 1
    args.local_rank = env_local_rank if env_local_rank >= 0 else args.local_rank
    if args.local_rank < 0:
        args.local_rank = 0

    if args.world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)
    return args.world_size > 1


def reduce_meter(meter, device: torch.device):
    if not is_dist_avail_and_initialized():
        return meter
    t = torch.tensor(
        [meter.correct, meter.total, meter.tp, meter.tn, meter.fp, meter.fn],
        device=device,
        dtype=torch.long,
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    meter.correct, meter.total, meter.tp, meter.tn, meter.fp, meter.fn = t.tolist()
    return meter


def reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


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


# 统一标签：0=real, 1=fake；兼容 {real,fake} 与 {0_real,1_fake}
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


# ---------------- QTT (Dynamic JPEG quant table transplant) ----------------
@dataclass(frozen=True)
class QEntry:
    y_table: Tuple[int, ...]         # len 64
    c_table: Tuple[int, ...]         # len 64
    subsampling: Optional[int]       # 0/1/2 or None


def _normalize_subsampling(val) -> Optional[int]:
    """
    Pillow may report subsampling as int (0/1/2) or str ('4:2:0').
    Return normalized int:
      0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0
    """
    if val is None:
        return None
    if isinstance(val, int):
        return val if val in (0, 1, 2) else None
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("4:4:4", "444"):
            return 0
        if v in ("4:2:2", "422"):
            return 1
        if v in ("4:2:0", "420"):
            return 2
    return None


def _safe_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB, handling alpha by compositing on white."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def build_qtable_bank(real_dir: str, max_real: int = 0):
    """
    Extract unique (Y-table, C-table, subsampling) entries from real JPEGs.
    Returns:
      entries: List[QEntry]
      freq:    List[int] (aligned with entries)
    """
    if not os.path.isdir(real_dir):
        raise FileNotFoundError(f"Real dir not found: {real_dir}")

    counts = defaultdict(int)
    files = []
    for fn in sorted(os.listdir(real_dir)):
        p = os.path.join(real_dir, fn)
        if os.path.isfile(p) and fn.lower().endswith((".jpg", ".jpeg")):
            files.append(p)
    if max_real and max_real > 0:
        files = files[: int(max_real)]

    used = 0
    for fp in files:
        try:
            with Image.open(fp) as im:
                q = getattr(im, "quantization", None)
                if not q or not isinstance(q, dict):
                    continue
                t0 = q.get(0)
                if t0 is None or len(t0) != 64:
                    continue
                t1 = q.get(1)
                if t1 is None or len(t1) != 64:
                    t1 = t0
                subs = _normalize_subsampling(im.info.get("subsampling"))
                entry = QEntry(tuple(int(x) for x in t0), tuple(int(x) for x in t1), subs)
                counts[entry] += 1
                used += 1
        except Exception:
            continue

    if not counts:
        raise RuntimeError("No quantization tables extracted. Check that your real dir contains valid JPEGs.")
    entries = list(counts.keys())
    freq = [counts[e] for e in entries]
    return entries, freq, used, len(files)


def build_cdf(freq: List[int], alpha: float = 0.5) -> List[float]:
    """Build CDF for weighted sampling with weights = freq^alpha."""
    alpha = float(alpha)
    weights = [float(f) ** alpha for f in freq]
    tot = sum(weights)
    if tot <= 0:
        raise RuntimeError("Invalid weight sum when building qtable CDF.")
    cdf = []
    acc = 0.0
    for w in weights:
        acc += w / tot
        cdf.append(acc)
    cdf[-1] = 1.0
    return cdf


def sample_entry_random(entries: List[QEntry], cdf: List[float]) -> QEntry:
    """Sample a QEntry according to cdf, using python's global RNG (seeded per worker)."""
    r = random.random()
    idx = bisect_left(cdf, r)
    if idx >= len(entries):
        idx = len(entries) - 1
    return entries[idx]


def qtt_reencode_pil(img: Image.Image, q: QEntry, quality_fallback: int = 95) -> Image.Image:
    """
    Decode -> encode JPEG using qtables -> decode back to PIL (RGB).
    IMPORTANT: Always call this starting from the original decoded image to avoid cumulative artifacts.
    """
    img = _safe_rgb(img)
    buf = io.BytesIO()
    save_kwargs = dict(
        format="JPEG",
        qtables=[list(q.y_table), list(q.c_table)],
        optimize=False,
        progressive=False,
        quality=int(quality_fallback),
    )
    if q.subsampling is not None:
        save_kwargs["subsampling"] = int(q.subsampling)
    img.save(buf, **save_kwargs)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def find_real_label_dir(train_root: str) -> str:
    """
    Locate the 'real' directory under train_root.
    Accepts folder names like: real / 0_real / 0.
    """
    for entry in os.listdir(train_root):
        p = os.path.join(train_root, entry)
        if not os.path.isdir(p):
            continue
        n = entry.lower()
        if ("real" in n) or n.startswith("0_") or n == "0":
            return p
    raise RuntimeError(f"Cannot locate real dir under train_root: {train_root}. "
                       f"Expected subdir like 'real' or '0_real'.")



def _is_label_dir(name: str) -> bool:
    n = name.lower()
    return ("real" in n) or ("fake" in n) or n in {"0", "1"} or n.startswith("0_") or n.startswith("1_")


def _root_has_label_dirs(root: str) -> bool:
    try:
        for entry in os.listdir(root):
            path = os.path.join(root, entry)
            if os.path.isdir(path) and _is_label_dir(entry):
                return True
    except FileNotFoundError:
        return False
    return False


def _find_subset_dirs(root: str):
    subsets = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if os.path.isdir(path) and _root_has_label_dirs(path):
            subsets.append((entry, path))
    return subsets


def _split_roots(val_root: str):
    roots = [r.strip() for r in val_root.split(",") if r.strip()]
    if not roots:
        raise ValueError("--val_root 不能为空")
    return roots


def _add_meter(dst: Meter, src: Meter):
    dst.correct += src.correct
    dst.total += src.total
    dst.tp += src.tp
    dst.tn += src.tn
    dst.fp += src.fp
    dst.fn += src.fn
    return dst


def _append_val_log(path: str, lines):
    if not path:
        return
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ---------------- Preprocess: Pad -> (Random|Center)Crop -> (Optional HFlip) -> ToTensor -> Normalize ----------------
def _safe_np_pad(arr: np.ndarray, pads, mode: str):
    """
    numpy pad 的 reflect/symmetric 对 pad 宽度有约束（pad 不能 >= 原尺寸）。
    为避免在小图上直接报错，这里在失败时自动降级到 edge。
    """
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
    """
    目标：尽量避免任何 resize 插值，从而不“洗掉”JPEG/QTT 相关的像素统计。
    - 若图像边长小于 crop_size：先 pad（默认 edge，避免 reflect 在小图上报错）到 >= crop_size
    - train: RandomCrop + 可选水平翻转
    - eval: CenterCrop
    """
    def __init__(
        self,
        crop_size: int,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        is_train: bool = True,
        hflip_prob: float = 0.5,
        pad_mode: str = "edge",  # constant/edge/reflect/symmetric
        jpeg_prob: float = 0.0,
        jpeg_min_quality: int = 60,
        jpeg_max_quality: int = 95,
        jpeg_rounds: int = 1,
    ):
        self.crop_size = int(crop_size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.is_train = bool(is_train)
        self.hflip_prob = float(hflip_prob)
        self.pad_mode = pad_mode
        self.jpeg_prob = float(jpeg_prob)
        self.jpeg_min_quality = int(jpeg_min_quality)
        self.jpeg_max_quality = int(jpeg_max_quality)
        self.jpeg_rounds = int(jpeg_rounds)

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

        # edge/reflect/symmetric：用 numpy.pad 做
        arr = np.array(img)  # HWC, uint8
        pads = ((top, bottom), (left, right), (0, 0))
        arr2 = _safe_np_pad(arr, pads, mode=self.pad_mode)
        return Image.fromarray(arr2)

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        th = tw = self.crop_size
        if w == tw and h == th:
            return img
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return img.crop((j, i, j + tw, i + th))

    def _center_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        th = tw = self.crop_size
        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))
        return img.crop((j, i, j + tw, i + th))

    def _jpeg_recompress(self, img: Image.Image, quality: int) -> Image.Image:
        quality = int(max(1, min(100, quality)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, subsampling=2)
        buf.seek(0)
        out = Image.open(buf)
        out.load()
        return out.convert("RGB")

    def _maybe_jpeg(self, img: Image.Image) -> Image.Image:
        if (not self.is_train) or self.jpeg_prob <= 0:
            return img
        if random.random() >= self.jpeg_prob:
            return img
        rounds = max(1, self.jpeg_rounds)
        out = img
        qmin = min(self.jpeg_min_quality, self.jpeg_max_quality)
        qmax = max(self.jpeg_min_quality, self.jpeg_max_quality)
        for _ in range(rounds):
            q = random.randint(qmin, qmax)
            out = self._jpeg_recompress(out, q)
        return out

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = self._pad_if_needed(img)

        if self.is_train:
            img = self._random_crop(img)
            if self.hflip_prob > 0 and random.random() < self.hflip_prob:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = self._center_crop(img)

        img = self._maybe_jpeg(img)

        arr = np.array(img).astype(np.float32) / 255.0  # HWC
        t = torch.from_numpy(arr).permute(2, 0, 1)      # CHW
        t = (t - self.mean) / self.std
        return t



class DynamicQTTTransform:
    """
    A lightweight wrapper that applies ONE random QTT re-encode (single-pass JPEG with sampled real qtables),
    then delegates to the base transform (crop/pad/normalize).
    Randomness is per-call, so it naturally happens at iteration-level.
    """
    def __init__(self, base_transform, qtt_entries: List[QEntry], qtt_cdf: List[float], quality_fallback: int = 95):
        self.base_transform = base_transform
        self.qtt_entries = qtt_entries
        self.qtt_cdf = qtt_cdf
        self.quality_fallback = int(quality_fallback)

    def __call__(self, img: Image.Image):
        q = sample_entry_random(self.qtt_entries, self.qtt_cdf)
        img = qtt_reencode_pil(img, q, quality_fallback=self.quality_fallback)
        return self.base_transform(img)


class TwoViewImageFolder(datasets.ImageFolder):
    """
    训练用：同一张图返回两份随机增强视图（crop/JPEG 链路）。
    """
    def __init__(self, root: str, transform1, transform2=None):
        super().__init__(root, transform=None)
        self.transform1 = transform1
        self.transform2 = transform2 if transform2 is not None else transform1

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        img1 = self.transform1(img) if self.transform1 is not None else img
        img2 = self.transform2(img) if self.transform2 is not None else img
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img1, img2, target


# ---------------- Model Heads ----------------
class LNDropLinearHead(nn.Module):
    """
    你不想引入多层 MLP 的情况下，一个对跨域更稳的 head：
    LayerNorm + Dropout + Linear
    """
    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.ln(x)))


# ---------------- Backbone unfreeze helpers ----------------
def _find_blocks_modulelist(backbone: nn.Module):
    """
    尽量兼容不同 ViT 命名：
    - model.vision_model.encoder.layers / layer
    - model.encoder.layers / layer
    - model.blocks (timm 风格)
    - model.layer / model.layers（如 DINOv3ViTModel）
    """
    candidate_paths = [
        ("vision_model", "encoder", "layers"),
        ("vision_model", "encoder", "layer"),
        ("encoder", "layers"),
        ("encoder", "layer"),
        ("blocks",),
        ("model", "blocks"),
        ("layer",),
        ("layers",),
    ]
    for path in candidate_paths:
        m = backbone
        ok = True
        for attr in path:
            if not hasattr(m, attr):
                ok = False
                break
            m = getattr(m, attr)
        if ok and isinstance(m, (nn.ModuleList, list, tuple)) and len(m) > 0:
            return m
    return None


def configure_backbone_trainability(backbone: nn.Module, mode: str, unfreeze_last_n: int = 0):
    """
    mode:
      - frozen: 全冻结（等价于你现在的 linear probe）
      - layernorm: 只训练所有 LayerNorm/Norm 参数（最稳的折中）
      - last_blocks: 解冻最后 unfreeze_last_n 个 block（最多建议 1~2）
    """
    mode = mode.lower()
    for p in backbone.parameters():
        p.requires_grad = False

    if mode == "frozen":
        return

    if mode == "layernorm":
        for name, p in backbone.named_parameters():
            n = name.lower()
            if ("norm" in n) or ("layernorm" in n) or (".ln" in n):
                p.requires_grad = True
        return

    if mode == "last_blocks":
        blocks = _find_blocks_modulelist(backbone)
        if blocks is None:
            raise RuntimeError("无法在 backbone 中定位 transformer blocks；请改用 --train_backbone layernorm 或 frozen。")
        n = int(unfreeze_last_n)
        if n <= 0:
            raise ValueError("--unfreeze_last_n 必须 > 0（当 --train_backbone last_blocks 时）。")
        for blk in list(blocks)[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
        # 同时把所有 norm 也打开（有些 norm 在 block 之外）
        for name, p in backbone.named_parameters():
            if ("norm" in name.lower()) or ("layernorm" in name.lower()):
                p.requires_grad = True
        return

    raise ValueError(f"Unknown --train_backbone mode: {mode}")


def backbone_is_trainable(backbone: nn.Module) -> bool:
    return any(p.requires_grad for p in backbone.parameters())


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


def maybe_apply_lora(backbone: nn.Module, args, is_main: bool):
    if not args.use_lora:
        return backbone
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as exc:
        raise RuntimeError("LoRA requires peft; please install/upgrade peft.") from exc

    target_modules = _parse_lora_target_modules(args.lora_target_modules)
    if not target_modules:
        target_modules = _infer_lora_target_modules(backbone)
    if not target_modules:
        raise RuntimeError(
            "LoRA enabled but no target modules matched. "
            "Please set --lora_target_modules (e.g., 'q_proj,v_proj' or 'query,value')."
        )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias=args.lora_bias,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    backbone = get_peft_model(backbone, lora_cfg)
    if is_main:
        if args.train_backbone != "frozen":
            print("[LoRA] train_backbone is not frozen; LoRA will train alongside unfrozen backbone params.")
        if hasattr(backbone, "print_trainable_parameters"):
            backbone.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in backbone.parameters())
            print(f"[LoRA] trainable params: {trainable}/{total} ({trainable / max(1, total):.2%})")
    return backbone


def extract_feats(outputs):
    """
    兼容常见 ViT、CLIP vision-only 等输出结构。
    选用优先级：image_embeds -> pooler_output -> CLS token。
    """
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


def get_trainable_state_dict(model: nn.Module):
    """
    仅保存 requires_grad=True 的参数（便于 resume；strict=False 加载）。
    """
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = model.state_dict()
    return {k: v for k, v in sd.items() if k in trainable}


# ---------------- Losses ----------------
def _gather_log_probs(logits: torch.Tensor, targets: torch.Tensor):
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    idx = torch.arange(logits.size(0), device=logits.device)
    log_pt = log_probs[idx, targets]
    pt = probs[idx, targets]
    return log_pt, pt


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None):
    log_pt, pt = _gather_log_probs(logits, targets)
    loss = -((1.0 - pt) ** gamma) * log_pt
    if alpha is not None:
        alpha_t = torch.where(targets == 1, torch.tensor(alpha, device=logits.device), torch.tensor(1.0 - alpha, device=logits.device))
        loss = loss * alpha_t
    return loss


def asymmetric_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_pos: float = 2.0,
    gamma_neg: float = 4.0,
):
    log_pt, pt = _gather_log_probs(logits, targets)
    gamma_t = torch.where(targets == 1, torch.tensor(gamma_pos, device=logits.device), torch.tensor(gamma_neg, device=logits.device))
    loss = -((1.0 - pt) ** gamma_t) * log_pt
    return loss


def _topk_mean(loss_vec: torch.Tensor, ratio: float) -> torch.Tensor:
    if loss_vec.numel() == 0:
        return loss_vec.mean()
    if ratio >= 1:
        k = min(int(ratio), loss_vec.numel())
    else:
        k = max(1, int(loss_vec.numel() * max(1e-6, min(1.0, ratio))))
    if k >= loss_vec.numel():
        return loss_vec.mean()
    topk = torch.topk(loss_vec, k=k, largest=True).values
    return topk.mean()


def reduce_ohem(loss_vec: torch.Tensor, targets: torch.Tensor, ratio: float, target: str = "fake", real_weight: float = 0.0):
    if loss_vec.numel() == 0:
        return loss_vec.mean()
    target = target.lower()
    if target == "fake":
        fake_mask = targets == 1
        real_mask = targets == 0
        if fake_mask.any():
            fake_loss = _topk_mean(loss_vec[fake_mask], ratio)
        else:
            fake_loss = loss_vec.mean()
        if real_weight > 0 and real_mask.any():
            real_loss = loss_vec[real_mask].mean()
            return fake_loss + real_weight * real_loss
        return fake_loss
    return _topk_mean(loss_vec, ratio)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, args):
    loss_type = args.loss_type.lower()
    if loss_type == "focal":
        alpha = args.focal_alpha if args.focal_alpha >= 0 else None
        loss_vec = focal_loss(logits, targets, gamma=args.focal_gamma, alpha=alpha)
        return loss_vec.mean()
    if loss_type == "asym_focal":
        loss_vec = asymmetric_focal_loss(logits, targets, gamma_pos=args.asym_gamma_pos, gamma_neg=args.asym_gamma_neg)
        return loss_vec.mean()
    if loss_type == "ohem":
        loss_vec = F.cross_entropy(logits, targets, reduction="none", label_smoothing=args.label_smoothing)
        return reduce_ohem(
            loss_vec, targets, ratio=args.ohem_ratio,
            target=args.ohem_target, real_weight=args.ohem_real_weight
        )
    if loss_type == "ce":
        return F.cross_entropy(logits, targets, label_smoothing=args.label_smoothing)
    raise ValueError(f"Unknown loss_type: {args.loss_type}")


def consistency_kl(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    log_p1 = F.log_softmax(logits1, dim=1)
    log_p2 = F.log_softmax(logits2, dim=1)
    p1 = log_p1.exp()
    p2 = log_p2.exp()
    kl_12 = F.kl_div(log_p1, p2, reduction="batchmean")
    kl_21 = F.kl_div(log_p2, p1, reduction="batchmean")
    return 0.5 * (kl_12 + kl_21)


# ---------------- Train/Eval ----------------
def evaluate(backbone, head, loader, use_amp: bool):
    backbone.eval()
    head.eval()
    meter = Meter()

    has_cuda = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (has_cuda and _cuda_supports_bf16()) else (torch.float16 if has_cuda else torch.float32)
    amp_device_type = "cuda" if has_cuda else "cpu"

    with torch.no_grad():
        for pixel_values, y in loader:
            pixel_values = pixel_values.to(next(backbone.parameters()).device, non_blocking=True)
            y = y.to(next(head.parameters()).device, non_blocking=True)
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(use_amp and has_cuda)):
                feats = extract_feats(backbone(pixel_values=pixel_values))
                logits = head(feats)
            meter.update(logits, y)
    return meter


def eval_val_entries(backbone, head, val_entries, use_amp: bool, prefix: str, device, distributed: bool, is_main: bool, show_confusion: bool = False, log_path: str = "", epoch: Optional[int] = None):
    results = []
    log_lines = []
    if epoch is not None:
        log_lines.append(f"Epoch {epoch:03d} {prefix}")
    else:
        log_lines.append(f"{prefix}")
    for entry in val_entries:
        if entry.get("subsets") is None:
            meter = evaluate(backbone, head, entry["loader"], use_amp=use_amp)
            if distributed:
                meter = reduce_meter(meter, device)
            results.append({"name": entry["name"], "meter": meter})
            if is_main:
                line = (f"{prefix} {entry['name']}: Acc {meter.acc*100:.2f}% | "
                        f"P {meter.precision*100:.2f}% | R {meter.recall*100:.2f}% | "
                        f"F1 {meter.f1*100:.2f}% (N={meter.total})")
                print(line)
                log_lines.append(line)
                if show_confusion:
                    log_lines.append("  Confusion: GT(real/fake) x Pred(real/fake)")
                    log_lines.append(f"  real: {meter.tn:>6d}  {meter.fp:>6d}")
                    log_lines.append(f"  fake: {meter.fn:>6d}  {meter.tp:>6d}")
                    print("  Confusion: GT(real/fake) x Pred(real/fake)")
                    print(f"  real: {meter.tn:>6d}  {meter.fp:>6d}")
                    print(f"  fake: {meter.fn:>6d}  {meter.tp:>6d}")
            continue

        total = Meter()
        subset_stats = []
        for sub in entry["subsets"]:
            meter = evaluate(backbone, head, sub["loader"], use_amp=use_amp)
            if distributed:
                meter = reduce_meter(meter, device)
            _add_meter(total, meter)
            subset_stats.append((sub["name"], meter.acc, meter.total))
        results.append({"name": entry["name"], "meter": total})
        if is_main:
            line = (f"{prefix} {entry['name']}: Acc {total.acc*100:.2f}% | "
                    f"P {total.precision*100:.2f}% | R {total.recall*100:.2f}% | "
                    f"F1 {total.f1*100:.2f}% (N={total.total})")
            print(line)
            log_lines.append(line)
            for name, acc, total_n in subset_stats:
                line = f"{prefix} {entry['name']}/{name}: Acc {acc*100:.2f}% (N={total_n})"
                print(line)
                log_lines.append(line)
            if show_confusion:
                log_lines.append("  Confusion: GT(real/fake) x Pred(real/fake)")
                log_lines.append(f"  real: {total.tn:>6d}  {total.fp:>6d}")
                log_lines.append(f"  fake: {total.fn:>6d}  {total.tp:>6d}")
                print("  Confusion: GT(real/fake) x Pred(real/fake)")
                print(f"  real: {total.tn:>6d}  {total.fp:>6d}")
                print(f"  fake: {total.fn:>6d}  {total.tp:>6d}")
    if is_main:
        log_lines.append("")
        _append_val_log(log_path, log_lines)
    return results


def train_one_epoch(backbone, head, loader, optimizer, use_amp: bool, args, use_consistency: bool, is_main: bool, epoch=None, total_epochs=None):
    bb_trainable = backbone_is_trainable(backbone)
    backbone.train(bb_trainable)  # frozen 时保持 eval 行为；可训时进入 train
    head.train()

    meter = Meter()
    total_loss = 0.0

    has_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and has_cuda))
    amp_dtype = torch.bfloat16 if (has_cuda and _cuda_supports_bf16()) else (torch.float16 if has_cuda else torch.float32)
    amp_device_type = "cuda" if has_cuda else "cpu"

    desc = f"Epoch {epoch}/{total_epochs} [train]" if (epoch is not None and total_epochs is not None) else "Train"
    loop = tqdm(loader, total=len(loader), desc=desc, leave=False, disable=not is_main)

    for batch in loop:
        if use_consistency:
            pixel_values1, pixel_values2, y = batch
            pixel_values1 = pixel_values1.to(next(backbone.parameters()).device, non_blocking=True)
            pixel_values2 = pixel_values2.to(next(backbone.parameters()).device, non_blocking=True)
            pixel_values = torch.cat([pixel_values1, pixel_values2], dim=0)
        else:
            pixel_values, y = batch
            pixel_values = pixel_values.to(next(backbone.parameters()).device, non_blocking=True)
        y = y.to(next(head.parameters()).device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(bb_trainable):
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(use_amp and has_cuda)):
                feats = extract_feats(backbone(pixel_values=pixel_values))
        if not bb_trainable:
            feats = feats.detach()

        with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(use_amp and has_cuda)):
            logits = head(feats)
            if use_consistency:
                logits1, logits2 = logits.chunk(2, dim=0)
                loss1 = compute_loss(logits1, y, args)
                loss2 = compute_loss(logits2, y, args)
                cls_loss = 0.5 * (loss1 + loss2)
                if args.consistency_weight > 0:
                    cons_loss = consistency_kl(logits1, logits2)
                    loss = cls_loss + args.consistency_weight * cons_loss
                else:
                    loss = cls_loss
                logits_for_meter = (logits1 + logits2) * 0.5
            else:
                loss = compute_loss(logits, y, args)
                logits_for_meter = logits

        if use_amp and has_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        meter.update(logits_for_meter, y)
        loop.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{meter.acc*100:.1f}%"})

    return total_loss, meter


def infer_feat_dim(backbone, device, crop_size: int):
    backbone.eval()
    dummy = torch.zeros(1, 3, crop_size, crop_size, device=device)
    with torch.no_grad():
        feats = extract_feats(backbone(pixel_values=dummy))
    return int(feats.shape[-1])


def get_mean_std(model_id: str):
    """
    只用来读取 mean/std；不使用 processor 的 resize/crop。
    """
    try:
        proc = AutoImageProcessor.from_pretrained(model_id)
        mean = tuple(getattr(proc, "image_mean", (0.485, 0.456, 0.406)))
        std = tuple(getattr(proc, "image_std", (0.229, 0.224, 0.225)))
        return mean, std
    except Exception:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def create_backbone(model_id: str):
    """
    自动选择合适的 backbone 实现：
    - CLIP 权重：用 vision-only 版本，避免需要 text input_ids
    - 其他：默认 AutoModel
    """
    try:
        cfg = AutoConfig.from_pretrained(model_id)
        model_type = getattr(cfg, "model_type", "")
    except Exception:
        cfg = None
        model_type = ""

    if model_type == "clip":
        return CLIPVisionModelWithProjection.from_pretrained(model_id)
    return AutoModel.from_pretrained(model_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--val_root", type=str, required=True, help="验证集根目录，支持逗号分隔多个路径")
    parser.add_argument("--model_id", type=str, default="dinov3-vitl16-pretrain-lvd1689m")

    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="head 学习率")
    parser.add_argument("--backbone_lr", type=float, default=5e-5, help="backbone 可训练部分学习率（LN/最后 block）")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="head weight decay")
    parser.add_argument("--backbone_weight_decay", type=float, default=0.01, help="backbone weight decay")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)

    # preprocess (关键：不 resize)
    parser.add_argument("--crop_size", type=int, default=224, help="pad/crop 到该尺寸，避免 resize 插值")
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--pad_mode", type=str, default="edge", choices=["constant", "edge", "reflect", "symmetric"])

    # dynamic QTT (iteration-level random qtable re-encode; single-pass JPEG only)
    parser.add_argument("--dynamic_qtt", type=int, default=1, choices=[0, 1],
                        help="1: enable online random QTT re-encode for training samples; 0: disable.")
    parser.add_argument("--qtt_alpha", type=float, default=0.5,
                        help="Sampling temperature for qtable bank: weight=freq^alpha. 0=uniform, 1=match freq.")
    parser.add_argument("--qtt_max_real_bank", type=int, default=0,
                        help="Use only first N real JPEGs to build qtable bank (0=all).")
    parser.add_argument("--qtt_quality_fallback", type=int, default=95,
                        help="Pillow JPEG save fallback quality (qtables still used).")

    # backbone tuning
    parser.add_argument("--train_backbone", type=str, default="layernorm", choices=["frozen", "layernorm", "last_blocks"])
    parser.add_argument("--unfreeze_last_n", type=int, default=2, help="当 train_backbone=last_blocks 时生效（建议 1~2）")
    parser.add_argument("--dropout", type=float, default=0.2)

    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="启用 LoRA 适配器（requires peft）")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="auto",
        help="LoRA 目标模块后缀，逗号分隔；'auto' 自动推断；'all-linear' 作用于所有 Linear",
    )
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "lora_only", "all"])

    # loss / OHEM
    parser.add_argument("--loss_type", type=str, default="ohem", choices=["ohem", "focal", "asym_focal", "ce"])
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=-1.0, help=">=0 时启用 alpha，常用 0.25 或 0.75")
    parser.add_argument("--asym_gamma_pos", type=float, default=2.0)
    parser.add_argument("--asym_gamma_neg", type=float, default=4.0)
    parser.add_argument("--ohem_ratio", type=float, default=0.5, help="top-k 比例(0~1)或数量(>1)")
    parser.add_argument("--ohem_target", type=str, default="fake", choices=["fake", "all"])
    parser.add_argument("--ohem_real_weight", type=float, default=0.2, help="ohem_target=fake 时保留 real 的损失权重")

    # consistency regularization
    parser.add_argument("--consistency_weight", type=float, default=0.0)

    # resume/save
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="../checkpoints")
    parser.add_argument("--val_log_path", type=str, default="", help="验证集结果日志路径")

    # jpeg consistency aug
    parser.add_argument("--jpeg_prob", type=float, default=0.0)
    parser.add_argument("--jpeg_min_quality", type=int, default=60)
    parser.add_argument("--jpeg_max_quality", type=int, default=95)
    parser.add_argument("--jpeg_rounds", type=int, default=1)
    args = parser.parse_args()

    distributed = init_distributed_mode(args)
    is_main = is_main_process()
    set_seed(args.seed, rank=get_rank())
    if torch.cuda.is_available():
        device_id = args.local_rank if distributed else 0
        device = torch.device("cuda", device_id)
    else:
        device = torch.device("cpu")
    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
    if not args.val_log_path:
        args.val_log_path = os.path.join(args.save_dir, "val_results.txt")

    # 1) backbone
    backbone = create_backbone(args.model_id)
    configure_backbone_trainability(backbone, mode=args.train_backbone, unfreeze_last_n=args.unfreeze_last_n)
    backbone = maybe_apply_lora(backbone, args, is_main=is_main)
    backbone = backbone.to(device)

    mean, std = get_mean_std(args.model_id)

    # 2) dataset
    train_tf = CropNoResizeTransform(
        crop_size=args.crop_size, mean=mean, std=std,
        is_train=True, hflip_prob=args.hflip_prob, pad_mode=args.pad_mode,
        jpeg_prob=args.jpeg_prob, jpeg_min_quality=args.jpeg_min_quality,
        jpeg_max_quality=args.jpeg_max_quality, jpeg_rounds=args.jpeg_rounds,
    )
    val_tf = CropNoResizeTransform(
        crop_size=args.crop_size, mean=mean, std=std,
        is_train=False, hflip_prob=0.0, pad_mode=args.pad_mode
    )

    # --- Dynamic QTT bank (build once; sample per-iteration in workers) ---
    train_transform = train_tf
    qtt_transform = None
    if args.dynamic_qtt == 1:
        # Build on rank-0 then broadcast to all ranks (avoids duplicated heavy I/O).
        qtt_entries = None
        qtt_cdf = None
        if (not distributed) or is_main:
            real_dir = find_real_label_dir(args.train_root)
            entries, freq, used, scanned = build_qtable_bank(real_dir, max_real=args.qtt_max_real_bank)
            cdf = build_cdf(freq, alpha=args.qtt_alpha)
            qtt_entries, qtt_cdf = entries, cdf
            if is_main:
                print(f"[dynamic_qtt] real_dir={real_dir} | scanned={scanned} usable={used} | unique_entries={len(entries)} | alpha={args.qtt_alpha}")
        if distributed:
            obj_list = [qtt_entries, qtt_cdf]
            dist.broadcast_object_list(obj_list, src=0)
            qtt_entries, qtt_cdf = obj_list
        if qtt_entries is None or qtt_cdf is None:
            raise RuntimeError("dynamic_qtt enabled but qtable bank is empty.")
        qtt_transform = DynamicQTTTransform(train_tf, qtt_entries, qtt_cdf, quality_fallback=args.qtt_quality_fallback)
        train_transform = qtt_transform

    use_consistency = args.consistency_weight > 0
    if use_consistency:
        if qtt_transform is not None:
            train_set = TwoViewImageFolder(args.train_root, transform1=train_tf, transform2=qtt_transform)
        else:
            train_set = TwoViewImageFolder(args.train_root, transform1=train_transform, transform2=train_transform)
    else:
        train_set = datasets.ImageFolder(args.train_root, transform=train_transform)
    val_roots = _split_roots(args.val_root)
    val_entries = []
    for root in val_roots:
        if _root_has_label_dirs(root):
            vset = datasets.ImageFolder(root, transform=val_tf)
            vset = standardize_binary_labels(vset)
            vsampler = DistributedSampler(vset, shuffle=False) if distributed else None
            vloader = DataLoader(
                vset, batch_size=args.batch_size, shuffle=False, sampler=vsampler,
                num_workers=args.num_workers, pin_memory=True, drop_last=False
            )
            val_entries.append(
                {"name": os.path.basename(os.path.normpath(root)), "root": root, "loader": vloader, "subsets": None}
            )
        else:
            subsets = _find_subset_dirs(root)
            if not subsets:
                raise ValueError(f"验证集目录下未找到包含 real/fake 的子集：{root}")
            subset_entries = []
            for sub_name, sub_root in subsets:
                vset = datasets.ImageFolder(sub_root, transform=val_tf)
                vset = standardize_binary_labels(vset)
                vsampler = DistributedSampler(vset, shuffle=False) if distributed else None
                vloader = DataLoader(
                    vset, batch_size=args.batch_size, shuffle=False, sampler=vsampler,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False
                )
                subset_entries.append({"name": sub_name, "root": sub_root, "loader": vloader})
            val_entries.append(
                {"name": os.path.basename(os.path.normpath(root)), "root": root, "subsets": subset_entries}
            )
    train_set = standardize_binary_labels(train_set)

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False,
        worker_init_fn=seed_worker, persistent_workers=(args.num_workers > 0)
    )

    # 3) head
    feat_dim = infer_feat_dim(backbone, device, crop_size=args.crop_size)
    head = LNDropLinearHead(in_dim=feat_dim, num_classes=2, dropout=args.dropout).to(device)

    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        if "head" in ckpt:
            head.load_state_dict(ckpt["head"], strict=True)
        if "backbone" in ckpt:
            backbone.load_state_dict(ckpt["backbone"], strict=False)
        if ckpt.get("use_lora", False) and not args.use_lora and is_main:
            print("[Resume] warning: checkpoint uses LoRA but --use_lora is not set.")
        best_acc = float(ckpt.get("best_acc", 0.0))
        if is_main:
            print(f"[Resume] loaded: {args.resume} (best_acc={best_acc*100:.2f}%)")

    if distributed:
        if backbone_is_trainable(backbone):
            backbone = DDP(backbone, device_ids=[args.local_rank], output_device=args.local_rank)
        head = DDP(head, device_ids=[args.local_rank], output_device=args.local_rank)

    # 4) optimizer (分组 lr)
    groups = []
    if backbone_is_trainable(backbone):
        bb_params = [p for p in backbone.parameters() if p.requires_grad]
        groups.append({"params": bb_params, "lr": args.backbone_lr, "weight_decay": args.backbone_weight_decay})
    groups.append({"params": head.parameters(), "lr": args.lr, "weight_decay": args.weight_decay})
    optimizer = torch.optim.AdamW(groups)

    # 5) train
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr_loss_sum, tr_meter = train_one_epoch(
            backbone, head, train_loader, optimizer, use_amp=args.amp,
            args=args, use_consistency=use_consistency, is_main=is_main, epoch=epoch, total_epochs=args.epochs
        )
        val_results = eval_val_entries(
            backbone, head, val_entries, use_amp=args.amp, prefix="Val",
            device=device, distributed=distributed, is_main=is_main,
            log_path=args.val_log_path, epoch=epoch
        )
        dt = time.time() - t0

        tr_loss_sum_t = torch.tensor(tr_loss_sum, device=device, dtype=torch.float32)
        tr_count_t = torch.tensor(tr_meter.total, device=device, dtype=torch.long)
        if distributed:
            tr_loss_sum_t = reduce_sum(tr_loss_sum_t)
            tr_count_t = reduce_sum(tr_count_t)
            tr_meter = reduce_meter(tr_meter, device)
        tr_loss = (tr_loss_sum_t / tr_count_t.clamp(min=1)).item()
        val_acc_mean = sum(r["meter"].acc for r in val_results) / max(1, len(val_results))

        if is_main:
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"train_loss={tr_loss:.4f} acc={tr_meter.acc*100:.2f}% | "
                  f"val_acc_mean={val_acc_mean*100:.2f}% | "
                  f"time={dt:.1f}s")

            if val_acc_mean > best_acc:
                best_acc = val_acc_mean

            save_path = os.path.join(args.save_dir, f"dinov3_detector_epoch{epoch:03d}.pt")
            backbone_to_save = backbone.module if isinstance(backbone, DDP) else backbone
            head_to_save = head.module if isinstance(head, DDP) else head
            torch.save(
                {
                    "head": {k: v.detach().cpu() for k, v in head_to_save.state_dict().items()},
                    "backbone": {k: v.detach().cpu() for k, v in get_trainable_state_dict(backbone_to_save).items()},
                    "best_acc": best_acc,
                    "model_id": args.model_id,
                    "feat_dim": feat_dim,
                    "train_backbone": args.train_backbone,
                    "unfreeze_last_n": args.unfreeze_last_n,
                    "use_lora": args.use_lora,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_target_modules": args.lora_target_modules,
                    "lora_bias": args.lora_bias,
                    "crop_size": args.crop_size,
                    "mean": mean,
                    "std": std,
                },
                save_path,
            )
            print(
                f"Saved: {save_path} (epoch {epoch:03d}, val_acc_mean={val_acc_mean*100:.2f}%, "
                f"best={best_acc*100:.2f}%)"
            )

    eval_val_entries(
        backbone, head, val_entries, use_amp=args.amp, prefix="Final",
        device=device, distributed=distributed, is_main=is_main,
        show_confusion=True, log_path=args.val_log_path
    )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
