import argparse
import os
import random
import time
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModel

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x


# ---------------- Utils ----------------
def set_seed(seed: int = 42, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def unwrap_ddp(model):
    return model.module if isinstance(model, DDP) else model


def reduce_meter(meter, device):
    if dist.is_initialized():
        t = torch.tensor([meter.correct, meter.total, meter.tp, meter.tn, meter.fp, meter.fn],
                         device=device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        meter.correct, meter.total, meter.tp, meter.tn, meter.fp, meter.fn = t.tolist()
    return meter


@dataclass
class Meter:
    correct: int = 0; total: int = 0
    tp: int = 0; tn: int = 0; fp: int = 0; fn: int = 0
    def update(self, logits: torch.Tensor, y: torch.Tensor):
        pred = logits.argmax(dim=1)
        self.correct += (pred == y).sum().item()
        self.total += y.numel()
        self.tp += ((pred == 1) & (y == 1)).sum().item()
        self.tn += ((pred == 0) & (y == 0)).sum().item()
        self.fp += ((pred == 1) & (y == 0)).sum().item()
        self.fn += ((pred == 0) & (y == 1)).sum().item()
    @property
    def acc(self): return self.correct / max(1, self.total)
    @property
    def precision(self): return self.tp / max(1, (self.tp + self.fp))
    @property
    def recall(self): return self.tp / max(1, (self.tp + self.fn))
    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / max(1e-12, (p + r))


def standardize_binary_labels(ds: datasets.ImageFolder):
    new_samples, new_targets = [], []
    for path, _ in ds.samples:
        # 支持 val_root 下再套一层方法目录（.../method/{real,fake}/xxx.jpg）
        parts = os.path.normpath(path).split(os.sep)
        y = None
        for part in reversed(parts[:-1]):  # 从父目录开始向上找标签
            cname = part.lower()
            if ("fake" in cname) or cname.startswith("1_") or cname == "1":
                y = 1; break
            if ("real" in cname) or cname.startswith("0_") or cname == "0":
                y = 0; break
        if y is None:
            raise ValueError(f"无法从文件夹名推断标签：'{path}'")
        new_samples.append((path, y))
        new_targets.append(y)
    ds.samples = new_samples
    if hasattr(ds, "imgs"):  # 兼容 torchvision 版本
        ds.imgs = new_samples
    ds.targets = new_targets
    ds.classes = ["real", "fake"]
    ds.class_to_idx = {"real": 0, "fake": 1}
    return ds


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


def _merge_val_results(val_results):
    total = Meter()
    for item in val_results:
        _add_meter(total, item["meter"])
    return total


def _append_val_log(path: str, lines):
    if not path:
        return
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def jpeg_compress_and_round(img, quality_min=50, quality_max=95):
    import io
    from PIL import Image
    import torchvision.transforms.functional as TF
    quality = random.randint(int(quality_min), int(quality_max))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as comp:
        comp = comp.convert("RGB")
    t = TF.to_tensor(comp) * 255.0
    t = t.round().clamp(0, 255).to(torch.uint8)
    return TF.to_pil_image(t)


def make_collate(processor, jpeg_prob=0.0, jpeg_range=(50, 95)):
    def _collate(batch):
        imgs, labels = list(zip(*batch))
        if jpeg_prob > 0:
            imgs = [
                jpeg_compress_and_round(img, *jpeg_range) if (random.random() < jpeg_prob) else img
                for img in imgs
            ]
        inputs = processor(images=list(imgs), return_tensors="pt")
        return inputs["pixel_values"], torch.tensor(labels, dtype=torch.long)
    return _collate



def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_band_ranges(s: str):
    # "1-2,3-4,11-14" -> [(1,2),(3,4),(11,14)]
    bands = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            a = b = int(part)
        else:
            a_str, b_str = part.split("-", 1)
            a, b = int(a_str.strip()), int(b_str.strip())
        bands.append((a, b))
    return bands



class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)


class InputAdapter(nn.Module):
    """Lightweight adapter to map artifact-driven channels -> 3-channel pseudo-RGB.

    Keep the pretrained ViT patch embedding untouched (expects 3 channels),
    and learn a small projection from 9-channel DCT features to 3 channels.
    """
    def __init__(self, in_channels: int = 9, out_channels: int = 3, hidden: int = 32):
        super().__init__()
        layers = []
        if hidden and hidden > 0:
            layers += [
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden, out_channels, kernel_size=1, bias=True),
            ]
        else:
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BackboneWithAdapter(nn.Module):
    """Wrap an adapter in front of a pretrained backbone."""
    def __init__(self, backbone: nn.Module, adapter: nn.Module | None = None):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def forward(self, pixel_values: torch.Tensor):
        if self.adapter is not None:
            pixel_values = self.adapter(pixel_values)
        return self.backbone(pixel_values=pixel_values)


def unfreeze_last_vit_blocks(backbone: nn.Module, n: int):
    """Unfreeze last n transformer blocks when the backbone is ViT-like.

    Supports common HuggingFace ViT naming: backbone.encoder.layer[i].
    If structure is unknown, it silently does nothing.
    """
    if n <= 0:
        return
    enc = getattr(backbone, "encoder", None)
    layers = getattr(enc, "layer", None) if enc is not None else None
    if layers is None:
        return
    total = len(layers)
    start = max(0, total - n)
    for i in range(start, total):
        for p in layers[i].parameters():
            p.requires_grad = True
    # also unfreeze layer norms if present (often helps stability)
    for name, p in backbone.named_parameters():
        lname = name.lower()
        if ("layernorm" in lname) or ("layer_norm" in lname):
            p.requires_grad = True

# ---------------- Train / Eval ----------------
@torch.no_grad()
def evaluate(backbone, head, loader, use_amp: bool):
    backbone.eval(); head.eval()
    meter = Meter()
    has_cuda = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if has_cuda else torch.float32
    amp_device_type = "cuda" if has_cuda else "cpu"
    for pixel_values, y in loader:
        pixel_values = pixel_values.to(next(backbone.parameters()).device, non_blocking=True)
        y = y.to(next(head.parameters()).device, non_blocking=True)
        with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(use_amp and has_cuda)):
            outputs = backbone(pixel_values)
            feats = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
            logits = head(feats)
        meter.update(logits, y)
    device = next(backbone.parameters()).device
    return reduce_meter(meter, device)


def eval_val_entries(backbone, head, val_entries, use_amp: bool, prefix: str, is_main: bool, log_path: str = "", epoch: int | None = None):
    results = []
    log_lines = []
    if epoch is not None:
        log_lines.append(f"Epoch {epoch:03d} {prefix}")
    else:
        log_lines.append(f"{prefix}")
    for entry in val_entries:
        if entry.get("subsets") is None:
            meter = evaluate(backbone, head, entry["loader"], use_amp=use_amp)
            results.append({"name": entry["name"], "meter": meter})
            if is_main:
                line = (f"{prefix} {entry['name']}: Acc {meter.acc*100:.2f}% | "
                        f"P {meter.precision*100:.2f}% | R {meter.recall*100:.2f}% | "
                        f"F1 {meter.f1*100:.2f}% (N={meter.total})")
                print(line)
                log_lines.append(line)
            continue

        total = Meter()
        subset_stats = []
        for sub in entry["subsets"]:
            meter = evaluate(backbone, head, sub["loader"], use_amp=use_amp)
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
    if is_main:
        log_lines.append("")
        _append_val_log(log_path, log_lines)
    return results


def train_one_epoch(backbone, head, loader, optimizer, use_amp: bool, epoch: int | None = None, total_epochs: int | None = None):
    backbone.train(); head.train()
    device = next(head.parameters()).device
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0], device=device))
    total_loss, meter = 0.0, Meter()
    has_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and has_cuda))
    amp_dtype = torch.bfloat16 if has_cuda else torch.float32
    amp_device_type = "cuda" if has_cuda else "cpu"

    disable = dist.is_initialized() and dist.get_rank() != 0
    loop = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}/{total_epochs} [train]" if (epoch and total_epochs) else "Train", leave=False, disable=disable)
    for pixel_values, y in loop:
        pixel_values = pixel_values.to(next(backbone.parameters()).device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(use_amp and has_cuda)):
            outputs = backbone(pixel_values)
            feats = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
            logits = head(feats)
            loss = ce(logits, y)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and has_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

        total_loss += loss.item() * y.size(0)
        meter.update(logits, y)
        if hasattr(loop, "set_postfix"):
            loop.set_postfix(loss=f"{total_loss / max(1, meter.total):.4f}", acc=f"{meter.acc*100:.2f}%")

    if dist.is_initialized():
        t = torch.tensor([meter.correct, meter.total, meter.tp, meter.tn, meter.fp, meter.fn],
                         device=device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        meter.correct, meter.total, meter.tp, meter.tn, meter.fp, meter.fn = t.tolist()
        loss_t = torch.tensor(total_loss, device=device)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        total_loss = loss_t.item()

    return total_loss / max(1, meter.total), meter


# ---------------- DCT Processor ----------------

class DCTBlockHFProcessorV2:
    """Multi-scale log-|DCT| band energies.

    Output: pixel_values shape [B, C, H, W], where
        C = 3 (Y/Cb/Cr) * len(scales) * len(bands)

    This stays in the DCT family (no FFT) but is more expressive and more robust than
    a single "artifact energy sum":
    - Band energies keep coarse spectral shape (multiple u+v bands).
    - Multi-scale is implemented by downsampling (area) before block-DCT, then
      upsampling energy maps back to the base resolution.
    """
    def __init__(
        self,
        size=224,
        train: bool = True,
        block: int = 8,
        align_crop: bool = False,
        scales=(1, 2),
        bands=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 14)),
        normalize_band: bool = True,
    ):
        if isinstance(size, int):
            H = W = size
        else:
            H = int(size["height"]); W = int(size["width"])
        assert H % block == 0 and W % block == 0, "Base H/W must be divisible by block."
        self.size = (H, W)
        self.train = train
        self.block = int(block)
        self.align_crop = bool(align_crop)

        # Scales: downsample factor. Each (H/scale, W/scale) must remain divisible by block.
        self.scales = tuple(int(s) for s in scales)
        for s in self.scales:
            assert s >= 1
            assert (H // s) % block == 0 and (W // s) % block == 0, f"Scale {s} invalid for H={H},W={W},block={block}"
        self.bands = tuple((int(a), int(b)) for a, b in bands)
        self.seen_sum_max = (self.block - 1) * 2  # 14 for block=8
        for a, b in self.bands:
            assert 0 <= a <= b <= self.seen_sum_max, f"Invalid band {a}-{b} for block={block}"

        self.normalize_band = bool(normalize_band)

        self.C = self._build_dct_matrix(self.block)
        self.Ct = self.C.t().contiguous()
        self.band_masks = [self._band_mask(self.block, a, b) for (a, b) in self.bands]
        self.out_channels = 3 * len(self.scales) * len(self.band_masks)

    def _build_dct_matrix(self, N: int):
        C = torch.zeros(N, N)
        for k in range(N):
            a = math.sqrt(1/N) if k == 0 else math.sqrt(2/N)
            for n in range(N):
                C[k, n] = a * math.cos(math.pi * (2*n + 1) * k / (2*N))
        return C

    def _band_mask(self, N: int, lo_sum: int, hi_sum: int):
        """Mask coefficients by u+v band, excluding DC."""
        m = torch.zeros(N, N, dtype=torch.bool)
        for u in range(N):
            for v in range(N):
                if (u == 0 and v == 0):
                    continue
                s = u + v
                if lo_sum <= s <= hi_sum:
                    m[u, v] = True
        return m

    def _crop(self, img):
        import torchvision.transforms.functional as TF
        H, W = self.size
        ih, iw = img.height, img.width
        if self.train:
            top = random.randint(0, max(0, ih - H))
            left = random.randint(0, max(0, iw - W))
            if self.align_crop:
                top = (top // self.block) * self.block
                left = (left // self.block) * self.block
        else:
            top = max(0, (ih - H) // 2)
            left = max(0, (iw - W) // 2)
        return TF.crop(img, top, left, H, W)

    def _rgb_to_ycbcr(self, t):
        r, g, b = t
        y  = 0.299*r + 0.587*g + 0.114*b
        cb = -0.1687*r - 0.3313*g + 0.5*b + 0.5
        cr = 0.5*r - 0.4187*g - 0.0813*b + 0.5
        # Convert to roughly JPEG-centered range
        return (y*255-128), (cb*255-128), (cr*255-128)

    def _dct_band_maps(self, x2d: torch.Tensor):
        """Compute band energy maps for one component at one scale.

        Returns list[tensor] each of shape [h, w].
        """
        B = self.block
        # x2d: [h,w]
        blocks = x2d.unfold(0, B, B).unfold(1, B, B)  # [h/B, w/B, B, B]
        C = self.C.to(x2d.device); Ct = self.Ct.to(x2d.device)
        coef = torch.matmul(torch.matmul(C, blocks), Ct)  # broadcast matmul
        mag = coef.abs()

        maps = []
        for m in self.band_masks:
            mm = m.to(x2d.device)
            denom = mm.sum().clamp(min=1)
            e = (mag * mm).sum(dim=(-1, -2))
            if self.normalize_band:
                e = e / denom
            emap = e.repeat_interleave(B, 0).repeat_interleave(B, 1)  # [h,w]
            maps.append(emap)
        return maps

    def __call__(self, images, return_tensors="pt"):
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF

        H, W = self.size
        feats = []
        for img in images:
            img = self._crop(img.convert("RGB"))
            t = TF.to_tensor(img)  # [3,H,W] in [0,1]
            y, cb, cr = self._rgb_to_ycbcr(t)

            chans = []
            for s in self.scales:
                if s == 1:
                    y_s, cb_s, cr_s = y, cb, cr
                    h_s, w_s = H, W
                else:
                    h_s, w_s = H // s, W // s
                    # Downsample by area to preserve energy statistics
                    y_s  = F.interpolate(y.unsqueeze(0).unsqueeze(0),  size=(h_s, w_s), mode="area").squeeze(0).squeeze(0)
                    cb_s = F.interpolate(cb.unsqueeze(0).unsqueeze(0), size=(h_s, w_s), mode="area").squeeze(0).squeeze(0)
                    cr_s = F.interpolate(cr.unsqueeze(0).unsqueeze(0), size=(h_s, w_s), mode="area").squeeze(0).squeeze(0)

                for comp in (y_s, cb_s, cr_s):
                    band_maps = self._dct_band_maps(comp)
                    for emap in band_maps:
                        if s != 1:
                            emap = F.interpolate(emap.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest").squeeze(0).squeeze(0)
                        chans.append(torch.log1p(emap))

            feats.append(torch.stack(chans, dim=0))
        return {"pixel_values": torch.stack(feats, dim=0)}


def adapt_patch_embed_channels(backbone, in_channels: int, device: str):
    patch_embed = getattr(backbone.embeddings, "patch_embeddings", None)
    proj = getattr(patch_embed, "projection", None) if patch_embed is not None else None
    conv = proj if isinstance(proj, nn.Conv2d) else patch_embed if isinstance(patch_embed, nn.Conv2d) else None
    if conv is None or conv.in_channels == in_channels:
        return
    new_conv = nn.Conv2d(in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                         stride=conv.stride, padding=conv.padding, bias=(conv.bias is not None)).to(device)
    with torch.no_grad():
        repeat = max(1, in_channels // conv.in_channels)
        new_conv.weight.zero_()
        for i in range(repeat):
            s = i * conv.in_channels; e = s + conv.in_channels
            if e > in_channels:
                break
            new_conv.weight[:, s:e] += conv.weight / repeat
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    if proj is not None and isinstance(patch_embed, nn.Module):
        patch_embed.projection = new_conv
    else:
        backbone.embeddings.patch_embeddings = new_conv
    backbone.config.num_channels = in_channels


def infer_feat_dim(backbone, input_mode, size, device, in_channels: int | None = None):
    if isinstance(size, dict):
        H = int(size.get("height", 224)); W = int(size.get("width", 224))
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        H, W = int(size[0]), int(size[1])
    else:
        H = W = int(size)
    C = int(in_channels) if (in_channels is not None) else (9 if input_mode == "jpeg_dct" else 3)
    dummy = torch.zeros(1, C, H, W, device=device)
    with torch.no_grad():
        out = backbone(dummy)
        feats = out.last_hidden_state[:, 1:, :].mean(dim=1)
    return feats.shape[-1]


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default="../datasets/train", required=True)
    parser.add_argument("--val_root", type=str, default="../datasets/val", required=True)
    parser.add_argument("--model_id", type=str, default="dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--input_mode", type=str, default="jpeg_dct", choices=["rgb", "jpeg_dct"])
    # Artifact-driven input: prefer an adapter (9->3) and keep patch embedding untouched.
    parser.add_argument("--use_input_adapter", action="store_true", help="Use a 9->3 adapter (recommended for vit-small).")
    parser.add_argument("--adapter_hidden", type=int, default=32, help="Adapter hidden width; set 0 for single 1x1 conv.")
    parser.add_argument("--unfreeze_last_n", type=int, default=4, help="Unfreeze last N transformer blocks (ViT-like backbones).")
    parser.add_argument("--crop_size", type=int, default=336)
    parser.add_argument("--align_crop", action="store_true", help="Align random crop offsets to 8-pixel blocks in DCT mode.")
    # DCT feature configuration
    parser.add_argument("--dct_scales", type=str, default="1,2", help="Comma-separated downsample scales for multi-scale DCT (e.g., 1,2).")
    parser.add_argument("--dct_bands", type=str, default="1-2,3-4,5-6,7-8,9-10,11-14", help="Comma-separated u+v band ranges for DCT (e.g., 1-2,3-4,...).")
    parser.add_argument("--dct_no_band_norm", action="store_true", help="Disable per-band coefficient-count normalization.")
    # Training-time JPEG augmentation (applied to all samples)
    parser.add_argument("--train_jpeg_prob", type=float, default=0.0, help="Probability to apply random JPEG re-encode during training.")
    parser.add_argument("--train_jpeg_min", type=int, default=80)
    parser.add_argument("--train_jpeg_max", type=int, default=95)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="../checkpoints")
    parser.add_argument("--save_best_prefix", type=str, default="dinov3_linear_head_best")
    parser.add_argument("--save_best_with_acc", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_true")
    args = parser.parse_args()

    distributed = ("WORLD_SIZE" in os.environ) and int(os.environ["WORLD_SIZE"]) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed, rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    val_results_path = os.path.join(args.save_dir, "val_results.txt")
    if is_main:
        with open(val_results_path, "w") as f:
            f.write("epoch,acc,precision,recall,f1\n")

    if (not args.resume) and (not args.no_auto_resume):
        default_ckpt = os.path.join(args.save_dir, f"{args.save_best_prefix}.pt")
        if os.path.isfile(default_ckpt):
            args.resume = default_ckpt
            if is_main:
                print(f"[Auto-Resume] Using checkpoint: {args.resume}")

    if args.input_mode == "jpeg_dct":
        dct_scales = parse_int_list(args.dct_scales)
        dct_bands = parse_band_ranges(args.dct_bands)
        processor_train = DCTBlockHFProcessorV2(size=args.crop_size, train=True, align_crop=args.align_crop,
                                              scales=dct_scales, bands=dct_bands, normalize_band=(not args.dct_no_band_norm))
        processor_val = DCTBlockHFProcessorV2(size=args.crop_size, train=False, align_crop=True,
                                            scales=dct_scales, bands=dct_bands, normalize_band=(not args.dct_no_band_norm))
    else:
        processor_train = AutoImageProcessor.from_pretrained(args.model_id)
        processor_val = processor_train


    backbone_raw = AutoModel.from_pretrained(args.model_id).to(device)

    # Determine DCT feature channel count dynamically (e.g., 3 * len(scales) * len(bands)).
    dct_in_channels = getattr(processor_train, 'out_channels', 9) if args.input_mode == 'jpeg_dct' else 3

    if args.input_mode == "jpeg_dct" and args.use_input_adapter:
        adapter = InputAdapter(in_channels=dct_in_channels, out_channels=3, hidden=args.adapter_hidden).to(device)
        backbone = BackboneWithAdapter(backbone_raw, adapter=adapter).to(device)

        # Full fine-tuning: keep all backbone layers trainable.
        for p in backbone_raw.parameters():
            p.requires_grad = True
        for p in adapter.parameters():
            p.requires_grad = True
    else:
        backbone = backbone_raw
        if args.input_mode == "jpeg_dct":
            adapt_patch_embed_channels(backbone, in_channels=dct_in_channels, device=device)
        # Full fine-tuning: keep all backbone layers trainable.
        for p in backbone.parameters():
            p.requires_grad = True

    train_set = standardize_binary_labels(datasets.ImageFolder(args.train_root))
    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
        collate_fn=make_collate(processor_train, jpeg_prob=args.train_jpeg_prob, jpeg_range=(args.train_jpeg_min, args.train_jpeg_max))
    )

    val_entries = []
    for root in _split_roots(args.val_root):
        if _root_has_label_dirs(root):
            vset = standardize_binary_labels(datasets.ImageFolder(root))
            v_sampler = DistributedSampler(vset, shuffle=False) if distributed else None
            vloader = DataLoader(
                vset, batch_size=args.batch_size, shuffle=False, sampler=v_sampler,
                num_workers=args.num_workers, pin_memory=True, collate_fn=make_collate(processor_val)
            )
            val_entries.append(
                {"name": os.path.basename(os.path.normpath(root)), "root": root, "loader": vloader, "subsets": None}
            )
        else:
            subsets = _find_subset_dirs(root)
            if not subsets:
                raise ValueError(f"--val_root 无法识别有效子集目录: {root}")
            subset_entries = []
            for sub_name, sub_root in subsets:
                vset = standardize_binary_labels(datasets.ImageFolder(sub_root))
                v_sampler = DistributedSampler(vset, shuffle=False) if distributed else None
                vloader = DataLoader(
                    vset, batch_size=args.batch_size, shuffle=False, sampler=v_sampler,
                    num_workers=args.num_workers, pin_memory=True, collate_fn=make_collate(processor_val)
                )
                subset_entries.append({"name": sub_name, "root": sub_root, "loader": vloader})
            val_entries.append(
                {"name": os.path.basename(os.path.normpath(root)), "root": root, "subsets": subset_entries}
            )

    size_for_feat = getattr(processor_val, "size", args.crop_size if args.input_mode == "jpeg_dct" else 224)
    feat_dim = infer_feat_dim(backbone, args.input_mode, size_for_feat, device, in_channels=dct_in_channels)
    head = LinearHead(in_dim=feat_dim, num_classes=2).to(device)

    def snapshot_model_state():
        bb = unwrap_ddp(backbone)
        hh = unwrap_ddp(head)
        return {
            "backbone": {k: v.detach().cpu() for k, v in bb.state_dict().items()},
            "head": {k: v.detach().cpu() for k, v in hh.state_dict().items()},
        }

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        if isinstance(ckpt, dict) and "backbone" in ckpt:
            miss_b, unexp_b = backbone.load_state_dict(ckpt["backbone"], strict=False)
        else:
            miss_b, unexp_b = [], []
        state_head = ckpt["head"] if isinstance(ckpt, dict) and "head" in ckpt else ckpt
        miss_h, unexp_h = head.load_state_dict(state_head, strict=False)
        if is_main:
            print(f"[Resume] Loaded {args.resume} (backbone missing={list(miss_b)}, unexpected={list(unexp_b)}; head missing={list(miss_h)}, unexpected={list(unexp_h)})")

    if distributed:
        backbone = DDP(backbone, device_ids=[local_rank] if torch.cuda.is_available() else None)
        head = DDP(head, device_ids=[local_rank] if torch.cuda.is_available() else None)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, list(backbone.parameters()) + list(head.parameters())),
        lr=args.lr, weight_decay=args.weight_decay
    )

    best_acc, best_state = 0.0, None
    epoch_prefix = args.save_best_prefix.replace("_best", "")
    if args.resume:
        val_results = eval_val_entries(
            backbone, head, val_entries, use_amp=args.amp, prefix="Val",
            is_main=is_main, log_path=val_results_path, epoch=0
        )
        base_meter = _merge_val_results(val_results)
        best_acc = base_meter.acc
        best_state = snapshot_model_state()
        if is_main:
            print(f"[Resume] Val Acc {best_acc*100:.2f}% | P {base_meter.precision*100:.2f}% | R {base_meter.recall*100:.2f}% | F1 {base_meter.f1*100:.2f}%")
            with open(val_results_path, "a") as f:
                f.write(f"0,{base_meter.acc:.6f},{base_meter.precision:.6f},{base_meter.recall:.6f},{base_meter.f1:.6f}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr_loss, tr_meter = train_one_epoch(backbone, head, train_loader, optimizer, use_amp=args.amp, epoch=epoch, total_epochs=args.epochs)
        val_results = eval_val_entries(
            backbone, head, val_entries, use_amp=args.amp, prefix="Val",
            is_main=is_main, log_path=val_results_path, epoch=epoch
        )
        val_meter = _merge_val_results(val_results)
        current_state = snapshot_model_state()
        if val_meter.acc > best_acc:
            best_acc = val_meter.acc
            best_state = current_state
        epoch_ckpt = os.path.join(args.save_dir, f"{epoch_prefix}_epoch{epoch:02d}.pt")
        if is_main:
            torch.save({**current_state, "model_id": args.model_id, "feat_dim": feat_dim, "epoch": epoch, "val_acc": val_meter.acc}, epoch_ckpt)
            print(f"Saved epoch checkpoint: {epoch_ckpt}")
            print(f"[Epoch {epoch:02d}] {time.time()-t0:.1f}s | train_acc {tr_meter.acc*100:.2f} | val_acc {val_meter.acc*100:.2f} | val_P {val_meter.precision*100:.2f} R {val_meter.recall*100:.2f} F1 {val_meter.f1*100:.2f}")
            with open(val_results_path, "a") as f:
                f.write(f"{epoch},{val_meter.acc:.6f},{val_meter.precision:.6f},{val_meter.recall:.6f},{val_meter.f1:.6f}\n")

    if best_state is not None:
        if args.save_best_with_acc:
            acc_str = f"{best_acc:.4f}"
            ckpt_name = f"{args.save_best_prefix}_seed{args.seed:02d}_acc{acc_str}.pt"
        else:
            ckpt_name = f"{args.save_best_prefix}.pt"
        ckpt_path = os.path.join(args.save_dir, ckpt_name)
        if is_main:
            torch.save({**best_state, "model_id": args.model_id, "feat_dim": feat_dim, "best_acc": best_acc},
                    ckpt_path)
            print(f"Saved: {ckpt_path} (best val_acc={best_acc*100:.2f})")
        unwrap_ddp(backbone).load_state_dict(best_state["backbone"])
        unwrap_ddp(head).load_state_dict(best_state["head"])

    final_results = eval_val_entries(
        backbone, head, val_entries, use_amp=args.amp, prefix="Final",
        is_main=is_main, log_path=val_results_path
    )
    final = _merge_val_results(final_results)
    if is_main:
        print("\nConfusion Matrix (rows=GT, cols=Pred):")
        print("            Pred: real    Pred: fake")
        print(f"GT: real     {final.tn:>6d}      {final.fp:>6d}")
        print(f"GT: fake     {final.fn:>6d}      {final.tp:>6d}")
        print(f"\nFinal Val Acc: {final.acc*100:.2f}% | P {final.precision*100:.2f}% | R {final.recall*100:.2f}% | F1 {final.f1*100:.2f}%")
        with open(val_results_path, "a") as f:
            f.write(f"final,{final.acc:.6f},{final.precision:.6f},{final.recall:.6f},{final.f1:.6f}\n")


if __name__ == "__main__":
    main()
