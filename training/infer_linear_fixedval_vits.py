import argparse
import os
import json
import csv
import math
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from dataclasses import dataclass

from transformers import AutoImageProcessor, AutoModel
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *args, **kwargs):
        return x


# ---------------- Meter & Utils ----------------
@dataclass
class Meter:
    correct:int=0; total:int=0
    tp:int=0; tn:int=0; fp:int=0; fn:int=0
    def update(self, logits: torch.Tensor, y: torch.Tensor):
        pred = logits.argmax(dim=1)
        self.correct += (pred==y).sum().item()
        self.total   += y.numel()
        self.tp += ((pred==1)&(y==1)).sum().item()
        self.tn += ((pred==0)&(y==0)).sum().item()
        self.fp += ((pred==1)&(y==0)).sum().item()
        self.fn += ((pred==0)&(y==1)).sum().item()
    @property
    def acc(self): return self.correct / max(1,self.total)
    @property
    def precision(self): return self.tp / max(1, (self.tp+self.fp))
    @property
    def recall(self): return self.tp / max(1, (self.tp+self.fn))
    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2*p*r/max(1e-12,(p+r))


# ---------------- Label standardization ----------------
def standardize_binary_labels(ds: datasets.ImageFolder):
    new_samples, new_targets = [], []
    for path, _ in ds.samples:
        parts = os.path.normpath(path).split(os.sep)
        y = None
        for part in reversed(parts[:-1]):
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
    if hasattr(ds, "imgs"):
        ds.imgs = new_samples
    ds.targets = new_targets
    ds.classes = ["real", "fake"]
    ds.class_to_idx = {"real":0, "fake":1}
    return ds


# ---------------- Processor Fallback ----------------
try:
    from torchvision.transforms import InterpolationMode
    import torchvision.transforms.functional as TF
except Exception:
    TF = None
    class InterpolationMode:
        BILINEAR = None

class SimpleImageProcessor:
    def __init__(self, size=224, image_mean=(0.485,0.456,0.406), image_std=(0.229,0.224,0.225), rescale_factor=1.0):
        if isinstance(size, dict):
            self.size = {"height": int(size.get("height", 224)), "width": int(size.get("width", 224))}
        elif isinstance(size, int):
            self.size = {"height": int(size), "width": int(size)}
        elif isinstance(size, (tuple, list)) and len(size)==2:
            self.size = {"height": int(size[0]), "width": int(size[1])}
        else:
            self.size = {"height": 224, "width": 224}
        self.image_mean = torch.tensor(image_mean).view(3,1,1)
        self.image_std  = torch.tensor(image_std).view(3,1,1)
        self.rescale_factor = float(rescale_factor)

    @staticmethod
    def from_pretrained(model_id_or_path: str):
        size = 224; mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225); rescale=1/255
        pp_path = os.path.join(model_id_or_path, 'preprocessor_config.json')
        if os.path.isfile(pp_path):
            try:
                with open(pp_path, 'r') as f:
                    cfg = json.load(f)
                if isinstance(cfg.get('size'), dict):
                    sz = cfg['size']
                    size = {"height": int(sz.get('height', 224)), "width": int(sz.get('width', 224))}
                elif isinstance(cfg.get('size'), int):
                    size = int(cfg['size'])
                mean = tuple(cfg.get('image_mean', mean))
                std  = tuple(cfg.get('image_std', std))
                rescale = float(cfg.get('rescale_factor', rescale))
            except Exception:
                pass
        else:
            cfg_path = os.path.join(model_id_or_path, 'config.json')
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, 'r') as f:
                        cfg = json.load(f)
                    if 'image_size' in cfg:
                        size = int(cfg['image_size'])
                except Exception:
                    pass
        return SimpleImageProcessor(size=size, image_mean=mean, image_std=std, rescale_factor=rescale)

    def __call__(self, images, return_tensors="pt"):
        tensors = []
        H, W = int(self.size['height']), int(self.size['width'])
        for img in images:
            if TF is None:
                import PIL.Image
                if isinstance(img, PIL.Image.Image):
                    img = img.convert('RGB')
                    import numpy as np
                    arr = np.array(img.resize((W, H)))
                    t = torch.from_numpy(arr).permute(2,0,1).float() * (self.rescale_factor if self.rescale_factor!=0 else 1.0)
                    t = (t - self.image_mean*255*self.rescale_factor) / (self.image_std*255*self.rescale_factor)
                else:
                    t = img
            else:
                import PIL.Image
                if isinstance(img, PIL.Image.Image):
                    img = img.convert('RGB')
                if isinstance(img, torch.Tensor):
                    img = TF.to_pil_image(img)
                img = TF.resize(img, [H, W], interpolation=InterpolationMode.BILINEAR)
                t = TF.to_tensor(img)
                if abs(self.rescale_factor - 1/255) > 1e-8:
                    t = t / (1/255) * self.rescale_factor
                t = (t - self.image_mean) / self.image_std
            tensors.append(t)
        batch = torch.stack(tensors, dim=0)
        return {"pixel_values": batch}


# ---------------- DCT Processor ----------------
class DCTBlockHFProcessorV2:
    """log-DCT artifact 能量，多频段，多通道，输出 shape [B, 9, H, W]"""
    def __init__(self, size=224, train=False, block=8):
        if isinstance(size, int):
            H = W = size
        else:
            H = size["height"]; W = size["width"]
        assert H % block == 0 and W % block == 0
        self.size = (H, W); self.train = train; self.block = block
        self.C = self._build_dct_matrix(block); self.Ct = self.C.t().contiguous()
        self.masks = [self._hf_mask(block, th) for th in (3, 5, 7)]

    def _build_dct_matrix(self, N):
        C = torch.zeros(N, N)
        for k in range(N):
            a = math.sqrt(1/N) if k == 0 else math.sqrt(2/N)
            for n in range(N):
                C[k, n] = a * math.cos(math.pi * (2*n + 1) * k / (2*N))
        return C

    def _hf_mask(self, N, th):
        m = torch.zeros(N, N, dtype=torch.bool)
        for u in range(N):
            for v in range(N):
                if u+v >= th and not (u == 0 and v == 0):
                    m[u, v] = True
        return m

    def _crop(self, img):
        import torchvision.transforms.functional as TF
        H, W = self.size
        ih, iw = img.height, img.width
        top = max(0, (ih - H) // 2)
        left = max(0, (iw - W) // 2)
        return TF.crop(img, top, left, H, W)

    def _rgb_to_ycbcr(self, t):
        r, g, b = t
        y  = 0.299*r + 0.587*g + 0.114*b
        cb = -0.1687*r - 0.3313*g + 0.5*b + 0.5
        cr = 0.5*r - 0.4187*g - 0.0813*b + 0.5
        return (y*255-128), (cb*255-128), (cr*255-128)

    def _dct_energy(self, x, mask):
        B = self.block
        blocks = x.unfold(0, B, B).unfold(1, B, B)
        C = self.C.to(x.device); Ct = self.Ct.to(x.device)
        coef = torch.matmul(torch.matmul(C, blocks), Ct)
        mag = coef.abs()
        e = (mag * mask.to(x.device)).sum(dim=(-1, -2))
        return e.repeat_interleave(B, 0).repeat_interleave(B, 1)

    def __call__(self, images, return_tensors="pt"):
        import torchvision.transforms.functional as TF
        feats = []
        for img in images:
            img = self._crop(img.convert("RGB"))
            t = TF.to_tensor(img)
            y, cb, cr = self._rgb_to_ycbcr(t)
            chans = []
            for comp in (y, cb, cr):
                for m in self.masks:
                    e = self._dct_energy(comp, m)
                    chans.append(torch.log1p(e))
            feats.append(torch.stack(chans))
        return {"pixel_values": torch.stack(feats)}


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


# ---------------- Data + Collate ----------------
def make_collate(processor, device):
    def _collate(batch):
        if len(batch[0]) == 3:
            imgs, labels, subclasses = list(zip(*batch))
        else:
            imgs, labels = list(zip(*batch))
            subclasses = [None] * len(imgs)
        inputs = processor(images=list(imgs), return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        labels = torch.tensor(labels, dtype=torch.long)
        return pixel_values, labels, list(subclasses)
    return _collate


def make_image_loader(allow_truncated: bool = False):
    def _loader(path: str):
        from PIL import Image, ImageFile
        if allow_truncated:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                return img.convert("RGB")
        except Exception as exc:
            raise OSError(f"Failed to load image: {path}. {exc}") from exc
    return _loader


def make_imagefolder(root: str, loader):
    try:
        return datasets.ImageFolder(root, loader=loader, allow_empty=True)
    except TypeError:
        return datasets.ImageFolder(root, loader=loader)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=bias)
    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def evaluate(backbone, head, loader, use_amp: bool, device: torch.device):
    backbone.eval(); head.eval()
    meter = Meter()
    has_cuda = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if has_cuda else torch.float32
    amp_device_type = "cuda" if has_cuda else "cpu"

    subclass_stats = {}
    all_paths = [p for p, _ in getattr(loader.dataset, 'samples', [])]
    dataset_root = getattr(loader.dataset, "root", None)
    if dataset_root is not None:
        dataset_root = os.path.abspath(dataset_root)
    path_cursor = 0
    per_image = []  # (filename, prob_fake)
    it = tqdm(loader, total=len(loader), desc="Eval", leave=False)
    for pixel_values, y, subclasses in it:
        pixel_values = pixel_values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(use_amp and has_cuda)):
            outputs = backbone(pixel_values=pixel_values)
            feats = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
            logits = head(feats)
        meter.update(logits, y)

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
        bsz = len(probs)
        for i in range(bsz):
            if path_cursor + i < len(all_paths):
                path = all_paths[path_cursor + i]
                if dataset_root:
                    abs_path = os.path.abspath(path)
                    if os.path.commonpath([abs_path, dataset_root]) == dataset_root:
                        fname = os.path.relpath(abs_path, dataset_root)
                    else:
                        fname = os.path.basename(path)
                else:
                    fname = os.path.basename(path)
            else:
                fname = f"sample_{path_cursor+i}"
            per_image.append((fname, float(probs[i])))
        path_cursor += bsz

        preds = logits.argmax(dim=1).detach().cpu()
        gts = y.detach().cpu()
        for cls_name, pred_i, gt_i in zip(subclasses, preds.tolist(), gts.tolist()):
            key = cls_name if cls_name is not None else "_unknown"
            corr, tot = subclass_stats.get(key, (0, 0))
            subclass_stats[key] = (corr + int(pred_i == gt_i), tot + 1)

    return meter, subclass_stats, per_image


def infer_feat_dim(backbone, processor, device):
    size = getattr(processor, "size", None)
    H = W = 224
    if isinstance(size, dict):
        if "shortest_edge" in size:
            H = W = int(size["shortest_edge"]) or 224
        elif "height" in size and "width" in size:
            H = int(size["height"]) or 224
            W = int(size["width"]) or 224
        elif "shortest_side" in size:
            H = W = int(size["shortest_side"]) or 224
    elif isinstance(size, int):
        H = W = int(size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        H, W = int(size[0]), int(size[1])
    C = 9 if isinstance(processor, DCTBlockHFProcessorV2) else 3
    dummy = torch.zeros(1, C, H, W, device=device)
    out = backbone(pixel_values=dummy)
    feats = out.last_hidden_state[:,1:,:].mean(dim=1)
    return feats.shape[-1]


def resolve_dataset_root(root: str) -> str:
    for sub in ["test","val","validation","eval","Eval","Validation","Test"]:
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


def _looks_like_category(name: str) -> bool:
    return bool(re.fullmatch(r"[a-z]+", name))


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

    if any(r or f for _, r, f, _ in subdir_flags):
        cat_like = sum(1 for sub, _, _, _ in subdir_flags if _looks_like_category(sub))
        if cat_like / max(1, len(subdirs)) >= 0.6:
            return [root]

    return [os.path.join(root, s) for s in subdirs]


def infer_acc_label(targets) -> str:
    label_set = set(int(x) for x in targets) if targets else set()
    has_real = 0 in label_set
    has_fake = 1 in label_set
    if has_fake and not has_real:
        return "Fake Acc"
    if has_real and not has_fake:
        return "Real Acc"
    return "Acc"

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
        "--data_roots", nargs='+', default=[
            "../datasets/test/",
        ], help="一个或多个测试集根目录；也可传入包含多行路径的 .txt 文件，或逗号分隔的路径字符串",
    )
    parser.add_argument("--model_id", type=str, default="dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--ckpt", type=str, default="checkpoints/dinov3_linear_head_best.pt")
    parser.add_argument("--input_mode", type=str, default="jpeg_dct", choices=["rgb", "jpeg_dct"])
    parser.add_argument("--crop_size", type=int, default=336)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--allow_truncated", action="store_true",
                        help="允许加载被截断的图片（PIL ImageFile.LOAD_TRUNCATED_IMAGES）")
    parser.add_argument("--split_mode", type=str, default="auto", choices=["auto", "subdir", "root"],
                        help="数据集拆分策略：auto 自动判断；subdir 逐子文件夹评估；root 将每个 data_root 视为单个数据集")
    parser.add_argument("--csv", type=str, default="eval_results_vits.csv", help="保存数据集汇总结果的CSV路径")
    parser.add_argument("--pred_csv", type=str, default="pred_results_vits.csv", help="保存每张图片预测结果的CSV路径")
    parser.add_argument("--log_txt", type=str, default="infer_linear_fixedval_vits_log.txt",
                        help="保存控制台输出的日志路径（.txt）")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_fp = open(args.log_txt, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_fp.write(f"{msg}\n")
        log_fp.flush()

    try:
        processor = AutoImageProcessor.from_pretrained(args.model_id) if args.input_mode == "rgb" else DCTBlockHFProcessorV2(size=args.crop_size, train=False)
    except Exception as e:
        log(f"[Warn] Processor init failed ({e}). Using SimpleImageProcessor fallback.")
        processor = SimpleImageProcessor.from_pretrained(args.model_id)

    backbone = AutoModel.from_pretrained(args.model_id).to(device)
    if args.input_mode == "jpeg_dct" and isinstance(processor, DCTBlockHFProcessorV2):
        adapt_patch_embed_channels(backbone, in_channels=9, device=device)
    for p in backbone.parameters():
        p.requires_grad = False

    feat_dim = infer_feat_dim(backbone, processor, device)
    head = LinearHead(in_dim=feat_dim, num_classes=2).to(device)

    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        alt = os.path.join(os.path.dirname(ckpt_path), "dinov3_linear_head_best_multi.pt")
        if os.path.isfile(alt):
            ckpt_path = alt
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "backbone" in ckpt:
            miss_b, unexp_b = backbone.load_state_dict(ckpt["backbone"], strict=False)
        else:
            miss_b, unexp_b = [], []
        state = ckpt["head"] if isinstance(ckpt, dict) and "head" in ckpt else ckpt
        missing, unexpected = head.load_state_dict(state, strict=False)
        log(f"Loaded checkpoint: {ckpt_path}; backbone missing={list(miss_b)}, unexpected={list(unexp_b)}; head missing={list(missing)}, unexpected={list(unexpected)}")
    else:
        log(f"[Warn] checkpoint not found: {args.ckpt}. Using randomly initialized head.")

    pin_mem = device.type == 'cuda'
    collate = make_collate(processor, device)
    loader_fn = make_image_loader(allow_truncated=args.allow_truncated)

    results = []
    pred_rows = []
    data_roots = expand_data_roots(args.data_roots)
    with open("acc.txt", "w+") as acc_file:
        for root in data_roots:
            log(f"\n[Dataset Root] {root}")
            eval_roots = infer_eval_roots(root, split_mode=args.split_mode)
            if not eval_roots:
                log(f"[Warn] no valid dataset roots found in: {root}")
            root_results = []
            for eval_root in eval_roots:
                sub_root = resolve_dataset_root(eval_root)
                if not os.path.isdir(sub_root):
                    log(f"[Skip] path not found: {sub_root}")
                    continue
                ds = make_imagefolder(sub_root, loader=loader_fn)
                ds = standardize_binary_labels(ds)
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=pin_mem, collate_fn=collate)
                if len(ds) == 0:
                    log(f"  {os.path.basename(os.path.normpath(eval_root))}: [Skip] empty dataset at {sub_root}")
                    continue
                meter, _, per_image = evaluate(backbone, head, loader, use_amp=args.amp, device=device)
                display_name = os.path.basename(os.path.normpath(eval_root))
                if os.path.normpath(eval_root) != os.path.normpath(root):
                    display_name = f"{os.path.basename(os.path.normpath(root))}/{display_name}"
                acc_label = infer_acc_label(getattr(ds, "targets", []))
                log(f"  {display_name}: {acc_label} {meter.acc*100:.2f}% (N={meter.total})")
                results.append((display_name, acc_label, meter.acc, meter.precision, meter.recall, meter.f1, meter.total))
                root_results.append((meter.acc, meter.total))
                for fname, prob in per_image:
                    acc_file.write(fname + '\t' + str(prob) + '\n')
                    pred_rows.append((display_name, os.path.basename(fname), prob))
            if root_results:
                all_acc = sum(acc for acc, _ in root_results) / max(1, len(root_results))
                total_n = sum(n for _, n in root_results)
                balanced_acc = sum(acc * n for acc, n in root_results) / max(1, total_n)
                log(f"  All Acc {all_acc*100:.2f}%")
                log(f"  Balanced Acc {balanced_acc*100:.2f}%")

    if results:
        log("\nSummary:")
        for name, acc_label, acc, p, r, f1, N in results:
            log(f"{name}: {acc_label} {acc*100:.2f}% | P {p*100:.2f}% | R {r*100:.2f}% | F1 {f1*100:.2f}%")

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "acc_label", "acc(%)", "precision(%)", "recall(%)", "f1(%)", "N"])
        for name, acc_label, acc, p, r, f1, N in results:
            writer.writerow([name, acc_label, f"{acc*100:.4f}", f"{p*100:.4f}", f"{r*100:.4f}", f"{f1*100:.4f}", N])
    log(f"\n[Saved] CSV written to: {args.csv}")

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
