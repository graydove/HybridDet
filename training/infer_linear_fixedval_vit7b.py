import argparse
import os
import json
import csv
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
    fake_keys = ["fake","ai","synth","gen","generated","diff","sd","midjourney","dalle","stable","novelai","civitai"]
    real_keys = ["real","photo","gt","natural"]
    new_samples, new_targets = [], []
    for path, _ in ds.samples:
        cname = os.path.basename(os.path.dirname(path)).lower()
        y = None
        if any(k in cname for k in fake_keys) or cname.startswith("1_") or cname == "1":
            y = 1
        elif any(k in cname for k in real_keys) or cname.startswith("0_") or cname == "0":
            y = 0
        # fallback: if class name exactly equals one of dataset-level buckets
        elif cname in ("0","1"):
            y = int(cname)
        else:
            # if there are only two classes total, map whichever is not 'real' as fake
            pass
        if y is None:
            # try a conservative fallback: treat non-real as fake
            y = 0 if ("real" in cname or "photo" in cname or cname.startswith("0_")) else 1
        new_samples.append((path, y))
        new_targets.append(y)
    ds.samples = new_samples
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


# ---------------- Data + Collate ----------------
def make_collate(processor, device):
    def _collate(batch):
        # Support (img, y, subclass) triplet
        if len(batch[0]) == 3:
            imgs, labels, subclasses = list(zip(*batch))
        else:
            imgs, labels = list(zip(*batch))
            subclasses = [None] * len(imgs)
        inputs = processor(images=list(imgs), return_tensors="pt")
        pixel_values = inputs["pixel_values"]  # keep CPU
        labels = torch.tensor(labels, dtype=torch.long)  # CPU
        return pixel_values, labels, list(subclasses)
    return _collate


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
    it = tqdm(loader, total=len(loader), desc="Eval", leave=False)
    for pixel_values, y, subclasses in it:
        pixel_values = pixel_values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(use_amp and has_cuda)):
            outputs = backbone(pixel_values=pixel_values)
            feats = (outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None
                     else outputs.last_hidden_state[:, 0, :])
            logits = head(feats)
        meter.update(logits, y)

        preds = logits.argmax(dim=1).detach().cpu()
        gts = y.detach().cpu()
        for cls_name, pred_i, gt_i in zip(subclasses, preds.tolist(), gts.tolist()):
            key = cls_name if cls_name is not None else "_unknown"
            corr, tot = subclass_stats.get(key, (0, 0))
            subclass_stats[key] = (corr + int(pred_i == gt_i), tot + 1)

    return meter, subclass_stats


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
    dummy = torch.zeros(1, 3, H, W, device=device)
    out = backbone(pixel_values=dummy)
    feats = (out.pooler_output if getattr(out, "pooler_output", None) is not None
             else out.last_hidden_state[:,0,:])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_roots", nargs='+', default=[
            "./datasets/GenImage/",
            "./datasets/RRDataset_final/",
        ], help="一个或多个测试集根目录（每个根目录下包含多个子文件夹，每个子文件夹内有 0_real 与 1_fake）",
    )
    parser.add_argument("--model_id", type=str, default="dinov3-vit7b16-pretrain-lvd1689m")
    parser.add_argument("--ckpt", type=str, default="checkpoints_1fc/dinov3_linear_head_best.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--csv", type=str, default="eval_results.csv", help="保存结果的CSV路径（默认：eval_results.csv）")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Processor with fallback
    try:
        processor = AutoImageProcessor.from_pretrained(args.model_id)
    except Exception as e:
        print(f"[Warn] AutoImageProcessor not available for '{args.model_id}' ({e}). Using SimpleImageProcessor fallback.")
        processor = SimpleImageProcessor.from_pretrained(args.model_id)

    # Backbone and linear head
    backbone = AutoModel.from_pretrained(args.model_id).to(device)
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
        state = ckpt.get("head", ckpt)
        missing, unexpected = head.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}; missing={list(missing)}, unexpected={list(unexpected)}")
    else:
        print(f"[Warn] checkpoint not found: {args.ckpt}. Using randomly initialized head.")

    pin_mem = device.type == 'cuda'
    collate = make_collate(processor, device)

    # Build datasets and evaluate
    results = []

    for root in args.data_roots:
        print(f"\n[Dataset Root] {root}")
        subdirs = list_immediate_subdirs(root)
        if not subdirs:
            print(f"[Warn] no subfolders found in: {root}")
        for sub in subdirs:
            sub_root = resolve_dataset_root(os.path.join(root, sub))
            if not os.path.isdir(sub_root):
                print(f"[Skip] path not found: {sub_root}")
                continue
            ds = datasets.ImageFolder(sub_root)
            ds = standardize_binary_labels(ds)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=pin_mem, collate_fn=collate)
            if len(ds) == 0:
                print(f"  {sub}: [Skip] empty dataset at {sub_root}")
                continue
            meter, _ = evaluate(backbone, head, loader, use_amp=args.amp, device=device)
            print(f"  {sub}: Acc {meter.acc*100:.2f}% (N={meter.total})")
            results.append((f"{os.path.basename(os.path.normpath(root))}/{sub}", meter.acc, meter.precision, meter.recall, meter.f1, meter.total))

    if results:
        print("\nSummary:")
        for name, acc, p, r, f1, N in results:
            print(f"{name}: Acc {acc*100:.2f}% | P {p*100:.2f}% | R {r*100:.2f}% | F1 {f1*100:.2f}%")

    # --- 将结果写入 CSV ---
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "acc(%)", "precision(%)", "recall(%)", "f1(%)", "N"])
        for name, acc, p, r, f1, N in results:
            writer.writerow([name, f"{acc*100:.4f}", f"{p*100:.4f}", f"{r*100:.4f}", f"{f1*100:.4f}", N])
    print(f"\n[Saved] CSV written to: {args.csv}")


if __name__ == "__main__":
    main()

# 请你修改上述代码，使得输出结果全部保存到csv文件中