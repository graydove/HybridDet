# save_logits.py
# 用法示例：
#   python save_logits.py --data_root ./datasets/GenImage --model_id dinov3-vitl16-pretrain-lvd1689m \
#                         --ckpt checkpoints/dinov3_linear_head_best.pt --out_csv logits.csv
import os
import csv
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from transformers import AutoImageProcessor, AutoModel
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, *_, **__): return x


def standardize_binary_labels(ds: datasets.ImageFolder):
    fake_keys = ["fake","ai","synth","gen","generated","diff","sd","midjourney","dalle","stable","novelai","civitai","1","1_"]
    real_keys = ["real","photo","gt","natural","0","0_"]
    new_samples, new_targets = [], []
    for path, _ in ds.samples:
        cname = os.path.basename(os.path.dirname(path)).lower()
        if any(k in cname for k in real_keys) or cname.startswith("0_") or cname == "0":
            y = 0
        elif any(k in cname for k in fake_keys) or cname.startswith("1_") or cname == "1":
            y = 1
        else:  # 不确定时：非“real”按 fake 处理
            y = 0 if ("real" in cname or "photo" in cname or cname.startswith("0_")) else 1
        new_samples.append((path, y)); new_targets.append(y)
    ds.samples, ds.targets = new_samples, new_targets
    ds.classes, ds.class_to_idx = ["real", "fake"], {"real":0, "fake":1}
    return ds


def make_collate(processor):
    def _collate(batch):
        imgs, ys = zip(*batch)
        inputs = processor(images=list(imgs), return_tensors="pt")
        return inputs["pixel_values"], torch.tensor(ys, dtype=torch.long)
    return _collate


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, bias: bool = True):
        super().__init__(); self.fc = nn.Linear(in_dim, num_classes, bias=bias)
    def forward(self, x): return self.fc(x)


@torch.no_grad()
def infer_feat_dim(backbone, processor):
    size = getattr(processor, "size", 224)
    if isinstance(size, dict):
        H = int(size.get("height", size.get("shortest_edge", 224)))
        W = int(size.get("width",  size.get("shortest_edge", 224)))
    elif isinstance(size, int):
        H = W = int(size)
    else:
        H = W = 224
    dummy = torch.zeros(1, 3, H, W, device=next(backbone.parameters()).device)
    out = backbone(pixel_values=dummy)
    feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:,0,:]
    return feats.shape[-1]


@torch.no_grad()
def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) 载入处理器与骨干
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    backbone  = AutoModel.from_pretrained(args.model_id).to(device)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad = False

    # 2) 线性头（从 ckpt 恢复）
    feat_dim = infer_feat_dim(backbone, processor)
    head = LinearHead(in_dim=feat_dim, num_classes=2).to(device).eval()
    if os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state = ckpt.get("head", ckpt)
        head.load_state_dict(state, strict=False)
        print(f"[OK] loaded head: {args.ckpt}")
    else:
        print(f"[Warn] checkpoint not found: {args.ckpt} (using randomly initialized head)")

    # 3) 数据集 & DataLoader（不打乱，保证样本次序与文件名对齐）
    ds = datasets.ImageFolder(args.data_root)
    ds = standardize_binary_labels(ds)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=make_collate(processor), pin_memory=(device.type=="cuda"))

    # 为了记录文件名
    filepaths = [p for p,_ in ds.samples]

    # 4) 推理并收集 logits / prob_fake
    rows = []
    idx = 0
    for pixel_values, y in tqdm(loader, total=len(loader), desc="Infer"):
        pixel_values = pixel_values.to(device, non_blocking=True)
        out = backbone(pixel_values=pixel_values)
        feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:,0,:]
        logits = head(feats)                     # [B, 2]
        probs  = torch.softmax(logits, dim=1)    # [B, 2]
        logit_fake = logits[:,1].cpu().tolist()
        logit_real = logits[:,0].cpu().tolist()
        prob_fake  = probs[:,1].cpu().tolist()
        labels     = y.cpu().tolist()
        B = len(labels)
        for i in range(B):
            fname = os.path.basename(filepaths[idx+i]) if idx+i < len(filepaths) else f"sample_{idx+i}"
            rows.append([fname, labels[i], logit_real[i], logit_fake[i], prob_fake[i]])
        idx += B

    # 5) 写 CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label(0=real,1=fake)", "logit_real", "logit_fake", "prob_fake"])
        for r in rows: w.writerow(r)
    print(f"[Saved] logits/probabilities -> {args.out_csv}  (N={len(rows)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="测试集根目录（含 0_real / 1_fake 子目录）")
    ap.add_argument("--model_id",  type=str, default="dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--ckpt",      type=str, default="checkpoints/dinov3_linear_head_best.pt")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--out_csv",   type=str, default="logits.csv")
    args = ap.parse_args()
    run(args)
