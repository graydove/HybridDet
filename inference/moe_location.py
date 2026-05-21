import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from moe_gate import (
    CropNoResizeTransform,
    DCTBlockHF9CenterCropTransform,
    _safe_np_pad,
    adapt_patch_embed_channels,
    collect_grouped_class_paths,
    find_group_roots,
    load_expert,
    set_seed,
    setup_logger,
)


def resolve_crop_size(arg_crop: int, ckpt_crop):
    if arg_crop and int(arg_crop) > 0:
        return int(arg_crop)
    if ckpt_crop and int(ckpt_crop) > 0:
        return int(ckpt_crop)
    return 336


def build_transform(input_mode: str, crop_size: int, mean, std, pad_mode: str):
    if input_mode == "dct9":
        return DCTBlockHF9CenterCropTransform(crop_size=crop_size)
    mean = mean or (0.485, 0.456, 0.406)
    std = std or (0.229, 0.224, 0.225)
    return CropNoResizeTransform(
        crop_size=crop_size,
        mean=mean,
        std=std,
        pad_mode=pad_mode,
    )


def compute_positions(length: int, window: int, stride: int):
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, stride))
    last = length - window
    if positions[-1] != last:
        positions.append(last)
    return positions


def pad_patch_to_size(img: Image.Image, crop_size: int, pad_mode: str):
    w, h = img.size
    pad_w = max(0, crop_size - w)
    pad_h = max(0, crop_size - h)
    if pad_w == 0 and pad_h == 0:
        return img

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    if pad_mode == "constant":
        return ImageOps.expand(img, border=(left, top, right, bottom), fill=0)

    arr = np.array(img)
    arr = _safe_np_pad(arr, ((top, bottom), (left, right), (0, 0)), mode=pad_mode)
    return Image.fromarray(arr)


def crop_window(img: Image.Image, left: int, top: int, crop_size: int, pad_mode: str):
    right = min(left + crop_size, img.width)
    bottom = min(top + crop_size, img.height)
    patch = img.crop((left, top, right, bottom))
    if patch.size != (crop_size, crop_size):
        patch = pad_patch_to_size(patch, crop_size=crop_size, pad_mode=pad_mode)
    return patch


def iter_fake_images(data_root: str, fake_dir_name: str):
    items = []
    for group_name, group_root in find_group_roots(data_root):
        items.extend(collect_grouped_class_paths(group_root, fake_dir_name, 1, group_name=group_name))
    items.sort(key=lambda x: x[0])
    return items


def colorize_prob_map(prob_map: np.ndarray):
    anchors_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    anchors_rgb = np.array(
        [
            [0, 0, 128],
            [0, 180, 255],
            [80, 255, 120],
            [255, 220, 0],
            [255, 0, 0],
        ],
        dtype=np.float32,
    )
    flat = np.clip(prob_map.astype(np.float32), 0.0, 1.0).reshape(-1)
    out = np.empty((flat.size, 3), dtype=np.float32)
    for channel in range(3):
        out[:, channel] = np.interp(flat, anchors_x, anchors_rgb[:, channel])
    return out.reshape(prob_map.shape + (3,)).astype(np.uint8)


def build_output_prefix(data_root: str, image_path: str, out_dir: str):
    data_root_path = Path(data_root).resolve()
    image_path_obj = Path(image_path).resolve()
    try:
        rel = image_path_obj.relative_to(data_root_path)
    except ValueError:
        rel = Path(image_path_obj.name)
    rel_no_suffix = rel.with_suffix("")
    return Path(out_dir) / rel_no_suffix


@torch.no_grad()
def infer_patch_probs(model, batch_tensors, device: torch.device, use_amp: bool):
    batch = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")):
        feats = model.forward_features(batch)
        logits = model.head(feats)
    probs = logits.float().softmax(dim=1)[:, 1]
    return probs.detach().cpu().numpy()


def save_heatmap_artifacts(original_img: Image.Image, prob_map: np.ndarray, prefix: Path):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prob_uint8 = np.clip(np.round(prob_map * 255.0), 0.0, 255.0).astype(np.uint8)
    heat_rgb = colorize_prob_map(prob_map)
    orig_rgb = np.array(original_img.convert("RGB"), dtype=np.float32)
    overlay = np.clip(orig_rgb * 0.45 + heat_rgb.astype(np.float32) * 0.55, 0.0, 255.0).astype(np.uint8)

    np.save(str(prefix) + "_prob.npy", prob_map.astype(np.float32))
    Image.fromarray(prob_uint8, mode="L").save(str(prefix) + "_prob_gray.png")
    Image.fromarray(heat_rgb, mode="RGB").save(str(prefix) + "_heatmap.png")
    Image.fromarray(overlay, mode="RGB").save(str(prefix) + "_overlay.png")


def process_single_image(
    model,
    transform,
    image_path: str,
    crop_size: int,
    stride: int,
    pad_mode: str,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    xs = compute_positions(width, crop_size, stride)
    ys = compute_positions(height, crop_size, stride)

    accum = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    windows = []
    batch_tensors = []
    batch_windows = []

    for top in ys:
        for left in xs:
            patch = crop_window(img, left=left, top=top, crop_size=crop_size, pad_mode=pad_mode)
            batch_tensors.append(transform(patch))
            batch_windows.append((left, top))

            if len(batch_tensors) < batch_size:
                continue

            probs = infer_patch_probs(model, batch_tensors, device=device, use_amp=use_amp)
            for (win_left, win_top), prob in zip(batch_windows, probs):
                right = min(win_left + crop_size, width)
                bottom = min(win_top + crop_size, height)
                accum[win_top:bottom, win_left:right] += float(prob)
                counts[win_top:bottom, win_left:right] += 1.0
                windows.append((win_left, win_top, right, bottom, float(prob)))
            batch_tensors.clear()
            batch_windows.clear()

    if batch_tensors:
        probs = infer_patch_probs(model, batch_tensors, device=device, use_amp=use_amp)
        for (win_left, win_top), prob in zip(batch_windows, probs):
            right = min(win_left + crop_size, width)
            bottom = min(win_top + crop_size, height)
            accum[win_top:bottom, win_left:right] += float(prob)
            counts[win_top:bottom, win_left:right] += 1.0
            windows.append((win_left, win_top, right, bottom, float(prob)))

    counts = np.maximum(counts, 1.0)
    prob_map = accum / counts
    return img, prob_map, windows


def main():
    ap = argparse.ArgumentParser(
        description="Sliding-window localization with the representation-driven expert."
    )
    ap.add_argument("--data_root", type=str, default="../datasets/test")
    ap.add_argument(
        "--representation_ckpt",
        "--ckpt",
        dest="ckpt",
        type=str,
        metavar="REPRESENTATION_CKPT",
        default="../checkpoints/representation_driven_expert.pt",
        help="Representation-driven expert checkpoint. The released file name is representation_driven_expert.pt.",
    )
    ap.add_argument("--input_mode", type=str, default="rgb", choices=["rgb", "dct9"])
    ap.add_argument("--crop_size", type=int, default=336)
    ap.add_argument("--stride", type=int, default=56)
    ap.add_argument("--pad_mode", type=str, default="reflect", choices=["constant", "edge", "reflect", "symmetric"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--fake_dir_name", type=str, default="1_fake")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="../outputs/location_eval")
    ap.add_argument("--log_file", type=str, default="location_eval.log")
    ap.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--summary_csv", type=str, default="image_summary.csv")
    ap.add_argument("--patch_csv", type=str, default="patch_scores.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_file = args.log_file if os.path.isabs(args.log_file) else os.path.join(args.out_dir, args.log_file)
    logger = setup_logger(log_file, level=args.log_level)

    logger.info("==== Sliding Localization Started ====")
    logger.info(f"cmdline: {' '.join(sys.argv)}")
    logger.info(f"data_root={args.data_root}")
    logger.info(f"ckpt={args.ckpt}")
    logger.info(f"stride={args.stride}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    model, model_id, feat_dim, ckpt_in_channels, mean, std, ckpt_crop = load_expert(args.ckpt, device=device, logger=logger)
    input_channels = 9 if args.input_mode == "dct9" else 3
    if ckpt_in_channels and ckpt_in_channels != input_channels:
        logger.warning(
            f"[Warn] checkpoint expects in_channels={ckpt_in_channels}, but input_mode={args.input_mode} uses {input_channels}"
        )
    adapt_patch_embed_channels(model.backbone, in_channels=input_channels, device=device)

    crop_size = resolve_crop_size(args.crop_size, ckpt_crop)
    transform = build_transform(args.input_mode, crop_size=crop_size, mean=mean, std=std, pad_mode=args.pad_mode)
    logger.info(
        f"model_id={model_id} feat_dim={feat_dim} input_mode={args.input_mode} crop_size={crop_size} pad_mode={args.pad_mode}"
    )

    items = iter_fake_images(args.data_root, args.fake_dir_name)
    if args.max_images and args.max_images > 0:
        items = items[: args.max_images]
    if not items:
        raise SystemExit(f"No fake images found under {args.data_root}")

    summary_csv = args.summary_csv if os.path.isabs(args.summary_csv) else os.path.join(args.out_dir, args.summary_csv)
    patch_csv = args.patch_csv if os.path.isabs(args.patch_csv) else os.path.join(args.out_dir, args.patch_csv)
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(patch_csv) or ".", exist_ok=True)

    total_windows = 0
    with open(summary_csv, "w", newline="", encoding="utf-8") as summary_fp, open(
        patch_csv, "w", newline="", encoding="utf-8"
    ) as patch_fp:
        summary_writer = csv.writer(summary_fp)
        patch_writer = csv.writer(patch_fp)
        summary_writer.writerow(
            [
                "group",
                "subgroup",
                "image_path",
                "width",
                "height",
                "num_windows",
                "prob_mean",
                "prob_min",
                "prob_max",
            ]
        )
        patch_writer.writerow(
            [
                "group",
                "subgroup",
                "image_path",
                "left",
                "top",
                "right",
                "bottom",
                "prob_fake",
            ]
        )

        for idx, (image_path, _label, group_name, subgroup_name) in enumerate(items, start=1):
            logger.info(f"[Image {idx}/{len(items)}] {image_path}")
            original_img, prob_map, windows = process_single_image(
                model=model,
                transform=transform,
                image_path=image_path,
                crop_size=crop_size,
                stride=args.stride,
                pad_mode=args.pad_mode,
                batch_size=args.batch_size,
                device=device,
                use_amp=args.amp,
            )
            prefix = build_output_prefix(args.data_root, image_path, args.out_dir)
            save_heatmap_artifacts(original_img, prob_map, prefix)

            for left, top, right, bottom, prob in windows:
                patch_writer.writerow([group_name, subgroup_name, image_path, left, top, right, bottom, prob])

            summary_writer.writerow(
                [
                    group_name,
                    subgroup_name,
                    image_path,
                    original_img.width,
                    original_img.height,
                    len(windows),
                    float(prob_map.mean()),
                    float(prob_map.min()),
                    float(prob_map.max()),
                ]
            )
            summary_fp.flush()
            patch_fp.flush()
            total_windows += len(windows)
            logger.info(
                f"[Done] windows={len(windows)} prob_mean={prob_map.mean():.6f} prob_min={prob_map.min():.6f} "
                f"prob_max={prob_map.max():.6f} output_prefix={prefix}"
            )

    logger.info(f"summary_csv={os.path.abspath(summary_csv)}")
    logger.info(f"patch_csv={os.path.abspath(patch_csv)}")
    logger.info(f"images={len(items)} total_windows={total_windows}")
    logger.info("==== Sliding Localization Finished ====")


if __name__ == "__main__":
    main()
