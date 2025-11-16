#!/usr/bin/env python3
"""Utility to run APGCC inference on an arbitrary folder of images."""
import argparse
import copy
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image

from config import cfg as base_cfg, merge_from_file
from datasets import build_dataset
from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run APGCC inference on a folder of images.")
    parser.add_argument(
        "--images_dir",
        required=True,
        type=Path,
        help="Directory that contains the images you want to count.",
    )
    parser.add_argument(
        "--checkpoint",
        default=Path("./apgcc/output/SHHA_best.pth"),
        type=Path,
        help="Path to the pretrained APGCC checkpoint (state dict).",
    )
    parser.add_argument(
        "--config",
        default=Path("./apgcc/configs/SHHA_test.yml"),
        type=Path,
        help="Config file that defines the SHHA inference pipeline.",
    )
    parser.add_argument(
        "--working_dir",
        default=Path("./apgcc/custom_data"),
        type=Path,
        help="Directory where the synthetic SHHA-style dataset will be created.",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("./apgcc/output"),
        type=Path,
        help="Directory where inference logs/results will be written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold used to select predicted crowd points.",
    )
    parser.add_argument(
        "--min_random_points",
        type=int,
        default=0,
        help="Minimum number of synthetic ground-truth points per image.",
    )
    parser.add_argument(
        "--max_random_points",
        type=int,
        default=10,
        help="Maximum number of synthetic ground-truth points per image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1229,
        help="Random seed used to place synthetic points.",
    )
    parser.add_argument(
        "--results_json",
        type=Path,
        default=None,
        help="Optional path to save the per-image predictions as JSON.",
    )
    return parser.parse_args()


def collect_image_paths(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
    if not images:
        raise FileNotFoundError(f"No images with supported extensions found in {images_dir}.")
    return sorted(images)


def write_list_file(entries: Sequence[Tuple[str, str]], list_path: Path) -> None:
    with open(list_path, "w", encoding="utf-8") as f:
        for rel_img, rel_label in entries:
            f.write(f"{rel_img} {rel_label}\n")


def prepare_shha_style_dataset(
    images_dir: Path,
    working_dir: Path,
    min_random_points: int,
    max_random_points: int,
    seed: int,
) -> Tuple[Path, List[Tuple[str, str]]]:
    if max_random_points < min_random_points:
        raise ValueError("max_random_points must be >= min_random_points")

    rng = random.Random(seed)
    dataset_root = working_dir.expanduser().resolve()
    image_store = dataset_root / "images"
    label_store = dataset_root / "labels"
    image_store.mkdir(parents=True, exist_ok=True)
    label_store.mkdir(parents=True, exist_ok=True)

    entries: List[Tuple[str, str]] = []
    for image_path in collect_image_paths(images_dir):
        dest_image = image_store / image_path.name
        if not dest_image.exists():
            try:
                dest_image.symlink_to(image_path.resolve())
            except OSError:
                shutil.copy2(image_path, dest_image)

        with Image.open(image_path) as img:
            width, height = img.size
        num_points = rng.randint(min_random_points, max_random_points)
        label_rel = f"labels/{image_path.stem}.txt"
        label_path = dataset_root / label_rel
        with open(label_path, "w", encoding="utf-8") as label_file:
            for _ in range(num_points):
                x = rng.uniform(0, max(width - 1, 1))
                y = rng.uniform(0, max(height - 1, 1))
                label_file.write(f"{x:.2f} {y:.2f}\n")

        entries.append((f"images/{image_path.name}", label_rel))

    write_list_file(entries, dataset_root / "train.list")
    write_list_file(entries, dataset_root / "test.list")
    return dataset_root, entries


def build_inference_cfg(
    config_path: Path,
    dataset_root: Path,
    output_dir: Path,
    threshold: float,
    weight_path: Path,
):
    cfg = copy.deepcopy(base_cfg)
    cfg = merge_from_file(cfg, str(config_path))
    cfg.DATASETS.DATASET = "SHHA_CUSTOM"
    cfg.DATASETS.DATA_ROOT = str(dataset_root)
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.TEST.THRESHOLD = threshold
    cfg.TEST.WEIGHT = str(weight_path)
    cfg.SOLVER.BATCH_SIZE = 1
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.VIS = False
    return cfg


def load_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg=cfg, training=False)
    model.to(device)

    checkpoint = torch.load(cfg.TEST.WEIGHT, map_location="cpu")
    model_dict = model.state_dict()
    compatible_state = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(compatible_state)
    model.load_state_dict(model_dict)
    return model, device


def predict_counts(model, device, data_loader, threshold: float):
    results = []
    model.eval()
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            outputs = model(samples)
            logits = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            keep = logits > threshold
            points = outputs['pred_points'][0][keep]
            count = int(keep.sum().item())
            image_name = targets[0]['name']
            results.append({
                "image": image_name,
                "predicted_count": count,
                "points": points.detach().cpu().tolist(),
            })
    return results


def main():
    args = parse_args()
    images_dir = args.images_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    dataset_root, entries = prepare_shha_style_dataset(
        images_dir,
        args.working_dir,
        args.min_random_points,
        args.max_random_points,
        args.seed,
    )
    print(f"Prepared synthetic SHHA dataset with {len(entries)} images at {dataset_root}.")

    cfg = build_inference_cfg(
        args.config,
        dataset_root,
        args.output_dir,
        args.threshold,
        checkpoint,
    )
    _, val_loader = build_dataset(cfg=cfg)
    model, device = load_model(cfg)
    results = predict_counts(model, device, val_loader, cfg.TEST.THRESHOLD)

    print("\nPredicted crowd counts:")
    for res in results:
        print(f"  {res['image']}: {res['predicted_count']}")
    total = sum(r['predicted_count'] for r in results)
    print(f"\nTotal predicted people across folder: {total}")

    if args.results_json is not None:
        args.results_json = args.results_json.expanduser().resolve()
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved predictions to {args.results_json}")


if __name__ == "__main__":
    main()
