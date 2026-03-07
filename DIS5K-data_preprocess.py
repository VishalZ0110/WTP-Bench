#!/usr/bin/env python
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

SRC_ROOT = Path("DIS5K")
TEST_SPLITS = [f"DIS-TE{i}" for i in range(1, 5)]

DST_HQ_DIR = Path("data/DIS5K/hq")
DST_SIL_DIR = Path("data/DIS5K/silhouette")
METADATA_CSV = Path("data/DIS5K/metadata.csv")
METADATA_SIL_CSV = Path("data/DIS5K/metadata_with_silhouette.csv")

LONG_EDGE = 200


def resize_long_edge(img: Image.Image, long_edge: int = LONG_EDGE) -> Image.Image:
    w, h = img.size
    scale = long_edge / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.ANTIALIAS
    return img.resize((new_w, new_h), resample)


def parse_class_name(filename: str) -> str:
    """Extract class_name from '<j>#<group>#<k>#<class>#<id>.ext'."""
    stem = Path(filename).stem
    parts = stem.split("#")
    return parts[3]


def to_black_on_white(img: Image.Image) -> Image.Image:
    """Convert a segmentation mask to black-on-white silhouette."""
    arr = np.array(img.convert("L"), dtype=np.uint8)
    arr[arr != 0] = 255
    arr = 255 - arr
    return Image.fromarray(arr)


def collect_files(split_dirs: list[str], subfolder: str) -> list[Path]:
    """Gather all files from the given subfolder across all splits, sorted."""
    files: list[Path] = []
    for split in split_dirs:
        folder = SRC_ROOT / split / subfolder
        if not folder.exists():
            print(f"  Warning: {folder} not found, skipping.")
            continue
        files.extend(sorted(f for f in folder.iterdir() if f.is_file()))
    return sorted(files, key=lambda p: p.name)


def main() -> None:
    os.makedirs(DST_HQ_DIR, exist_ok=True)
    os.makedirs(DST_SIL_DIR, exist_ok=True)

    im_files = collect_files(TEST_SPLITS, "im")
    gt_files = collect_files(TEST_SPLITS, "gt")
    print(f"Found {len(im_files)} images and {len(gt_files)} GT masks.")

    gt_by_stem: dict[str, Path] = {f.stem: f for f in gt_files}

    metadata_rows: list[list[str]] = []
    metadata_sil_rows: list[list[str]] = []

    for src_path in tqdm(im_files, desc="Processing"):
        filename = src_path.name
        label = parse_class_name(filename)
        stem = src_path.stem

        img = Image.open(src_path).convert("RGB")
        img_resized = resize_long_edge(img)
        hq_path = DST_HQ_DIR / filename
        img_resized.save(hq_path)

        metadata_rows.append([str(hq_path), label])

        gt_src = gt_by_stem.get(stem)
        if gt_src is not None:
            gt_img = Image.open(gt_src).convert("L")
            gt_resized = resize_long_edge(gt_img)
            sil = to_black_on_white(gt_resized)
            sil_filename = stem + ".png"
            sil_path = DST_SIL_DIR / sil_filename
            sil.save(sil_path)

            metadata_sil_rows.append([str(hq_path), label, str(sil_path)])

    with open(METADATA_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(metadata_rows)

    with open(METADATA_SIL_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "silhouette_path"])
        writer.writerows(metadata_sil_rows)

    print(f"\nDone — {len(metadata_rows)} HQ images, {len(metadata_sil_rows)} silhouettes.")
    print(f"  {METADATA_CSV}")
    print(f"  {METADATA_SIL_CSV}")


if __name__ == "__main__":
    main()
