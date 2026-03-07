#!/usr/bin/env python
"""
Preprocess CUB-200-2011 dataset into WTP-Bench format.

1. Copy HQ images from cub/CUB_200_2011/images/ → data/cub/hq/
   and produce data/cub/metadata.csv (image_path, label).
2. Copy segmentation masks from cub/segmentations/ → data/cub/silhouette/
   and produce data/cub/metadata_with_silhouette.csv (image_path, label, silhouette_path).
3. Convert copied silhouettes to black-on-white: threshold non-zero → 255, then invert.
"""
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

from PIL import Image
import numpy as np


SRC_IMAGES_DIR = Path("cub/CUB_200_2011/images")
SRC_SEG_DIR = Path("cub/segmentations")

DST_HQ_DIR = Path("data/cub/hq")
DST_SIL_DIR = Path("data/cub/silhouette")

METADATA_CSV = Path("data/cub/metadata.csv")
METADATA_SIL_CSV = Path("data/cub/metadata_with_silhouette.csv")


def folder_name_to_label(folder_name: str) -> str:
    """'001.Black_footed_Albatross' → 'Black Footed Albatross'"""
    after_dot = folder_name.split(".", 1)[1]
    return " ".join(word.capitalize() for word in after_dot.split("_"))


def collect_folder_images(src_root: Path) -> list[tuple[str, str, Path]]:
    """Return sorted list of (folder_name, filename, full_path) for every image."""
    entries: list[tuple[str, str, Path]] = []
    for folder in sorted(src_root.iterdir()):
        if not folder.is_dir():
            continue
        for img_file in sorted(folder.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                entries.append((folder.name, img_file.name, img_file))
    return entries


def copy_images_and_build_metadata(
    src_root: Path,
    dst_dir: Path,
) -> list[dict[str, str]]:
    """Copy images into a flat destination dir and return metadata rows."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    entries = collect_folder_images(src_root)
    rows: list[dict[str, str]] = []

    for folder_name, filename, src_path in entries:
        label = folder_name_to_label(folder_name)
        dst_path = dst_dir / filename
        shutil.copy2(src_path, dst_path)
        rows.append({
            "image_path": str(dst_path),
            "label": label,
        })

    return rows


def convert_silhouettes(dst_dir: Path) -> None:
    """Threshold non-zero pixels to 255 then invert (black silhouette on white)."""
    for img_path in sorted(dst_dir.iterdir()):
        if not img_path.is_file():
            continue
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        arr[arr != 0] = 255
        arr = 255 - arr
        Image.fromarray(arr).save(img_path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # --- Step 1: HQ images ---
    print(f"Copying HQ images from {SRC_IMAGES_DIR} → {DST_HQ_DIR} ...")
    hq_rows = copy_images_and_build_metadata(SRC_IMAGES_DIR, DST_HQ_DIR)
    write_csv(METADATA_CSV, ["image_path", "label"], hq_rows)
    print(f"  {len(hq_rows)} images copied, metadata written to {METADATA_CSV}")

    # --- Step 2: Segmentation → silhouette images ---
    print(f"Copying segmentations from {SRC_SEG_DIR} → {DST_SIL_DIR} ...")
    sil_rows = copy_images_and_build_metadata(SRC_SEG_DIR, DST_SIL_DIR)
    print(f"  {len(sil_rows)} images copied")

    # Build silhouette metadata by matching HQ ↔ silhouette via filename stem
    hq_by_stem: dict[str, str] = {}
    for row in hq_rows:
        stem = Path(row["image_path"]).stem
        hq_by_stem[stem] = row["image_path"]

    sil_meta_rows: list[dict[str, str]] = []
    for row in sil_rows:
        stem = Path(row["image_path"]).stem
        hq_path = hq_by_stem.get(stem, "")
        sil_meta_rows.append({
            "image_path": hq_path,
            "label": row["label"],
            "silhouette_path": row["image_path"],
        })

    write_csv(METADATA_SIL_CSV, ["image_path", "label", "silhouette_path"], sil_meta_rows)
    print(f"  Silhouette metadata written to {METADATA_SIL_CSV}")

    # --- Step 3: Convert silhouettes to black-on-white ---
    print(f"Converting silhouettes to black-on-white ...")
    convert_silhouettes(DST_SIL_DIR)
    print("  Done.")

    print(f"\nSummary: {len(hq_rows)} HQ images, {len(sil_rows)} silhouettes processed.")


if __name__ == "__main__":
    main()
