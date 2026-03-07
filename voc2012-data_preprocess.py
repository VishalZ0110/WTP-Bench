#!/usr/bin/env python
"""
Preprocess VOC2012 train/val data into WTP-Bench format.

For each image that has a SegmentationClass mask:
  1. Parse the XML annotation and find the object with the largest bounding-box area.
  2. Expand the bounding box by 1.2x (clamped to image boundaries).
  3. Crop the JPEG image to that region → data/voc/hq/
  4. Crop the SegmentationClass mask, convert to silhouette
     (object class → black, everything else → white) → data/voc/silhouette/
  5. Write metadata CSVs.
"""
from __future__ import annotations

import csv
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

VOC_ROOT = Path("VOC2012/VOC2012_train_val")
ANN_DIR = VOC_ROOT / "Annotations"
IMG_DIR = VOC_ROOT / "JPEGImages"
SEG_DIR = VOC_ROOT / "SegmentationClass"

DST_HQ_DIR = Path("data/voc/hq")
DST_SIL_DIR = Path("data/voc/silhouette")
METADATA_CSV = Path("data/voc/metadata.csv")
METADATA_SIL_CSV = Path("data/voc/metadata_with_silhouette.csv")

BBOX_SCALE = 1.2

VOC_CLASS_TO_IDX = {
    "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5,
    "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10,
    "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14, "person": 15,
    "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20,
}


def parse_annotation(xml_path: Path) -> dict | None:
    """Parse a VOC XML annotation and return the object with the largest BB area."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    best_obj = None
    best_area = -1

    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in VOC_CLASS_TO_IDX:
            continue
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        area = (xmax - xmin) * (ymax - ymin)
        if area > best_area:
            best_area = area
            best_obj = {
                "name": name,
                "class_idx": VOC_CLASS_TO_IDX[name],
                "bbox": (xmin, ymin, xmax, ymax),
                "area": area,
            }

    return best_obj


def get_expanded_bbox(bbox: tuple[int, int, int, int], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    cx, cy = xmin + w / 2, ymin + h / 2
    new_w, new_h = w * BBOX_SCALE, h * BBOX_SCALE
    x1 = max(0, int(cx - new_w / 2))
    y1 = max(0, int(cy - new_h / 2))
    x2 = min(img_w, int(cx + new_w / 2))
    y2 = min(img_h, int(cy + new_h / 2))
    return x1, y1, x2, y2


def make_silhouette(seg_arr: np.ndarray, class_idx: int) -> Image.Image:
    """Object class pixels → black (0), everything else → white (255)."""
    sil = np.full(seg_arr.shape, 255, dtype=np.uint8)
    sil[seg_arr == class_idx] = 0
    return Image.fromarray(sil)


def main() -> None:
    seg_stems = {p.stem for p in SEG_DIR.iterdir() if p.suffix == ".png"}
    print(f"Found {len(seg_stems)} segmentation masks")

    DST_HQ_DIR.mkdir(parents=True, exist_ok=True)
    DST_SIL_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    metadata_sil_rows = []
    skipped = 0

    for stem in tqdm(sorted(seg_stems), desc="Processing"):
        xml_path = ANN_DIR / f"{stem}.xml"
        img_path = IMG_DIR / f"{stem}.jpg"
        seg_path = SEG_DIR / f"{stem}.png"

        if not xml_path.exists() or not img_path.exists():
            skipped += 1
            continue

        best_obj = parse_annotation(xml_path)
        if best_obj is None:
            skipped += 1
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        x1, y1, x2, y2 = get_expanded_bbox(best_obj["bbox"], img_w, img_h)

        cropped_img = img.crop((x1, y1, x2, y2))
        hq_path = DST_HQ_DIR / f"{stem}.jpg"
        cropped_img.save(hq_path)

        seg = Image.open(seg_path)
        seg_arr = np.array(seg)
        sil = make_silhouette(seg_arr, best_obj["class_idx"])
        sil_cropped = sil.crop((x1, y1, x2, y2))
        sil_path = DST_SIL_DIR / f"{stem}.png"
        sil_cropped.save(sil_path)

        label = best_obj["name"]
        metadata_rows.append([str(hq_path), label])
        metadata_sil_rows.append([str(hq_path), label, str(sil_path)])

    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(metadata_rows)

    with open(METADATA_SIL_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "silhouette_path"])
        writer.writerows(metadata_sil_rows)

    print(f"\nProcessed {len(metadata_rows)} images (skipped {skipped})")
    print(f"  HQ images:    {DST_HQ_DIR}")
    print(f"  Silhouettes:  {DST_SIL_DIR}")
    print(f"  metadata.csv: {METADATA_CSV}")
    print(f"  metadata_with_silhouette.csv: {METADATA_SIL_CSV}")


if __name__ == "__main__":
    main()
