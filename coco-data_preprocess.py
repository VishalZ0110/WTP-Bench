import os
import csv
import json
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask
from tqdm import tqdm

ANNOTATIONS_PATH = "annotations/instances_val2014.json"
SRC_DIR = "val2014"
HQ_DIR = "data/coco/hq"
SIL_DIR = "data/coco/silhouette"
METADATA_CSV = "data/coco/metadata.csv"
METADATA_SIL_CSV = "data/coco/metadata_with_silhouette.csv"
BBOX_SCALE = 1.2

INCLUSIVE_CLASSES = [
    "person", "bicycle", "airplane", "elephant", "giraffe",
    "umbrella", "scissors", "horse", "fire hydrant",
    "motorcycle", "bird", "bear", "bus", "skateboard", "surfboard",
    "tennis racket", "baseball bat", "banana", "donut", "broccoli",
    "pizza", "chair", "toilet", "potted plant", "kite", "teddy bear", "cow",
    "zebra", "cat", "dog", "sheep",
    "hot dog", "carrot",
    "cup", "bowl", "vase",
    "couch", "bench", "bed",
    "car", "truck", "boat", "train",
    "sandwich", "cake",
]


def get_expanded_bbox(bbox, img_w, img_h, scale=BBOX_SCALE):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    new_w, new_h = w * scale, h * scale
    x1 = max(0, int(cx - new_w / 2))
    y1 = max(0, int(cy - new_h / 2))
    x2 = min(img_w, int(cx + new_w / 2))
    y2 = min(img_h, int(cy + new_h / 2))
    return x1, y1, x2, y2


def build_image_id_to_annotation(instances):
    """For each image_id, keep the annotation with the largest area."""
    image_id_to_ann = {}
    for ann in instances["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_id_to_ann or ann["area"] > image_id_to_ann[img_id]["area"]:
            image_id_to_ann[img_id] = ann

    image_id_to_info = {img["id"]: img for img in instances["images"]}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in instances["categories"]}

    for img_id, ann in image_id_to_ann.items():
        ann["file_name"] = image_id_to_info[img_id]["file_name"]
        ann["category"] = cat_id_to_name[ann["category_id"]]

    # Filter to inclusive classes only
    image_id_to_ann = {
        k: v for k, v in image_id_to_ann.items() if v["category"] in INCLUSIVE_CLASSES
    }
    return image_id_to_ann


def make_silhouette(ann, img_w, img_h):
    """Object = black (0), background = white (255)."""
    seg_data = ann["segmentation"]
    if isinstance(seg_data, dict):
        rle = coco_mask.frPyObjects(seg_data, img_h, img_w)
        bin_mask = coco_mask.decode(rle)
        return Image.fromarray(np.where(bin_mask, 0, 255).astype(np.uint8))

    mask = Image.new("L", (img_w, img_h), 255)
    draw = ImageDraw.Draw(mask)
    for seg in seg_data:
        poly = [(float(seg[i]), float(seg[i + 1])) for i in range(0, len(seg), 2)]
        if len(poly) < 3:
            continue
        draw.polygon(poly, fill=0)
    return mask


def main():
    print(f"Loading annotations from {ANNOTATIONS_PATH} ...")
    with open(ANNOTATIONS_PATH) as f:
        instances = json.load(f)

    image_id_to_ann = build_image_id_to_annotation(instances)
    print(f"Annotations after filtering: {len(image_id_to_ann)}")

    os.makedirs(HQ_DIR, exist_ok=True)
    os.makedirs(SIL_DIR, exist_ok=True)

    metadata_rows = []
    metadata_sil_rows = []

    for img_id, ann in tqdm(image_id_to_ann.items(), desc="Processing"):
        file_name = ann["file_name"]
        src_path = os.path.join(SRC_DIR, file_name)
        if not os.path.exists(src_path):
            continue

        img = Image.open(src_path).convert("RGB")
        img_w, img_h = img.size
        label = ann["category"]
        x1, y1, x2, y2 = get_expanded_bbox(ann["bbox"], img_w, img_h)

        stem = os.path.splitext(file_name)[0]

        cropped = img.crop((x1, y1, x2, y2))
        hq_path = os.path.join(HQ_DIR, f"{stem}.jpg")
        cropped.save(hq_path)

        sil = make_silhouette(ann, img_w, img_h)
        sil_cropped = sil.crop((x1, y1, x2, y2))
        sil_path = os.path.join(SIL_DIR, f"{stem}.png")
        sil_cropped.save(sil_path)

        metadata_rows.append([hq_path, label])
        metadata_sil_rows.append([hq_path, label, sil_path])

    with open(METADATA_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(metadata_rows)

    with open(METADATA_SIL_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "silhouette_path"])
        writer.writerows(metadata_sil_rows)

    print(f"Processed {len(metadata_rows)} images")
    print(f"  metadata.csv: {METADATA_CSV}")
    print(f"  metadata_with_silhouette.csv: {METADATA_SIL_CSV}")


if __name__ == "__main__":
    main()
