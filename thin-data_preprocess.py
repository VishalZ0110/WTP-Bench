#!/usr/bin/env python
"""
Preprocess ThinObject5K train split into WTP-Bench format.

For each image listed in ThinObject5K/list/train.txt:
  1. Copy the RGB image → data/thin/hq/
  2. Convert the grayscale mask to a silhouette
     (non-zero → black, zero → white) → data/thin/silhouette/
  3. Write metadata CSVs.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

THIN_ROOT = Path("ThinObject5K")
LIST_FILE = THIN_ROOT / "list" / "train.txt"
IMG_DIR = THIN_ROOT / "images"
MASK_DIR = THIN_ROOT / "masks"

DST_HQ_DIR = Path("data/thin/hq")
DST_SIL_DIR = Path("data/thin/silhouette")
METADATA_CSV = Path("data/thin/metadata.csv")
METADATA_SIL_CSV = Path("data/thin/metadata_with_silhouette.csv")

_CLASS_RE = re.compile(r"^(.+?)_PNG")

KEEP_CLASSES = {
    "Anaconda", "Ant", "Apple", "Armchair", "Arrow Bow", "Ax", "Backpack",
    "Barbell", "Bassinet", "Bathtub", "Bed", "Bee", "Bicycle", "Birds",
    "Boat", "Boots", "Bridge", "Broom", "Bucket", "Bulldozer", "Butterfly",
    "Cactus", "Cage", "Camel", "Cannon", "Canoe", "Carriage", "Cat",
    "Chain Saw", "Chair", "Chicken", "Cobra", "Cock", "Comb", "Cooking Pan",
    "Corkscrew", "Cow", "Crab", "Crane", "Crocodile", "Crow", "Crown",
    "Crutch", "Darts", "Deer", "Dog", "Donkey", "Dragonfly", "Dresser",
    "Drone", "Drum", "Eagle", "Eiffel Tower", "Elephants", "Excavator",
    "Fan", "Fir Tree", "Fire Truck", "Fish", "Flags", "Flamingo", "Flippers",
    "Football Goal", "Fork", "Frog", "Frying Pan", "Gas Mask", "Giraffe",
    "Glider", "Goat", "Golden Cup", "Grasshopper", "Grenade Launcher",
    "Grill", "Gun", "Hairbrush", "Hammer", "Hammock", "Hand Saw", "Handcuffs",
    "Harp", "Headphones", "Hedgehog", "Helicopter", "Horseshoe", "Ice Axe",
    "Iron", "Jail", "Jeep", "Kangaroo", "Katana", "Kayak", "Kettle", "Key",
    "Kick Scooter", "Ladder", "Ladybug", "Lemur", "Lifebuoy", "Lizard",
    "Lobster", "Louboutin", "Mantis", "Megaphone", "Microphone", "Moose",
    "Mop", "Mortar", "Mosquito", "Motorcycle", "Octopus", "Ostrich", "Paddle",
    "Padlock", "Palm Tree", "Parachute", "Pear", "Piano", "Pineapple",
    "Plier", "Plunger", "Pram", "Punching Bag", "Radio", "Rat Mouse",
    "Raven", "Roller Skates", "Rose", "Samovar", "Sandals", "Scales",
    "Scissors", "Scooter", "Scorpion", "Screwdriver", "Seahorse",
    "Sewing Machine", "Ship", "Shopping Cart", "Shovel", "Shrimps",
    "Skeleton", "Sled", "Snails", "Snake", "Sofa", "Spear", "Spider",
    "Spoon", "Squid", "Starfish", "Statue Of Liberty", "Stethoscope",
    "Stork", "Street Light", "Submarine", "Suitcase", "Sunflower",
    "Sunglasses", "Swan", "Sword", "Syringe", "Table", "Tank", "Tap",
    "Telephone Booth", "Telescope", "Tent", "Tie", "Time Bomb", "Titanic",
    "Toothbrush", "Tractor", "Tram", "Trampoline", "Treadmill", "Tripod",
    "Trolleybus", "Trombone", "Trumpet Saxophone", "Tulip", "Turkey",
    "Turtle", "Typewriter", "Umbrella", "Vacuum Cleaner", "Vase",
    "Video Camera", "Violin", "Wedding Cake", "Wheelchair", "Wineglass",
    "Wrench",
}


def extract_label(filename: str) -> str:
    """'air_pump_PNG1.png' → 'Air Pump'"""
    m = _CLASS_RE.match(filename)
    raw = m.group(1) if m else filename.rsplit(".", 1)[0]
    return " ".join(w.capitalize() for w in raw.split("_"))


def make_silhouette(mask_path: Path) -> Image.Image:
    """Non-zero → black (0), zero → white (255)."""
    arr = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    sil = np.where(arr > 0, 0, 255).astype(np.uint8)
    return Image.fromarray(sil)


def main() -> None:
    with open(LIST_FILE) as f:
        stems = [line.strip().rsplit(".", 1)[0] for line in f if line.strip()]
    print(f"Found {len(stems)} entries in {LIST_FILE}")

    DST_HQ_DIR.mkdir(parents=True, exist_ok=True)
    DST_SIL_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[list[str]] = []
    metadata_sil_rows: list[list[str]] = []
    skipped = 0

    for stem in tqdm(stems, desc="Processing"):
        img_path = IMG_DIR / f"{stem}.jpg"
        mask_path = MASK_DIR / f"{stem}.png"

        if not img_path.exists() or not mask_path.exists():
            skipped += 1
            continue

        label = extract_label(stem)

        if label not in KEEP_CLASSES:
            skipped += 1
            continue

        img = Image.open(img_path).convert("RGB")
        hq_path = DST_HQ_DIR / f"{stem}.jpg"
        img.save(hq_path)

        sil = make_silhouette(mask_path)
        sil_path = DST_SIL_DIR / f"{stem}.png"
        sil.save(sil_path)

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
