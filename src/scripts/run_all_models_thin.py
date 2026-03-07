#!/usr/bin/env python
"""
Run all model inference on the ThinObject5K dataset.
Mirrors run_all_models_dis5k.py but uses ThinObject5K data paths and a 193-class prompt.
Classes filtered to only those with visually distinctive silhouettes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

DEFAULT_BACKEND = "hf"
REQUIRED_KEYS_BY_BACKEND = {
    "hf": ["model_id", "dtype", "device", "max_new_tokens"],
    "openrouter": ["model_id", "max_new_tokens"],
    "anthropic": ["model_id", "max_new_tokens"],
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_model_configs(cfg_dir: Path) -> list[dict[str, object]]:
    if not cfg_dir.exists():
        raise SystemExit(f"Config dir not found: {cfg_dir}")
    configs = sorted(cfg_dir.rglob("*.json"))
    if not configs:
        raise SystemExit(f"No model config JSON files found in {cfg_dir}")

    models: list[dict[str, object]] = []
    for path in configs:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("enabled", True) is False:
            continue
        backend = str(cfg.get("backend", DEFAULT_BACKEND)).lower()
        required = REQUIRED_KEYS_BY_BACKEND.get(backend, REQUIRED_KEYS_BY_BACKEND[DEFAULT_BACKEND])
        missing = [k for k in required if k not in cfg]
        if missing:
            raise SystemExit(f"Missing keys in {path}: {', '.join(missing)}")
        cfg["backend"] = backend
        cfg["_filename"] = path.stem
        cfg["_tag"] = path.parent.name if path.parent != cfg_dir else ""
        models.append(cfg)

    if not models:
        raise SystemExit(f"No enabled model configs found in {cfg_dir}")
    return models


def safe_name(model_id: str) -> str:
    return model_id.lower().replace("/", "_")


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def run_phase(
    *,
    phase_name: str,
    input_csv: str,
    image_col: str,
    out_dir: str,
    models: list[dict[str, object]],
    prompt: str,
    base_dir: str,
    limit: int,
    topk: int,
    skip_eval: bool,
) -> None:
    failed = []
    for cfg in tqdm(models, desc=f"[{phase_name}] Models", unit="model"):
        model_id = str(cfg["model_id"])
        backend = str(cfg.get("backend", DEFAULT_BACKEND)).lower()
        tag = cfg.get("_tag", "")
        model_out_dir = Path(out_dir) / tag if tag else Path(out_dir)
        model_out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = str(model_out_dir / f"predictions_{safe_name(model_id)}.csv")

        if Path(output_csv).exists():
            tqdm.write(f"[{phase_name}] Skipping {model_id} (predictions already exist)")
            continue

        tqdm.write(f"[{phase_name}] Running inference on {model_id}...")
        try:
            if backend == "openrouter":
                cmd = [
                    sys.executable,
                    "src/scripts/vlm_infer_openrouter.py",
                    "--input-csv",
                    input_csv,
                    "--output-csv",
                    output_csv,
                    "--prompt",
                    prompt,
                    "--model-id",
                    model_id,
                    "--base-dir",
                    base_dir,
                    "--image-col",
                    image_col,
                    "--max-new-tokens",
                    str(cfg["max_new_tokens"]),
                    "--limit",
                    str(limit),
                ]
                if "temperature" in cfg:
                    cmd.extend(["--temperature", str(cfg["temperature"])])
                if "api_base" in cfg:
                    cmd.extend(["--api-base", str(cfg["api_base"])])
                if "api_key_env" in cfg:
                    cmd.extend(["--api-key-env", str(cfg["api_key_env"])])
                if "batch_size" in cfg:
                    cmd.extend(["--batch-size", str(cfg["batch_size"])])
                if "referer" in cfg:
                    cmd.extend(["--referer", str(cfg["referer"])])
                if "title" in cfg:
                    cmd.extend(["--title", str(cfg["title"])])
                if "reasoning_effort" in cfg:
                    cmd.extend(["--reasoning-effort", str(cfg["reasoning_effort"])])
                run_cmd(cmd)
            elif backend == "anthropic":
                cmd = [
                    sys.executable,
                    "src/scripts/vlm_infer_anthropic.py",
                    "--input-csv",
                    input_csv,
                    "--output-csv",
                    output_csv,
                    "--prompt",
                    prompt,
                    "--model-id",
                    model_id,
                    "--base-dir",
                    base_dir,
                    "--max-new-tokens",
                    str(cfg["max_new_tokens"]),
                ]
                if "temperature" in cfg:
                    cmd.extend(["--temperature", str(cfg["temperature"])])
                run_cmd(cmd)
            elif backend == "hf":
                run_cmd(
                    [
                        sys.executable,
                        "src/scripts/vlm_infer_template.py",
                        "--input-csv",
                        input_csv,
                        "--output-csv",
                        output_csv,
                        "--prompt",
                        prompt,
                        "--model-id",
                        model_id,
                        "--base-dir",
                        base_dir,
                        "--image-col",
                        image_col,
                        "--device",
                        str(cfg["device"]),
                        "--dtype",
                        str(cfg["dtype"]),
                        "--max-new-tokens",
                        str(cfg["max_new_tokens"]),
                        "--limit",
                        str(limit),
                        "--batch-size",
                        str(cfg.get("batch_size", 64)),
                    ]
                )
            else:
                tqdm.write(f"[{phase_name}] WARNING: Unknown backend '{backend}' for {model_id}. Skipping...")
                continue
        except subprocess.CalledProcessError as e:
            tqdm.write(f"[{phase_name}] FAILED: {model_id} (exit code {e.returncode}), skipping...")
            failed.append(model_id)
            continue

        if not skip_eval:
            run_cmd(
                [
                    sys.executable,
                    "src/scripts/eval_predictions.py",
                    "--pred-csv",
                    output_csv,
                    "--topk",
                    str(topk),
                ]
            )

    if failed:
        tqdm.write(f"\n[{phase_name}] {len(failed)} model(s) failed: {', '.join(failed)}")


def main() -> None:
    model_config_dir = Path(os.environ.get("MODEL_CONFIG_DIR", "configs"))
    models = load_model_configs(model_config_dir)

    models_filter = os.environ.get("MODELS")
    if models_filter:
        needles = [m.strip().lower() for m in models_filter.split(",") if m.strip()]
        include = [n for n in needles if not n.startswith("!")]
        exclude = [n[1:] for n in needles if n.startswith("!")]

        def _matches(cfg, terms):
            mid = str(cfg["model_id"]).lower()
            fname = str(cfg.get("_filename", "")).lower()
            tag = str(cfg.get("_tag", "")).lower()
            return any(t in mid or t in fname or t == tag for t in terms)

        if include:
            models = [cfg for cfg in models if _matches(cfg, include)]
        if exclude:
            models = [cfg for cfg in models if not _matches(cfg, exclude)]
        if not models:
            raise SystemExit(f"No models matched filter: {models_filter}")
        print(f"Filtered to {len(models)} model(s): {', '.join(str(m['model_id']) for m in models)}")

    base_dir = os.environ.get("BASE_DIR", ".")
    object_class_list = [
        "Anaconda",
        "Ant",
        "Apple",
        "Armchair",
        "Arrow Bow",
        "Ax",
        "Backpack",
        "Barbell",
        "Bassinet",
        "Bathtub",
        "Bed",
        "Bee",
        "Bicycle",
        "Birds",
        "Boat",
        "Boots",
        "Bridge",
        "Broom",
        "Bucket",
        "Bulldozer",
        "Butterfly",
        "Cactus",
        "Cage",
        "Camel",
        "Cannon",
        "Canoe",
        "Carriage",
        "Cat",
        "Chain Saw",
        "Chair",
        "Chicken",
        "Cobra",
        "Cock",
        "Comb",
        "Cooking Pan",
        "Corkscrew",
        "Cow",
        "Crab",
        "Crane",
        "Crocodile",
        "Crow",
        "Crown",
        "Crutch",
        "Darts",
        "Deer",
        "Dog",
        "Donkey",
        "Dragonfly",
        "Dresser",
        "Drone",
        "Drum",
        "Eagle",
        "Eiffel Tower",
        "Elephants",
        "Excavator",
        "Fan",
        "Fir Tree",
        "Fire Truck",
        "Fish",
        "Flags",
        "Flamingo",
        "Flippers",
        "Football Goal",
        "Fork",
        "Frog",
        "Frying Pan",
        "Gas Mask",
        "Giraffe",
        "Glider",
        "Goat",
        "Golden Cup",
        "Grasshopper",
        "Grenade Launcher",
        "Grill",
        "Gun",
        "Hairbrush",
        "Hammer",
        "Hammock",
        "Hand Saw",
        "Handcuffs",
        "Harp",
        "Headphones",
        "Hedgehog",
        "Helicopter",
        "Horseshoe",
        "Ice Axe",
        "Iron",
        "Jail",
        "Jeep",
        "Kangaroo",
        "Katana",
        "Kayak",
        "Kettle",
        "Key",
        "Kick Scooter",
        "Ladder",
        "Ladybug",
        "Lemur",
        "Lifebuoy",
        "Lizard",
        "Lobster",
        "Louboutin",
        "Mantis",
        "Megaphone",
        "Microphone",
        "Moose",
        "Mop",
        "Mortar",
        "Mosquito",
        "Motorcycle",
        "Octopus",
        "Ostrich",
        "Paddle",
        "Padlock",
        "Palm Tree",
        "Parachute",
        "Pear",
        "Piano",
        "Pineapple",
        "Plier",
        "Plunger",
        "Pram",
        "Punching Bag",
        "Radio",
        "Rat Mouse",
        "Raven",
        "Roller Skates",
        "Rose",
        "Samovar",
        "Sandals",
        "Scales",
        "Scissors",
        "Scooter",
        "Scorpion",
        "Screwdriver",
        "Seahorse",
        "Sewing Machine",
        "Ship",
        "Shopping Cart",
        "Shovel",
        "Shrimps",
        "Skeleton",
        "Sled",
        "Snails",
        "Snake",
        "Sofa",
        "Spear",
        "Spider",
        "Spoon",
        "Squid",
        "Starfish",
        "Statue Of Liberty",
        "Stethoscope",
        "Stork",
        "Street Light",
        "Submarine",
        "Suitcase",
        "Sunflower",
        "Sunglasses",
        "Swan",
        "Sword",
        "Syringe",
        "Table",
        "Tank",
        "Tap",
        "Telephone Booth",
        "Telescope",
        "Tent",
        "Tie",
        "Time Bomb",
        "Titanic",
        "Toothbrush",
        "Tractor",
        "Tram",
        "Trampoline",
        "Treadmill",
        "Tripod",
        "Trolleybus",
        "Trombone",
        "Trumpet Saxophone",
        "Tulip",
        "Turkey",
        "Turtle",
        "Typewriter",
        "Umbrella",
        "Vacuum Cleaner",
        "Vase",
        "Video Camera",
        "Violin",
        "Wedding Cake",
        "Wheelchair",
        "Wineglass",
        "Wrench",
    ]
    n_classes = len(object_class_list)
    classes_str = ", ".join(object_class_list)
    default_prompt = (
        f"Identify the main object in this image. Choose from these {n_classes} classes:\n"
        f"{classes_str}\n"
        f"Respond with only the exact class name from the list above."
    )

    prompt = os.environ.get("PROMPT", default_prompt)
    limit = _env_int("LIMIT", 0)
    topk = _env_int("TOPK", 1)
    skip_eval = _env_bool("SKIP_EVAL", False)

    run_phase(
        phase_name="THIN-HQ",
        input_csv="data/thin/metadata.csv",
        image_col="image_path",
        out_dir="predictions/thin/hq",
        models=models,
        prompt=prompt,
        base_dir=base_dir,
        limit=limit,
        topk=topk,
        skip_eval=skip_eval,
    )

    run_phase(
        phase_name="THIN-SILHOUETTE",
        input_csv="data/thin/metadata_with_silhouette.csv",
        image_col="silhouette_path",
        out_dir="predictions/thin/silhouette",
        models=models,
        prompt=prompt,
        base_dir=base_dir,
        limit=limit,
        topk=topk,
        skip_eval=skip_eval,
    )


if __name__ == "__main__":
    main()
