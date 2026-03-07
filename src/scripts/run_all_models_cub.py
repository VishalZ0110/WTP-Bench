#!/usr/bin/env python
"""
Run all model inference on the CUB-200-2011 bird dataset.
Mirrors run_all_models.py but uses CUB data paths and a bird-identification prompt.
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
    bird_species_list = [
    "Black footed Albatross",
    "Laysan Albatross",
    "Sooty Albatross",
    "Groove billed Ani",
    "Crested Auklet",
    "Least Auklet",
    "Parakeet Auklet",
    "Rhinoceros Auklet",
    "Brewer Blackbird",
    "Red winged Blackbird",
    "Rusty Blackbird",
    "Yellow headed Blackbird",
    "Bobolink",
    "Indigo Bunting",
    "Lazuli Bunting",
    "Painted Bunting",
    "Cardinal",
    "Spotted Catbird",
    "Gray Catbird",
    "Yellow breasted Chat",
    "Eastern Towhee",
    "Chuck will Widow",
    "Brandt Cormorant",
    "Red faced Cormorant",
    "Pelagic Cormorant",
    "Bronzed Cowbird",
    "Shiny Cowbird",
    "Brown Creeper",
    "American Crow",
    "Fish Crow",
    "Black billed Cuckoo",
    "Mangrove Cuckoo",
    "Yellow billed Cuckoo",
    "Gray crowned Rosy Finch",
    "Purple Finch",
    "Northern Flicker",
    "Acadian Flycatcher",
    "Great Crested Flycatcher",
    "Least Flycatcher",
    "Olive sided Flycatcher",
    "Scissor tailed Flycatcher",
    "Vermilion Flycatcher",
    "Yellow bellied Flycatcher",
    "Frigatebird",
    "Northern Fulmar",
    "Gadwall",
    "American Goldfinch",
    "European Goldfinch",
    "Boat tailed Grackle",
    "Eared Grebe",
    "Horned Grebe",
    "Pied billed Grebe",
    "Western Grebe",
    "Blue Grosbeak",
    "Evening Grosbeak",
    "Pine Grosbeak",
    "Rose breasted Grosbeak",
    "Pigeon Guillemot",
    "California Gull",
    "Glaucous winged Gull",
    "Heermann Gull",
    "Herring Gull",
    "Ivory Gull",
    "Ring billed Gull",
    "Slaty backed Gull",
    "Western Gull",
    "Anna Hummingbird",
    "Ruby throated Hummingbird",
    "Rufous Hummingbird",
    "Green Violetear",
    "Long tailed Jaeger",
    "Pomarine Jaeger",
    "Blue Jay",
    "Florida Jay",
    "Green Jay",
    "Dark eyed Junco",
    "Tropical Kingbird",
    "Gray Kingbird",
    "Belted Kingfisher",
    "Green Kingfisher",
    "Pied Kingfisher",
    "Ringed Kingfisher",
    "White breasted Kingfisher",
    "Red legged Kittiwake",
    "Horned Lark",
    "Pacific Loon",
    "Mallard",
    "Western Meadowlark",
    "Hooded Merganser",
    "Red breasted Merganser",
    "Mockingbird",
    "Nighthawk",
    "Clark Nutcracker",
    "White breasted Nuthatch",
    "Baltimore Oriole",
    "Hooded Oriole",
    "Orchard Oriole",
    "Scott Oriole",
    "Ovenbird",
    "Brown Pelican",
    "White Pelican",
    "Western Wood Pewee",
    "Sayornis",
    "American Pipit",
    "Whip poor Will",
    "Horned Puffin",
    "Common Raven",
    "White necked Raven",
    "American Redstart",
    "Geococcyx",
    "Loggerhead Shrike",
    "Great Grey Shrike",
    "Baird Sparrow",
    "Black throated Sparrow",
    "Brewer Sparrow",
    "Chipping Sparrow",
    "Clay colored Sparrow",
    "House Sparrow",
    "Field Sparrow",
    "Fox Sparrow",
    "Grasshopper Sparrow",
    "Harris Sparrow",
    "Henslow Sparrow",
    "Le Conte Sparrow",
    "Lincoln Sparrow",
    "Nelson Sharp tailed Sparrow",
    "Savannah Sparrow",
    "Seaside Sparrow",
    "Song Sparrow",
    "Tree Sparrow",
    "Vesper Sparrow",
    "White crowned Sparrow",
    "White throated Sparrow",
    "Cape Glossy Starling",
    "Bank Swallow",
    "Barn Swallow",
    "Cliff Swallow",
    "Tree Swallow",
    "Scarlet Tanager",
    "Summer Tanager",
    "Artic Tern",
    "Black Tern",
    "Caspian Tern",
    "Common Tern",
    "Elegant Tern",
    "Forsters Tern",
    "Least Tern",
    "Green tailed Towhee",
    "Brown Thrasher",
    "Sage Thrasher",
    "Black capped Vireo",
    "Blue headed Vireo",
    "Philadelphia Vireo",
    "Red eyed Vireo",
    "Warbling Vireo",
    "White eyed Vireo",
    "Yellow throated Vireo",
    "Bay breasted Warbler",
    "Black and white Warbler",
    "Black throated Blue Warbler",
    "Blue winged Warbler",
    "Canada Warbler",
    "Cape May Warbler",
    "Cerulean Warbler",
    "Chestnut sided Warbler",
    "Golden winged Warbler",
    "Hooded Warbler",
    "Kentucky Warbler",
    "Magnolia Warbler",
    "Mourning Warbler",
    "Myrtle Warbler",
    "Nashville Warbler",
    "Orange crowned Warbler",
    "Palm Warbler",
    "Pine Warbler",
    "Prairie Warbler",
    "Prothonotary Warbler",
    "Swainson Warbler",
    "Tennessee Warbler",
    "Wilson Warbler",
    "Worm eating Warbler",
    "Yellow Warbler",
    "Northern Waterthrush",
    "Louisiana Waterthrush",
    "Bohemian Waxwing",
    "Cedar Waxwing",
    "American Three toed Woodpecker",
    "Pileated Woodpecker",
    "Red bellied Woodpecker",
    "Red cockaded Woodpecker",
    "Red headed Woodpecker",
    "Downy Woodpecker",
    "Bewick Wren",
    "Cactus Wren",
    "Carolina Wren",
    "House Wren",
    "Marsh Wren",
    "Rock Wren",
    "Winter Wren",
    "Common Yellowthroat"
]
    n_classes = len(bird_species_list)
    classes_str = ", ".join(bird_species_list)
    default_prompt = (
        f"Identify the bird species in this image. Choose from these {n_classes} classes:\n"
        f"{classes_str}\n"
        f"Respond with only the exact class name from the list above."
    )

    prompt = os.environ.get("PROMPT", default_prompt)
    limit = _env_int("LIMIT", 0)
    topk = _env_int("TOPK", 1)
    skip_eval = _env_bool("SKIP_EVAL", False)

    run_phase(
        phase_name="CUB-HQ",
        input_csv="data/cub/metadata.csv",
        image_col="image_path",
        out_dir="predictions/cub/hq",
        models=models,
        prompt=prompt,
        base_dir=base_dir,
        limit=limit,
        topk=topk,
        skip_eval=skip_eval,
    )

    run_phase(
        phase_name="CUB-SILHOUETTE",
        input_csv="data/cub/metadata_with_silhouette.csv",
        image_col="silhouette_path",
        out_dir="predictions/cub/silhouette",
        models=models,
        prompt=prompt,
        base_dir=base_dir,
        limit=limit,
        topk=topk,
        skip_eval=skip_eval,
    )


if __name__ == "__main__":
    main()
