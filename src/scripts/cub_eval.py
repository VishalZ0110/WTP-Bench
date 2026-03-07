#!/usr/bin/env python3
"""CUB-200-2011 Evaluation Script — ImageNet-style batch evaluation with family-level metrics.

200 bird species grouped into families by the last word of the species name.
  e.g.  "Black Footed Albatross"  ->  family "Albatross"
        "Laysan Albatross"        ->  family "Albatross"

Scoring tiers
  Exact species  – normalised prediction matches normalised label (with fuzzy tolerance)
  Family match   – prediction names a different species from the same family
  Failed         – model refused / produced garbage
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ============================================================================
# Family Taxonomy (hardcoded)
# ============================================================================

SPECIES_TO_FAMILY: dict[str, str] = {
    "Acadian Flycatcher": "Flycatcher",
    "American Crow": "Crow",
    "American Goldfinch": "Goldfinch",
    "American Pipit": "Pipit",
    "American Redstart": "Redstart",
    "American Three Toed Woodpecker": "Woodpecker",
    "Anna Hummingbird": "Hummingbird",
    "Artic Tern": "Tern",
    "Baird Sparrow": "Sparrow",
    "Baltimore Oriole": "Oriole",
    "Bank Swallow": "Swallow",
    "Barn Swallow": "Swallow",
    "Bay Breasted Warbler": "Warbler",
    "Belted Kingfisher": "Kingfisher",
    "Bewick Wren": "Wren",
    "Black And White Warbler": "Warbler",
    "Black Billed Cuckoo": "Cuckoo",
    "Black Capped Vireo": "Vireo",
    "Black Footed Albatross": "Albatross",
    "Black Tern": "Tern",
    "Black Throated Blue Warbler": "Warbler",
    "Black Throated Sparrow": "Sparrow",
    "Blue Grosbeak": "Grosbeak",
    "Blue Headed Vireo": "Vireo",
    "Blue Jay": "Jay",
    "Blue Winged Warbler": "Warbler",
    "Boat Tailed Grackle": "Grackle",
    "Bobolink": "Bobolink",
    "Bohemian Waxwing": "Waxwing",
    "Brandt Cormorant": "Cormorant",
    "Brewer Blackbird": "Blackbird",
    "Brewer Sparrow": "Sparrow",
    "Bronzed Cowbird": "Cowbird",
    "Brown Creeper": "Creeper",
    "Brown Pelican": "Pelican",
    "Brown Thrasher": "Thrasher",
    "Cactus Wren": "Wren",
    "California Gull": "Gull",
    "Canada Warbler": "Warbler",
    "Cape Glossy Starling": "Starling",
    "Cape May Warbler": "Warbler",
    "Cardinal": "Cardinal",
    "Carolina Wren": "Wren",
    "Caspian Tern": "Tern",
    "Cedar Waxwing": "Waxwing",
    "Cerulean Warbler": "Warbler",
    "Chestnut Sided Warbler": "Warbler",
    "Chipping Sparrow": "Sparrow",
    "Chuck Will Widow": "Widow",
    "Clark Nutcracker": "Nutcracker",
    "Clay Colored Sparrow": "Sparrow",
    "Cliff Swallow": "Swallow",
    "Common Raven": "Raven",
    "Common Tern": "Tern",
    "Common Yellowthroat": "Yellowthroat",
    "Crested Auklet": "Auklet",
    "Dark Eyed Junco": "Junco",
    "Downy Woodpecker": "Woodpecker",
    "Eared Grebe": "Grebe",
    "Eastern Towhee": "Towhee",
    "Elegant Tern": "Tern",
    "European Goldfinch": "Goldfinch",
    "Evening Grosbeak": "Grosbeak",
    "Field Sparrow": "Sparrow",
    "Fish Crow": "Crow",
    "Florida Jay": "Jay",
    "Forsters Tern": "Tern",
    "Fox Sparrow": "Sparrow",
    "Frigatebird": "Frigatebird",
    "Gadwall": "Gadwall",
    "Geococcyx": "Geococcyx",
    "Glaucous Winged Gull": "Gull",
    "Golden Winged Warbler": "Warbler",
    "Grasshopper Sparrow": "Sparrow",
    "Gray Catbird": "Catbird",
    "Gray Crowned Rosy Finch": "Finch",
    "Gray Kingbird": "Kingbird",
    "Great Crested Flycatcher": "Flycatcher",
    "Great Grey Shrike": "Shrike",
    "Green Jay": "Jay",
    "Green Kingfisher": "Kingfisher",
    "Green Tailed Towhee": "Towhee",
    "Green Violetear": "Violetear",
    "Groove Billed Ani": "Ani",
    "Harris Sparrow": "Sparrow",
    "Heermann Gull": "Gull",
    "Henslow Sparrow": "Sparrow",
    "Herring Gull": "Gull",
    "Hooded Merganser": "Merganser",
    "Hooded Oriole": "Oriole",
    "Hooded Warbler": "Warbler",
    "Horned Grebe": "Grebe",
    "Horned Lark": "Lark",
    "Horned Puffin": "Puffin",
    "House Sparrow": "Sparrow",
    "House Wren": "Wren",
    "Indigo Bunting": "Bunting",
    "Ivory Gull": "Gull",
    "Kentucky Warbler": "Warbler",
    "Laysan Albatross": "Albatross",
    "Lazuli Bunting": "Bunting",
    "Le Conte Sparrow": "Sparrow",
    "Least Auklet": "Auklet",
    "Least Flycatcher": "Flycatcher",
    "Least Tern": "Tern",
    "Lincoln Sparrow": "Sparrow",
    "Loggerhead Shrike": "Shrike",
    "Long Tailed Jaeger": "Jaeger",
    "Louisiana Waterthrush": "Waterthrush",
    "Magnolia Warbler": "Warbler",
    "Mallard": "Mallard",
    "Mangrove Cuckoo": "Cuckoo",
    "Marsh Wren": "Wren",
    "Mockingbird": "Mockingbird",
    "Mourning Warbler": "Warbler",
    "Myrtle Warbler": "Warbler",
    "Nashville Warbler": "Warbler",
    "Nelson Sharp Tailed Sparrow": "Sparrow",
    "Nighthawk": "Nighthawk",
    "Northern Flicker": "Flicker",
    "Northern Fulmar": "Fulmar",
    "Northern Waterthrush": "Waterthrush",
    "Olive Sided Flycatcher": "Flycatcher",
    "Orange Crowned Warbler": "Warbler",
    "Orchard Oriole": "Oriole",
    "Ovenbird": "Ovenbird",
    "Pacific Loon": "Loon",
    "Painted Bunting": "Bunting",
    "Palm Warbler": "Warbler",
    "Parakeet Auklet": "Auklet",
    "Pelagic Cormorant": "Cormorant",
    "Philadelphia Vireo": "Vireo",
    "Pied Billed Grebe": "Grebe",
    "Pied Kingfisher": "Kingfisher",
    "Pigeon Guillemot": "Guillemot",
    "Pileated Woodpecker": "Woodpecker",
    "Pine Grosbeak": "Grosbeak",
    "Pine Warbler": "Warbler",
    "Pomarine Jaeger": "Jaeger",
    "Prairie Warbler": "Warbler",
    "Prothonotary Warbler": "Warbler",
    "Purple Finch": "Finch",
    "Red Bellied Woodpecker": "Woodpecker",
    "Red Breasted Merganser": "Merganser",
    "Red Cockaded Woodpecker": "Woodpecker",
    "Red Eyed Vireo": "Vireo",
    "Red Faced Cormorant": "Cormorant",
    "Red Headed Woodpecker": "Woodpecker",
    "Red Legged Kittiwake": "Kittiwake",
    "Red Winged Blackbird": "Blackbird",
    "Rhinoceros Auklet": "Auklet",
    "Ring Billed Gull": "Gull",
    "Ringed Kingfisher": "Kingfisher",
    "Rock Wren": "Wren",
    "Rose Breasted Grosbeak": "Grosbeak",
    "Ruby Throated Hummingbird": "Hummingbird",
    "Rufous Hummingbird": "Hummingbird",
    "Rusty Blackbird": "Blackbird",
    "Sage Thrasher": "Thrasher",
    "Savannah Sparrow": "Sparrow",
    "Sayornis": "Sayornis",
    "Scarlet Tanager": "Tanager",
    "Scissor Tailed Flycatcher": "Flycatcher",
    "Scott Oriole": "Oriole",
    "Seaside Sparrow": "Sparrow",
    "Shiny Cowbird": "Cowbird",
    "Slaty Backed Gull": "Gull",
    "Song Sparrow": "Sparrow",
    "Sooty Albatross": "Albatross",
    "Spotted Catbird": "Catbird",
    "Summer Tanager": "Tanager",
    "Swainson Warbler": "Warbler",
    "Tennessee Warbler": "Warbler",
    "Tree Sparrow": "Sparrow",
    "Tree Swallow": "Swallow",
    "Tropical Kingbird": "Kingbird",
    "Vermilion Flycatcher": "Flycatcher",
    "Vesper Sparrow": "Sparrow",
    "Warbling Vireo": "Vireo",
    "Western Grebe": "Grebe",
    "Western Gull": "Gull",
    "Western Meadowlark": "Meadowlark",
    "Western Wood Pewee": "Pewee",
    "Whip Poor Will": "Will",
    "White Breasted Kingfisher": "Kingfisher",
    "White Breasted Nuthatch": "Nuthatch",
    "White Crowned Sparrow": "Sparrow",
    "White Eyed Vireo": "Vireo",
    "White Necked Raven": "Raven",
    "White Pelican": "Pelican",
    "White Throated Sparrow": "Sparrow",
    "Wilson Warbler": "Warbler",
    "Winter Wren": "Wren",
    "Worm Eating Warbler": "Warbler",
    "Yellow Bellied Flycatcher": "Flycatcher",
    "Yellow Billed Cuckoo": "Cuckoo",
    "Yellow Breasted Chat": "Chat",
    "Yellow Headed Blackbird": "Blackbird",
    "Yellow Throated Vireo": "Vireo",
    "Yellow Warbler": "Warbler",
}


def get_taxonomy() -> tuple[dict[str, str], dict[str, set[str]], set[str]]:
    """Derive normalised lookup dicts from the hardcoded map.

    Returns
    -------
    species_to_family : {normalised_species: family}
    family_to_species : {family: set of normalised species}
    known_species     : set of normalised species names
    """
    species_to_family: dict[str, str] = {}
    family_to_species: dict[str, set[str]] = defaultdict(set)

    for species, family in SPECIES_TO_FAMILY.items():
        sp_norm = normalize_text(species)
        fam_norm = normalize_text(family)
        species_to_family[sp_norm] = fam_norm
        family_to_species[fam_norm].add(sp_norm)

    known_species = set(species_to_family.keys())
    return species_to_family, dict(family_to_species), known_species


# ============================================================================
# Text normalisation & VLM response extraction
# ============================================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[.,!?;:\"'`(){}[\]]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_FAIL_PATTERNS = re.compile(
    r"unanswerable|unknown|cannot identify|can'?t identify|"
    r"i'?m not able|not possible to identify|no bird|there is no bird|"
    r"cannot determine|can'?t determine|cannot provide|"
    r"not enough information|not clear what species",
    re.IGNORECASE,
)

_GARBAGE = re.compile(r"^[\d.:/\s]+$")
_NON_LATIN = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")
_MARKDOWN_BOLD = re.compile(r"\*\*(.+?)\*\*")

_SENTENCE_PATTERNS = [
    re.compile(
        r"(?:the bird|this bird|it|this|that|the species|the image)\s+"
        r"(?:is|appears to be|looks like|seems to be|depicts|shows|features)"
        r"\s+(?:a\s+|an\s+|the\s+)?(.+)",
        re.I,
    ),
    re.compile(r"(?:it'?s|that'?s)\s+(?:a\s+|an\s+)?(.+)", re.I),
]

_EXPLICIT_PATTERNS = [
    re.compile(r"(?:common\s+name|answer|species|bird)\s*[:=]\s*(.+)", re.I),
]


def extract_bird_name(response: str, known_species: set[str]) -> str:
    """Pull a bird species name out of a raw VLM response."""
    if not response or not isinstance(response, str):
        return ""

    text = response.strip()

    if _FAIL_PATTERNS.search(text):
        return ""

    if _NON_LATIN.search(text):
        latin_parts = re.findall(r"[A-Za-z][A-Za-z\s\-']+", text)
        text = max(latin_parts, key=len).strip() if latin_parts else ""
        if not text:
            return ""

    bold_skip = {"answer", "answer:", "bird", "species", "common name", "common name:"}
    bold_matches = _MARKDOWN_BOLD.findall(text)
    if bold_matches:
        for candidate in reversed(bold_matches):
            cleaned = candidate.strip().rstrip(".,!?;:")
            if cleaned.lower() not in bold_skip and len(cleaned) > 1:
                text = cleaned
                break

    text = text.split("\n")[0].strip()

    for pat in _EXPLICIT_PATTERNS:
        m = pat.search(text)
        if m:
            text = m.group(1).strip()
            break

    for pat in _SENTENCE_PATTERNS:
        m = pat.search(text)
        if m:
            text = m.group(1).strip()
            break

    text = re.sub(r"^[:/]+\s*", "", text)
    text = re.sub(r"^(?:a|an|the|some|this|that)\s+", "", text, flags=re.I)
    text = re.sub(r"\s*\(.*?\)\s*", " ", text)
    text = text.rstrip(".,!?;:")
    text = text.strip()

    if not text or _GARBAGE.match(text):
        return ""

    if len(text) > 60:
        text_n = normalize_text(text)
        for sp in sorted(known_species, key=len, reverse=True):
            if sp in text_n:
                return sp
        return ""

    return text


# ============================================================================
# Matching helpers
# ============================================================================

def fuzzy_match(pred: str, label: str, threshold: float = 0.85) -> bool:
    """Exact match with fuzzy tolerance for typos."""
    pred_norm = normalize_text(pred)
    label_norm = normalize_text(label)

    if pred_norm == label_norm:
        return True

    return SequenceMatcher(None, pred_norm, label_norm).ratio() >= threshold


def family_match(
    pred_text: str,
    label_norm: str,
    species_to_family: dict[str, str],
    family_to_species: dict[str, set[str]],
) -> bool:
    """Check whether the prediction belongs to the same family as the label."""
    label_family = species_to_family.get(label_norm, "")
    if not label_family:
        return False

    pred_norm = normalize_text(pred_text)

    # 1) prediction is a known species in the same family
    pred_family = species_to_family.get(pred_norm, "")
    if pred_family == label_family:
        return True

    # 2) last word of prediction matches the family word
    pred_last = pred_norm.split()[-1] if pred_norm.split() else ""
    if pred_last == label_family:
        return True

    # 3) fuzzy-match prediction against any sibling species
    siblings = family_to_species.get(label_family, set())
    for sibling in siblings:
        if SequenceMatcher(None, pred_norm, sibling).ratio() >= 0.85:
            return True

    return False


# ============================================================================
# Per-file evaluation
# ============================================================================

def evaluate_predictions(
    pred_csv: Path,
    species_to_family: dict[str, str],
    family_to_species: dict[str, set[str]],
    known_species: set[str],
) -> Dict:
    """Evaluate a single prediction CSV and return metrics."""
    with open(pred_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    exact = 0
    family_only = 0
    failed = 0
    per_group: dict[str, dict[str, int]] = defaultdict(
        lambda: {"exact": 0, "group_only": 0, "failed": 0, "total": 0}
    )

    for row in rows:
        label_raw = (row.get("label") or "").strip()
        pred_raw = (row.get("prediction") or "").strip()

        label_norm = normalize_text(label_raw)
        group = species_to_family.get(label_norm, "unknown")
        per_group[group]["total"] += 1

        extracted = extract_bird_name(pred_raw, known_species)
        pred_norm = normalize_text(extracted)

        if not pred_norm:
            per_group[group]["failed"] += 1
            failed += 1
            continue

        if fuzzy_match(extracted, label_raw):
            per_group[group]["exact"] += 1
            exact += 1
        elif family_match(extracted, label_norm, species_to_family, family_to_species):
            per_group[group]["group_only"] += 1
            family_only += 1

    return {
        "exact_acc": exact / total if total else 0.0,
        "exact_count": exact,
        "family_acc": (exact + family_only) / total if total else 0.0,
        "family_count": exact + family_only,
        "family_only_count": family_only,
        "failed_count": failed,
        "total_samples": total,
        "per_group": dict(per_group),
    }


# ============================================================================
# Batch evaluation & reporting
# ============================================================================

def batch_evaluate(pred_dir: Path, output_dir: Path):
    """Evaluate all model predictions and generate an analysis report."""
    species_to_family, family_to_species, known_species = get_taxonomy()

    print(f"Loaded taxonomy: {len(known_species)} species across {len(family_to_species)} families\n")

    all_csvs = sorted(pred_dir.rglob("*.csv"))
    if not all_csvs:
        print(f"No prediction files found in {pred_dir}")
        return

    print(f"Evaluating {len(all_csvs)} model predictions ...\n")

    results: List[Dict] = []
    hq_group_agg: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"exact": 0, "total": 0})
    )
    sil_group_agg: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"exact": 0, "total": 0})
    )
    hq_dataset_counts: dict[str, int] = {}
    sil_dataset_counts: dict[str, int] = {}

    for csv_file in all_csvs:
        phase = "HQ" if "/hq/" in str(csv_file) else "Silhouette"
        model_type = csv_file.parent.name
        model_name = csv_file.stem.replace("predictions_", "").replace("_", "/")

        metrics = evaluate_predictions(csv_file, species_to_family, family_to_species, known_species)

        results.append({
            "Model": model_name,
            "Type": model_type,
            "Phase": phase,
            "Exact Acc": metrics["exact_acc"],
            "Exact Count": f"{metrics['exact_count']}/{metrics['total_samples']}",
            "Family Acc": metrics["family_acc"],
            "Family Count": f"{metrics['family_count']}/{metrics['total_samples']}",
            "Family Only": metrics["family_only_count"],
            "Failed": metrics["failed_count"],
            "Total": metrics["total_samples"],
        })

        target_agg = hq_group_agg if phase == "HQ" else sil_group_agg
        target_counts = hq_dataset_counts if phase == "HQ" else sil_dataset_counts
        for grp, counts in metrics["per_group"].items():
            target_agg[model_type][grp]["exact"] += counts["exact"]
            target_agg[model_type][grp]["total"] += counts["total"]
            if grp not in target_counts:
                target_counts[grp] = counts["total"]

    df = pd.DataFrame(results)
    df_hq = df[df["Phase"] == "HQ"].copy()
    df_sil = df[df["Phase"] == "Silhouette"].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = output_dir / "cub200_evaluation_results.csv"
    df.to_csv(csv_out, index=False)

    generate_analysis_report(
        df_hq, df_sil, output_dir, family_to_species,
        hq_group_agg, sil_group_agg, hq_dataset_counts, sil_dataset_counts,
    )

    print(f"\nResults saved to : {csv_out}")
    print(f"Analysis saved to: {output_dir / 'analysis_full.txt'}")


def generate_analysis_report(
    df_hq: pd.DataFrame,
    df_sil: pd.DataFrame,
    output_dir: Path,
    family_to_species: dict[str, set[str]],
    hq_group_agg: dict = None,
    sil_group_agg: dict = None,
    hq_dataset_counts: dict = None,
    sil_dataset_counts: dict = None,
):
    """Generate a comprehensive text report."""
    W = 130
    analysis_file = output_dir / "analysis_full.txt"

    with open(analysis_file, "w") as f:
        f.write(f"Loaded {len(df_hq)} HQ models, {len(df_sil)} silhouette models\n")
        f.write(f"Family taxonomy: {len(family_to_species)} families\n\n")

        _write_phase_table(f, df_hq, "CUB-200 ACCURACY — HQ IMAGES", W)
        _write_phase_table(f, df_sil, "CUB-200 ACCURACY — SILHOUETTE IMAGES", W)

        if not df_hq.empty and not df_sil.empty:
            f.write("=" * W + "\n")
            f.write("  HQ vs SILHOUETTE COMPARISON\n")
            f.write("=" * W + "\n")
            f.write(
                f"{'Model':<55s} {'Type':<12s} "
                f"{'HQ Exact':>10s} {'Sil Exact':>10s} {'Delta':>8s} "
                f"{'HQ Fam':>10s} {'Sil Fam':>10s} {'Delta':>8s}\n"
            )
            f.write("-" * W + "\n")

            comp = df_hq.merge(df_sil, on=["Model", "Type"], suffixes=("_hq", "_sil"))
            comp["D_exact"] = comp["Exact Acc_hq"] - comp["Exact Acc_sil"]
            comp["D_fam"] = comp["Family Acc_hq"] - comp["Family Acc_sil"]

            for _, row in comp.sort_values("Exact Acc_hq", ascending=False).iterrows():
                f.write(
                    f"{row['Model']:<55s} {row['Type']:<12s} "
                    f"{row['Exact Acc_hq']:>9.2%} {row['Exact Acc_sil']:>10.2%} {row['D_exact']:>+7.2%} "
                    f"{row['Family Acc_hq']:>9.2%} {row['Family Acc_sil']:>10.2%} {row['D_fam']:>+7.2%}\n"
                )

            f.write("\n")

        if hq_group_agg:
            _write_group_accuracy_table(
                f, hq_group_agg, hq_dataset_counts or {},
                "FAMILY ACCURACY — HQ (aggregated across models)",
                "Family", W,
            )

        if sil_group_agg:
            _write_group_accuracy_table(
                f, sil_group_agg, sil_dataset_counts or {},
                "FAMILY ACCURACY — SILHOUETTE (aggregated across models)",
                "Family", W,
            )


def _write_group_accuracy_table(
    f,
    group_agg: dict[str, dict[str, dict[str, int]]],
    dataset_counts: dict[str, int],
    title: str,
    group_label: str,
    width: int,
):
    """Write a per-group accuracy table aggregated across models, sorted descending."""
    f.write("=" * width + "\n")
    f.write(f"  {title}\n")
    f.write("=" * width + "\n")

    for model_type in ["proprietary", "legacy", "remote"]:
        if model_type not in group_agg:
            continue

        type_data = group_agg[model_type]
        f.write(f"\n  {model_type.upper()}:\n")
        f.write(
            f"  {group_label:<25s} {'Accuracy':>10s} {'Correct':>10s}"
            f" {'Total':>10s} {'In Dataset':>12s}\n"
        )
        f.write(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12}\n")

        sorted_groups = sorted(
            type_data.items(),
            key=lambda x: x[1]["exact"] / x[1]["total"] if x[1]["total"] > 0 else 0,
            reverse=True,
        )

        for grp_name, counts in sorted_groups:
            acc = counts["exact"] / counts["total"] if counts["total"] > 0 else 0
            in_ds = dataset_counts.get(grp_name, counts["total"])
            f.write(
                f"  {grp_name:<25s} {acc:>9.2%}"
                f" {counts['exact']:>10d} {counts['total']:>10d} {in_ds:>12d}\n"
            )

    f.write("\n")


def _write_phase_table(f, df: pd.DataFrame, title: str, width: int):
    """Write a single-phase results table."""
    f.write("=" * width + "\n")
    f.write(f"  {title}\n")
    f.write("=" * width + "\n")
    f.write(
        f"{'Model':<55s} {'Type':<12s} "
        f"{'Exact Acc':>10s} {'Exact':>10s} "
        f"{'Family Acc':>10s} {'Family':>10s} "
        f"{'Fam Only':>9s} {'Failed':>8s}\n"
    )
    f.write("-" * width + "\n")

    for _, row in df.sort_values("Exact Acc", ascending=False).iterrows():
        f.write(
            f"{row['Model']:<55s} {row['Type']:<12s} "
            f"{row['Exact Acc']:>9.2%} {row['Exact Count']:>10s} "
            f"{row['Family Acc']:>9.2%} {row['Family Count']:>10s} "
            f"{row['Family Only']:>9d} {row['Failed']:>8d}\n"
        )

    # Averages by model type
    f.write("\n")
    for model_type in ["proprietary", "legacy", "remote"]:
        type_df = df[df["Type"] == model_type]
        if len(type_df) > 0:
            f.write(
                f"  [AVG {model_type.upper():<12s}] {' ' * 52} "
                f"{type_df['Exact Acc'].mean():>9.2%} {' ' * 10} "
                f"{type_df['Family Acc'].mean():>9.2%}\n"
            )

    f.write("\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CUB-200-2011 predictions with exact and family-level metrics."
    )
    parser.add_argument(
        "--pred-csv", type=Path,
        help="Single prediction CSV to evaluate",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch-evaluate all predictions under predictions/cub/",
    )
    parser.add_argument(
        "--pred-dir", type=Path, default=Path("predictions/cub"),
        help="Root directory for batch mode (default: predictions/cub/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/cub"),
        help="Output directory for batch results",
    )
    args = parser.parse_args()

    species_to_family, family_to_species, known_species = get_taxonomy()

    if args.batch:
        batch_evaluate(args.pred_dir, args.output_dir)

    elif args.pred_csv:
        if not args.pred_csv.exists():
            print(f"Error: {args.pred_csv} not found")
            return

        metrics = evaluate_predictions(args.pred_csv, species_to_family, family_to_species, known_species)

        print()
        print("=" * 80)
        print("  CUB-200 EVALUATION RESULTS")
        print("=" * 80)
        print(f"  File: {args.pred_csv.name}")
        print(f"  Taxonomy: {len(known_species)} species, {len(family_to_species)} families\n")
        print(f"  {'Metric':<30s} {'Value':>15s}")
        print("-" * 80)
        print(f"  {'Exact Species Accuracy':<30s} {metrics['exact_acc']:>14.2%}")
        print(f"  {'Exact Count':<30s} {metrics['exact_count']:>10d} / {metrics['total_samples']}")
        print(f"  {'Family Match Accuracy':<30s} {metrics['family_acc']:>14.2%}")
        print(f"  {'Family Match Count':<30s} {metrics['family_count']:>10d} / {metrics['total_samples']}")
        print(f"  {'  (family-only matches)':<30s} {metrics['family_only_count']:>10d}")
        print(f"  {'Failed / Refused':<30s} {metrics['failed_count']:>10d} / {metrics['total_samples']}")
        print("=" * 80)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
