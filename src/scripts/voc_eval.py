#!/usr/bin/env python3
"""Pascal VOC 2012 Evaluation Script — batch evaluation with super-category-level metrics.

20 object classes grouped into 4 super-categories (standard VOC groupings).
  e.g.  "cat"        ->  super-category "Animal"
        "aeroplane"  ->  super-category "Vehicle"

Scoring tiers
  Exact category  – normalised prediction matches normalised label (with fuzzy tolerance)
  Super-category  – prediction names a different category from the same super-category
  Failed          – model refused / produced garbage
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
# Super-category Taxonomy (hardcoded)
# ============================================================================

CATEGORY_TO_SUPERCATEGORY: dict[str, str] = {
    "aeroplane": "Vehicle",
    "bicycle": "Vehicle",
    "bird": "Animal",
    "boat": "Vehicle",
    "bottle": "Indoor",
    "bus": "Vehicle",
    "car": "Vehicle",
    "cat": "Animal",
    "chair": "Indoor",
    "cow": "Animal",
    "diningtable": "Indoor",
    "dog": "Animal",
    "horse": "Animal",
    "motorbike": "Vehicle",
    "person": "Person",
    "pottedplant": "Indoor",
    "sheep": "Animal",
    "sofa": "Indoor",
    "train": "Vehicle",
    "tvmonitor": "Indoor",
}


def get_taxonomy() -> tuple[dict[str, str], dict[str, set[str]], set[str]]:
    """Derive normalised lookup dicts from the hardcoded map.

    Returns
    -------
    category_to_supercat   : {normalised_category: normalised_super_category}
    supercat_to_categories : {normalised_super_category: set of normalised categories}
    known_categories       : set of normalised category names
    """
    category_to_supercat: dict[str, str] = {}
    supercat_to_categories: dict[str, set[str]] = defaultdict(set)

    for cat, supercat in CATEGORY_TO_SUPERCATEGORY.items():
        cat_norm = normalize_text(cat)
        sc_norm = normalize_text(supercat)
        category_to_supercat[cat_norm] = sc_norm
        supercat_to_categories[sc_norm].add(cat_norm)

    known_categories = set(category_to_supercat.keys())
    return category_to_supercat, dict(supercat_to_categories), known_categories


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
    r"i'?m not able|not possible to identify|"
    r"cannot determine|can'?t determine|cannot provide|"
    r"not enough information|not clear what",
    re.IGNORECASE,
)

_GARBAGE = re.compile(r"^[\d.:/\s]+$")
_NON_LATIN = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")
_MARKDOWN_BOLD = re.compile(r"\*\*(.+?)\*\*")

_SENTENCE_PATTERNS = [
    re.compile(
        r"(?:the object|this object|the item|this item|it|this|that|the image|the picture)\s+"
        r"(?:is|appears to be|looks like|seems to be|depicts|shows|features)"
        r"\s+(?:a\s+|an\s+|the\s+)?(.+)",
        re.I,
    ),
    re.compile(r"(?:it'?s|that'?s)\s+(?:a\s+|an\s+)?(.+)", re.I),
]

_EXPLICIT_PATTERNS = [
    re.compile(r"(?:answer|object|item|category)\s*[:=]\s*(.+)", re.I),
]


def extract_object_name(response: str, known_categories: set[str]) -> str:
    """Pull an object category name out of a raw VLM response."""
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

    bold_skip = {"answer", "answer:", "object", "category", "item"}
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

    if len(text) > 80:
        text_n = normalize_text(text)
        for cat in sorted(known_categories, key=len, reverse=True):
            if cat in text_n:
                return cat
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


def supercategory_match(
    pred_text: str,
    label_norm: str,
    category_to_supercat: dict[str, str],
    supercat_to_categories: dict[str, set[str]],
) -> bool:
    """Check whether the prediction belongs to the same super-category as the label."""
    label_supercat = category_to_supercat.get(label_norm, "")
    if not label_supercat:
        return False

    pred_norm = normalize_text(pred_text)

    pred_supercat = category_to_supercat.get(pred_norm, "")
    if pred_supercat == label_supercat:
        return True

    siblings = supercat_to_categories.get(label_supercat, set())
    for sibling in siblings:
        if SequenceMatcher(None, pred_norm, sibling).ratio() >= 0.85:
            return True

    return False


# ============================================================================
# Per-file evaluation
# ============================================================================

def evaluate_predictions(
    pred_csv: Path,
    category_to_supercat: dict[str, str],
    supercat_to_categories: dict[str, set[str]],
    known_categories: set[str],
) -> Dict:
    """Evaluate a single prediction CSV and return metrics."""
    with open(pred_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    exact = 0
    supercat_only = 0
    failed = 0
    per_group: dict[str, dict[str, int]] = defaultdict(
        lambda: {"exact": 0, "group_only": 0, "failed": 0, "total": 0}
    )

    for row in rows:
        label_raw = (row.get("label") or "").strip()
        pred_raw = (row.get("prediction") or "").strip()

        label_norm = normalize_text(label_raw)
        group = category_to_supercat.get(label_norm, "unknown")
        per_group[group]["total"] += 1

        extracted = extract_object_name(pred_raw, known_categories)
        pred_norm = normalize_text(extracted)

        if not pred_norm:
            per_group[group]["failed"] += 1
            failed += 1
            continue

        if fuzzy_match(extracted, label_raw):
            per_group[group]["exact"] += 1
            exact += 1
        elif supercategory_match(
            extracted, label_norm, category_to_supercat, supercat_to_categories
        ):
            per_group[group]["group_only"] += 1
            supercat_only += 1

    return {
        "exact_acc": exact / total if total else 0.0,
        "exact_count": exact,
        "supercat_acc": (exact + supercat_only) / total if total else 0.0,
        "supercat_count": exact + supercat_only,
        "supercat_only_count": supercat_only,
        "failed_count": failed,
        "total_samples": total,
        "per_group": dict(per_group),
    }


# ============================================================================
# Batch evaluation & reporting
# ============================================================================

def batch_evaluate(pred_dir: Path, output_dir: Path):
    """Evaluate all model predictions and generate an analysis report."""
    category_to_supercat, supercat_to_categories, known_categories = get_taxonomy()

    print(
        f"Loaded taxonomy: {len(known_categories)} categories "
        f"across {len(supercat_to_categories)} super-categories\n"
    )

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

        metrics = evaluate_predictions(
            csv_file, category_to_supercat, supercat_to_categories, known_categories
        )

        results.append({
            "Model": model_name,
            "Type": model_type,
            "Phase": phase,
            "Exact Acc": metrics["exact_acc"],
            "Exact Count": f"{metrics['exact_count']}/{metrics['total_samples']}",
            "SuperCat Acc": metrics["supercat_acc"],
            "SuperCat Count": f"{metrics['supercat_count']}/{metrics['total_samples']}",
            "SuperCat Only": metrics["supercat_only_count"],
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
    csv_out = output_dir / "voc_evaluation_results.csv"
    df.to_csv(csv_out, index=False)

    generate_analysis_report(
        df_hq, df_sil, output_dir, supercat_to_categories,
        hq_group_agg, sil_group_agg, hq_dataset_counts, sil_dataset_counts,
    )

    print(f"\nResults saved to : {csv_out}")
    print(f"Analysis saved to: {output_dir / 'analysis_full.txt'}")


def generate_analysis_report(
    df_hq: pd.DataFrame,
    df_sil: pd.DataFrame,
    output_dir: Path,
    supercat_to_categories: dict[str, set[str]],
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
        f.write(f"Super-category taxonomy: {len(supercat_to_categories)} super-categories\n\n")

        _write_phase_table(f, df_hq, "VOC ACCURACY — HQ IMAGES", W)
        _write_phase_table(f, df_sil, "VOC ACCURACY — SILHOUETTE IMAGES", W)

        if not df_hq.empty and not df_sil.empty:
            f.write("=" * W + "\n")
            f.write("  HQ vs SILHOUETTE COMPARISON\n")
            f.write("=" * W + "\n")
            f.write(
                f"{'Model':<55s} {'Type':<12s} "
                f"{'HQ Exact':>10s} {'Sil Exact':>10s} {'Delta':>8s} "
                f"{'HQ SCat':>10s} {'Sil SCat':>10s} {'Delta':>8s}\n"
            )
            f.write("-" * W + "\n")

            comp = df_hq.merge(df_sil, on=["Model", "Type"], suffixes=("_hq", "_sil"))
            comp["D_exact"] = comp["Exact Acc_hq"] - comp["Exact Acc_sil"]
            comp["D_supercat"] = comp["SuperCat Acc_hq"] - comp["SuperCat Acc_sil"]

            for _, row in comp.sort_values("Exact Acc_hq", ascending=False).iterrows():
                f.write(
                    f"{row['Model']:<55s} {row['Type']:<12s} "
                    f"{row['Exact Acc_hq']:>9.2%} {row['Exact Acc_sil']:>10.2%} {row['D_exact']:>+7.2%} "
                    f"{row['SuperCat Acc_hq']:>9.2%} {row['SuperCat Acc_sil']:>10.2%} {row['D_supercat']:>+7.2%}\n"
                )

            f.write("\n")

        if hq_group_agg:
            _write_group_accuracy_table(
                f, hq_group_agg, hq_dataset_counts or {},
                "SUPER-CATEGORY ACCURACY — HQ (aggregated across models)",
                "Super-Category", W,
            )

        if sil_group_agg:
            _write_group_accuracy_table(
                f, sil_group_agg, sil_dataset_counts or {},
                "SUPER-CATEGORY ACCURACY — SILHOUETTE (aggregated across models)",
                "Super-Category", W,
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
        f"{'SCat Acc':>10s} {'SCat':>10s} "
        f"{'SCat Only':>9s} {'Failed':>8s}\n"
    )
    f.write("-" * width + "\n")

    for _, row in df.sort_values("Exact Acc", ascending=False).iterrows():
        f.write(
            f"{row['Model']:<55s} {row['Type']:<12s} "
            f"{row['Exact Acc']:>9.2%} {row['Exact Count']:>10s} "
            f"{row['SuperCat Acc']:>9.2%} {row['SuperCat Count']:>10s} "
            f"{row['SuperCat Only']:>9d} {row['Failed']:>8d}\n"
        )

    f.write("\n")
    for model_type in ["proprietary", "legacy", "remote"]:
        type_df = df[df["Type"] == model_type]
        if len(type_df) > 0:
            f.write(
                f"  [AVG {model_type.upper():<12s}] {' ' * 52} "
                f"{type_df['Exact Acc'].mean():>9.2%} {' ' * 10} "
                f"{type_df['SuperCat Acc'].mean():>9.2%}\n"
            )

    f.write("\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pascal VOC predictions with exact and super-category-level metrics."
    )
    parser.add_argument(
        "--pred-csv", type=Path,
        help="Single prediction CSV to evaluate",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch-evaluate all predictions under predictions/voc/",
    )
    parser.add_argument(
        "--pred-dir", type=Path, default=Path("predictions/voc"),
        help="Root directory for batch mode (default: predictions/voc/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/voc"),
        help="Output directory for batch results",
    )
    args = parser.parse_args()

    category_to_supercat, supercat_to_categories, known_categories = get_taxonomy()

    if args.batch:
        batch_evaluate(args.pred_dir, args.output_dir)

    elif args.pred_csv:
        if not args.pred_csv.exists():
            print(f"Error: {args.pred_csv} not found")
            return

        metrics = evaluate_predictions(
            args.pred_csv, category_to_supercat, supercat_to_categories, known_categories
        )

        print()
        print("=" * 80)
        print("  VOC EVALUATION RESULTS")
        print("=" * 80)
        print(f"  File: {args.pred_csv.name}")
        print(
            f"  Taxonomy: {len(known_categories)} categories, "
            f"{len(supercat_to_categories)} super-categories\n"
        )
        print(f"  {'Metric':<30s} {'Value':>15s}")
        print("-" * 80)
        print(f"  {'Exact Category Accuracy':<30s} {metrics['exact_acc']:>14.2%}")
        print(f"  {'Exact Count':<30s} {metrics['exact_count']:>10d} / {metrics['total_samples']}")
        print(f"  {'Super-Category Accuracy':<30s} {metrics['supercat_acc']:>14.2%}")
        print(f"  {'Super-Category Count':<30s} {metrics['supercat_count']:>10d} / {metrics['total_samples']}")
        print(f"  {'  (super-cat-only matches)':<30s} {metrics['supercat_only_count']:>10d}")
        print(f"  {'Failed / Refused':<30s} {metrics['failed_count']:>10d} / {metrics['total_samples']}")
        print("=" * 80)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
