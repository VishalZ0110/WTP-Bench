#!/usr/bin/env python3
"""ImageNet100 Evaluation Script"""
from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# ============================================================================
# ImageNet100 Animal Taxonomy
# ============================================================================

FAMILY_TAXONOMY = {
    # Dogs
    "Afghan hound": "dog", "Bernese mountain dog": "dog", "Blenheim spaniel": "dog",
    "Border collie": "dog", "Bouvier des Flandres": "dog", "Dandie Dinmont": "dog",
    "EntleBucher": "dog", "German short-haired pointer": "dog", "Gordon setter": "dog",
    "Great Pyrenees": "dog", "Irish setter": "dog", "Irish water spaniel": "dog",
    "Irish wolfhound": "dog", "Japanese spaniel": "dog", "Kerry blue terrier": "dog",
    "Newfoundland": "dog", "Pekinese": "dog", "Rhodesian ridgeback": "dog",
    "Samoyed": "dog", "Scottish deerhound": "dog", "Shih-Tzu": "dog",
    "Tibetan terrier": "dog", "Welsh springer spaniel": "dog", "Yorkshire terrier": "dog",
    "affenpinscher": "dog", "bluetick": "dog", "briard": "dog", "dingo": "dog",
    "groenendael": "dog", "komondor": "dog", "miniature poodle": "dog", "papillon": "dog",

    # Big cats
    "cheetah": "big cat", "leopard": "big cat", "tiger": "big cat",

    # Canines
    "African hunting dog": "canine", "Arctic fox": "fox", "kit fox": "fox",
    "red fox": "fox", "red wolf": "canine",

    # Beetles
    "ground beetle": "beetle", "long-horned beetle": "beetle",
    "rhinoceros beetle": "beetle", "tiger beetle": "beetle", "weevil": "beetle",

    # Spiders
    "barn spider": "spider", "black widow": "spider", "garden spider": "spider",
    "tarantula": "spider", "wolf spider": "spider",

    # Snakes
    "boa constrictor": "snake", "diamondback": "snake", "garter snake": "snake",
    "green mamba": "snake", "hognose snake": "snake", "horned viper": "snake",
    "Indian cobra": "snake", "king snake": "snake", "night snake": "snake",
    "rock python": "snake", "sidewinder": "snake",

    # Reptiles
    "African crocodile": "crocodile", "American alligator": "alligator",
    "Komodo dragon": "lizard", "box turtle": "turtle", "common iguana": "iguana",

    # Amphibians
    "European fire salamander": "salamander", "bullfrog": "frog",
    "spotted salamander": "salamander", "tailed frog": "frog",

    # Insects
    "admiral": "butterfly", "cabbage butterfly": "butterfly", "monarch": "butterfly",
    "cockroach": "cockroach", "cricket": "cricket", "grasshopper": "grasshopper",
    "leafhopper": "grasshopper",

    # Arthropods
    "centipede": "centipede", "harvestman": "arachnid", "tick": "arachnid",
    "walking stick": "insect",

    # Birds
    "bittern": "bird", "black swan": "bird", "brambling": "bird", "bustard": "bird",
    "chickadee": "bird", "dowitcher": "bird", "junco": "bird", "kite": "bird",
    "limpkin": "bird", "red-backed sandpiper": "bird", "redshank": "bird",
    "ruffed grouse": "bird",

    # Marine
    "conch": "mollusk", "sea anemone": "cnidarian", "sea slug": "mollusk",
    "flatworm": "worm", "nematode": "worm", "trilobite": "arthropod",
    "great white shark": "shark",
}


# ============================================================================
# Evaluation Functions
# ============================================================================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    return ' '.join(text.split())


def fuzzy_match(pred: str, label: str, threshold: float = 0.85) -> bool:
    """Check if prediction matches label with fuzzy tolerance."""
    pred_norm = normalize_text(pred)
    label_norm = normalize_text(label)

    if pred_norm == label_norm:
        return True

    ratio = SequenceMatcher(None, pred_norm, label_norm).ratio()
    return ratio >= threshold


def extract_family(text: str) -> str:
    """Extract animal family from text using taxonomy and fuzzy matching."""
    text_norm = normalize_text(text)

    # Exact match in taxonomy
    for class_name, family in FAMILY_TAXONOMY.items():
        if normalize_text(class_name) == text_norm:
            return family

    # Fuzzy match against all known classes
    best_match = None
    best_ratio = 0.0
    for class_name, family in FAMILY_TAXONOMY.items():
        ratio = SequenceMatcher(None, text_norm, normalize_text(class_name)).ratio()
        if ratio > best_ratio and ratio >= 0.85:
            best_ratio = ratio
            best_match = family

    if best_match:
        return best_match

    # Keyword mapping for generic terms
    keyword_map = {
        'hound': 'dog', 'spaniel': 'dog', 'terrier': 'dog', 'retriever': 'dog',
        'setter': 'dog', 'pointer': 'dog', 'poodle': 'dog', 'collie': 'dog', 'dog': 'dog',
        'tiger': 'big cat', 'leopard': 'big cat', 'cheetah': 'big cat', 'cat': 'big cat',
        'fox': 'fox', 'wolf': 'canine',
        'beetle': 'beetle', 'weevil': 'beetle',
        'spider': 'spider', 'arachnid': 'spider',
        'snake': 'snake', 'viper': 'snake', 'cobra': 'snake', 'python': 'snake', 'boa': 'snake',
        'crocodile': 'crocodile', 'alligator': 'alligator',
        'lizard': 'lizard', 'iguana': 'iguana',
        'salamander': 'salamander', 'frog': 'frog',
        'butterfly': 'butterfly', 'moth': 'butterfly',
        'cockroach': 'cockroach', 'cricket': 'cricket', 'grasshopper': 'grasshopper',
        'centipede': 'centipede', 'millipede': 'centipede',
        'bird': 'bird', 'swan': 'bird', 'grouse': 'bird',
        'shark': 'shark', 'fish': 'fish',
        'mollusk': 'mollusk', 'slug': 'mollusk', 'snail': 'mollusk',
        'worm': 'worm',
    }

    for keyword, family in keyword_map.items():
        if keyword in text_norm:
            return family

    return text_norm


def get_family_for_class(class_name: str) -> str:
    """Get family for a known ImageNet class."""
    return FAMILY_TAXONOMY.get(class_name, extract_family(class_name))


def evaluate_predictions(pred_csv: Path) -> Dict:
    """Evaluate predictions with exact and family-level metrics."""
    df = pd.read_csv(pred_csv)

    total_samples = len(df)
    exact_matches = 0
    family_matches = 0

    for _, row in df.iterrows():
        label = str(row['label']).strip()
        prediction = str(row['prediction']).strip()

        is_exact = fuzzy_match(prediction, label)

        label_family = get_family_for_class(label)
        pred_family = extract_family(prediction)
        is_family = (label_family == pred_family) or \
                   (label_family in pred_family) or \
                   (pred_family in label_family)

        if is_exact:
            exact_matches += 1
            family_matches += 1
        elif is_family:
            family_matches += 1

    return {
        'total_acc': exact_matches / total_samples if total_samples > 0 else 0,
        'total_count': exact_matches,
        'family_acc': family_matches / total_samples if total_samples > 0 else 0,
        'family_count': family_matches,
        'total_samples': total_samples,
    }


# ============================================================================
# Batch Evaluation & Reporting
# ============================================================================

def batch_evaluate(pred_dir: Path, output_dir: Path):
    """Evaluate all models and generate analysis report."""
    all_csvs = list(pred_dir.rglob("*.csv"))

    if not all_csvs:
        print(f"No prediction files found in {pred_dir}")
        return

    print(f"Evaluating {len(all_csvs)} model predictions...\n")

    results = []
    for csv_file in sorted(all_csvs):
        phase = "HQ" if "/hq/" in str(csv_file) else "Silhouette"
        model_type = csv_file.parent.name
        model_name = csv_file.stem.replace("predictions_", "").replace("_", "/")

        metrics = evaluate_predictions(csv_file)

        results.append({
            'Model': model_name,
            'Type': model_type,
            'Phase': phase,
            'Total Acc': metrics['total_acc'],
            'Total Count': f"{metrics['total_count']}/{metrics['total_samples']}",
            'Family Acc': metrics['family_acc'],
            'Family Count': f"{metrics['family_count']}/{metrics['total_samples']}",
        })

    df = pd.DataFrame(results)
    df_hq = df[df['Phase'] == 'HQ'].copy()
    df_sil = df[df['Phase'] == 'Silhouette'].copy()

    # Save CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "imagenet100_evaluation_results.csv"
    df.to_csv(csv_file, index=False)

    # Generate analysis report
    generate_analysis_report(df_hq, df_sil, output_dir)

    print(f"\n✓ Results saved to: {csv_file}")
    print(f"✓ Analysis saved to: {output_dir / 'analysis_full.txt'}")


def generate_analysis_report(df_hq: pd.DataFrame, df_sil: pd.DataFrame, output_dir: Path):
    """Generate comprehensive analysis report."""
    analysis_file = output_dir / "analysis_full.txt"

    with open(analysis_file, 'w') as f:
        f.write(f"Loaded {len(df_hq)} HQ models, {len(df_sil)} silhouette models\n\n")

        # HQ Results
        f.write("=" * 120 + "\n")
        f.write("  IMAGENET100 ACCURACY — HQ IMAGES\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Model':<50s} {'Type':<12s} {'Total Acc':>12s} {'Total':>10s} {'Family Acc':>12s} {'Family':>10s}\n")
        f.write("-" * 120 + "\n")

        for _, row in df_hq.sort_values('Total Acc', ascending=False).iterrows():
            f.write(f"{row['Model']:<50s} {row['Type']:<12s} "
                   f"{row['Total Acc']:>11.2%} {row['Total Count']:>10s} "
                   f"{row['Family Acc']:>11.2%} {row['Family Count']:>10s}\n")

        # Averages by type
        f.write("\n")
        for model_type in ['proprietary', 'legacy', 'remote']:
            type_df = df_hq[df_hq['Type'] == model_type]
            if len(type_df) > 0:
                f.write(f"[AVG {model_type.upper()}] {' ' * 51} "
                       f"{type_df['Total Acc'].mean():>11.2%} {' ' * 10} "
                       f"{type_df['Family Acc'].mean():>11.2%}\n")

        # Silhouette Results
        f.write("\n" + "=" * 120 + "\n")
        f.write("  IMAGENET100 ACCURACY — SILHOUETTE IMAGES\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Model':<50s} {'Type':<12s} {'Total Acc':>12s} {'Total':>10s} {'Family Acc':>12s} {'Family':>10s}\n")
        f.write("-" * 120 + "\n")

        for _, row in df_sil.sort_values('Total Acc', ascending=False).iterrows():
            f.write(f"{row['Model']:<50s} {row['Type']:<12s} "
                   f"{row['Total Acc']:>11.2%} {row['Total Count']:>10s} "
                   f"{row['Family Acc']:>11.2%} {row['Family Count']:>10s}\n")

        f.write("\n")
        for model_type in ['proprietary', 'legacy', 'remote']:
            type_df = df_sil[df_sil['Type'] == model_type]
            if len(type_df) > 0:
                f.write(f"[AVG {model_type.upper()}] {' ' * 51} "
                       f"{type_df['Total Acc'].mean():>11.2%} {' ' * 10} "
                       f"{type_df['Family Acc'].mean():>11.2%}\n")

        # HQ vs Silhouette Comparison
        f.write("\n" + "=" * 120 + "\n")
        f.write("  HQ vs SILHOUETTE COMPARISON (Total Accuracy)\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Model':<50s} {'Type':<12s} {'HQ Acc':>12s} {'Sil Acc':>12s} {'Δ':>10s}\n")
        f.write("-" * 120 + "\n")

        comparison = df_hq.merge(df_sil, on=['Model', 'Type'], suffixes=('_hq', '_sil'))
        comparison['Delta'] = comparison['Total Acc_hq'] - comparison['Total Acc_sil']

        for _, row in comparison.sort_values('Total Acc_hq', ascending=False).iterrows():
            f.write(f"{row['Model']:<50s} {row['Type']:<12s} "
                   f"{row['Total Acc_hq']:>11.2%} {row['Total Acc_sil']:>11.2%} "
                   f"{row['Delta']:+10.2%}\n")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ImageNet100 predictions with exact and family-level metrics."
    )
    parser.add_argument(
        "--pred-csv", type=Path,
        help="Single prediction CSV to evaluate"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch evaluate all predictions in predictions_imagenet100/"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_imagenet"),
        help="Output directory for batch evaluation results"
    )
    args = parser.parse_args()

    if args.batch:
        pred_dir = Path("predictions_imagenet100")
        batch_evaluate(pred_dir, args.output_dir)
    elif args.pred_csv:
        if not args.pred_csv.exists():
            print(f"Error: {args.pred_csv} not found")
            return

        results = evaluate_predictions(args.pred_csv)

        print("\n" + "="*80)
        print("IMAGENET100 EVALUATION RESULTS")
        print("="*80)
        print(f"File: {args.pred_csv.name}\n")
        print(f"{'Metric':<30s} {'Value':>15s}")
        print("-" * 80)
        print(f"{'Total Accuracy (Exact)':<30s} {results['total_acc']:>14.2%}")
        print(f"{'Total Count':<30s} {results['total_count']:>10d} / {results['total_samples']}")
        print(f"{'Family Match Accuracy':<30s} {results['family_acc']:>14.2%}")
        print(f"{'Family Match Count':<30s} {results['family_count']:>10d} / {results['total_samples']}")
        print("="*80)


if __name__ == "__main__":
    main()