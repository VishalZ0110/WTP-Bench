#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

for dir in predictions/DIS5K/hq/legacy predictions/DIS5K/hq/remote predictions/DIS5K/hq/proprietary predictions/DIS5K/hq; do
    if [ -d "$dir" ]; then
        for file in "$dir"/*.csv; do
            [ -f "$file" ] || continue
            echo "═══════════════════════════════════════════════════"
            echo "  $file"
            PYTHONPATH=src python src/scripts/eval_predictions_dis5k.py --pred-csv "$file"
        done
    fi
done

for dir in predictions/DIS5K/silhouette/legacy predictions/DIS5K/silhouette/remote predictions/DIS5K/silhouette/proprietary predictions/DIS5K/silhouette; do
    if [ -d "$dir" ]; then
        for file in "$dir"/*.csv; do
            [ -f "$file" ] || continue
            echo "═══════════════════════════════════════════════════"
            echo "  $file"
            PYTHONPATH=src python src/scripts/eval_predictions_dis5k.py --pred-csv "$file"
        done
    fi
done
