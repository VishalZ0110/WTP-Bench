#!/bin/bash
cd "$(dirname "$0")/.."
echo "============================================"
echo "  ThinObject5K Evaluation — HQ images"
echo "============================================"
for dir in predictions/thin/hq/legacy predictions/thin/hq/remote predictions/thin/hq/proprietary predictions/thin/hq; do
    if [ -d "$dir" ]; then
        for file in "$dir"/*.csv; do
            [ -f "$file" ] || continue
            echo ""
            echo ">>> $(basename "$file")"
            PYTHONPATH=src python src/scripts/eval_predictions_thin.py --pred-csv "$file"
        done
    fi
done

echo ""
echo "============================================"
echo "  ThinObject5K Evaluation — Silhouette images"
echo "============================================"
for dir in predictions/thin/silhouette/legacy predictions/thin/silhouette/remote predictions/thin/silhouette/proprietary predictions/thin/silhouette; do
    if [ -d "$dir" ]; then
        for file in "$dir"/*.csv; do
            [ -f "$file" ] || continue
            echo ""
            echo ">>> $(basename "$file")"
            PYTHONPATH=src python src/scripts/eval_predictions_thin.py --pred-csv "$file"
        done
    fi
done
