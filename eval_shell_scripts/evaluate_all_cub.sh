#!/bin/bash
cd "$(dirname "$0")/.."
echo "============================================"
echo "  CUB-200 Evaluation — HQ images"
echo "============================================"
for file in predictions/cub/hq/legacy/*.csv; do
    echo ""
    echo ">>> $(basename "$file")"
    PYTHONPATH=src python src/scripts/eval_predictions_cub.py --pred-csv "$file"
done

echo ""
echo "============================================"
echo "  CUB-200 Evaluation — Silhouette images"
echo "============================================"
for file in predictions/cub/silhouette/legacy/*.csv; do
    echo ""
    echo ">>> $(basename "$file")"
    PYTHONPATH=src python src/scripts/eval_predictions_cub.py --pred-csv "$file"
done
