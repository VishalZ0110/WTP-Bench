#!/bin/bash
cd "$(dirname "$0")/.."
for file in predictions/hq/legacy/*.csv; do
    echo "Evaluating $file"
    PYTHONPATH=src python src/scripts/eval_predictions.py --pred-csv $file
done

for file in predictions/silhouette/legacy/*.csv; do
    echo "Evaluating $file"
    PYTHONPATH=src python src/scripts/eval_predictions.py --pred-csv $file
done