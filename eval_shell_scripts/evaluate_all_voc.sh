#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

for dir in predictions/voc/hq/legacy predictions/voc/hq/remote predictions/voc/hq/proprietary predictions/voc/hq; do
    if [ -d "$dir" ]; then
        for file in "$dir"/*.csv; do
            [ -f "$file" ] || continue
            echo "═══════════════════════════════════════════════════"
            echo "  $file"
            PYTHONPATH=src python src/scripts/eval_predictions_voc.py --pred-csv "$file"
        done
    fi
done

for dir in predictions/voc/silhouette/legacy predictions/voc/silhouette/remote predictions/voc/silhouette/proprietary predictions/voc/silhouette; do
    if [ -d "$dir" ]; then
        for file in "$dir"/*.csv; do
            [ -f "$file" ] || continue
            echo "═══════════════════════════════════════════════════"
            echo "  $file"
            PYTHONPATH=src python src/scripts/eval_predictions_voc.py --pred-csv "$file"
        done
    fi
done
