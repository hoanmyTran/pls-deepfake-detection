#!/usr/bin/env bash
set -euo pipefail
CHECKPOINT="${CHECKPOINT:-results/pls_xlsr300m/model_weights.pt}"
python evaluate_datasets.py --config config.yaml --datasets-file datasets.yaml --checkpoint "$CHECKPOINT" "$@"
