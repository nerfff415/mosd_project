#!/usr/bin/env bash
# Full pipeline. Usage: ./run_all.sh [config]   (default: configs/main.yaml; pilot: configs/pilot.yaml)
set -euo pipefail
CONFIG="${1:-configs/main.yaml}"
PY="${PYTHON:-.venv/bin/python}"
"$PY" -m tsi.data --config "$CONFIG"
"$PY" -m tsi.masks --config "$CONFIG"
"$PY" -m tsi.kalman --config "$CONFIG"
"$PY" -m tsi.chronos_imputer --config "$CONFIG"
"$PY" -m tsi.csdi_imputer --config "$CONFIG"
"$PY" -m tsi.evaluate --config "$CONFIG"
