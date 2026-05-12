#!/bin/bash
set -euo pipefail

# === SETTINGS ===
DATA_DIR="./data"

# === Step 1: Generate label files ===
echo "[→] Running make_combined.py ..."
python "${DATA_DIR}/make_combined.py"
echo ""

# === Step 2: Copy matching images ===
echo "[→] Running data_selection.py ..."
python "${DATA_DIR}/data_selection.py"
echo ""

echo "[✓] Data preparation complete."