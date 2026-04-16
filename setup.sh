#!/bin/bash
# run this once after cloning, before submitting any slurm jobs
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs data/raw data/sft eval checkpoints
pip install -r requirements.txt
python scripts/download_data.py
echo "ready"
