#!/usr/bin/env bash
set -euo pipefail

SECONDS=0

mkdir -p experiments/results
mkdir -p experiments/logs

echo "1) Downloading dataset..."
python data/get_data.py

echo "2) Running simple non-DL baseline..."
python -m src.baselines

echo "3) Training small CNN from scratch..."
python -m src.train --model_name scratch --epochs 5 --lr 0.001 --patience 3

echo "4) Evaluating small CNN..."
python -m src.eval --model_name scratch --weights_path experiments/logs/scratch_best.pt --split val

echo "5) Training MobileNetV3..."
python -m src.train --model_name mobilenetv3 --epochs 12 --lr 0.0003 --patience 4

echo "6) Evaluating MobileNetV3..."
python -m src.eval --model_name mobilenetv3 --weights_path experiments/logs/mobilenetv3_best.pt --split val

echo "7) Running NLP component..."
python -m src.run_nlp

echo "8) Training RL component from validation predictions..."
python -m src.rl_agent --predictions_path experiments/results/mobilenetv3_val_predictions.json --episodes 1000

echo "9) Building Question 2 summary..."
python -m src.summarize_results

RUNTIME_SECONDS=$SECONDS
RUNTIME_MINUTES=$(python - <<PY
secs = ${RUNTIME_SECONDS}
print(round(secs / 60.0, 2))
PY
)

python - <<PY
import json
from pathlib import Path

runtime_seconds = ${RUNTIME_SECONDS}
runtime_minutes = float(${RUNTIME_MINUTES})

payload = {
    "total_runtime_seconds": runtime_seconds,
    "total_runtime_minutes": runtime_minutes,
    "within_90_min_limit": runtime_minutes <= 90.0
}

path = Path("experiments/results/pipeline_runtime.json")
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print("Saved pipeline runtime to experiments/results/pipeline_runtime.json")
print(f"Total runtime: {runtime_minutes:.2f} minutes")
print(f"Within 90-minute limit: {runtime_minutes <= 90.0}")
PY

echo "10) Refreshing Question 2 summary with runtime..."
python -m src.summarize_results

echo "Done."