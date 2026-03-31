#!/usr/bin/env bash
set -e

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

echo "7) Training NLP component..."
python -m src.run_nlp

echo "8) Training RL component from validation predictions..."
python -m src.rl_agent --predictions_path experiments/results/mobilenetv3_val_predictions.json

echo "Done."