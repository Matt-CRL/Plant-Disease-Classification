
---

# 📄 2. Model Card

```md
# Plant Disease Classification System

## Model Overview
This system combines:
- CNN (MobileNetV3) for image classification
- NLP module for plant-genus classification and explanation
- RL agent for decision thresholding

## Intended Use
- Assist users in identifying plant diseases
- Educational and decision-support tool
- NOT for professional agricultural diagnosis

## Dataset
- PlantVillage dataset (Kaggle)
- 15 plant disease classes
- Images of plant leaves under controlled conditions

## Metrics

### Vision Models
| Model | Accuracy | Macro-F1 |
|------|--------|---------|
| Logistic Regression | 68.27% | 0.6406 |
| Scratch CNN | 91.92% | 0.9009 |
| MobileNetV3 | 99.32% | 0.9919 |

### NLP
- Accuracy: 92.86%
- Macro-F1: 0.9259

### RL
- Mean success rate: ~0.99
- 3-seed training for stability

## Evaluation
- Confusion matrices available in `experiments/results/`
- Ablation studies included (learning rate variations)

## Limitations
- May not generalize to real-world environments
- Cannot detect unseen diseases
- Limited to known plant disease classes
- Sensitive to image quality and lighting

## Ethical Considerations
- Predictions may be incorrect
- Should not replace expert advice
- Users must interpret results cautiously

## Deployment
- Runs locally using PyTorch
- Designed for local inference using PyTorch
- Full pipeline reproducible via `run.sh`