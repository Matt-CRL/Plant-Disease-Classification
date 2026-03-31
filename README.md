# Plant-Disease-Classification (CNN + NLP + RL)
6INTELSYS Final Project
#
**Team Members:**
- Matt Lara - Project Lead / Integration
- Javie Alfaro - Modeling Lead
- Joalnes Vender - Data & Ethics Lead

## 📌 Overview
This project implements a hybrid AI system for plant disease classification using:
- **CNN (MobileNetV3)** for image classification
- **NLP module** for plant-genus validation and explanation generation
- **Reinforcement Learning (RL)** for decision thresholding (accept/reject predictions)

The system classifies **15 plant disease classes** from the PlantVillage dataset.

---

## 🎯 Objectives
- Accurately classify plant diseases from leaf images
- Improve reliability using NLP validation
- Enhance decision safety using RL

---

## 📊 Results Summary

### Vision Models
| Model | Accuracy | Macro-F1 |
|------|--------|---------|
| Logistic Regression | 68.27% | 0.6406 |
| Scratch CNN | 91.92% | 0.9009 |
| MobileNetV3 | **99.32%** | **0.9919** |

### NLP
- Accuracy: 92.86%
- Macro-F1: 0.9259

### RL
- Mean success rate: ~0.99
- Trained over 3 seeds
- Learning curves stored in `experiments/results/`

### Runtime
- Total pipeline runtime: **14.12 minutes**
- Within required ≤90 minutes

---

# Data

This project uses the PlantVillage dataset downloaded via KaggleHub.

The dataset is not committed to the repository.
Run:

```bash
python data/get_data.py


