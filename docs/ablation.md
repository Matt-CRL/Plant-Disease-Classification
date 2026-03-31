# Ablation Study

## Objective
The goal of this ablation study is to evaluate how different hyperparameter configurations affect the performance of the MobileNetV3 model for plant disease classification.

---

## Experiment Setup
- Model: MobileNetV3 (pretrained)
- Dataset: PlantVillage (15 classes)
- Input size: 224x224
- Optimizer: Adam
- Batch size: 32
- Evaluation metrics: Accuracy, Macro-F1

Two experiments were conducted by varying the learning rate.

---

## Ablation Experiments

### 1. Learning Rate = 0.001
- Faster convergence during early epochs
- Slightly less stable validation performance
- Occasional fluctuations in training loss

### 2. Learning Rate = 0.0003 (Baseline Configuration)
- Slower but more stable convergence
- Higher final validation accuracy
- Better generalization performance

---

## Results Summary

| Configuration | Learning Rate | Observation |
|--------------|-------------|------------|
| Ablation 1 | 0.001 | Faster training but less stable |
| Ablation 2 | 0.0003 | More stable, better performance |

---

## Analysis

The higher learning rate (0.001) allowed the model to learn quickly but introduced instability during training. In contrast, the lower learning rate (0.0003) resulted in smoother convergence and better validation performance.

This indicates that a smaller learning rate is more suitable for fine-tuning pretrained models such as MobileNetV3.

---

## Conclusion

The ablation study demonstrates that:
- Learning rate significantly impacts training stability
- Lower learning rates improve generalization in this task

Thus, the final model uses a learning rate of **0.0003**.