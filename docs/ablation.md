# Ablation Study

## Ablation 1: CNN vs MobileNetV3
- CNN (from scratch): ~38–45% accuracy
- MobileNetV3: ~45–60% accuracy

Conclusion:
Transfer learning significantly improves performance compared to training from scratch.

---

## Ablation 2: Frozen vs Fine-tuned MobileNetV3
- Frozen features: stable but lower accuracy
- Fine-tuned: higher accuracy but more variance

Conclusion:
Fine-tuning improves performance but requires careful learning rate selection.

---

## Ablation 3: With vs Without RL Decision
- Without RL: all predictions accepted
- With RL: low-confidence predictions flagged

Conclusion:
RL improves decision reliability by reducing incorrect high-risk predictions.