# Ethics Statement

## Risks

### 1. Misdiagnosis
The model may incorrectly classify plant diseases, which could lead to improper treatment.

### 2. Over-Reliance on AI
Users may trust predictions without validation from experts.

### 3. Dataset Bias
The PlantVillage dataset contains controlled images that may not reflect real-world variability.

---

## Mitigations

- Provide confidence scores for all predictions
- Use RL decision layer to filter risky predictions
- Clearly state that the system is assistive only

---

## Intended Use
- Educational tool
- Decision-support system for plant disease identification

---

## Limitations
- Cannot detect internal or chemical plant conditions
- Performance may decrease in varying lighting or backgrounds

---

## Privacy
- No personal or sensitive data is collected
- Dataset contains only plant images

---

## Fairness
- Model trained on limited dataset
- May not generalize equally across all plant conditions

---

## Misuse Considerations
- Should not be used as sole basis for agricultural decisions
- Not suitable for medical or professional diagnosis