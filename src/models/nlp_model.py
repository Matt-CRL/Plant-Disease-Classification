from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictionExplanation:
    predicted_label: str
    confidence: float
    explanation: str
    summary: str
    recommendation: str


class PlantExplanationGenerator:
    """
    Lightweight NLP component for turning model predictions into
    simple, user-friendly text explanations.
    """

    FRIENDLY_LABELS = {
        "Pepper__bell___Bacterial_spot": "Bell Pepper Bacterial Spot",
        "Pepper__bell___healthy": "Healthy Bell Pepper",
        "Potato___Early_blight": "Potato Early Blight",
        "Potato___healthy": "Healthy Potato",
        "Potato___Late_blight": "Potato Late Blight",
        "Tomato_Bacterial_spot": "Tomato Bacterial Spot",
        "Tomato_Early_blight": "Tomato Early Blight",
        "Tomato_healthy": "Healthy Tomato",
        "Tomato_Late_blight": "Tomato Late Blight",
        "Tomato_Leaf_Mold": "Tomato Leaf Mold",
        "Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Spider Mites",
        "Tomato_Target_Spot": "Tomato Target Spot",
        "Tomato_Tomato_YellowLeaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
        "Tomato_Tomato_mosaic_virus": "Tomato Mosaic Virus",
    }

    SHORT_DESCRIPTIONS = {
        "Pepper__bell___Bacterial_spot": "The leaf shows signs commonly associated with bacterial spot in bell pepper.",
        "Pepper__bell___healthy": "The leaf appears healthy and shows no strong signs of visible disease.",
        "Potato___Early_blight": "The leaf shows visible patterns consistent with early blight in potato.",
        "Potato___healthy": "The leaf appears healthy and does not show strong visible disease symptoms.",
        "Potato___Late_blight": "The leaf shows visible symptoms consistent with late blight in potato.",
        "Tomato_Bacterial_spot": "The leaf shows symptoms consistent with bacterial spot in tomato.",
        "Tomato_Early_blight": "The leaf has visible signs commonly associated with tomato early blight.",
        "Tomato_healthy": "The tomato leaf appears healthy based on its visible surface condition.",
        "Tomato_Late_blight": "The tomato leaf shows symptoms that match late blight patterns.",
        "Tomato_Leaf_Mold": "The leaf shows visible characteristics associated with tomato leaf mold.",
        "Tomato_Septoria_leaf_spot": "The tomato leaf shows patterns consistent with Septoria leaf spot.",
        "Tomato_Spider_mites_Two_spotted_spider_mite": "The leaf shows visible damage associated with spider mites.",
        "Tomato_Target_Spot": "The leaf shows circular damage patterns consistent with target spot.",
        "Tomato_Tomato_YellowLeaf_Curl_Virus": "The leaf shows visible symptoms associated with yellow leaf curl virus.",
        "Tomato_Tomato_mosaic_virus": "The leaf shows visible symptoms associated with mosaic virus.",
    }

    RECOMMENDATIONS = {
        "healthy": "Recommendation: The plant appears healthy. Continue regular monitoring and proper care.",
        "diseased": "Recommendation: This result should be used as decision support only. Consider checking the plant further or consulting an agricultural expert before treatment.",
    }

    def get_friendly_label(self, predicted_label: str) -> str:
        return self.FRIENDLY_LABELS.get(predicted_label, predicted_label.replace("_", " "))

    def get_description(self, predicted_label: str) -> str:
        return self.SHORT_DESCRIPTIONS.get(
            predicted_label,
            "The image shows visible patterns that the model associates with this class."
        )

    def is_healthy(self, predicted_label: str) -> bool:
        return "healthy" in predicted_label.lower()

    def generate(self, predicted_label: str, confidence: float) -> PredictionExplanation:
        friendly_label = self.get_friendly_label(predicted_label)
        description = self.get_description(predicted_label)
        confidence_pct = confidence * 100.0

        explanation = f"Predicted disease: {friendly_label} with {confidence_pct:.2f}% confidence."
        summary = description

        if self.is_healthy(predicted_label):
            recommendation = self.RECOMMENDATIONS["healthy"]
        else:
            recommendation = self.RECOMMENDATIONS["diseased"]

        return PredictionExplanation(
            predicted_label=predicted_label,
            confidence=confidence,
            explanation=explanation,
            summary=summary,
            recommendation=recommendation,
        )