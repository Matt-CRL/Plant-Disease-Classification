from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.models.nlp_model import PlantExplanationGenerator
from src.utils.io import save_json


def build_nlp_dataset() -> Tuple[List[str], List[str]]:
    pepper_texts = [
        "my pepper plant has spots",
        "bell pepper leaves are turning yellow",
        "capsicum leaf has bacterial spot",
        "my siling plant looks sick",
        "pepper leaf has dark patches",
        "the bell pepper plant is healthy",
        "green pepper leaves have damage",
        "pepper plant has leaf disease",
        "capsicum leaves are curling",
        "my pepper crop has infected leaves",
        "pepper leaf has holes and spots",
        "bell pepper plant has brown lesions",
        "capsicum plant looks weak",
        "siling leaves are unhealthy",
        "pepper leaf is damaged",
    ]

    potato_texts = [
        "my potato plant has blight",
        "potato leaves have brown spots",
        "the patatas plant looks sick",
        "potato crop has leaf infection",
        "potato leaf is turning yellow",
        "my spud plant has disease",
        "potato leaves are drying",
        "patatas leaf has dark lesions",
        "potato plant is healthy",
        "potato leaf has fungus signs",
        "spud leaves have many spots",
        "potato crop looks unhealthy",
        "potato leaves are wilting",
        "patatas plant has early blight",
        "potato leaf is damaged",
    ]

    tomato_texts = [
        "my tomato plant has leaf mold",
        "tomato leaves have yellow spots",
        "kamatis plant looks sick",
        "my tomato crop has bacterial spot",
        "tomato leaf is curling",
        "the tomato plant is healthy",
        "kamatis leaves are infected",
        "tomato leaf has blight",
        "cherry tomato plant has disease",
        "tomato leaves are turning brown",
        "kamatis leaf has lesions",
        "tomato plant looks unhealthy",
        "tomato crop has virus symptoms",
        "cherry tomato leaves have spots",
        "tomato leaf is damaged",
    ]

    texts = pepper_texts + potato_texts + tomato_texts
    labels = (
        ["pepper"] * len(pepper_texts)
        + ["potato"] * len(potato_texts)
        + ["tomato"] * len(tomato_texts)
    )
    return texts, labels


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def train_text_classifier(random_state: int = 42) -> Dict:
    texts, labels = build_nlp_dataset()
    cleaned_texts = [clean_text(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts,
        labels,
        test_size=0.30,
        random_state=random_state,
        stratify=labels,
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100.0
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    labels_order = ["pepper", "potato", "tomato"]
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    return {
        "model": pipeline,
        "metrics": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "labels_order": labels_order,
            "confusion_matrix": cm.tolist(),
            "test_size": len(y_test),
            "random_state": random_state,
        },
    }


def evaluate_match_logic() -> Dict:
    """
    Small auxiliary evaluation showing how an NLP-derived plant genus
    can be matched against the CNN predicted label genus.
    """
    samples = [
        ("my tomato plant has yellow spots", "Tomato_Early_blight", 1),
        ("kamatis leaves are infected", "Tomato_Late_blight", 1),
        ("my potato leaf has brown lesions", "Potato___Early_blight", 1),
        ("the bell pepper plant has spots", "Pepper__bell___Bacterial_spot", 1),
        ("my tomato plant has yellow spots", "Potato___Early_blight", 0),
        ("potato leaf is damaged", "Tomato_healthy", 0),
        ("pepper plant has disease", "Tomato_Bacterial_spot", 0),
        ("kamatis leaf has lesions", "Pepper__bell___healthy", 0),
    ]

    genus_keywords = {
        "pepper": ["pepper", "capsicum", "siling", "bell pepper"],
        "potato": ["potato", "patatas", "spud"],
        "tomato": ["tomato", "kamatis", "cherry tomato"],
    }

    def infer_genus(text: str) -> str:
        t = clean_text(text)
        for genus, words in genus_keywords.items():
            for word in words:
                if clean_text(word) in t:
                    return genus
        return "unknown"

    y_true: List[int] = []
    y_pred: List[int] = []

    for text, predicted_label, expected in samples:
        inferred = infer_genus(text)
        predicted_genus = predicted_label.split("_")[0].lower()
        match = 1 if inferred == predicted_genus else 0
        y_true.append(expected)
        y_pred.append(match)

    acc = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "num_samples": len(samples),
    }


def build_example_explanations() -> List[Dict]:
    generator = PlantExplanationGenerator()
    examples = [
        ("Tomato_Early_blight", 0.92),
        ("Potato___healthy", 0.88),
        ("Pepper__bell___Bacterial_spot", 0.95),
    ]

    outputs: List[Dict] = []
    for label, conf in examples:
        exp = generator.generate(label, conf)
        outputs.append(
            {
                "predicted_label": exp.predicted_label,
                "confidence": exp.confidence,
                "explanation": exp.explanation,
                "summary": exp.summary,
                "recommendation": exp.recommendation,
            }
        )
    return outputs


def main() -> None:
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    clf_result = train_text_classifier(random_state=42)
    match_metrics = evaluate_match_logic()
    explanations = build_example_explanations()

    payload = {
        "component": "auxiliary_nlp_text_classifier_and_explainer",
        "task": "plant genus text classification + explanation generation",
        "classification_metrics": clf_result["metrics"],
        "cnn_match_metrics": match_metrics,
        "example_explanations": explanations,
    }

    save_json(payload, results_dir / "nlp_metrics.json")

    print("Saved NLP metrics to experiments/results/nlp_metrics.json")
    print(f"NLP Accuracy: {clf_result['metrics']['accuracy']:.2f}%")
    print(f"NLP Macro-F1: {clf_result['metrics']['macro_f1']:.4f}")
    print(f"Match Accuracy: {match_metrics['accuracy']:.2f}%")
    print(f"Match Macro-F1: {match_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()