from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.gradcam import GradCAM, overlay_heatmap_on_image
from src.models.mobilenetv3_model import build_mobilenetv3_small
from src.models.nlp_model import PlantExplanationGenerator
from src.rl_agent import ThresholdRLAgent
from src.utils.io import load_json


def load_model(weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    classes = checkpoint["classes"]

    model = build_mobilenetv3_small(
        num_classes=len(classes),
        freeze_features=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes


def load_rl_agent(qtable_path: str) -> ThresholdRLAgent:
    data = load_json(qtable_path)
    agent = ThresholdRLAgent()
    agent.q_table = np.array(data["q_table"], dtype=np.float32)
    agent.epsilon = 0.0
    return agent


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def predict_image(image: Image.Image, model, classes, device: torch.device):
    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    pred_idx = int(pred_idx.cpu().item())
    conf = float(conf.cpu().item())
    pred_label = classes[pred_idx]

    topk_probs, topk_idxs = torch.topk(probs, k=min(3, len(classes)))
    topk = [
        {
            "label": classes[int(i.cpu().item())],
            "confidence": float(p.cpu().item()),
        }
        for p, i in zip(topk_probs, topk_idxs)
    ]

    return pred_label, conf, pred_idx, topk, input_tensor


# ✅ THIS IS THE MISSING FUNCTION
def run_inference(
    image_path: str,
    weights_path: str = "experiments/logs/mobilenetv3_best.pt",
    qtable_path: str = "experiments/logs/rl_qtable.json",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, classes = load_model(weights_path, device)
    rl_agent = load_rl_agent(qtable_path)
    nlp_generator = PlantExplanationGenerator()

    image = Image.open(image_path).convert("RGB")

    pred_label, conf, pred_idx, topk, input_tensor = predict_image(
        image, model, classes, device
    )

    action = rl_agent.choose_action(conf, 1, training=False)
    final_status = "ACCEPTED" if action == 1 else "REVIEW_NEEDED"

    explanation = nlp_generator.generate(pred_label, conf)

    gradcam = GradCAM(model, model.features[-1])
    heatmap = gradcam.generate(input_tensor, pred_idx)
    overlay = overlay_heatmap_on_image(image, heatmap)

    overlay_path = "experiments/results/last_inference_gradcam.png"
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    overlay.save(overlay_path)

    return {
        "predicted_label": pred_label,
        "confidence": conf,
        "rl_decision": final_status,
        "top3_predictions": topk,
        "explanation": explanation.explanation,
        "summary": explanation.summary,
        "recommendation": explanation.recommendation,
        "gradcam_path": overlay_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()

    result = run_inference(args.image_path)
    print(result)


if __name__ == "__main__":
    main()