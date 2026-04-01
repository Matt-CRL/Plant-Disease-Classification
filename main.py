from __future__ import annotations

import tempfile

import gradio as gr
from PIL import Image

from src.infer import run_inference


def predict_app(image):
    if image is None:
        return "No image uploaded.", "", "", None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        result = run_inference(tmp.name)

    main_text = "\n".join([
        f"Predicted Label: {result['predicted_label']}",
        f"Confidence: {result['confidence']:.4f}",
        f"RL Decision: {result['rl_decision']}",
        "",
        result["explanation"],
    ])

    extra_text = "\n".join([
        result["summary"],
        "",

        "",
        "Top-3 Predictions:",
        *[f"- {item['label']}: {item['confidence']:.4f}" for item in result["top3_predictions"]],
    ])

    gradcam_image = Image.open(result["gradcam_path"])
    return main_text, extra_text, result["recommendation"], gradcam_image


demo = gr.Interface(
    fn=predict_app,
    inputs=gr.Image(type="pil", label="Leaf Image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Explanation"),
        gr.Textbox(label="Recommendation"),
        gr.Image(type="pil", label="Grad-CAM"),
    ],
    title="Plant Disease Classification System",
    description="MobileNetV3 + Lightweight NLP Explanation + RL Safety Decision",
)

if __name__ == "__main__":
    demo.launch()