from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.utils.io import load_json, save_json


def safe_load(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"missing": True, "path": path}
    return load_json(p)


def main() -> None:
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "question_2_summary": {
            "vision_non_dl_baseline": safe_load("experiments/results/logreg_val_metrics.json"),
            "vision_scratch_cnn": safe_load("experiments/results/scratch_val_metrics.json"),
            "vision_mobilenetv3": safe_load("experiments/results/mobilenetv3_val_metrics.json"),
            "nlp_component": safe_load("experiments/results/nlp_metrics.json"),
            "rl_component": safe_load("experiments/results/rl_metrics.json"),
            "pipeline_runtime": safe_load("experiments/results/pipeline_runtime.json"),
        }
    }

    save_json(summary, results_dir / "q2_summary.json")
    print("Saved Question 2 summary to experiments/results/q2_summary.json")


if __name__ == "__main__":
    main()