from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    target_dir = data_dir / "PlantVillage"

    data_dir.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        print(f"Dataset already exists at: {target_dir}")
        return

    print("Downloading PlantVillage dataset from KaggleHub...")
    downloaded_path = Path(kagglehub.dataset_download("emmarex/plantdisease"))

    # Find the correct PlantVillage folder safely
    candidate_1 = downloaded_path / "PlantVillage"
    candidate_2 = downloaded_path

    if candidate_1.exists() and any(candidate_1.iterdir()):
        source_dir = candidate_1
    elif candidate_2.exists() and any(candidate_2.iterdir()):
        source_dir = candidate_2
    else:
        raise FileNotFoundError("Could not locate the PlantVillage dataset folder.")

    # If downloaded_path itself already contains a nested PlantVillage,
    # normalize by copying only the actual class folder level
    nested = source_dir / "PlantVillage"
    if nested.exists() and any(nested.iterdir()):
        source_dir = nested

    print(f"Copying dataset to: {target_dir}")
    shutil.copytree(source_dir, target_dir)
    print("Dataset ready.")


if __name__ == "__main__":
    main()