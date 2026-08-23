"""Sanity check for the project environment.

Run it after setting up the venv (or from PyCharm with the 'check_env' configuration)
to confirm OpenCV, NumPy and matplotlib are importable and that every sample image
in the project actually decodes.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".avif"}


def main() -> int:
    print(f"python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"opencv      {cv2.__version__}")
    print(f"numpy       {np.__version__}")
    print(f"matplotlib  {matplotlib.__version__}")
    print()

    images = sorted(
        candidate_path
        for candidate_path in DATA_DIR.rglob("*")
        if candidate_path.suffix.lower() in SUFFIXES
        and ".ipynb_checkpoints" not in candidate_path.parts
    )
    failed_images = []
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        relative_path = image_path.relative_to(PROJECT_ROOT)
        if image is None:
            failed_images.append(str(relative_path))
            print(f"  FAIL  {relative_path}  — cv2.imread returned None")
        else:
            image_height, image_width = image.shape[:2]
            print(f"  ok    {str(relative_path):38} {image_width}x{image_height}  {image.dtype}")

    print()
    if failed_images:
        print(f"{len(failed_images)} image(s) could not be decoded: {', '.join(failed_images)}")
        return 1
    print(f"All {len(images)} images decoded fine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
