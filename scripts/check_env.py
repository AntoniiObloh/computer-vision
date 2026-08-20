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

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".avif"}


def main() -> int:
    print(f"python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"opencv      {cv2.__version__}")
    print(f"numpy       {np.__version__}")
    print(f"matplotlib  {matplotlib.__version__}")
    print()

    images = sorted(
        p
        for p in DATA_DIR.rglob("*")
        if p.suffix.lower() in SUFFIXES
        and ".ipynb_checkpoints" not in p.parts
    )
    failed = []
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        rel = path.relative_to(ROOT)
        if img is None:
            failed.append(str(rel))
            print(f"  FAIL  {rel}  — cv2.imread returned None")
        else:
            h, w = img.shape[:2]
            print(f"  ok    {str(rel):38} {w}x{h}  {img.dtype}")

    print()
    if failed:
        print(f"{len(failed)} image(s) could not be decoded: {', '.join(failed)}")
        return 1
    print(f"All {len(images)} images decoded fine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
