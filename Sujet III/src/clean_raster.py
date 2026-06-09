from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def clean_rendered_plan(
    input_png: Path,
    output_png: Path,
    threshold: int = 220,
    min_component_area: int = 20,
    morph_open_kernel: int = 3,
    morph_close_kernel: int = 0,
) -> Path:
    img = cv2.imread(str(input_png), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire {input_png}")

    _, bw = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    inv = 255 - bw

    if morph_open_kernel and morph_open_kernel > 1:
        k = np.ones((morph_open_kernel, morph_open_kernel), np.uint8)
        inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k)

    if morph_close_kernel and morph_close_kernel > 1:
        k = np.ones((morph_close_kernel, morph_close_kernel), np.uint8)
        inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    cleaned = np.zeros_like(inv)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == i] = 255

    final = 255 - cleaned
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_png), final)
    return output_png
