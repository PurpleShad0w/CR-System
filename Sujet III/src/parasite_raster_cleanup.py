from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _as_bool_mask(img_l: Image.Image, threshold: int) -> np.ndarray:
    return np.array(img_l) < threshold


def _runs_1d(values: np.ndarray):
    n = len(values)
    i = 0
    while i < n:
        if not values[i]:
            i += 1
            continue
        j = i + 1
        while j < n and values[j]:
            j += 1
        yield i, j
        i = j


def _local_perp_support_horizontal(mask: np.ndarray, y: int, x0: int, x1: int, radius: int) -> np.ndarray:
    h, _ = mask.shape
    ya = max(0, y - radius)
    yb = min(h, y + radius + 1)
    return mask[ya:yb, x0:x1].sum(axis=0)


def _local_perp_support_vertical(mask: np.ndarray, x: int, y0: int, y1: int, radius: int) -> np.ndarray:
    _, w = mask.shape
    xa = max(0, x - radius)
    xb = min(w, x + radius + 1)
    return mask[y0:y1, xa:xb].sum(axis=1)


def _local_density(mask: np.ndarray, y: int, x0: int, x1: int, radius: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros(x1 - x0, dtype=np.int32)
    for k, x in enumerate(range(x0, x1)):
        ya = max(0, y - radius)
        yb = min(h, y + radius + 1)
        xa = max(0, x - radius)
        xb = min(w, x + radius + 1)
        out[k] = int(mask[ya:yb, xa:xb].sum())
    return out


def _local_density_vertical(mask: np.ndarray, x: int, y0: int, y1: int, radius: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros(y1 - y0, dtype=np.int32)
    for k, y in enumerate(range(y0, y1)):
        ya = max(0, y - radius)
        yb = min(h, y + radius + 1)
        xa = max(0, x - radius)
        xb = min(w, x + radius + 1)
        out[k] = int(mask[ya:yb, xa:xb].sum())
    return out


def _remove_long_hairlines(mask: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    h, w = mask.shape
    max_dim = max(h, w)
    min_run = max(int(cfg.get("min_run_length_px", 34)), int(max_dim * float(cfg.get("min_run_length_ratio", 0.018))))
    support_radius = int(cfg.get("support_radius_px", 3))
    max_mean = float(cfg.get("max_mean_perpendicular_support", 2.35))
    max_median = float(cfg.get("max_median_perpendicular_support", 2.0))
    density_radius = int(cfg.get("density_radius_px", 10))
    dense_min = int(cfg.get("min_dense_neighborhood_ink", 32))

    remove = np.zeros_like(mask, dtype=bool)

    if bool(cfg.get("remove_long_horizontal", True)):
        for y in range(h):
            for x0, x1 in _runs_1d(mask[y, :]):
                if x1 - x0 < min_run:
                    continue
                support = _local_perp_support_horizontal(mask, y, x0, x1, support_radius)
                # A parasite line is long, thin and weakly supported in perpendicular direction.
                candidate = (support <= max_median)
                if float(support.mean()) <= max_mean:
                    density = _local_density(mask, y, x0, x1, density_radius)
                    candidate &= density < dense_min
                    remove[y, x0:x1] |= candidate

    if bool(cfg.get("remove_long_vertical", True)):
        for x in range(w):
            for y0, y1 in _runs_1d(mask[:, x]):
                if y1 - y0 < min_run:
                    continue
                support = _local_perp_support_vertical(mask, x, y0, y1, support_radius)
                candidate = (support <= max_median)
                if float(support.mean()) <= max_mean:
                    density = _local_density_vertical(mask, x, y0, y1, density_radius)
                    candidate &= density < dense_min
                    remove[y0:y1, x] |= candidate

    cleaned = mask.copy()
    cleaned[remove] = False
    return cleaned


def _components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for y in range(h):
        xs = np.where(mask[y] & ~visited[y])[0]
        for x in xs:
            if visited[y, x] or not mask[y, x]:
                continue
            q = deque([(x, y)])
            visited[y, x] = True
            pix = []
            while q:
                cx, cy = q.popleft()
                pix.append((cx, cy))
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        if nx == cx and ny == cy:
                            continue
                        if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((nx, ny))
            comps.append(pix)
    return comps


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    keep = np.zeros_like(mask, dtype=bool)
    for comp in _components(mask):
        if len(comp) >= min_area:
            for x, y in comp:
                keep[y, x] = True
    return keep


def _crop_to_content(img: Image.Image, mask: np.ndarray, margin: int) -> Image.Image:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img
    w, h = img.size
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w, int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h, int(ys.max()) + margin + 1)
    return img.crop((x0, y0, x1, y1))


def clean_parasite_lines_from_image(input_png: str | Path, output_png: str | Path, cfg: dict[str, Any] | None = None) -> Path:
    """Remove long isolated parasite lines while preserving architectural details.

    Intended use: run on the clean 2D rendering before the shadow-only 2.25D pass.
    This is conservative by design: it targets long, thin, weakly-supported horizontal/vertical lines,
    not small doors/windows/furniture-like details.
    """
    cfg = cfg or {}
    input_png = Path(input_png)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    img_l = Image.open(input_png).convert("L")
    arr = np.array(img_l)
    threshold = int(cfg.get("threshold", 245))
    mask = _as_bool_mask(img_l, threshold)

    cleaned_mask = _remove_long_hairlines(mask, cfg)
    if bool(cfg.get("remove_small_components", True)):
        cleaned_mask = _remove_small_components(cleaned_mask, int(cfg.get("min_component_area_px", 12)))

    out = np.full_like(arr, 255)
    out[cleaned_mask] = arr[cleaned_mask]
    out_img = Image.fromarray(out, mode="L").convert("RGB")
    if bool(cfg.get("crop_to_content", True)):
        out_img = _crop_to_content(out_img, cleaned_mask, int(cfg.get("crop_margin_px", 32)))
    out_img.save(output_png)
    return output_png
