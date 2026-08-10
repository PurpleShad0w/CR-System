from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _components(ink: np.ndarray):
    h, w = ink.shape
    visited = np.zeros_like(ink, dtype=bool)
    comps = []
    for y in range(h):
        xs = np.where(ink[y] & ~visited[y])[0]
        for x in xs:
            if visited[y, x] or not ink[y, x]:
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
                        if 0 <= nx < w and 0 <= ny < h and ink[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((nx, ny))
            xs2 = [p[0] for p in pix]
            ys2 = [p[1] for p in pix]
            comps.append({"pixels": pix, "area": len(pix), "bbox": (min(xs2), min(ys2), max(xs2), max(ys2))})
    return comps


def _inflate_bbox(bbox, ratio, w, h):
    x0, y0, x1, y1 = bbox
    m = int(max(x1 - x0, y1 - y0, 1) * ratio)
    return max(0, x0 - m), max(0, y0 - m), min(w - 1, x1 + m), min(h - 1, y1 + m)


def _union(a, b):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _touches(a, b):
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def cleanup_rendered_plan(path: str | Path, out_path: str | Path | None = None, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or {}
    path = Path(path)
    out_path = Path(out_path) if out_path else path
    img = Image.open(path).convert("L")
    arr = np.array(img)
    ink = arr < int(cfg.get("threshold", 245))
    h, w = ink.shape
    comps = _components(ink)
    if not comps:
        img.save(out_path)
        return out_path
    min_area = int(cfg.get("remove_component_area_lt", 6))
    seed_min = int(cfg.get("seed_min_area", 18))
    relevant = [c for c in comps if c["area"] >= min_area]
    seeds = [c for c in relevant if c["area"] >= seed_min]
    if not seeds:
        img.save(out_path)
        return out_path
    seed = max(seeds, key=lambda c: c["area"])
    bbox = seed["bbox"]
    kept = {id(seed)}
    for _ in range(int(cfg.get("iterative_expansion_passes", 5))):
        expanded = _inflate_bbox(bbox, float(cfg.get("main_bbox_expand_ratio", 0.14)), w, h)
        changed = False
        for comp in relevant:
            if id(comp) in kept:
                continue
            if _touches(comp["bbox"], expanded):
                bbox = _union(bbox, comp["bbox"])
                kept.add(id(comp))
                changed = True
        if not changed:
            break
    keep = np.zeros_like(ink, dtype=bool)
    for comp in relevant:
        if id(comp) in kept:
            for x, y in comp["pixels"]:
                keep[y, x] = True
    out = np.full_like(arr, 255)
    out[keep] = arr[keep]
    if bool(cfg.get("crop_to_content", True)) and keep.any():
        ys, xs = np.where(keep)
        m = int(cfg.get("crop_margin_px", 30))
        out = out[max(0, ys.min() - m):min(h, ys.max() + m + 1), max(0, xs.min() - m):min(w, xs.max() + m + 1)]
    Image.fromarray(out).save(out_path)
    return out_path
