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
            comps.append({
                "pixels": pix,
                "area": len(pix),
                "bbox": (min(xs2), min(ys2), max(xs2), max(ys2)),
                "center": ((min(xs2) + max(xs2)) / 2.0, (min(ys2) + max(ys2)) / 2.0),
            })
    return comps


def _inflate_bbox(bbox, ratio, img_w, img_h):
    x0, y0, x1, y1 = bbox
    dx = max(x1 - x0, 1)
    dy = max(y1 - y0, 1)
    m = int(max(dx, dy) * ratio)
    return max(0, x0 - m), max(0, y0 - m), min(img_w - 1, x1 + m), min(img_h - 1, y1 + m)


def _union_bbox(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)


def _touches_or_inside(comp_bbox, bbox):
    x0, y0, x1, y1 = bbox
    cx0, cy0, cx1, cy1 = comp_bbox
    return not (cx1 < x0 or cx0 > x1 or cy1 < y0 or cy0 > y1)


def cleanup_main_plan_image(image_path: str | Path, out_path: str | Path | None = None, rules: dict[str, Any] | None = None) -> Path:
    """Nettoyage post-rendu calibré pour conserver le rendu v1.

    Au lieu de rerouter toute la logique CAD, ce post-process retire les groupes graphiques très éloignés
    du plan principal : barres verticales, cartouches isolés, petits traits à droite/haut, etc.
    """
    rules = rules or {}
    cfg = rules.get("post_cleanup", rules) or {}
    image_path = Path(image_path)
    out_path = Path(out_path) if out_path else image_path

    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    threshold = int(cfg.get("threshold", 245))
    ink = arr < threshold
    h, w = ink.shape

    comps = _components(ink)
    if not comps:
        img.save(out_path)
        return out_path

    min_area = int(cfg.get("remove_component_area_lt", 8))
    seed_min_area = int(cfg.get("seed_min_area", 25))
    relevant = [c for c in comps if c["area"] >= min_area]
    seeds = [c for c in relevant if c["area"] >= seed_min_area]
    if not seeds:
        img.save(out_path)
        return out_path

    # Le bâtiment principal du v1 est généralement le groupe qui contient le plus d'encre.
    seed = max(seeds, key=lambda c: c["area"])
    main_bbox = seed["bbox"]
    kept = {id(seed)}

    ratio = float(cfg.get("main_bbox_expand_ratio", 0.12))
    passes = int(cfg.get("iterative_expansion_passes", 4))
    for _ in range(max(1, passes)):
        expanded = _inflate_bbox(main_bbox, ratio, w, h)
        changed = False
        for comp in relevant:
            if id(comp) in kept:
                continue
            if _touches_or_inside(comp["bbox"], expanded):
                kept.add(id(comp))
                main_bbox = _union_bbox(main_bbox, comp["bbox"])
                changed = True
        if not changed:
            break

    keep_mask = np.zeros_like(ink, dtype=bool)
    for comp in relevant:
        if id(comp) in kept:
            for x, y in comp["pixels"]:
                keep_mask[y, x] = True

    out = np.full_like(arr, 255)
    out[keep_mask] = arr[keep_mask]

    if bool(cfg.get("crop_to_content", True)) and keep_mask.any():
        ys, xs = np.where(keep_mask)
        margin = int(cfg.get("crop_margin_px", 28))
        x0 = max(0, xs.min() - margin)
        x1 = min(w, xs.max() + margin + 1)
        y0 = max(0, ys.min() - margin)
        y1 = min(h, ys.max() + margin + 1)
        out = out[y0:y1, x0:x1]

    Image.fromarray(out).save(out_path)
    return out_path
