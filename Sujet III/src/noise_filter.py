from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .classify_physical import classify_physical


def _points_ok(points) -> bool:
    return isinstance(points, list) and len(points) >= 2


def _entity_cells(row, min_x, min_y, scale_x, scale_y, grid_size: int) -> set[tuple[int, int]]:
    pts = row.get("points", [])
    cells: set[tuple[int, int]] = set()
    if _points_ok(pts):
        for x, y in pts:
            ix = int((x - min_x) * scale_x)
            iy = int((y - min_y) * scale_y)
            ix = max(0, min(grid_size - 1, ix))
            iy = max(0, min(grid_size - 1, iy))
            cells.add((ix, iy))
    else:
        cx = (float(row.get("bbox_min_x", 0)) + float(row.get("bbox_max_x", 0))) / 2
        cy = (float(row.get("bbox_min_y", 0)) + float(row.get("bbox_max_y", 0))) / 2
        ix = int((cx - min_x) * scale_x)
        iy = int((cy - min_y) * scale_y)
        cells.add((max(0, min(grid_size - 1, ix)), max(0, min(grid_size - 1, iy))))
    return cells


def _largest_grid_component(cells: set[tuple[int, int]], connectivity: int = 8) -> set[tuple[int, int]]:
    if not cells:
        return set()
    remaining = set(cells)
    neigh4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    neigh8 = neigh4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    neighbors = neigh8 if connectivity == 8 else neigh4
    best: set[tuple[int, int]] = set()
    while remaining:
        start = remaining.pop()
        comp = {start}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for dx, dy in neighbors:
                nb = (x + dx, y + dy)
                if nb in remaining:
                    remaining.remove(nb)
                    comp.add(nb)
                    q.append(nb)
        if len(comp) > len(best):
            best = comp
    return best


def _dilate_cells(cells: set[tuple[int, int]], margin: int, grid_size: int) -> set[tuple[int, int]]:
    if margin <= 0:
        return cells
    out = set(cells)
    for x, y in list(cells):
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    out.add((nx, ny))
    return out


def _drop_noisy_layers(df: pd.DataFrame, rules: dict[str, Any]) -> pd.Series:
    noise = rules.get("noise", {}) or {}
    if not bool(noise.get("drop_noisy_layers", True)) or df.empty:
        return pd.Series(False, index=df.index)
    min_entities = int(noise.get("noisy_layer_min_entities", 80))
    small_ratio_thr = float(noise.get("noisy_layer_small_ratio", 0.72))
    physical_ratio_max = float(noise.get("noisy_layer_physical_ratio_max", 0.20))
    min_len_unclassified = float(noise.get("min_length_for_unclassified", 12.0))
    layer_flags = {}
    for layer, g in df.groupby("layer"):
        if len(g) < min_entities:
            layer_flags[layer] = False
            continue
        small_ratio = ((g["length"].fillna(0) < min_len_unclassified) | (g["bbox_diag"].fillna(0) < min_len_unclassified)).mean()
        physical_ratio = g["feature_class"].isin(["wall", "door", "window", "stairs"]).mean()
        layer_flags[layer] = bool(small_ratio >= small_ratio_thr and physical_ratio <= physical_ratio_max)
    return df["layer"].map(layer_flags).fillna(False).astype(bool)


def clean_physical_entities(df: pd.DataFrame, rules: dict[str, Any] | None, decisions: dict[str, Any] | None) -> pd.DataFrame:
    rules = rules or {}
    noise = rules.get("noise", {}) or {}
    out = classify_physical(df, rules, decisions)
    if out.empty:
        return out

    physical_classes = set(noise.get("physical_classes", ["wall", "door", "window", "stairs"]))
    if bool(noise.get("keep_only_physical_classes", True)):
        out = out[out["feature_class"].isin(physical_classes)].copy()
    else:
        out = out[out["decision"] == "keep"].copy()

    out = out[out["points"].apply(_points_ok)].copy()
    if out.empty:
        return out

    if bool(noise.get("drop_tiny_entities", True)):
        min_phys = float(noise.get("min_length_for_physical", 3.0))
        min_diag = float(noise.get("min_bbox_diag", 2.0))
        out = out[(out["length"].fillna(0) >= min_phys) & (out["bbox_diag"].fillna(0) >= min_diag)].copy()

    noisy_layers = _drop_noisy_layers(out, rules)
    if noisy_layers.any():
        out = out[~noisy_layers].copy()

    if bool(noise.get("keep_largest_spatial_component", True)) and len(out) > 0:
        grid_size = int(noise.get("spatial_grid_size", 420))
        min_x = float(out["bbox_min_x"].min())
        min_y = float(out["bbox_min_y"].min())
        max_x = float(out["bbox_max_x"].max())
        max_y = float(out["bbox_max_y"].max())
        dx = max(max_x - min_x, 1.0)
        dy = max(max_y - min_y, 1.0)
        scale_x = (grid_size - 1) / dx
        scale_y = (grid_size - 1) / dy
        entity_cells = []
        all_cells = set()
        for _, row in out.iterrows():
            cells = _entity_cells(row, min_x, min_y, scale_x, scale_y, grid_size)
            entity_cells.append(cells)
            all_cells.update(cells)
        largest = _largest_grid_component(all_cells, int(noise.get("spatial_connectivity", 8)))
        largest = _dilate_cells(largest, int(noise.get("spatial_component_margin_cells", 2)), grid_size)
        keep_mask = [bool(cells & largest) for cells in entity_cells]
        out = out.loc[keep_mask].copy()
    return out


def raster_cleanup_image(path: str | Path, rules: dict[str, Any] | None = None) -> Path:
    rules = rules or {}
    noise = rules.get("noise", {}) or {}
    if not bool(noise.get("raster_cleanup", True)):
        return Path(path)
    path = Path(path)
    img = Image.open(path).convert("L")
    arr = np.array(img)
    ink = arr < 245

    h, w = ink.shape
    visited = np.zeros_like(ink, dtype=bool)
    remove_lt = int(noise.get("raster_remove_components_area_lt", 60))
    keep_largest = bool(noise.get("raster_keep_largest_component", True))
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
            comps.append(pix)
    if not comps:
        return path
    keep = np.zeros_like(ink, dtype=bool)
    largest = max(comps, key=len)
    for comp in comps:
        if len(comp) >= remove_lt and (not keep_largest or comp is largest):
            for x, y in comp:
                keep[y, x] = True
    out = np.full_like(arr, 255)
    out[keep] = arr[keep]

    if bool(noise.get("raster_crop_to_content", True)) and keep.any():
        ys, xs = np.where(keep)
        m = int(noise.get("raster_crop_margin_px", 20))
        x0, x1 = max(0, xs.min() - m), min(w, xs.max() + m + 1)
        y0, y1 = max(0, ys.min() - m), min(h, ys.max() + m + 1)
        out = out[y0:y1, x0:x1]
    Image.fromarray(out).save(path)
    return path
