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


def _bbox_from_df(df: pd.DataFrame) -> tuple[float, float, float, float]:
    return (
        float(df["bbox_min_x"].min()),
        float(df["bbox_min_y"].min()),
        float(df["bbox_max_x"].max()),
        float(df["bbox_max_y"].max()),
    )


def _inflate_bbox(bbox: tuple[float, float, float, float], ratio: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    dx = max(x1 - x0, 1.0)
    dy = max(y1 - y0, 1.0)
    m = max(dx, dy) * ratio
    return x0 - m, y0 - m, x1 + m, y1 + m


def _bbox_intersects(row: pd.Series, bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    return not (
        float(row.get("bbox_max_x", 0)) < x0
        or float(row.get("bbox_min_x", 0)) > x1
        or float(row.get("bbox_max_y", 0)) < y0
        or float(row.get("bbox_min_y", 0)) > y1
    )


def _center_in_bbox(row: pd.Series, bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    cx = (float(row.get("bbox_min_x", 0)) + float(row.get("bbox_max_x", 0))) / 2.0
    cy = (float(row.get("bbox_min_y", 0)) + float(row.get("bbox_max_y", 0))) / 2.0
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _soft_noisy_layer_mask(df: pd.DataFrame, rules: dict[str, Any]) -> pd.Series:
    cfg = rules.get("balanced_noise", {}) or {}
    if not bool(cfg.get("drop_noisy_nonphysical_layers", True)) or df.empty:
        return pd.Series(False, index=df.index)
    min_entities = int(cfg.get("noisy_layer_min_entities", 180))
    small_ratio_thr = float(cfg.get("noisy_layer_small_ratio", 0.86))
    flags = {}
    for layer, g in df.groupby("layer"):
        if len(g) < min_entities:
            flags[layer] = False
            continue
        small_ratio = ((g["length"].fillna(0) < 4.0) | (g["bbox_diag"].fillna(0) < 4.0)).mean()
        # Pas de suppression si la couche contient beaucoup d'objets déjà classés physiques.
        physical_ratio = g["feature_class"].isin(["wall", "door", "window", "stairs"]).mean()
        flags[layer] = bool(small_ratio >= small_ratio_thr and physical_ratio < 0.12)
    return df["layer"].map(flags).fillna(False).astype(bool)


def clean_balanced_entities(df: pd.DataFrame, rules: dict[str, Any] | None, decisions: dict[str, Any] | None) -> pd.DataFrame:
    """Nettoyage équilibré.

    Différence avec v3 :
    - ne rend pas tout le DWG ;
    - ne garde pas seulement les murs ;
    - restaure les petits détails architecturaux proches de l'empreinte du bâtiment ;
    - ne fait plus de raster_keep_largest_component.
    """
    rules = rules or {}
    cfg = rules.get("balanced_noise", {}) or {}
    manual_policy = rules.get("manual_drop_policy", {}) or {}

    out = classify_physical(df, rules, decisions)
    if out.empty:
        return out
    out = out[out["points"].apply(_points_ok)].copy()
    if out.empty:
        return out

    physical_classes = set(cfg.get("keep_physical_classes", ["wall", "door", "window", "stairs"]))
    rescue_classes = set(manual_policy.get("rescue_classes", list(physical_classes)))
    allow_drop_rescue = bool(manual_policy.get("allow_physical_rescue_from_dropped_layers", True))

    is_physical = out["feature_class"].isin(physical_classes)
    if allow_drop_rescue:
        # Si layer_decisions.yaml a dropé un layer CLO/MENUI, on rescape l'entité si elle est physique et pas hard_noise.
        physical_keep = is_physical & ~out["hard_noise"].fillna(False) & out["feature_class"].isin(rescue_classes)
    else:
        physical_keep = is_physical & out["decision"].eq("keep")

    # Empreinte de référence : murs physiques, sinon tous physiques.
    core = out[physical_keep & out["feature_class"].eq("wall")].copy()
    if core.empty:
        core = out[physical_keep].copy()
    if core.empty:
        return core

    footprint = _bbox_from_df(core)
    near_bbox = _inflate_bbox(footprint, float(cfg.get("footprint_buffer_ratio", 0.035)))
    far_bbox = _inflate_bbox(footprint, float(cfg.get("far_buffer_ratio", 0.08)))

    # Récupération contrôlée : les portes/fenêtres ratées sont souvent des arcs/lignes/blocs courts proches du bâti.
    rescue_types = set(map(str.upper, cfg.get("rescue_entity_types", [])))
    min_len = float(cfg.get("rescue_min_length", 1.5))
    min_diag = float(cfg.get("rescue_min_bbox_diag", 1.0))
    x0, y0, x1, y1 = footprint
    max_plan_dim = max(x1 - x0, y1 - y0, 1.0)
    max_rescue_len = max_plan_dim * float(cfg.get("rescue_max_length_ratio", 0.20))

    entity_type = out["entity_type"].fillna("").astype(str).str.upper()
    near = out.apply(lambda r: _bbox_intersects(r, near_bbox), axis=1)
    rescue_candidate = (
        bool(cfg.get("rescue_near_footprint", True))
        & near
        & entity_type.isin(rescue_types)
        & ~out["hard_noise"].fillna(False)
        & (out["length"].fillna(0) >= min_len)
        & (out["bbox_diag"].fillna(0) >= min_diag)
        & (out["length"].fillna(0) <= max_rescue_len)
    )

    keep = physical_keep | rescue_candidate
    clean = out[keep].copy()

    if bool(cfg.get("drop_noisy_nonphysical_layers", True)) and not clean.empty:
        noisy_mask = _soft_noisy_layer_mask(clean, rules)
        # On ne supprime que ce qui a été sauvé comme détail, jamais les classes physiques détectées.
        noisy_nonphysical = noisy_mask & ~clean["feature_class"].isin(physical_classes)
        clean = clean[~noisy_nonphysical].copy()

    if bool(cfg.get("drop_far_from_footprint", True)) and not clean.empty:
        close_enough = clean.apply(lambda r: _center_in_bbox(r, far_bbox), axis=1)
        clean = clean[close_enough].copy()

    # Classe visuelle pour les détails sauvés.
    clean["render_class"] = clean["feature_class"]
    clean.loc[~clean["feature_class"].isin(physical_classes), "render_class"] = "detail"
    return clean


def raster_cleanup_image(path: str | Path, rules: dict[str, Any] | None = None) -> Path:
    rules = rules or {}
    cfg = rules.get("balanced_noise", {}) or {}
    if not bool(cfg.get("raster_cleanup", True)):
        return Path(path)

    path = Path(path)
    img = Image.open(path).convert("L")
    arr = np.array(img)
    ink = arr < 245
    h, w = ink.shape
    visited = np.zeros_like(ink, dtype=bool)
    remove_lt = int(cfg.get("raster_remove_components_area_lt", 18))
    keep_largest = bool(cfg.get("raster_keep_largest_component", False))

    comps: list[list[tuple[int, int]]] = []
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

    if bool(cfg.get("raster_crop_to_content", True)) and keep.any():
        ys, xs = np.where(keep)
        m = int(cfg.get("raster_crop_margin_px", 28))
        x0, x1 = max(0, xs.min() - m), min(w, xs.max() + m + 1)
        y0, y1 = max(0, ys.min() - m), min(h, ys.max() + m + 1)
        out = out[y0:y1, x0:x1]

    Image.fromarray(out).save(path)
    return path
