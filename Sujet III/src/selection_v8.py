from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or default


def normalize_decisions(decisions: dict[str, Any] | None) -> dict[str, list[str]]:
    decisions = decisions or {}
    return {
        "keep": sorted(set(map(str, decisions.get("keep", []) or []))),
        "drop": sorted(set(map(str, decisions.get("drop", []) or []))),
        "undecided": sorted(set(map(str, decisions.get("undecided", []) or []))),
    }


def _contains_any(value: str, needles: list[str]) -> bool:
    value_u = str(value or "").upper()
    return any(str(n).upper() in value_u for n in needles)


def _blob(df: pd.DataFrame) -> pd.Series:
    out = pd.Series("", index=df.index)
    for col in ["layer", "source_block", "block_name", "entity_type"]:
        if col in df.columns:
            out = out + " " + df[col].fillna("").astype(str)
    return out


def _bbox(df: pd.DataFrame) -> tuple[float, float, float, float]:
    return float(df.bbox_min_x.min()), float(df.bbox_min_y.min()), float(df.bbox_max_x.max()), float(df.bbox_max_y.max())


def _inflate_bbox(bbox: tuple[float, float, float, float], ratio: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    m = max(x1 - x0, y1 - y0, 1.0) * ratio
    return x0 - m, y0 - m, x1 + m, y1 + m


def _intersects(row: pd.Series, bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    return not (row.bbox_max_x < x0 or row.bbox_min_x > x1 or row.bbox_max_y < y0 or row.bbox_min_y > y1)


def _cells(row: pd.Series, minx: float, miny: float, sx: float, sy: float, g: int, dilate: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    points = row.points if isinstance(row.points, list) and len(row.points) >= 2 else [((row.bbox_min_x + row.bbox_max_x) / 2, (row.bbox_min_y + row.bbox_max_y) / 2)]
    for x, y in points:
        ix = max(0, min(g - 1, int((x - minx) * sx)))
        iy = max(0, min(g - 1, int((y - miny) * sy)))
        for dx in range(-dilate, dilate + 1):
            for dy in range(-dilate, dilate + 1):
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < g and 0 <= ny < g:
                    cells.add((nx, ny))
    return cells


def _components(cell_to_rows: dict[tuple[int, int], set[int]]) -> list[set[int]]:
    remain = set(cell_to_rows)
    comps: list[set[int]] = []
    neigh = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while remain:
        start = remain.pop()
        q = deque([start])
        cc = {start}
        while q:
            x, y = q.popleft()
            for dx, dy in neigh:
                nb = (x + dx, y + dy)
                if nb in remain:
                    remain.remove(nb)
                    cc.add(nb)
                    q.append(nb)
        rows: set[int] = set()
        for c in cc:
            rows.update(cell_to_rows[c])
        comps.append(rows)
    return comps


def _main_plan_rows(df: pd.DataFrame, cfg: dict[str, Any]) -> set[int]:
    if df.empty:
        return set()
    g = int(cfg.get("main_component_grid_size", 700))
    dil = int(cfg.get("main_component_dilation_cells", 1))
    minx, miny, maxx, maxy = _bbox(df)
    sx = (g - 1) / max(maxx - minx, 1.0)
    sy = (g - 1) / max(maxy - miny, 1.0)
    cell_to_rows: dict[tuple[int, int], set[int]] = defaultdict(set)
    for idx, row in df.iterrows():
        for c in _cells(row, minx, miny, sx, sy, g, dil):
            cell_to_rows[c].add(idx)
    comps = _components(cell_to_rows)
    if not comps:
        return set(df.index)

    scored: list[tuple[float, set[int]]] = []
    for rows in comps:
        gdf = df.loc[list(rows)]
        # v1-like scoring: prefer ink-rich, dense drawing group, not isolated legend/barcodes.
        x0, y0, x1, y1 = _bbox(gdf)
        bbox_area = max((x1 - x0) * (y1 - y0), 1.0)
        total_len = float(gdf.length.fillna(0).sum())
        density = total_len / (bbox_area ** 0.5)
        small_ratio = ((gdf.length.fillna(0) < 2.0) | (gdf.bbox_diag.fillna(0) < 2.0)).mean()
        score = total_len + density * 20.0 - small_ratio * len(gdf) * 2.0
        scored.append((score, rows))
    scored.sort(reverse=True, key=lambda x: x[0])
    best_score = scored[0][0]
    keep_rows = set(scored[0][1])
    ratio = float(cfg.get("keep_components_score_ratio", 0.18))
    for score, rows in scored[1:]:
        if best_score > 0 and score / best_score >= ratio:
            keep_rows.update(rows)
    return keep_rows


def select_entities_v8(df: pd.DataFrame, rules: dict[str, Any] | None, decisions: dict[str, Any] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = rules or {}
    cfg = rules.get("selection", {}) or {}
    decisions = normalize_decisions(decisions)
    out = df.copy()
    if out.empty:
        return out, out

    out = out[out.points.apply(lambda p: isinstance(p, list) and len(p) >= 2)].copy()
    if out.empty:
        return out, out

    blob = _blob(out)
    entity_type = out.entity_type.fillna("").astype(str).str.upper()
    layer = out.layer.fillna("").astype(str)

    hard_drop = entity_type.isin(set(map(str.upper, cfg.get("hard_drop_entity_types", []) or [])))
    hard_drop |= blob.apply(lambda x: _contains_any(x, cfg.get("hard_drop_layer_keywords", []) or []))

    manual_keep = layer.isin(decisions["keep"])
    manual_drop = layer.isin(decisions["drop"])

    min_base_len = float(cfg.get("min_base_length", 0.8))
    min_base_diag = float(cfg.get("min_base_bbox_diag", 0.4))
    too_small = (out.length.fillna(0) < min_base_len) & (out.bbox_diag.fillna(0) < min_base_diag)

    base_keep = ~hard_drop & ~too_small
    if bool(cfg.get("respect_manual_drop_layers", True)):
        base_keep &= ~manual_drop
    if not bool(cfg.get("render_undecided_layers", True)):
        base_keep &= manual_keep

    base = out[base_keep].copy()

    # Main plan selection removes far-away barcode/cartouche components without semantic filtering.
    if bool(cfg.get("remove_far_artifacts_before_render", True)) and not base.empty:
        main_rows = _main_plan_rows(base, cfg)
        base = base.loc[list(main_rows)].copy() if main_rows else base

    # Detail rescue: bring back doors/windows/small features from dropped layers if they are near the selected plan.
    rescue = pd.DataFrame(columns=out.columns)
    if bool(cfg.get("allow_detail_rescue_from_dropped_layers", True)) and not base.empty:
        main_bbox = _inflate_bbox(_bbox(base), float(cfg.get("rescue_bbox_expand_ratio", 0.10)))
        rescue_types = set(map(str.upper, cfg.get("rescue_geometry_types", []) or []))
        rescue_kw = blob.apply(lambda x: _contains_any(x, cfg.get("detail_rescue_keywords", []) or []))
        near = out.apply(lambda row: _intersects(row, main_bbox), axis=1)
        max_dim = max(main_bbox[2] - main_bbox[0], main_bbox[3] - main_bbox[1], 1.0)
        max_len = max_dim * float(cfg.get("max_rescue_length_ratio", 0.16))
        rescue_mask = (
            ~hard_drop
            & near
            & entity_type.isin(rescue_types)
            & (out.length.fillna(0) >= float(cfg.get("min_rescue_length", 0.6)))
            & (out.bbox_diag.fillna(0) >= float(cfg.get("min_rescue_bbox_diag", 0.35)))
            & (out.length.fillna(0) <= max_len)
            & (rescue_kw | manual_drop)
        )
        rescue = out[rescue_mask].copy()

    selected = pd.concat([base, rescue], axis=0).drop_duplicates("row_id")
    selected["is_rescued_detail"] = selected.row_id.isin(set(rescue.row_id)) if not rescue.empty else False
    debug = out.copy()
    debug["selected_v8"] = debug.row_id.isin(set(selected.row_id))
    debug["hard_drop_v8"] = hard_drop
    debug["manual_drop_v8"] = manual_drop
    return selected, debug
