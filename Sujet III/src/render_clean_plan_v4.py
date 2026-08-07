from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .noise_filter_balanced import clean_balanced_entities, raster_cleanup_image


def _iter_lines(points, closed: bool):
    if not isinstance(points, list) or len(points) < 2:
        return
    pts = points + [points[0]] if closed and points[0] != points[-1] else points
    for a, b in zip(pts[:-1], pts[1:]):
        yield a, b


def _style(render_class: str, cfg: dict[str, Any]) -> tuple[str, float, str, int]:
    if bool(cfg.get("draw_debug_colors", False)):
        colors = {"wall": "black", "door": "#d95f02", "window": "#1b9e77", "stairs": "#7570b3", "detail": "#666666"}
        return colors.get(render_class, "#444444"), 0.75, "-", 5
    if render_class == "wall":
        return cfg.get("wall_color", "#050505"), float(cfg.get("wall_linewidth", 1.25)), "-", 3
    if render_class == "door":
        return cfg.get("detail_color", "#151515"), float(cfg.get("door_linewidth", 0.70)), "-", 6
    if render_class == "window":
        return cfg.get("detail_color", "#151515"), float(cfg.get("window_linewidth", 0.70)), cfg.get("window_linestyle", "-"), 7
    if render_class == "stairs":
        return cfg.get("detail_color", "#151515"), float(cfg.get("stairs_linewidth", 0.55)), "-", 5
    return cfg.get("detail_color", "#151515"), float(cfg.get("detail_linewidth", 0.55)), "-", 4


def render_clean_plan_v4(df: pd.DataFrame, out_path: str | Path, rules: dict[str, Any] | None = None, decisions: dict[str, Any] | None = None) -> Path:
    rules = rules or {}
    cfg = rules.get("render", {}) or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_balanced_entities(df, rules, decisions)

    fig_size = float(cfg.get("figure_size", 12))
    dpi = int(cfg.get("dpi", 300))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    fig.patch.set_facecolor(cfg.get("background", "white"))
    ax.set_facecolor(cfg.get("background", "white"))

    all_x, all_y = [], []
    order = {"detail": 1, "wall": 2, "stairs": 3, "door": 4, "window": 5}
    if not clean.empty:
        clean = clean.assign(_order=clean["render_class"].map(order).fillna(1)).sort_values("_order")

    for _, row in clean.iterrows():
        pts = row["points"]
        closed = bool(row.get("closed", False))
        color, lw, ls, zo = _style(str(row.get("render_class", "detail")), cfg)
        for (x1, y1), (x2, y2) in _iter_lines(pts, closed):
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=ls, zorder=zo, solid_capstyle="round")
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        dx = max(max_x - min_x, 1.0)
        dy = max(max_y - min_y, 1.0)
        m = float(cfg.get("margins_ratio", 0.018))
        ax.set_xlim(min_x - dx * m, max_x + dx * m)
        ax.set_ylim(min_y - dy * m, max_y + dy * m)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0, facecolor=fig.get_facecolor())
    plt.close(fig)
    raster_cleanup_image(out_path, rules)
    return out_path
