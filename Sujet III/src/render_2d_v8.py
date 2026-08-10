from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .selection_v8 import select_entities_v8
from .raster_cleanup_v8 import cleanup_rendered_plan


def _iter_lines(points, closed: bool):
    if not isinstance(points, list) or len(points) < 2:
        return
    pts = points + [points[0]] if closed and points[0] != points[-1] else points
    for a, b in zip(pts[:-1], pts[1:]):
        yield a, b


def _line_width(row, cfg: dict[str, Any]) -> float:
    default = float(cfg.get("linewidth_default", 0.52))
    if bool(row.get("is_rescued_detail", False)):
        default *= float(cfg.get("rescued_linewidth_multiplier", 0.90))
    if not bool(cfg.get("use_entity_lineweight", True)):
        return default
    lw = row.get("lineweight")
    try:
        if lw and float(lw) > 0:
            return max(float(cfg.get("linewidth_min", 0.30)), min(float(cfg.get("linewidth_max", 1.55)), float(lw) / 100.0))
    except Exception:
        pass
    return default


def render_2d_v8(df: pd.DataFrame, out_path: str | Path, rules: dict[str, Any] | None = None, decisions: dict[str, Any] | None = None) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    rules = rules or {}
    cfg = rules.get("render_2d", {}) or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected, debug = select_entities_v8(df, rules, decisions)

    fig_size = float(cfg.get("figure_size", 12))
    dpi = int(cfg.get("dpi", 300))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    fig.patch.set_facecolor(cfg.get("background", "white"))
    ax.set_facecolor(cfg.get("background", "white"))

    xs, ys = [], []
    for _, row in selected.iterrows():
        lw = _line_width(row, cfg)
        for (x1, y1), (x2, y2) in _iter_lines(row.points, bool(row.get("closed", False))):
            ax.plot([x1, x2], [y1, y2], color=cfg.get("foreground", "black"), linewidth=lw, solid_capstyle="round")
            xs.extend([x1, x2])
            ys.extend([y1, y2])

    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dx = max(max_x - min_x, 1.0)
        dy = max(max_y - min_y, 1.0)
        m = float(cfg.get("margins_ratio", 0.025))
        ax.set_xlim(min_x - dx * m, max_x + dx * m)
        ax.set_ylim(min_y - dy * m, max_y + dy * m)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0, facecolor=fig.get_facecolor())
    plt.close(fig)

    if bool((rules.get("post_cleanup", {}) or {}).get("enabled", True)):
        cleanup_rendered_plan(out_path, out_path, rules.get("post_cleanup", {}) or {})
    return out_path, selected, debug
