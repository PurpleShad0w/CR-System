from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .layer_classifier import classify_entities


def _iter_lines(points: list[tuple[float, float]], closed: bool):
    if not points or len(points) < 2:
        return
    pts = points + [points[0]] if closed and points[0] != points[-1] else points
    for a, b in zip(pts[:-1], pts[1:]):
        yield a, b


def filter_renderable(df: pd.DataFrame, rules: dict[str, Any] | None, decisions: dict[str, Any] | None) -> pd.DataFrame:
    classified = classify_entities(df, rules, decisions)
    if classified.empty:
        return classified
    keep = classified[classified["decision"] == "keep"].copy()
    keep = keep[keep["points"].apply(lambda p: isinstance(p, list) and len(p) >= 2)]
    return keep


def render_clean_plan(
    df: pd.DataFrame,
    out_path: str | Path,
    rules: dict[str, Any] | None = None,
    decisions: dict[str, Any] | None = None,
    preview: bool = False,
) -> Path:
    rules = rules or {}
    render_cfg = rules.get("render", {}) or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = filter_renderable(df, rules, decisions)
    max_entities = int(render_cfg.get("max_preview_entities", 20000))
    if preview and len(keep) > max_entities:
        keep = keep.sort_values("length", ascending=False).head(max_entities)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=int(render_cfg.get("dpi", 300)))
    fig.patch.set_facecolor(render_cfg.get("background", "white"))
    ax.set_facecolor(render_cfg.get("background", "white"))

    fg = render_cfg.get("foreground", "black")
    lw_scale = float(render_cfg.get("lineweight_scaling", 1.0))
    all_x, all_y = [], []

    for _, row in keep.iterrows():
        pts = row["points"]
        closed = bool(row.get("closed", False))
        lw = row.get("lineweight")
        width = 0.6
        try:
            if lw and float(lw) > 0:
                width = max(0.4, min(2.5, float(lw) / 100.0 * lw_scale))
        except Exception:
            pass
        for (x1, y1), (x2, y2) in _iter_lines(pts, closed):
            ax.plot([x1, x2], [y1, y2], color=fg, linewidth=width, solid_capstyle="round")
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        dx = max(max_x - min_x, 1.0)
        dy = max(max_y - min_y, 1.0)
        m = float(render_cfg.get("margins_ratio", 0.03))
        ax.set_xlim(min_x - dx * m, max_x + dx * m)
        ax.set_ylim(min_y - dy * m, max_y + dy * m)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
