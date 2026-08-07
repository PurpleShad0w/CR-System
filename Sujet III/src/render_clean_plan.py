from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .feature_classifier import classify_features


def _iter_lines(points: list[tuple[float, float]], closed: bool):
    if not points or len(points) < 2:
        return
    pts = points + [points[0]] if closed and points[0] != points[-1] else points
    for a, b in zip(pts[:-1], pts[1:]):
        yield a, b


def filter_renderable(df: pd.DataFrame, rules: dict[str, Any] | None, decisions: dict[str, Any] | None) -> pd.DataFrame:
    classified = classify_features(df, rules, decisions)
    if classified.empty:
        return classified
    keep = classified[classified["decision"] == "keep"].copy()
    keep = keep[keep["points"].apply(lambda p: isinstance(p, list) and len(p) >= 2)]
    return keep


def _style_for_class(feature_class: str, cfg: dict[str, Any]) -> tuple[str, float, int]:
    debug = bool(cfg.get("draw_debug_colors", False))
    if debug:
        palette = {
            "wall": "black",
            "door": "#d95f02",
            "window": "#1b9e77",
            "stairs": "#7570b3",
            "furniture": "#999999",
            "other": "#444444",
        }
        return palette.get(feature_class, "#444444"), 0.8, 3
    if feature_class == "wall":
        return cfg.get("wall_color", "#111111"), float(cfg.get("wall_linewidth", 1.4)), 5
    if feature_class == "door":
        return cfg.get("door_color", "#111111"), float(cfg.get("door_linewidth", 0.9)), 7
    if feature_class == "window":
        return cfg.get("window_color", "#111111"), float(cfg.get("window_linewidth", 0.9)), 8
    if feature_class == "stairs":
        return cfg.get("stairs_color", "#222222"), float(cfg.get("stairs_linewidth", 0.75)), 6
    return cfg.get("other_color", "#555555"), float(cfg.get("other_linewidth", 0.45)), 2


def render_clean_plan(
    df: pd.DataFrame,
    out_path: str | Path,
    rules: dict[str, Any] | None = None,
    decisions: dict[str, Any] | None = None,
    preview: bool = False,
) -> Path:
    rules = rules or {}
    cfg = rules.get("render", {}) or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = filter_renderable(df, rules, decisions)
    if not bool(cfg.get("draw_other_kept", True)):
        keep = keep[keep["feature_class"].isin(["wall", "door", "window", "stairs"])]
    max_entities = int(cfg.get("max_preview_entities", 40000))
    if preview and len(keep) > max_entities:
        keep = keep.sort_values("length", ascending=False).head(max_entities)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=int(cfg.get("dpi", 300)))
    fig.patch.set_facecolor(cfg.get("background", "white"))
    ax.set_facecolor(cfg.get("background", "white"))
    all_x, all_y = [], []

    # ordre de rendu : autres, murs, escaliers, portes/fenêtres au-dessus
    order = {"other": 1, "furniture": 1, "wall": 2, "stairs": 3, "door": 4, "window": 5}
    keep = keep.assign(_draw_order=keep["feature_class"].map(order).fillna(1)).sort_values("_draw_order")

    for _, row in keep.iterrows():
        pts = row["points"]
        closed = bool(row.get("closed", False))
        fc = str(row.get("feature_class", "other"))
        color, width, zorder = _style_for_class(fc, cfg)
        linestyle = "-"
        if fc == "window":
            linestyle = "--"
        for (x1, y1), (x2, y2) in _iter_lines(pts, closed):
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, linestyle=linestyle, zorder=zorder, solid_capstyle="round")
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        dx = max(max_x - min_x, 1.0)
        dy = max(max_y - min_y, 1.0)
        m = float(cfg.get("margins_ratio", 0.025))
        ax.set_xlim(min_x - dx * m, max_x + dx * m)
        ax.set_ylim(min_y - dy * m, max_y + dy * m)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
