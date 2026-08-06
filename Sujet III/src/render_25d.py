from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .render_clean_plan import filter_renderable


def export_25d_html(
    df: pd.DataFrame,
    out_path: str | Path,
    rules: dict[str, Any] | None = None,
    decisions: dict[str, Any] | None = None,
) -> Path:
    """Génère une vue 2.5D HTML interactive.

    On extrude les polylignes gardées les plus plausibles en murs. Le rendu reste
    volontairement léger : il sert de prévisualisation, pas de BIM.
    """
    import plotly.graph_objects as go

    rules = rules or {}
    cfg = rules.get("render_25d", {}) or {}
    wall_height = float(cfg.get("wall_height", 3000.0))
    z_scale = float(cfg.get("z_scale", 0.25))
    line_width = float(cfg.get("line_width", 1.4))
    opacity = float(cfg.get("opacity", 0.32))
    z_top = wall_height * z_scale

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = filter_renderable(df, rules, decisions)
    if "wall_score" in keep.columns:
        keep = keep[keep["wall_score"] >= float((rules.get("classification", {}) or {}).get("wall_score_threshold", 0.55))]

    fig = go.Figure()
    for _, row in keep.iterrows():
        pts = row["points"]
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        closed = bool(row.get("closed", False))
        pts2 = pts + [pts[0]] if closed and pts[0] != pts[-1] else pts
        xs = [p[0] for p in pts2]
        ys = [p[1] for p in pts2]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[0] * len(xs), mode="lines", line=dict(color="black", width=line_width), showlegend=False))
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[z_top] * len(xs), mode="lines", line=dict(color="black", width=line_width), showlegend=False))
        for a, b in zip(pts2[:-1], pts2[1:]):
            fig.add_trace(go.Mesh3d(
                x=[a[0], b[0], b[0], a[0]],
                y=[a[1], b[1], b[1], a[1]],
                z=[0, 0, z_top, z_top],
                i=[0, 0], j=[1, 2], k=[2, 3],
                color="lightgrey", opacity=opacity, showlegend=False,
            ))
    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path
