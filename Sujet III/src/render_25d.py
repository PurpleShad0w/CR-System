from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .render_clean_plan import filter_renderable, _iter_lines


def render_aerial_25d(
    df: pd.DataFrame,
    out_path: str | Path,
    rules: dict[str, Any] | None = None,
    decisions: dict[str, Any] | None = None,
    preview: bool = False,
) -> Path:
    """Vue aérienne 2.5D.

    Ce n'est volontairement pas une scène 3D. On garde une vue top-down, type plan
    d'architecte propre, avec un très léger décalage d'ombre pour donner du relief.
    """
    rules = rules or {}
    cfg = rules.get("aerial_25d", {}) or {}
    render_cfg = rules.get("render", {}) or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keep = filter_renderable(df, rules, decisions)
    keep = keep[keep["feature_class"].isin(["wall", "door", "window", "stairs", "other"])]
    max_entities = int(render_cfg.get("max_preview_entities", 40000))
    if preview and len(keep) > max_entities:
        keep = keep.sort_values("length", ascending=False).head(max_entities)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=int(render_cfg.get("dpi", 300)))
    fig.patch.set_facecolor(render_cfg.get("background", "white"))
    ax.set_facecolor(render_cfg.get("background", "white"))

    all_x, all_y = [], []
    shadow_offset = float(cfg.get("shadow_offset_px", 5))
    shadow_alpha = float(cfg.get("shadow_alpha", 0.16))
    draw_shadow = bool(cfg.get("draw_soft_shadow", True))

    walls = keep[keep["feature_class"].eq("wall")]
    features = keep[~keep["feature_class"].eq("wall")]

    # Ombre très légère, top-down. Elle remplace l'ancienne extrusion trop haute.
    if draw_shadow:
        for _, row in walls.iterrows():
            pts = row["points"]
            closed = bool(row.get("closed", False))
            for (x1, y1), (x2, y2) in _iter_lines(pts, closed):
                ax.plot([x1 + shadow_offset, x2 + shadow_offset], [y1 - shadow_offset, y2 - shadow_offset],
                        color="black", alpha=shadow_alpha, linewidth=float(cfg.get("wall_linewidth", 2.2)) + 1.2,
                        zorder=1, solid_capstyle="round")

    for _, row in walls.iterrows():
        pts = row["points"]
        closed = bool(row.get("closed", False))
        for (x1, y1), (x2, y2) in _iter_lines(pts, closed):
            ax.plot([x1, x2], [y1, y2], color=cfg.get("wall_color", "#111111"),
                    linewidth=float(cfg.get("wall_linewidth", 2.2)), zorder=3, solid_capstyle="round")
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

    # Portes/fenêtres/escalier au-dessus. Les ouvertures sont claires pour ne pas être noyées dans les murs.
    for _, row in features.iterrows():
        pts = row["points"]
        closed = bool(row.get("closed", False))
        fc = str(row.get("feature_class", "other"))
        if fc == "door":
            color = cfg.get("feature_color", "#222222")
            width = float(cfg.get("feature_linewidth", 1.0))
            linestyle = "-"
            zorder = 6
        elif fc == "window":
            color = cfg.get("feature_color", "#222222")
            width = float(cfg.get("feature_linewidth", 1.0))
            linestyle = "--"
            zorder = 7
        elif fc == "stairs":
            color = cfg.get("feature_color", "#222222")
            width = float(cfg.get("feature_linewidth", 0.9))
            linestyle = "-"
            zorder = 5
        else:
            color = "#666666"
            width = 0.35
            linestyle = "-"
            zorder = 2
        for (x1, y1), (x2, y2) in _iter_lines(pts, closed):
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, linestyle=linestyle, zorder=zorder, solid_capstyle="round")
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        dx = max(max_x - min_x, 1.0)
        dy = max(max_y - min_y, 1.0)
        m = float(render_cfg.get("margins_ratio", 0.025))
        ax.set_xlim(min_x - dx * m, max_x + dx * m)
        ax.set_ylim(min_y - dy * m, max_y + dy * m)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# Compatibilité avec l'appel v1 : export_25d_html génère maintenant une image aérienne PNG si out_path se termine par .png,
# sinon un HTML minimal qui référence le PNG voisin.
def export_25d_html(df: pd.DataFrame, out_path: str | Path, rules: dict[str, Any] | None = None, decisions: dict[str, Any] | None = None) -> Path:
    out_path = Path(out_path)
    if out_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return render_aerial_25d(df, out_path, rules=rules, decisions=decisions)
    png_path = out_path.with_suffix(".png")
    render_aerial_25d(df, png_path, rules=rules, decisions=decisions)
    out_path.write_text(f"""<!doctype html><html><body style='margin:0;background:white;'><img src='{png_path.name}' style='width:100%;height:auto;display:block;'></body></html>""", encoding="utf-8")
    return out_path
