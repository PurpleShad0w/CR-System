from __future__ import annotations

from pathlib import Path
import json
import ezdxf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import math


SKIP_TYPES = {
    "TEXT",
    "MTEXT",
    "DIMENSION",
    "LEADER",
    "MULTILEADER",
    "MLEADER",
    "HATCH",
    "FIELD",
    "DIMASSOC",
    "ACAD_PROXY_OBJECT",
    "WIPEOUT",
    "TABLE",
    "IMAGE",
    "UNDERLAY",
}


def _safe_layer(entity) -> str:
    try:
        return entity.dxf.layer
    except Exception:
        return "0"


def _arc_points(center, radius, start_angle_deg, end_angle_deg, n=64):
    start = math.radians(start_angle_deg)
    end = math.radians(end_angle_deg)
    if end < start:
        end += 2 * math.pi
    angles = np.linspace(start, end, n)
    cx, cy = center.x, center.y
    return np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
    ])


def _circle_points(center, radius, n=96):
    angles = np.linspace(0.0, 2 * math.pi, n)
    cx, cy = center.x, center.y
    return np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
    ])


def _polyline_points(entity):
    pts = []
    try:
        for v in entity.vertices:
            loc = v.dxf.location
            pts.append((float(loc.x), float(loc.y)))
    except Exception:
        pass
    return pts


def _entity_to_segments(entity):
    dxftype = entity.dxftype()

    if dxftype in SKIP_TYPES:
        return []

    if dxftype == "LINE":
        s = entity.dxf.start
        e = entity.dxf.end
        return [[(s.x, s.y), (e.x, e.y)]]

    if dxftype == "LWPOLYLINE":
        pts = [(float(p[0]), float(p[1])) for p in entity.get_points("xy")]
        if len(pts) < 2:
            return []
        segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
        if entity.closed:
            segs.append([pts[-1], pts[0]])
        return segs

    if dxftype == "POLYLINE":
        pts = _polyline_points(entity)
        if len(pts) < 2:
            return []
        segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
        if getattr(entity, "is_closed", False):
            segs.append([pts[-1], pts[0]])
        return segs

    if dxftype == "ARC":
        pts = _arc_points(
            entity.dxf.center,
            float(entity.dxf.radius),
            float(entity.dxf.start_angle),
            float(entity.dxf.end_angle),
            n=72,
        )
        return [[tuple(pts[i]), tuple(pts[i + 1])] for i in range(len(pts) - 1)]

    if dxftype == "CIRCLE":
        pts = _circle_points(entity.dxf.center, float(entity.dxf.radius), n=96)
        return [[tuple(pts[i]), tuple(pts[i + 1])] for i in range(len(pts) - 1)]

    return []


def _collect_segments(layout):
    segments = []
    counts = {}
    for entity in layout:
        dxftype = entity.dxftype()
        counts[dxftype] = counts.get(dxftype, 0) + 1

        try:
            if dxftype == "INSERT":
                for sub in entity.virtual_entities():
                    segments.extend(_entity_to_segments(sub))
            else:
                segments.extend(_entity_to_segments(entity))
        except Exception:
            pass
    return segments, counts


def _save_segments_png(segments, out_path: Path):
    if not segments:
        return False

    xs = []
    ys = []
    for seg in segments:
        for p in seg:
            xs.append(p[0])
            ys.append(p[1])

    if not xs or not ys:
        return False

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if min_x == max_x or min_y == max_y:
        return False

    fig, ax = plt.subplots(figsize=(12, 12))
    lc = LineCollection(segments, colors="black", linewidths=0.25)
    ax.add_collection(lc)

    mx = (max_x - min_x) * 0.02
    my = (max_y - min_y) * 0.02
    ax.set_xlim(min_x - mx, max_x + mx)
    ax.set_ylim(min_y - my, max_y + my)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return True


def explore_layouts(dxf_path: Path, out_dir: Path) -> dict:
    doc = ezdxf.readfile(dxf_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"layouts": []}

    # modelspace
    msp = doc.modelspace()
    segments, counts = _collect_segments(msp)
    png_path = out_dir / "modelspace.png"
    _save_segments_png(segments, png_path)
    report["layouts"].append({
        "name": "modelspace",
        "segment_count": len(segments),
        "entity_counts": counts,
        "image": str(png_path),
    })

    # paperspace layouts
    for layout in doc.layouts:
        if layout.name.lower() == "model":
            continue
        segments, counts = _collect_segments(layout)
        safe_name = layout.name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        png_path = out_dir / f"{safe_name}.png"
        _save_segments_png(segments, png_path)
        report["layouts"].append({
            "name": layout.name,
            "segment_count": len(segments),
            "entity_counts": counts,
            "image": str(png_path),
        })

    report_path = out_dir / "layout_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
