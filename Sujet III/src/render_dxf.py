from __future__ import annotations

from pathlib import Path
import math
import json

import ezdxf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


DRAWABLE_TYPES = {
    "LINE",
    "LWPOLYLINE",
    "POLYLINE",
    "ARC",
    "CIRCLE",
    "ELLIPSE",
    "SPLINE",
    "INSERT",
}

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


def _layer_should_drop(layer: str, rules: dict) -> bool:
    keep_layers = {x.lower() for x in rules.get("keep_layers", [])}
    if keep_layers and layer.lower() in keep_layers:
        return False

    for token in rules.get("drop_layers_contains", []):
        if token.lower() in layer.lower():
            return True
    return False


def _frange(start: float, stop: float, count: int) -> np.ndarray:
    return np.linspace(start, stop, count)


def _arc_points(center, radius, start_angle_deg, end_angle_deg, n=64):
    start = math.radians(start_angle_deg)
    end = math.radians(end_angle_deg)
    if end < start:
        end += 2 * math.pi
    angles = _frange(start, end, n)
    cx, cy = center.x, center.y
    pts = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
    ])
    return pts


def _circle_points(center, radius, n=96):
    angles = _frange(0.0, 2 * math.pi, n)
    cx, cy = center.x, center.y
    pts = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
    ])
    return pts


def _ellipse_points(entity, n=96):
    center = entity.dxf.center
    major = entity.dxf.major_axis
    ratio = entity.dxf.ratio
    start_param = float(entity.dxf.start_param)
    end_param = float(entity.dxf.end_param)

    major_vec = np.array([major.x, major.y], dtype=float)
    major_len = np.linalg.norm(major_vec)
    if major_len == 0:
        return np.empty((0, 2))

    u = major_vec / major_len
    v = np.array([-u[1], u[0]])

    if end_param < start_param:
        end_param += 2 * math.pi

    t = _frange(start_param, end_param, n)
    a = major_len
    b = major_len * ratio

    c = np.array([center.x, center.y], dtype=float)
    pts = np.array([
        c + (a * math.cos(tt)) * u + (b * math.sin(tt)) * v
        for tt in t
    ])
    return pts


def _lwpolyline_points(entity):
    pts = []
    for p in entity.get_points("xy"):
        pts.append((float(p[0]), float(p[1])))
    return pts


def _polyline_points(entity):
    pts = []
    try:
        for v in entity.vertices:
            loc = v.dxf.location
            pts.append((float(loc.x), float(loc.y)))
    except Exception:
        pass
    return pts


def _spline_points(entity, n=100):
    try:
        pts = list(entity.flattening(0.5))
        return np.array([(p.x, p.y) for p in pts], dtype=float)
    except Exception:
        return np.empty((0, 2))


def _entity_to_segments(entity):
    dxftype = entity.dxftype()

    if dxftype == "LINE":
        s = entity.dxf.start
        e = entity.dxf.end
        return [[(s.x, s.y), (e.x, e.y)]]

    if dxftype == "LWPOLYLINE":
        pts = _lwpolyline_points(entity)
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
        pts = _circle_points(
            entity.dxf.center,
            float(entity.dxf.radius),
            n=96,
        )
        return [[tuple(pts[i]), tuple(pts[i + 1])] for i in range(len(pts) - 1)]

    if dxftype == "ELLIPSE":
        pts = _ellipse_points(entity, n=96)
        if len(pts) < 2:
            return []
        return [[tuple(pts[i]), tuple(pts[i + 1])] for i in range(len(pts) - 1)]

    if dxftype == "SPLINE":
        pts = _spline_points(entity, n=100)
        if len(pts) < 2:
            return []
        return [[tuple(pts[i]), tuple(pts[i + 1])] for i in range(len(pts) - 1)]

    return []


def _collect_segments(
    entity,
    rules: dict,
    segments: list,
    stats: dict,
    depth: int = 0,
    max_depth: int = 8,
):
    dxftype = entity.dxftype()
    layer = _safe_layer(entity)

    stats["seen"][dxftype] = stats["seen"].get(dxftype, 0) + 1

    if _layer_should_drop(layer, rules):
        stats["dropped_by_layer"][layer] = stats["dropped_by_layer"].get(layer, 0) + 1
        return

    if dxftype in SKIP_TYPES:
        stats["skipped"][dxftype] = stats["skipped"].get(dxftype, 0) + 1
        return

    if dxftype not in DRAWABLE_TYPES:
        stats["unsupported"][dxftype] = stats["unsupported"].get(dxftype, 0) + 1
        return

    if dxftype == "INSERT":
        if depth >= max_depth:
            stats["insert_depth_limit"] += 1
            return
        try:
            virtuals = list(entity.virtual_entities())
        except Exception:
            stats["insert_virtual_fail"] += 1
            return

        if not virtuals:
            stats["empty_insert"] += 1
            return

        for sub in virtuals:
            _collect_segments(
                sub,
                rules,
                segments,
                stats,
                depth=depth + 1,
                max_depth=max_depth,
            )
        return

    segs = _entity_to_segments(entity)
    if segs:
        segments.extend(segs)
        stats["drawn"][dxftype] = stats["drawn"].get(dxftype, 0) + len(segs)
    else:
        stats["empty_geom"][dxftype] = stats["empty_geom"].get(dxftype, 0) + 1


def render_dxf_to_png(
    dxf_path: Path,
    png_path: Path,
    rules: dict,
    dpi: int = 300,
    debug_json: Path | None = None,
) -> Path:
    """
    Signature compatible avec pipeline.py :
        render_dxf_to_png(
            dxf_path=...,
            png_path=...,
            rules=rules,
            dpi=...,
            debug_json=...,
        )
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    segments = []
    stats = {
        "seen": {},
        "drawn": {},
        "skipped": {},
        "unsupported": {},
        "empty_geom": {},
        "dropped_by_layer": {},
        "insert_virtual_fail": 0,
        "insert_depth_limit": 0,
        "empty_insert": 0,
    }

    for entity in msp:
        _collect_segments(entity, rules, segments, stats)

    if debug_json is not None:
        debug_json.parent.mkdir(parents=True, exist_ok=True)
        debug_json.write_text(
            json.dumps(
                {
                    "segment_count": len(segments),
                    "stats": stats,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    if not segments:
        raise RuntimeError(
            "Aucune géométrie drawable n'a été trouvée après filtrage. "
            "Vérifie render_debug.json pour voir quels types / layers sont ignorés."
        )

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    lc = LineCollection(segments, colors="black", linewidths=0.6)
    ax.add_collection(lc)

    xs = []
    ys = []
    for seg in segments:
        for p in seg:
            xs.append(p[0])
            ys.append(p[1])

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if min_x == max_x or min_y == max_y:
        raise RuntimeError("Bornes invalides : géométrie dégénérée.")

    margin_x = (max_x - min_x) * 0.02
    margin_y = (max_y - min_y) * 0.02

    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)
    return png_path
