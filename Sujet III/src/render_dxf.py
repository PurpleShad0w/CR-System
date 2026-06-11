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


def _segment_bbox(seg):
    (x1, y1), (x2, y2) = seg
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _bbox_expand(bbox, margin):
    x1, y1, x2, y2 = bbox
    return x1 - margin, y1 - margin, x2 + margin, y2 + margin


def _bbox_intersects(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _segment_length(seg):
    (x1, y1), (x2, y2) = seg
    return math.hypot(x2 - x1, y2 - y1)


def _merge_bbox(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)


def _cluster_segments(segments, proximity=200.0):
    """
    Regroupe naïvement les segments par proximité spatiale en utilisant
    leurs bounding boxes dilatées.
    """
    items = []
    for seg in segments:
        bbox = _bbox_expand(_segment_bbox(seg), proximity)
        items.append({"seg": seg, "bbox": bbox})

    clusters = []

    for item in items:
        attached = []
        for idx, cluster in enumerate(clusters):
            if _bbox_intersects(item["bbox"], cluster["bbox"]):
                attached.append(idx)

        if not attached:
            clusters.append({
                "segments": [item["seg"]],
                "bbox": item["bbox"],
            })
        else:
            first = attached[0]
            clusters[first]["segments"].append(item["seg"])
            clusters[first]["bbox"] = _merge_bbox(clusters[first]["bbox"], item["bbox"])

            # fusionne les clusters multiples si nécessaire
            for other_idx in reversed(attached[1:]):
                clusters[first]["segments"].extend(clusters[other_idx]["segments"])
                clusters[first]["bbox"] = _merge_bbox(clusters[first]["bbox"], clusters[other_idx]["bbox"])
                del clusters[other_idx]

    return clusters


def _score_cluster(cluster):
    segs = cluster["segments"]
    bbox = cluster["bbox"]
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area = width * height
    total_length = sum(_segment_length(s) for s in segs)
    n = len(segs)

    # score simple :
    # - beaucoup de segments
    # - grande emprise
    # - beaucoup de longueur totale
    return (total_length * 1.0) + (n * 5.0) + (math.sqrt(area) * 2.0)


def _select_main_cluster(segments, proximity=200.0):
    clusters = _cluster_segments(segments, proximity=proximity)
    if not clusters:
        return segments, None, []

    scored = [(c, _score_cluster(c)) for c in clusters]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_cluster = scored[0][0]
    return best_cluster["segments"], best_cluster["bbox"], scored


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

    main_segments, main_bbox, scored_clusters = _select_main_cluster(segments, proximity=200.0)

    if debug_json is not None:
        debug_payload = {
            "segment_count_total": len(segments),
            "segment_count_main_cluster": len(main_segments),
            "main_bbox": main_bbox,
            "cluster_count": len(scored_clusters),
            "cluster_scores": [
                {
                    "score": score,
                    "segment_count": len(cluster["segments"]),
                    "bbox": cluster["bbox"],
                }
                for cluster, score in scored_clusters[:10]
            ],
            "stats": stats,
        }
        debug_json.parent.mkdir(parents=True, exist_ok=True)
        debug_json.write_text(
            json.dumps(debug_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    segments = main_segments

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    lc = LineCollection(segments, colors="black", linewidths=0.2)
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
