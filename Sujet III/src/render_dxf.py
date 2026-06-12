from __future__ import annotations

from pathlib import Path
import math
import json
from dataclasses import dataclass

import ezdxf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import cv2

from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend


# Pour le rendu principal du modelspace, on ne garde volontairement
# que le linework structurel + arcs (portes, arrondis utiles).
PRIMARY_DRAWABLE_TYPES = {
    "LINE",
    "LWPOLYLINE",
    "POLYLINE",
    "ARC",
    "SPLINE",
    "INSERT",
}

# Compat layout conservé si besoin, mais plus utilisé comme voie principale.
LAYOUT_DRAWABLE_TYPES = {
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
    "VIEWPORT",
    "POINT",
    "ACAD_TABLE",
}


@dataclass
class WorldBBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return max(1e-9, self.max_x - self.min_x)

    @property
    def height(self) -> float:
        return max(1e-9, self.max_y - self.min_y)


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


def _polyline_to_segments(pts):
    if len(pts) < 2:
        return []
    return [[tuple(pts[i]), tuple(pts[i + 1])] for i in range(len(pts) - 1)]


def _segments_bbox(segments) -> WorldBBox:
    xs = []
    ys = []
    for seg in segments:
        for p in seg:
            xs.append(p[0])
            ys.append(p[1])

    if not xs or not ys:
        raise RuntimeError("Impossible de calculer la bbox : aucun segment.")

    return WorldBBox(
        min_x=min(xs),
        min_y=min(ys),
        max_x=max(xs),
        max_y=max(ys),
    )


def _world_to_px(x: float, y: float, bbox: WorldBBox, width_px: int, height_px: int):
    nx = (x - bbox.min_x) / bbox.width
    ny = (y - bbox.min_y) / bbox.height

    px = int(round(nx * (width_px - 1)))
    py = int(round((1.0 - ny) * (height_px - 1)))
    px = max(0, min(width_px - 1, px))
    py = max(0, min(height_px - 1, py))
    return px, py


def _px_to_world_bbox(x: int, y: int, w: int, h: int, bbox: WorldBBox, width_px: int, height_px: int):
    x1n = x / max(width_px - 1, 1)
    x2n = (x + w) / max(width_px - 1, 1)

    y_top_n = y / max(height_px - 1, 1)
    y_bottom_n = (y + h) / max(height_px - 1, 1)

    world_min_x = bbox.min_x + x1n * bbox.width
    world_max_x = bbox.min_x + x2n * bbox.width

    world_max_y = bbox.min_y + (1.0 - y_top_n) * bbox.height
    world_min_y = bbox.min_y + (1.0 - y_bottom_n) * bbox.height

    return WorldBBox(
        min_x=world_min_x,
        min_y=world_min_y,
        max_x=world_max_x,
        max_y=world_max_y,
    )


def _segment_midpoint(seg):
    (x1, y1), (x2, y2) = seg
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _segment_length(seg):
    (x1, y1), (x2, y2) = seg
    return math.hypot(x2 - x1, y2 - y1)


def _compute_size_filters(global_bbox: WorldBBox):
    ref = max(global_bbox.width, global_bbox.height)
    return {
        "min_arc_radius": ref * 0.0020,
        "min_segment_length": ref * 0.0006,
    }


def _arc_points(center, radius, start_angle_deg, end_angle_deg, n=20):
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


def _spline_points(entity):
    try:
        pts = list(entity.flattening(1.0))
        return np.array([(p.x, p.y) for p in pts], dtype=float)
    except Exception:
        return np.empty((0, 2))


def _filter_short_segments(segs, min_len):
    out = []
    for s in segs:
        if _segment_length(s) >= min_len:
            out.append(s)
    return out


def _entity_to_segments(entity, geom_filters):
    dxftype = entity.dxftype()

    if dxftype == "LINE":
        s = entity.dxf.start
        e = entity.dxf.end
        seg = [(float(s.x), float(s.y)), (float(e.x), float(e.y))]
        if _segment_length(seg) < geom_filters["min_segment_length"]:
            return []
        return [seg]

    if dxftype == "LWPOLYLINE":
        pts = _lwpolyline_points(entity)
        segs = _polyline_to_segments(pts)
        if entity.closed and len(pts) >= 2:
            segs.append([pts[-1], pts[0]])
        return _filter_short_segments(segs, geom_filters["min_segment_length"])

    if dxftype == "POLYLINE":
        pts = _polyline_points(entity)
        segs = _polyline_to_segments(pts)
        if getattr(entity, "is_closed", False) and len(pts) >= 2:
            segs.append([pts[-1], pts[0]])
        return _filter_short_segments(segs, geom_filters["min_segment_length"])

    if dxftype == "ARC":
        radius = float(entity.dxf.radius)
        if radius < geom_filters["min_arc_radius"]:
            return []
        pts = _arc_points(
            entity.dxf.center,
            radius,
            float(entity.dxf.start_angle),
            float(entity.dxf.end_angle),
            n=16,
        )
        segs = _polyline_to_segments(pts)
        return _filter_short_segments(segs, geom_filters["min_segment_length"])

    if dxftype == "SPLINE":
        pts = _spline_points(entity)
        segs = _polyline_to_segments(pts)
        return _filter_short_segments(segs, geom_filters["min_segment_length"])

    return []


def _initial_bbox_from_linework(dxf_path: Path, rules: dict) -> WorldBBox:
    """
    Première bbox basée exclusivement sur lignes/polylignes/arcs/splines,
    pour éviter que les ellipses polluent l'échelle globale.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    raw_segments = []

    for entity in msp:
        dxftype = entity.dxftype()
        layer = _safe_layer(entity)

        if _layer_should_drop(layer, rules):
            continue

        if dxftype == "LINE":
            s = entity.dxf.start
            e = entity.dxf.end
            raw_segments.append([(float(s.x), float(s.y)), (float(e.x), float(e.y))])

        elif dxftype == "LWPOLYLINE":
            pts = _lwpolyline_points(entity)
            raw_segments.extend(_polyline_to_segments(pts))
            if entity.closed and len(pts) >= 2:
                raw_segments.append([pts[-1], pts[0]])

        elif dxftype == "POLYLINE":
            pts = _polyline_points(entity)
            raw_segments.extend(_polyline_to_segments(pts))
            if getattr(entity, "is_closed", False) and len(pts) >= 2:
                raw_segments.append([pts[-1], pts[0]])

        elif dxftype == "ARC":
            radius = float(entity.dxf.radius)
            if radius > 0:
                pts = _arc_points(
                    entity.dxf.center,
                    radius,
                    float(entity.dxf.start_angle),
                    float(entity.dxf.end_angle),
                    n=12,
                )
                raw_segments.extend(_polyline_to_segments(pts))

    if not raw_segments:
        raise RuntimeError("Impossible d'établir la bbox initiale à partir du linework.")

    return _segments_bbox(raw_segments)


def _collect_segments(
    entity,
    rules: dict,
    segments: list,
    stats: dict,
    geom_filters: dict,
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

    if dxftype not in PRIMARY_DRAWABLE_TYPES:
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
                geom_filters,
                depth=depth + 1,
                max_depth=max_depth,
            )
        return

    segs = _entity_to_segments(entity, geom_filters)
    if segs:
        segments.extend(segs)
        stats["drawn"][dxftype] = stats["drawn"].get(dxftype, 0) + len(segs)
    else:
        stats["empty_geom"][dxftype] = stats["empty_geom"].get(dxftype, 0) + 1


def _choose_main_component_bbox(segments, debug: dict | None = None) -> WorldBBox:
    bbox = _segments_bbox(segments)

    target_long_side = 2400
    if bbox.width >= bbox.height:
        width_px = target_long_side
        height_px = max(800, int(round(target_long_side * bbox.height / bbox.width)))
    else:
        height_px = target_long_side
        width_px = max(800, int(round(target_long_side * bbox.width / bbox.height)))

    canvas = np.zeros((height_px, width_px), dtype=np.uint8)

    for seg in segments:
        (x1, y1), (x2, y2) = seg
        p1 = _world_to_px(x1, y1, bbox, width_px, height_px)
        p2 = _world_to_px(x2, y2, bbox, width_px, height_px)
        cv2.line(canvas, p1, p2, 255, 1, lineType=cv2.LINE_8)

    # Connexion modérée surtout horizontale
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (61, 9))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))

    work = cv2.dilate(canvas, kernel_h, iterations=1)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((work > 0).astype(np.uint8), connectivity=8)

    candidates = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if w < 120 or h < 70 or area < 2000:
            continue

        seg_count = 0
        total_len = 0.0
        for seg in segments:
            mx, my = _segment_midpoint(seg)
            px, py = _world_to_px(mx, my, bbox, width_px, height_px)
            if 0 <= px < width_px and 0 <= py < height_px and labels[py, px] == i:
                seg_count += 1
                total_len += _segment_length(seg)

        if seg_count < 500:
            continue

        aspect = w / max(h, 1)
        fill_ratio = area / max(w * h, 1)

        top_touch = 1 if y <= int(0.02 * height_px) else 0
        right_touch = 1 if (x + w) >= int(0.98 * width_px) else 0

        # On valorise :
        # - grande largeur
        # - ratio horizontal
        # - nombre de segments important
        # On pénalise :
        # - composants pleins/compacts
        # - composants collés aux bords (souvent détails parasites)
        score = (
            w * 12.0
            + aspect * 1800.0
            + math.sqrt(area) * 10.0
            + seg_count * 0.08
            - fill_ratio * 1200.0
            - top_touch * 2500.0
            - right_touch * 1800.0
        )

        candidates.append({
            "label": i,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "aspect": aspect,
            "fill_ratio": fill_ratio,
            "seg_count": seg_count,
            "total_len": total_len,
            "top_touch": top_touch,
            "right_touch": right_touch,
            "score": score,
        })

    if not candidates:
        if debug is not None:
            debug["component_selection"] = {
                "mode": "fallback_global_bbox",
                "width_px": width_px,
                "height_px": height_px,
                "component_count": 0,
            }
        return bbox

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    margin_x = int(round(best["w"] * 0.05))
    margin_y = int(round(best["h"] * 0.08))

    bx = max(0, best["x"] - margin_x)
    by = max(0, best["y"] - margin_y)
    bw = min(width_px - bx, best["w"] + 2 * margin_x)
    bh = min(height_px - by, best["h"] + 2 * margin_y)

    best_world_bbox = _px_to_world_bbox(bx, by, bw, bh, bbox, width_px, height_px)

    if debug is not None:
        debug["component_selection"] = {
            "mode": "connected_components_lowres",
            "width_px": width_px,
            "height_px": height_px,
            "component_count": len(candidates),
            "selected_component": best,
            "top_components": candidates[:10],
            "selected_world_bbox": {
                "min_x": best_world_bbox.min_x,
                "min_y": best_world_bbox.min_y,
                "max_x": best_world_bbox.max_x,
                "max_y": best_world_bbox.max_y,
            },
        }

    return best_world_bbox


def _filter_segments_in_bbox(segments, bbox: WorldBBox):
    kept = []
    for seg in segments:
        mx, my = _segment_midpoint(seg)
        if bbox.min_x <= mx <= bbox.max_x and bbox.min_y <= my <= bbox.max_y:
            kept.append(seg)
    return kept


def render_layout_to_png(
    dxf_path: Path,
    png_path: Path,
    layout_name: str,
    dpi: int = 300,
) -> Path:
    """
    Compatibilité conservée, mais ce n'est plus la voie principale.
    """
    doc = ezdxf.readfile(dxf_path)

    target_layout = None
    for layout in doc.layouts:
        if layout.name == layout_name:
            target_layout = layout
            break

    if target_layout is None:
        raise ValueError(f'Layout "{layout_name}" introuvable.')

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("white")

    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    Frontend(ctx, backend).draw_layout(target_layout, finalize=True)

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


def render_dxf_to_png(
    dxf_path: Path,
    png_path: Path,
    rules: dict,
    dpi: int = 300,
    debug_json: Path | None = None,
) -> Path:
    """
    Rendu principal du modelspace :
    - ignore ellipses/circles pour éviter le bruit massif
    - sélectionne la zone dominante à partir du linework structurel
    """
    initial_bbox = _initial_bbox_from_linework(dxf_path, rules)
    geom_filters = _compute_size_filters(initial_bbox)

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
        "geom_filters": geom_filters,
    }

    for entity in msp:
        _collect_segments(entity, rules, segments, stats, geom_filters)

    if not segments:
        raise RuntimeError(
            "Aucune géométrie drawable n'a été trouvée après filtrage. "
            "Vérifie render_debug.json pour voir quels types / layers sont ignorés."
        )

    debug = {
        "segment_count_total": len(segments),
        "initial_bbox": {
            "min_x": initial_bbox.min_x,
            "min_y": initial_bbox.min_y,
            "max_x": initial_bbox.max_x,
            "max_y": initial_bbox.max_y,
        },
        "stats": stats,
    }

    selected_bbox = _choose_main_component_bbox(segments, debug=debug)
    filtered_segments = _filter_segments_in_bbox(segments, selected_bbox)

    # Si la sélection est trop faible, on élargit légèrement autour de la bbox
    if len(filtered_segments) < max(2000, int(0.03 * len(segments))):
        pad_x = selected_bbox.width * 0.15
        pad_y = selected_bbox.height * 0.15
        expanded = WorldBBox(
            min_x=selected_bbox.min_x - pad_x,
            min_y=selected_bbox.min_y - pad_y,
            max_x=selected_bbox.max_x + pad_x,
            max_y=selected_bbox.max_y + pad_y,
        )
        filtered_segments = _filter_segments_in_bbox(segments, expanded)
        debug["fallback_used"] = "expanded_selected_bbox"
        selected_bbox = expanded
    else:
        debug["fallback_used"] = None

    debug["segment_count_selected"] = len(filtered_segments)
    debug["selected_bbox"] = {
        "min_x": selected_bbox.min_x,
        "min_y": selected_bbox.min_y,
        "max_x": selected_bbox.max_x,
        "max_y": selected_bbox.max_y,
    }

    if debug_json is not None:
        debug_json.parent.mkdir(parents=True, exist_ok=True)
        debug_json.write_text(
            json.dumps(debug, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    bbox = _segments_bbox(filtered_segments)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    lc = LineCollection(filtered_segments, colors="black", linewidths=0.18)
    ax.add_collection(lc)

    margin_x = bbox.width * 0.02
    margin_y = bbox.height * 0.02

    ax.set_xlim(bbox.min_x - margin_x, bbox.max_x + margin_x)
    ax.set_ylim(bbox.min_y - margin_y, bbox.max_y + margin_y)
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
