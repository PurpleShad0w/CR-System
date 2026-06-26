from __future__ import annotations

from pathlib import Path
import math
import json
from dataclasses import dataclass
from collections import defaultdict

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


# ============================================================
# Helpers généraux
# ============================================================

def _safe_layer(entity) -> str:
    try:
        return entity.dxf.layer
    except Exception:
        return "0"


def _layer_token_match(layer: str, token: str) -> bool:
    """
    Matching prudent pour éviter que 'SUR' matche 'sur allège'.
    """
    layer_l = layer.lower()
    token_l = token.lower()

    if len(token_l) <= 4:
        patterns = [
            f"-{token_l}-",
            f"_{token_l}_",
            f"/{token_l}/",
            f" {token_l} ",
        ]
        if any(p in layer_l for p in patterns):
            return True

        if layer_l.startswith(token_l + "-") or layer_l.endswith("-" + token_l):
            return True

        return False

    return token_l in layer_l


def _layer_is_architectural_after_crop(layer: str) -> bool:
    """
    Filtrage sémantique APRES bbox :
    - on garde les couches de cloisons / murs / portes / circulation
    - on enlève les couches de surfaces / mobilier / plafonds / toiture / présentation
    """
    layer_l = layer.lower()

    drop_tokens = (
        "kdm-sur-",
        "kdm-mob-",
        "kdm-age-",
        "kdm-sec-",
        "kdm-cvc-",
        "kdm-fpl-",
        "kdm-fxp-",
        "kdm-plo-",
        "kdm-sol-",
        "kdm-rvm-",
        "kdm-sig-",
        "kdm-xrf-",
        "kdm-prz-",
        "plf-",
        "vpa",
        "facade",
        "fa-",
        "toit",
        "couv",
        "beton",
        "brique",
        "trame",
        "hach",
        "texte",
        "cote",
        "dim",
        "legende",
        "légende",
        "cartouche",
        "tableau",
    )

    for tok in drop_tokens:
        if tok in layer_l:
            return False

    keep_tokens = (
        "kdm-clo-",
        "mur",
        "porte",
        "menuis",
        "baie",
        "cir",
        "gaine",
        "esc",
        "asc",
        "noyau",
        "san",
    )

    for tok in keep_tokens:
        if tok in layer_l:
            return True

    # la couche 0 peut contenir de la géométrie utile dans les blocs
    if layer_l == "0":
        return True

    # par défaut, on ne garde pas
    return False


def _layer_should_drop(layer: str, rules: dict) -> bool:
    keep_layers = {x.lower() for x in rules.get("keep_layers", [])}
    if keep_layers and layer.lower() in keep_layers:
        return False

    for token in rules.get("drop_layers_contains", []):
        if _layer_token_match(layer, token):
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


def _segment_angle_deg(seg):
    (x1, y1), (x2, y2) = seg
    ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
    return ang


def _is_axisish(seg, tol_deg: float = 12.0) -> bool:
    ang = _segment_angle_deg(seg)
    return (
        abs(ang - 0.0) <= tol_deg
        or abs(ang - 90.0) <= tol_deg
        or abs(ang - 180.0) <= tol_deg
    )


def _compute_size_filters(global_bbox: WorldBBox):
    ref = max(global_bbox.width, global_bbox.height)
    return {
        "min_arc_radius": ref * 0.0020,
        "min_segment_length": ref * 0.0006,

        # seuils anti-triangle / éventails
        "triangle_min_bbox": ref * 0.08,
        "triangle_min_area": (ref * ref) * 0.0020,

        # utilisé seulement pour jeter des très longues diagonales individuelles
        "long_diag_len": ref * 0.06,

        # NOUVEAU : seuil plus bas pour détecter les éventails
        "fan_diag_len": ref * 0.012,
        "fan_endpoint_radius": ref * 0.012,
        "fan_min_segments": 4,
        "fan_min_angle_spread_deg": 30.0,

        # nettoyage petits composants
        "cc_min_area_px": 50,
        "cc_min_width_px": 10,
        "cc_min_height_px": 10,
    }


# ============================================================
# Géométrie supplémentaire pour anti-triangles
# ============================================================

def _poly_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _polygon_area(pts):
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _angle_at(p_prev, p, p_next):
    ax, ay = p_prev[0] - p[0], p_prev[1] - p[1]
    bx, by = p_next[0] - p[0], p_next[1] - p[1]
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na == 0 or nb == 0:
        return 180.0
    c = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(c))


def _is_large_triangle_closed(pts, filters):
    """
    Détecte une polyline fermée triangulaire/quasi triangulaire suffisamment grande.
    """
    if len(pts) < 3:
        return False

    uniq = pts[:]
    if len(uniq) >= 2 and uniq[0] == uniq[-1]:
        uniq = uniq[:-1]

    if len(uniq) not in (3, 4):
        return False

    minx, miny, maxx, maxy = _poly_bbox(uniq)
    w = maxx - minx
    h = maxy - miny
    area = _polygon_area(uniq)

    if max(w, h) < filters["triangle_min_bbox"]:
        return False
    if area < filters["triangle_min_area"]:
        return False

    if len(uniq) == 3:
        return True

    # cas quadrilatère quasi triangle
    angles = []
    for i in range(len(uniq)):
        angles.append(_angle_at(uniq[i - 1], uniq[i], uniq[(i + 1) % len(uniq)]))
    if any(a > 170 for a in angles):
        return True

    return False


def _is_large_open_vshape(pts, filters):
    """
    Détecte un grand 'V' ouvert parasite.
    """
    if len(pts) != 3:
        return False

    s1 = [pts[0], pts[1]]
    s2 = [pts[1], pts[2]]

    l1 = _segment_length(s1)
    l2 = _segment_length(s2)
    if l1 < filters["long_diag_len"] or l2 < filters["long_diag_len"]:
        return False

    if _is_axisish(s1) or _is_axisish(s2):
        return False

    ang = _angle_at(pts[0], pts[1], pts[2])
    if ang < 15 or ang > 120:
        return False

    minx, miny, maxx, maxy = _poly_bbox(pts)
    if max(maxx - minx, maxy - miny) < filters["triangle_min_bbox"]:
        return False

    return True


def _remove_triangle_fans_once(segments, filters, debug: dict | None = None):
    """
    Supprime les paquets de diagonales obliques convergeant vers une même zone.
    Détection basée sur les MILIEUX des segments.
    Un seul passage.
    """
    if not segments:
        return segments

    candidate_idx = []
    candidate_angles = {}
    candidate_midpoints = {}

    for i, seg in enumerate(segments):
        length = _segment_length(seg)
        angle = _segment_angle_deg(seg)

        if length < filters["fan_diag_len"]:
            continue
        if _is_axisish(seg, tol_deg=4.0):
            continue

        candidate_idx.append(i)
        candidate_angles[i] = angle
        candidate_midpoints[i] = _segment_midpoint(seg)

    if not candidate_idx:
        if debug is not None:
            debug["candidate_count"] = 0
            debug["fan_clusters"] = 0
            debug["segments_removed"] = 0
        return segments

    r = filters["fan_endpoint_radius"]
    cell = max(r, 1e-9)

    midpoint_bins = defaultdict(list)

    def _bin_key(p):
        return (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)))

    for i in candidate_idx:
        midpoint_bins[_bin_key(candidate_midpoints[i])].append(i)

    fan_segment_ids = set()
    fan_clusters = 0
    visited_bins = set()

    for bk in list(midpoint_bins.keys()):
        if bk in visited_bins:
            continue

        neighborhood = []
        bx, by = bk
        local_bins = []

        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                nb = (bx + dx, by + dy)
                if nb in midpoint_bins:
                    neighborhood.extend(midpoint_bins[nb])
                    local_bins.append(nb)

        for nb in local_bins:
            visited_bins.add(nb)

        uniq = sorted(set(neighborhood))
        if len(uniq) < filters["fan_min_segments"]:
            continue

        angles = [candidate_angles[i] for i in uniq]
        angle_buckets = set(int(a // 15) for a in angles)
        if len(angle_buckets) < 3:
            continue

        spread = max(angles) - min(angles)
        if spread < filters["fan_min_angle_spread_deg"]:
            continue

        fan_clusters += 1
        fan_segment_ids.update(uniq)

    if debug is not None:
        debug["candidate_count"] = len(candidate_idx)
        debug["fan_clusters"] = fan_clusters
        debug["segments_removed"] = len(fan_segment_ids)

    if not fan_segment_ids:
        return segments

    return [seg for i, seg in enumerate(segments) if i not in fan_segment_ids]


def _remove_triangle_fans(segments, filters, debug: dict | None = None):
    """
    Plusieurs passes successives pour nettoyer les éventails résiduels.
    On resserre légèrement les seuils à chaque passe.
    """
    if not segments:
        return segments

    current = segments
    passes = []

    for pass_idx in range(3):
        pass_debug = {}

        local_filters = dict(filters)
        if pass_idx == 0:
            local_filters["fan_diag_len"] = filters["fan_diag_len"]
            local_filters["fan_endpoint_radius"] = filters["fan_endpoint_radius"]
            local_filters["fan_min_segments"] = 4
            local_filters["fan_min_angle_spread_deg"] = 30.0
        elif pass_idx == 1:
            local_filters["fan_diag_len"] = filters["fan_diag_len"] * 0.75
            local_filters["fan_endpoint_radius"] = filters["fan_endpoint_radius"] * 1.50
            local_filters["fan_min_segments"] = 3
            local_filters["fan_min_angle_spread_deg"] = 20.0
        else:
            local_filters["fan_diag_len"] = filters["fan_diag_len"] * 0.60
            local_filters["fan_endpoint_radius"] = filters["fan_endpoint_radius"] * 1.80
            local_filters["fan_min_segments"] = 3
            local_filters["fan_min_angle_spread_deg"] = 15.0

        before = len(current)
        current = _remove_triangle_fans_once(current, local_filters, debug=pass_debug)
        removed = before - len(current)

        passes.append({
            "pass_index": pass_idx + 1,
            "filters": {
                "fan_diag_len": local_filters["fan_diag_len"],
                "fan_endpoint_radius": local_filters["fan_endpoint_radius"],
                "fan_min_segments": local_filters["fan_min_segments"],
                "fan_min_angle_spread_deg": local_filters["fan_min_angle_spread_deg"],
            },
            "candidate_count": pass_debug.get("candidate_count", 0),
            "fan_clusters": pass_debug.get("fan_clusters", 0),
            "segments_removed": pass_debug.get("segments_removed", 0),
            "segments_before": before,
            "segments_after": len(current),
        })

        # si la passe n'enlève presque plus rien, on arrête
        if removed < 20:
            break

    if debug is not None:
        total_removed = len(segments) - len(current)
        debug["triangle_fan_cleanup"] = {
            "passes": passes,
            "segments_removed_total": total_removed,
        }

    return current


# ============================================================
# DXF -> segments
# ============================================================

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
        if not pts:
            return []

        closed = bool(entity.closed)

        # filtre triangle parasite
        if closed and _is_large_triangle_closed(pts + [pts[0]], geom_filters):
            return []

        if (not closed) and _is_large_open_vshape(pts, geom_filters):
            return []

        segs = _polyline_to_segments(pts)
        if entity.closed and len(pts) >= 2:
            segs.append([pts[-1], pts[0]])
        return _filter_short_segments(segs, geom_filters["min_segment_length"])

    if dxftype == "POLYLINE":
        pts = _polyline_points(entity)
        if not pts:
            return []

        closed = bool(getattr(entity, "is_closed", False))

        if closed and _is_large_triangle_closed(pts + [pts[0]], geom_filters):
            return []

        if (not closed) and _is_large_open_vshape(pts, geom_filters):
            return []

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

        elif dxftype == "SPLINE":
            pts = _spline_points(entity)
            raw_segments.extend(_polyline_to_segments(pts))

    if not raw_segments:
        raise RuntimeError("Impossible d'établir la bbox initiale à partir du linework.")

    return _segments_bbox(raw_segments)


def _collect_segments(
    entity,
    rules: dict,
    segments: list,
    segment_layers: list,
    stats: dict,
    geom_filters: dict,
    depth: int = 0,
    max_depth: int = 8

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
                segment_layers,
                stats,
                geom_filters,
                depth=depth + 1,
                max_depth=max_depth,
            )
        return

    segs = _entity_to_segments(entity, geom_filters)
    if segs:
        segments.extend(segs)
        segment_layers.extend([layer] * len(segs))
        stats["drawn"][dxftype] = stats["drawn"].get(dxftype, 0) + len(segs)
    else:
        stats["empty_geom"][dxftype] = stats["empty_geom"].get(dxftype, 0) + 1


# ============================================================
# Sélection de la bbox principale
# ============================================================

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

    def _interval_overlap(a1, a2, b1, b2):
        inter = max(0, min(a2, b2) - max(a1, b1))
        return inter

    best_x1 = best["x"]
    best_y1 = best["y"]
    best_x2 = best["x"] + best["w"]
    best_y2 = best["y"] + best["h"]

    merged = [best]

    for c in candidates[1:]:
        cx1 = c["x"]
        cy1 = c["y"]
        cx2 = c["x"] + c["w"]
        cy2 = c["y"] + c["h"]

        # si le composant touche à la fois le haut et la droite,
        # on se méfie fortement (typique des tableaux/légendes détachés)
        if c["top_touch"] and c["right_touch"]:
            continue

        x_overlap = _interval_overlap(best_x1, best_x2, cx1, cx2)
        y_overlap = _interval_overlap(best_y1, best_y2, cy1, cy2)

        x_gap = max(0, max(best_x1, cx1) - min(best_x2, cx2))
        y_gap = max(0, max(best_y1, cy1) - min(best_y2, cy2))

        # On fusionne seulement si le composant est réellement voisin :
        # - soit il chevauche bien verticalement et est proche en horizontal,
        # - soit il chevauche bien horizontalement et est proche en vertical.
        cond_horizontal_neighbor = (y_overlap >= 0.20 * min(best["h"], c["h"])) and (x_gap <= 0.30 * best["w"])
        cond_vertical_neighbor = (x_overlap >= 0.20 * min(best["w"], c["w"])) and (y_gap <= 0.30 * best["h"])

        if cond_horizontal_neighbor or cond_vertical_neighbor:
            merged.append(c)


    # Fusion des rectangles sélectionnés
    min_x = min(c["x"] for c in merged)
    min_y = min(c["y"] for c in merged)
    max_x = max(c["x"] + c["w"] for c in merged)
    max_y = max(c["y"] + c["h"] for c in merged)

    w = max_x - min_x
    h = max_y - min_y

    margin_x = int(round(w * 0.05))
    margin_y = int(round(h * 0.08))

    bx = max(0, min_x - margin_x)
    by = max(0, min_y - margin_y)
    bw = min(width_px - bx, w + 2 * margin_x)
    bh = min(height_px - by, h + 2 * margin_y)

    best_world_bbox = _px_to_world_bbox(bx, by, bw, bh, bbox, width_px, height_px)

    if debug is not None:
        debug["component_selection"] = {
            "mode": "connected_components_lowres_merged",
            "width_px": width_px,
            "height_px": height_px,
            "component_count": len(candidates),
            "selected_component": best,
            "merged_components": merged,
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


# ============================================================
# Nettoyage léger des petites composantes (version memory-safe)
# ============================================================

def _remove_small_components(segments, filters, debug: dict | None = None):
    """
    Supprime les très petites composantes isolées sans créer un masque complet
    par segment.
    """
    if not segments:
        return segments

    bbox = _segments_bbox(segments)

    target_long_side = 1400
    if bbox.width >= bbox.height:
        width_px = target_long_side
        height_px = max(500, int(round(target_long_side * bbox.height / bbox.width)))
    else:
        height_px = target_long_side
        width_px = max(500, int(round(target_long_side * bbox.width / bbox.height)))

    canvas = np.zeros((height_px, width_px), dtype=np.uint8)

    seg_midpoints_px = []
    for seg in segments:
        (x1, y1), (x2, y2) = seg
        p1 = _world_to_px(x1, y1, bbox, width_px, height_px)
        p2 = _world_to_px(x2, y2, bbox, width_px, height_px)
        cv2.line(canvas, p1, p2, 255, 1, lineType=cv2.LINE_8)

        mx, my = _segment_midpoint(seg)
        pm = _world_to_px(mx, my, bbox, width_px, height_px)
        seg_midpoints_px.append(pm)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    work = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
        (work > 0).astype(np.uint8),
        connectivity=8
    )

    kept_labels = set()
    for i in range(1, num_labels):
        x = int(stats_cc[i, cv2.CC_STAT_LEFT])
        y = int(stats_cc[i, cv2.CC_STAT_TOP])
        w = int(stats_cc[i, cv2.CC_STAT_WIDTH])
        h = int(stats_cc[i, cv2.CC_STAT_HEIGHT])
        area = int(stats_cc[i, cv2.CC_STAT_AREA])

        if area < filters["cc_min_area_px"]:
            continue
        if w < filters["cc_min_width_px"] and h < filters["cc_min_height_px"]:
            continue

        kept_labels.add(i)

    kept_segments = []
    for seg, (px, py) in zip(segments, seg_midpoints_px):
        lbl = labels[py, px]
        if lbl in kept_labels:
            kept_segments.append(seg)

    if debug is not None:
        debug["small_component_cleanup"] = {
            "segments_before": len(segments),
            "segments_after": len(kept_segments),
            "component_count": int(num_labels - 1),
            "kept_component_count": len(kept_labels),
            "width_px": width_px,
            "height_px": height_px,
        }

    return kept_segments


# ============================================================
# Compatibilité layout
# ============================================================

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


# ============================================================
# API publique principale
# ============================================================

def render_dxf_to_png(
    dxf_path: Path,
    png_path: Path,
    rules: dict,
    dpi: int = 300,
    debug_json: Path | None = None,
) -> Path:
    """
    Rendu principal du modelspace :
    - exactement la structure de ta version de référence
    - + filtre anti-éventails triangulaires
    - + nettoyage léger des petites composantes
    """
    initial_bbox = _initial_bbox_from_linework(dxf_path, rules)
    geom_filters = _compute_size_filters(initial_bbox)

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    segments = []
    segment_layers = []
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
        _collect_segments(entity, rules, segments, segment_layers, stats, geom_filters)

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

    cleaned_for_bbox = _remove_triangle_fans(segments, geom_filters, debug=debug)

    selected_bbox = _choose_main_component_bbox(segments, debug=debug)

    filtered_pairs = [
        (seg, layer)
        for seg, layer in zip(segments, segment_layers)
        if selected_bbox.min_x <= _segment_midpoint(seg)[0] <= selected_bbox.max_x
        and selected_bbox.min_y <= _segment_midpoint(seg)[1] <= selected_bbox.max_y
    ]

    filtered_segments = [seg for seg, _ in filtered_pairs]

    # Filtrage architectural APRES crop :
    # on enlève les couches toiture/plafond/surface/etc. sans casser la bbox.
    arch_pairs = [
        (seg, layer)
        for seg, layer in filtered_pairs
        if _layer_is_architectural_after_crop(layer)
    ]

    # On n'applique ce filtre que s'il garde une quantité raisonnable de géométrie.
    # Sinon on reste sur le crop brut.
    if len(arch_pairs) >= max(1200, int(0.18 * len(filtered_pairs))):
        filtered_pairs = arch_pairs
        filtered_segments = [seg for seg, _ in filtered_pairs]
        debug["post_crop_layer_filter"] = {
            "mode": "architectural_after_crop",
            "segments_before": len([seg for seg, _ in zip(segments, segment_layers)
                                    if selected_bbox.min_x <= _segment_midpoint(seg)[0] <= selected_bbox.max_x
                                    and selected_bbox.min_y <= _segment_midpoint(seg)[1] <= selected_bbox.max_y]),
            "segments_after": len(filtered_segments),
        }
    else:
        debug["post_crop_layer_filter"] = {
            "mode": "skipped_not_enough_geometry",
            "segments_after": len(filtered_segments),
        }

    # Si la sélection est trop faible, on élargit légèrement autour de la bbox
    if len(filtered_segments) < max(2000, int(0.03 * len(cleaned_for_bbox))):
        pad_x = selected_bbox.width * 0.15
        pad_y = selected_bbox.height * 0.15
        expanded = WorldBBox(
            min_x=selected_bbox.min_x - pad_x,
            min_y=selected_bbox.min_y - pad_y,
            max_x=selected_bbox.max_x + pad_x,
            max_y=selected_bbox.max_y + pad_y,
        )
        filtered_segments = _filter_segments_in_bbox(cleaned_for_bbox, expanded)
        debug["fallback_used"] = "expanded_selected_bbox"
        selected_bbox = expanded
    else:
        debug["fallback_used"] = None

    # SECOND PASS anti-éventails sur la géométrie déjà recadrée.
    second_pass_filters = dict(geom_filters)
    second_pass_filters["fan_diag_len"] = geom_filters["fan_diag_len"] * 0.85
    second_pass_filters["fan_endpoint_radius"] = geom_filters["fan_endpoint_radius"] * 1.20
    second_pass_filters["fan_min_segments"] = 3
    second_pass_filters["fan_min_angle_spread_deg"] = 25.0

    filtered_segments = _remove_triangle_fans(filtered_segments, second_pass_filters, debug=debug)


    # PASS 1 anti-éventails avec filtres globaux
    filtered_segments = _remove_triangle_fans(filtered_segments, geom_filters, debug=debug)

    # PASS 2 anti-éventails avec filtres locaux recalculés sur la bbox déjà recadrée
    local_bbox = _segments_bbox(filtered_segments)
    local_ref = max(local_bbox.width, local_bbox.height)

    local_filters = dict(geom_filters)
    local_filters["fan_diag_len"] = local_ref * 0.010
    local_filters["fan_endpoint_radius"] = local_ref * 0.015
    local_filters["fan_min_segments"] = 4
    local_filters["fan_min_angle_spread_deg"] = 35.0

    filtered_segments = _remove_triangle_fans(filtered_segments, local_filters, debug=debug)

    # Nettoyage léger des petits objets détachés
    filtered_segments = _remove_small_components(filtered_segments, local_filters, debug=debug)

    if not filtered_segments:
        raise RuntimeError("Tous les segments ont été supprimés après nettoyage.")

    cleaned_bbox = _segments_bbox(filtered_segments)

    debug["segment_count_selected"] = len(filtered_segments)
    debug["selected_bbox_after_cleanup"] = {
        "min_x": cleaned_bbox.min_x,
        "min_y": cleaned_bbox.min_y,
        "max_x": cleaned_bbox.max_x,
        "max_y": cleaned_bbox.max_y,
    }

    if debug_json is not None:
        debug_json.parent.mkdir(parents=True, exist_ok=True)
        debug_json.write_text(
            json.dumps(debug, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    lc = LineCollection(filtered_segments, colors="black", linewidths=0.18)
    ax.add_collection(lc)

    margin_x = cleaned_bbox.width * 0.02
    margin_y = cleaned_bbox.height * 0.02

    ax.set_xlim(cleaned_bbox.min_x - margin_x, cleaned_bbox.max_x + margin_x)
    ax.set_ylim(cleaned_bbox.min_y - margin_y, cleaned_bbox.max_y + margin_y)
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
