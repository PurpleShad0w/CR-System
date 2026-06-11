from __future__ import annotations

from pathlib import Path
import math
import ezdxf


SUPPORTED_TYPES = {
    "LINE",
    "LWPOLYLINE",
    "POLYLINE",
    "ARC",
    "CIRCLE",
    "ELLIPSE",
    "SPLINE",
    "INSERT",   # on les traite par explosion contrôlée
}

DROP_TYPES = {
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


def _entity_should_drop(entity, rules: dict) -> bool:
    dxftype = entity.dxftype()

    if dxftype in DROP_TYPES:
        return True

    if dxftype not in SUPPORTED_TYPES:
        return True

    layer = _safe_layer(entity)
    if _layer_should_drop(layer, rules):
        return True

    small_cfg = rules.get("drop_small_entities", {})
    if small_cfg.get("enabled", False) and dxftype == "LINE":
        try:
            start = entity.dxf.start
            end = entity.dxf.end
            length = math.dist((start.x, start.y), (end.x, end.y))
            if length < float(small_cfg.get("min_length", 0)):
                return True
        except Exception:
            return True

    return False


def _copy_basic_entity(src_entity, dst_msp, forced_layer: str | None = None):
    dxftype = src_entity.dxftype()
    layer = forced_layer or _safe_layer(src_entity)

    if dxftype == "LINE":
        return dst_msp.add_line(
            src_entity.dxf.start,
            src_entity.dxf.end,
            dxfattribs={"layer": layer},
        )

    if dxftype == "LWPOLYLINE":
        points = list(src_entity.get_points("xyb"))
        return dst_msp.add_lwpolyline(
            points,
            format="xyb",
            close=src_entity.closed,
            dxfattribs={"layer": layer},
        )

    if dxftype == "POLYLINE":
        try:
            points = [v.dxf.location for v in src_entity.vertices]
            return dst_msp.add_polyline2d(
                points,
                close=getattr(src_entity, "is_closed", False),
                dxfattribs={"layer": layer},
            )
        except Exception:
            return None

    if dxftype == "ARC":
        return dst_msp.add_arc(
            center=src_entity.dxf.center,
            radius=src_entity.dxf.radius,
            start_angle=src_entity.dxf.start_angle,
            end_angle=src_entity.dxf.end_angle,
            dxfattribs={"layer": layer},
        )

    if dxftype == "CIRCLE":
        return dst_msp.add_circle(
            center=src_entity.dxf.center,
            radius=src_entity.dxf.radius,
            dxfattribs={"layer": layer},
        )

    if dxftype == "ELLIPSE":
        return dst_msp.add_ellipse(
            center=src_entity.dxf.center,
            major_axis=src_entity.dxf.major_axis,
            ratio=src_entity.dxf.ratio,
            start_param=src_entity.dxf.start_param,
            end_param=src_entity.dxf.end_param,
            dxfattribs={"layer": layer},
        )

    if dxftype == "SPLINE":
        try:
            spline = dst_msp.add_spline(dxfattribs={"layer": layer})
            spline.control_points = src_entity.control_points
            return spline
        except Exception:
            return None

    return None


def _flatten_insert(insert_entity, dst_msp, rules: dict, depth: int = 0, max_depth: int = 5) -> int:
    """
    Explose un INSERT depuis le document source (où les block definitions existent encore),
    puis copie uniquement les primitives utiles.
    """
    if depth > max_depth:
        return 0

    copied_count = 0
    parent_layer = _safe_layer(insert_entity)

    try:
        virtuals = list(insert_entity.virtual_entities())
    except Exception:
        return 0

    for sub in virtuals:
        dxftype = sub.dxftype()

        # Hérite du layer du sous-objet si présent, sinon du parent insert
        effective_layer = _safe_layer(sub)
        if not effective_layer or effective_layer == "0":
            effective_layer = parent_layer

        # Si le layer du contenu doit être supprimé, on saute
        if _layer_should_drop(effective_layer, rules):
            continue

        # On retire explicitement les types de bruit
        if dxftype in DROP_TYPES:
            continue

        # Si on rencontre un INSERT imbriqué, on recurse
        if dxftype == "INSERT":
            copied_count += _flatten_insert(sub, dst_msp, rules, depth + 1, max_depth)
            continue

        # On ne garde que les primitives supportées
        if dxftype not in {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"}:
            continue

        copied = _copy_basic_entity(sub, dst_msp, forced_layer=effective_layer)
        if copied is not None:
            copied_count += 1

    return copied_count


def filter_dxf(input_dxf: Path, output_dxf: Path, rules: dict) -> Path:
    src_doc = ezdxf.readfile(input_dxf)
    src_msp = src_doc.modelspace()

    dst_doc = ezdxf.new(dxfversion=src_doc.dxfversion)
    dst_msp = dst_doc.modelspace()

    kept = 0
    skipped = 0
    kept_by_type = {}
    skipped_by_type = {}

    for entity in src_msp:
        dxftype = entity.dxftype()

        if _entity_should_drop(entity, rules):
            skipped += 1
            skipped_by_type[dxftype] = skipped_by_type.get(dxftype, 0) + 1
            continue

        if dxftype == "INSERT":
            n = _flatten_insert(entity, dst_msp, rules)
            if n > 0:
                kept += n
                kept_by_type["INSERT_EXPLODED"] = kept_by_type.get("INSERT_EXPLODED", 0) + n
            else:
                skipped += 1
                skipped_by_type["INSERT"] = skipped_by_type.get("INSERT", 0) + 1
            continue

        copied = _copy_basic_entity(entity, dst_msp)
        if copied is not None:
            kept += 1
            kept_by_type[dxftype] = kept_by_type.get(dxftype, 0) + 1
        else:
            skipped += 1
            skipped_by_type[dxftype] = skipped_by_type.get(dxftype, 0) + 1

    output_dxf.parent.mkdir(parents=True, exist_ok=True)
    dst_doc.saveas(output_dxf)

    print(f"[filter_dxf] kept={kept} skipped={skipped}")
    print(f"[filter_dxf] kept_by_type={kept_by_type}")
    print(f"[filter_dxf] skipped_by_type={skipped_by_type}")

    return output_dxf
