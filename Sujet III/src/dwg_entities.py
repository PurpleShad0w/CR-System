from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import ezdxf
import numpy as np
import pandas as pd


LINEAR_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"}
TEXTUAL_TYPES = {"TEXT", "MTEXT", "DIMENSION", "LEADER", "MULTILEADER", "TABLE"}


@dataclass
class ConversionConfig:
    backend: str = "oda"
    exe_path: str = ""
    version: str = "ACAD2018"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _xy(point: Any) -> tuple[float, float]:
    return (_safe_float(point[0]), _safe_float(point[1]))


def _bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _length(points: list[tuple[float, float]], closed: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    pts = points + [points[0]] if closed and points[0] != points[-1] else points
    return float(sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:])))


def _polygon_area(points: list[tuple[float, float]], closed: bool = False) -> float:
    if len(points) < 3:
        return 0.0
    pts = points + [points[0]] if points[0] != points[-1] else points
    if not closed and points[0] != points[-1]:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _approx_arc(center: Any, radius: float, start_deg: float, end_deg: float, n: int = 32) -> list[tuple[float, float]]:
    if end_deg < start_deg:
        end_deg += 360.0
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), max(8, n))
    cx, cy = _xy(center)
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]


def _approx_circle(center: Any, radius: float, n: int = 80) -> list[tuple[float, float]]:
    cx, cy = _xy(center)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=True)
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]


def _entity_points(entity: Any) -> tuple[list[tuple[float, float]], bool]:
    t = entity.dxftype()
    try:
        if t == "LINE":
            return [_xy(entity.dxf.start), _xy(entity.dxf.end)], False
        if t == "LWPOLYLINE":
            pts = [(float(x), float(y)) for x, y, *_ in entity.get_points("xy")]
            return pts, bool(entity.closed)
        if t == "POLYLINE":
            pts = [_xy(v.dxf.location) for v in entity.vertices]
            return pts, bool(getattr(entity, "is_closed", False))
        if t == "ARC":
            return _approx_arc(entity.dxf.center, float(entity.dxf.radius), float(entity.dxf.start_angle), float(entity.dxf.end_angle)), False
        if t == "CIRCLE":
            return _approx_circle(entity.dxf.center, float(entity.dxf.radius)), True
        if t == "ELLIPSE":
            pts = [_xy(p) for p in entity.flattening(distance=2.0)]
            return pts, bool(getattr(entity, "closed", False))
        if t == "SPLINE":
            pts = [_xy(p) for p in entity.flattening(distance=2.0)]
            return pts, False
    except Exception:
        return [], False
    return [], False


def _basic_record(entity: Any, index: int, source: str = "modelspace") -> dict[str, Any]:
    layer = getattr(entity.dxf, "layer", "0") if hasattr(entity, "dxf") else "0"
    typ = entity.dxftype()
    handle = getattr(entity.dxf, "handle", None) if hasattr(entity, "dxf") else None
    parent_id = getattr(entity.dxf, "owner", None) if hasattr(entity, "dxf") else None
    points, closed = _entity_points(entity)
    x0, y0, x1, y1 = _bbox(points)
    length = _length(points, closed=closed)
    area = _polygon_area(points, closed=closed)
    color = getattr(entity.dxf, "color", None) if hasattr(entity, "dxf") else None
    lineweight = getattr(entity.dxf, "lineweight", None) if hasattr(entity, "dxf") else None
    return {
        "row_id": index,
        "entity_id": str(handle or f"entity_{index}"),
        "parent_id": str(parent_id or ""),
        "source": source,
        "layer": str(layer),
        "entity_type": typ,
        "points": points,
        "closed": bool(closed),
        "n_points": len(points),
        "bbox_min_x": x0,
        "bbox_min_y": y0,
        "bbox_max_x": x1,
        "bbox_max_y": y1,
        "bbox_width": x1 - x0,
        "bbox_height": y1 - y0,
        "length": length,
        "area": area,
        "color": color,
        "lineweight": lineweight,
        "is_textual": typ in TEXTUAL_TYPES,
        "is_linear": typ in LINEAR_TYPES,
    }


def convert_dwg_to_dxf(dwg_path: str | Path, out_dir: str | Path, cfg: Optional[ConversionConfig] = None) -> Path:
    """Convertit un DWG en DXF via ODA/Teigha si nécessaire.

    Remarque : ce wrapper reste volontairement simple. Si ton projet possède déjà
    une fonction de conversion DWG->DXF, garde-la et branche seulement le DXF
    généré sur extract_entities_df().
    """
    cfg = cfg or ConversionConfig()
    dwg_path = Path(dwg_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if dwg_path.suffix.lower() == ".dxf":
        return dwg_path
    if dwg_path.suffix.lower() != ".dwg":
        raise ValueError(f"Extension non supportée: {dwg_path.suffix}")
    if not cfg.exe_path or not Path(cfg.exe_path).exists():
        raise FileNotFoundError("Convertisseur ODA introuvable. Renseigne converter.exe_path dans config/default_rules.yaml.")
    cmd = [cfg.exe_path, str(dwg_path.parent), str(out_dir), cfg.version, "DXF", "0", "1", str(dwg_path.name)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    candidates = list(out_dir.glob(dwg_path.stem + "*.dxf"))
    if not candidates:
        raise FileNotFoundError(f"Aucun DXF généré pour {dwg_path}")
    return candidates[0]


def extract_entities_df(path: str | Path, converter: Optional[dict[str, Any]] = None, include_blocks: bool = True) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".dwg":
        cfg = ConversionConfig(**(converter or {}))
        tmp_dir = Path(tempfile.mkdtemp(prefix="sujet3_dwg_"))
        path = convert_dwg_to_dxf(path, tmp_dir, cfg)
    doc = ezdxf.readfile(path)
    records: list[dict[str, Any]] = []
    modelspace = doc.modelspace()
    for i, entity in enumerate(modelspace):
        if entity.dxftype() == "INSERT" and include_blocks:
            try:
                for j, virtual in enumerate(entity.virtual_entities()):
                    records.append(_basic_record(virtual, len(records), source=f"block:{getattr(entity.dxf, 'name', '')}"))
            except Exception:
                records.append(_basic_record(entity, len(records), source="modelspace"))
        else:
            records.append(_basic_record(entity, len(records), source="modelspace"))
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["row_id", "entity_id", "layer", "entity_type", "points"])
    return df


def layer_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["bbox_area"] = tmp["bbox_width"].abs() * tmp["bbox_height"].abs()
    return (
        tmp.groupby("layer")
        .agg(
            n_entities=("entity_id", "count"),
            entity_types=("entity_type", lambda s: ", ".join(sorted(set(map(str, s))))),
            total_length=("length", "sum"),
            total_area=("area", "sum"),
            bbox_area=("bbox_area", "sum"),
            n_textual=("is_textual", "sum"),
            n_linear=("is_linear", "sum"),
        )
        .reset_index()
        .sort_values(["n_entities", "total_length"], ascending=False)
    )
