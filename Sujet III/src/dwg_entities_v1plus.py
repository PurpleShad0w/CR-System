from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import ezdxf
import numpy as np
import pandas as pd

LINEAR_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"}


@dataclass
class ConversionConfig:
    backend: str = "oda"
    exe_path: str = ""
    version: str = "ACAD2018"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return default if v is None else float(v)
    except Exception:
        return default


def _xy(point: Any) -> tuple[float, float]:
    return _safe_float(point[0]), _safe_float(point[1])


def _bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _length(points: list[tuple[float, float]], closed: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    pts = points + [points[0]] if closed and points[0] != points[-1] else points
    return float(sum(math.dist(a, b) for a, b in zip(pts[:-1], pts[1:])))


def _arc(center: Any, radius: float, start_deg: float, end_deg: float, n: int = 40) -> list[tuple[float, float]]:
    if end_deg < start_deg:
        end_deg += 360.0
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), max(10, n))
    cx, cy = _xy(center)
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]


def _circle(center: Any, radius: float, n: int = 96) -> list[tuple[float, float]]:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=True)
    cx, cy = _xy(center)
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]


def _entity_points(entity: Any) -> tuple[list[tuple[float, float]], bool]:
    t = entity.dxftype()
    try:
        if t == "LINE":
            return [_xy(entity.dxf.start), _xy(entity.dxf.end)], False
        if t == "LWPOLYLINE":
            return [(float(x), float(y)) for x, y, *_ in entity.get_points("xy")], bool(entity.closed)
        if t == "POLYLINE":
            return [_xy(v.dxf.location) for v in entity.vertices], bool(getattr(entity, "is_closed", False))
        if t == "ARC":
            return _arc(entity.dxf.center, float(entity.dxf.radius), float(entity.dxf.start_angle), float(entity.dxf.end_angle)), False
        if t == "CIRCLE":
            return _circle(entity.dxf.center, float(entity.dxf.radius)), True
        if t == "ELLIPSE":
            return [_xy(p) for p in entity.flattening(distance=1.5)], bool(getattr(entity, "closed", False))
        if t == "SPLINE":
            return [_xy(p) for p in entity.flattening(distance=1.5)], False
    except Exception:
        return [], False
    return [], False


def _record(entity: Any, index: int, source: str = "modelspace", source_block: str = "") -> dict[str, Any]:
    layer = getattr(entity.dxf, "layer", "0") if hasattr(entity, "dxf") else "0"
    typ = entity.dxftype()
    points, closed = _entity_points(entity)
    x0, y0, x1, y1 = _bbox(points)
    length = _length(points, closed)
    return {
        "row_id": index,
        "entity_id": str(getattr(entity.dxf, "handle", None) or f"entity_{index}"),
        "parent_id": str(getattr(entity.dxf, "owner", "") or ""),
        "source": source,
        "source_block": source_block,
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
        "bbox_diag": math.hypot(x1 - x0, y1 - y0),
        "length": length,
        "color": getattr(entity.dxf, "color", None) if hasattr(entity, "dxf") else None,
        "lineweight": getattr(entity.dxf, "lineweight", None) if hasattr(entity, "dxf") else None,
        "is_linear": typ in LINEAR_TYPES,
    }


def convert_dwg_to_dxf(dwg_path: str | Path, out_dir: str | Path, cfg: Optional[ConversionConfig] = None) -> Path:
    cfg = cfg or ConversionConfig()
    dwg_path = Path(dwg_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if dwg_path.suffix.lower() == ".dxf":
        return dwg_path
    if not cfg.exe_path or not Path(cfg.exe_path).exists():
        raise FileNotFoundError("Convertisseur ODA introuvable. Renseigne converter.exe_path dans la config.")
    subprocess.run([
        cfg.exe_path,
        str(dwg_path.parent),
        str(out_dir),
        cfg.version,
        "DXF",
        "0",
        "1",
        str(dwg_path.name),
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    candidates = list(out_dir.glob(dwg_path.stem + "*.dxf"))
    if not candidates:
        raise FileNotFoundError(f"Aucun DXF généré pour {dwg_path}")
    return candidates[0]


def extract_entities_df(path: str | Path, converter: Optional[dict[str, Any]] = None, include_blocks: bool = True) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".dwg":
        path = convert_dwg_to_dxf(path, Path(tempfile.mkdtemp(prefix="sujet3_dwg_")), ConversionConfig(**(converter or {})))
    doc = ezdxf.readfile(path)
    records: list[dict[str, Any]] = []
    for entity in doc.modelspace():
        if entity.dxftype() == "INSERT" and include_blocks:
            block_name = str(getattr(entity.dxf, "name", ""))
            try:
                for virtual in entity.virtual_entities():
                    records.append(_record(virtual, len(records), source="block_virtual", source_block=block_name))
            except Exception:
                records.append(_record(entity, len(records), source="insert", source_block=block_name))
        else:
            records.append(_record(entity, len(records)))
    return pd.DataFrame(records)
