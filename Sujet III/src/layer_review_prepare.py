from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import re
import yaml
import numpy as np

import ezdxf
import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from src.utils import load_yaml, ensure_dir
from src.render_dxf import (
    _initial_bbox_from_linework,
    _compute_size_filters,
    _collect_segments,
    _segments_bbox,
)


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE)
    return name[:180] or "layer"


def _segment_length(seg) -> float:
    (x1, y1), (x2, y2) = seg
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _render_layer_preview(segments: list, out_png: Path, dpi: int = 120) -> bool:
    """
    Génère une preview robuste pour un layer.

    Points importants :
    - ne pas utiliser bbox_inches='tight' ;
    - gérer les bbox dégénérées ou très allongées ;
    - ne jamais faire planter toute la préparation à cause d'un seul layer.
    """
    if not segments:
        return False

    try:
        bbox = _segments_bbox(segments)
    except Exception:
        return False

    min_x = float(bbox.min_x)
    min_y = float(bbox.min_y)
    max_x = float(bbox.max_x)
    max_y = float(bbox.max_y)

    if not all(np.isfinite(v) for v in [min_x, min_y, max_x, max_y]):
        return False

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)

    span_x = max_x - min_x
    span_y = max_y - min_y

    # Cas dégénéré : layer avec une ligne unique horizontale/verticale.
    base_span = max(abs(span_x), abs(span_y), 1.0)

    if span_x <= 1e-9:
        span_x = base_span * 0.05

    if span_y <= 1e-9:
        span_y = base_span * 0.05

    # Évite les ratios extrêmes qui font parfois planter bbox/tile côté PIL.
    max_aspect = 8.0
    aspect = span_x / max(span_y, 1e-9)

    if aspect > max_aspect:
        span_y = span_x / max_aspect
    elif aspect < 1.0 / max_aspect:
        span_x = span_y / max_aspect

    margin_x = span_x * 0.05
    margin_y = span_y * 0.05

    x1 = cx - span_x / 2.0 - margin_x
    x2 = cx + span_x / 2.0 + margin_x
    y1 = cy - span_y / 2.0 - margin_y
    y2 = cy + span_y / 2.0 + margin_y

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    try:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        lc = LineCollection(segments, colors="black", linewidths=0.25)
        ax.add_collection(lc)

        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # Important :
        # ne PAS utiliser bbox_inches="tight" ici.
        # Sur certains layers dégénérés, PIL peut lever :
        # SystemError: tile cannot extend outside image
        fig.savefig(
            out_png,
            dpi=dpi,
            facecolor="white",
            pad_inches=0,
        )
        return True

    except Exception:
        return False

    finally:
        plt.close(fig)


def prepare_layer_review(
    dxf_path: Path,
    config_path: Path,
    out_dir: Path,
    max_preview_layers: int = 250,
) -> None:
    rules = load_yaml(config_path)

    out_dir = ensure_dir(out_dir)
    previews_dir = ensure_dir(out_dir / "previews")

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
        _collect_segments(
            entity,
            rules,
            segments,
            segment_layers,
            stats,
            geom_filters,
        )

    layer_segments: dict[str, list] = {}
    for seg, layer in zip(segments, segment_layers):
        layer_segments.setdefault(layer, []).append(seg)

    rows = []
    for layer, segs in layer_segments.items():
        try:
            bbox = _segments_bbox(segs)
            total_length = sum(_segment_length(s) for s in segs)
            rows.append({
                "layer": layer,
                "segment_count": len(segs),
                "total_length": total_length,
                "min_x": bbox.min_x,
                "min_y": bbox.min_y,
                "max_x": bbox.max_x,
                "max_y": bbox.max_y,
                "width": bbox.width,
                "height": bbox.height,
                "preview": str(previews_dir / f"{_safe_filename(layer)}.png"),
            })
        except Exception:
            rows.append({
                "layer": layer,
                "segment_count": len(segs),
                "total_length": 0,
                "min_x": "",
                "min_y": "",
                "max_x": "",
                "max_y": "",
                "width": "",
                "height": "",
                "preview": "",
            })

    rows.sort(key=lambda r: (r["segment_count"], r["total_length"]), reverse=True)

    manifest_csv = out_dir / "layers_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "layer",
            "segment_count",
            "total_length",
            "min_x",
            "min_y",
            "max_x",
            "max_y",
            "width",
            "height",
            "preview",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_json = out_dir / "layers_manifest.json"
    manifest_json.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Génère les previews des layers principaux.
    # Une preview qui échoue ne doit pas arrêter toute la préparation.
    skipped_previews = []

    for row in rows[:max_preview_layers]:
        layer = row["layer"]
        preview_path = previews_dir / f"{_safe_filename(layer)}.png"

        try:
            ok = _render_layer_preview(layer_segments[layer], preview_path)
            if not ok:
                skipped_previews.append({
                    "layer": layer,
                    "reason": "preview_generation_returned_false",
                })
        except Exception as e:
            skipped_previews.append({
                "layer": layer,
                "reason": repr(e),
            })

    skipped_path = out_dir / "skipped_previews.json"
    skipped_path.write_text(
        json.dumps(skipped_previews, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    decisions_path = out_dir / "layer_decisions.yaml"
    if not decisions_path.exists():
        decisions = {
            "keep": [],
            "drop": [],
            "undecided": [row["layer"] for row in rows],
        }
        decisions_path.write_text(
            yaml.safe_dump(decisions, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    print(f"[layer_review_prepare] manifest: {manifest_csv}")
    print(f"[layer_review_prepare] decisions: {decisions_path}")
    print(f"[layer_review_prepare] previews: {previews_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/default_rules.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/work/layer_review"))
    parser.add_argument("--max-preview-layers", type=int, default=250)
    args = parser.parse_args()

    prepare_layer_review(
        dxf_path=args.dxf,
        config_path=args.config,
        out_dir=args.out_dir,
        max_preview_layers=args.max_preview_layers,
    )


if __name__ == "__main__":
    main()