from __future__ import annotations

from pathlib import Path
import math
import ezdxf
from ezdxf.addons import Importer


def _layer_should_drop(layer: str, rules: dict) -> bool:
    keep_layers = {x.lower() for x in rules.get("keep_layers", [])}
    if keep_layers and layer.lower() in keep_layers:
        return False

    for token in rules.get("drop_layers_contains", []):
        if token.lower() in layer.lower():
            return True
    return False


def _entity_should_drop(entity, rules: dict) -> bool:
    if entity.dxftype() in set(rules.get("drop_entity_types", [])):
        return True

    layer = getattr(entity.dxf, "layer", "0")
    if _layer_should_drop(layer, rules):
        return True

    small_cfg = rules.get("drop_small_entities", {})
    if small_cfg.get("enabled", False):
        min_length = float(small_cfg.get("min_length", 0))
        if entity.dxftype() == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            length = math.dist((start.x, start.y), (end.x, end.y))
            if length < min_length:
                return True

    return False


def filter_dxf(input_dxf: Path, output_dxf: Path, rules: dict) -> Path:
    src_doc = ezdxf.readfile(input_dxf)
    src_msp = src_doc.modelspace()

    dst_doc = ezdxf.new(dxfversion=src_doc.dxfversion)
    dst_msp = dst_doc.modelspace()

    importer = Importer(src_doc, dst_doc)

    for entity in src_msp:
        if _entity_should_drop(entity, rules):
            continue
        importer.import_entity(entity, dst_msp)

    importer.finalize()
    output_dxf.parent.mkdir(parents=True, exist_ok=True)
    dst_doc.saveas(output_dxf)
    return output_dxf
