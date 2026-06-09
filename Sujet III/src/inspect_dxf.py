from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import ezdxf


def inspect_dxf(dxf_path: Path, output_json: Path | None = None) -> dict:
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    entity_counter = Counter()
    layer_counter = Counter()

    for e in msp:
        entity_counter[e.dxftype()] += 1
        layer_name = getattr(e.dxf, "layer", "0")
        layer_counter[layer_name] += 1

    result = {
        "file": str(dxf_path),
        "entity_types": dict(entity_counter),
        "layers": dict(layer_counter),
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result
