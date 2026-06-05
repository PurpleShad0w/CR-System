from __future__ import annotations

from pathlib import Path
import argparse

from src.utils import load_yaml, ensure_dir
from src.convert_dwg import convert_dwg_to_dxf
from src.inspect_dxf import inspect_dxf
from src.filter_dxf import filter_dxf
from src.render_dxf import render_dxf_to_png
from src.clean_raster import clean_rendered_plan

from src.convert_dwg import ODAFileConverter, LibreDWGConverter

def build_converter(cfg: dict):
    backend = cfg["converter"]["backend"].lower()
    exe_path = Path(cfg["converter"]["exe_path"])

    if backend == "oda":
        return ODAFileConverter(
            exe_path=exe_path,
            version=cfg["converter"].get("version", "ACAD2018"),
        )
    if backend == "libredwg":
        return LibreDWGConverter(exe_path=exe_path)

    raise ValueError(f"Unsupported converter backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dwg", type=Path, required=True)
    parser.add_argument("--oda-exe", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/default_rules.yaml"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/work"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    args = parser.parse_args()

    rules = load_yaml(args.config)
    work_dir = ensure_dir(args.work_dir)
    output_dir = ensure_dir(args.output_dir)

    dwg_input_dir = ensure_dir(work_dir / "dwg_in")
    dxf_output_dir = ensure_dir(work_dir / "dxf_out")

    local_dwg = dwg_input_dir / args.input_dwg.name
    local_dwg.write_bytes(args.input_dwg.read_bytes())

    convert_dwg_to_dxf(
        oda_exe=args.oda_exe,
        input_dir=dwg_input_dir,
        output_dir=dxf_output_dir,
    )

    converter = build_converter(rules)
    dxf_path = converter.convert(args.input_dwg, dxf_output_dir)
    inspect_json = work_dir / "inspection.json"
    inspect_dxf(dxf_path, inspect_json)

    filtered_dxf = work_dir / "filtered.dxf"
    filter_dxf(dxf_path, filtered_dxf, rules)

    rendered_png = output_dir / "rendered.png"
    render_dxf_to_png(
        filtered_dxf,
        rendered_png,
        dpi=int(rules.get("render", {}).get("dpi", 300)),
    )

    if rules.get("raster_cleanup", {}).get("enabled", True):
        cleaned_png = output_dir / "cleaned.png"
        clean_rendered_plan(
            rendered_png,
            cleaned_png,
            threshold=int(rules["raster_cleanup"].get("threshold", 220)),
            min_component_area=int(rules["raster_cleanup"].get("min_component_area", 20)),
            morph_open_kernel=int(rules["raster_cleanup"].get("morph_open_kernel", 3)),
            morph_close_kernel=int(rules["raster_cleanup"].get("morph_close_kernel", 0)),
        )

    print("OK")


if __name__ == "__main__":
    main()
