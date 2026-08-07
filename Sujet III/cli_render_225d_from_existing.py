from __future__ import annotations

import argparse
from src.yaml_utils import load_yaml
from src.raster_225d import render_225d_from_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Produit une vue 2.25D subtile depuis un rendu PNG existant")
    parser.add_argument("input_png", help="Ex: output/rendered_clean_v1plus_from_existing.png")
    parser.add_argument("--rules", default="config/default_rules_225d.yaml")
    parser.add_argument("--out", default="output/rendered_clean_225d.png")
    args = parser.parse_args()
    rules = load_yaml(args.rules, default={})
    render_225d_from_image(args.input_png, args.out, (rules.get("render_225d") or {}))
    print(f"OK - vue 2.25D: {args.out}")


if __name__ == "__main__":
    main()
