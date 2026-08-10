from __future__ import annotations

import argparse
from pathlib import Path

from src.yaml_utils import load_yaml
from src.parasite_raster_cleanup import clean_parasite_lines_from_image
from src.shadow225_v9 import render_shadow225_from_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Nettoie parasites puis applique le shadow-only 2.25D")
    parser.add_argument("input_png", help="Ex: output/rendered_clean_v8_2d.png")
    parser.add_argument("--rules", default="config/default_rules_shadow225_v9.yaml")
    parser.add_argument("--out-2d", default="output/rendered_clean_v9_2d_cleaned.png")
    parser.add_argument("--out-225d", default="output/rendered_clean_v9_shadow225.png")
    args = parser.parse_args()
    rules = load_yaml(args.rules, default={})
    cleaned = clean_parasite_lines_from_image(args.input_png, args.out_2d, rules.get("parasite_cleanup", {}) or {})
    render_shadow225_from_image(cleaned, args.out_225d, rules.get("shadow_225d", {}) or {})
    print(f"OK - rendu 2D nettoyé: {args.out_2d}")
    print(f"OK - rendu shadow-only 2.25D: {args.out_225d}")


if __name__ == "__main__":
    main()
