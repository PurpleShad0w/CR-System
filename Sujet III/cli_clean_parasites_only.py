from __future__ import annotations

import argparse
from src.yaml_utils import load_yaml
from src.parasite_raster_cleanup import clean_parasite_lines_from_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Nettoie les lignes parasites d'un rendu 2D, sans appliquer le shadow pass")
    parser.add_argument("input_png", help="Ex: output/rendered_clean_v8_2d.png")
    parser.add_argument("--rules", default="config/default_rules_shadow225_v9.yaml")
    parser.add_argument("--out", default="output/rendered_clean_v9_2d_cleaned.png")
    args = parser.parse_args()
    rules = load_yaml(args.rules, default={})
    clean_parasite_lines_from_image(args.input_png, args.out, rules.get("parasite_cleanup", {}) or {})
    print(f"OK - rendu 2D nettoyé: {args.out}")


if __name__ == "__main__":
    main()
