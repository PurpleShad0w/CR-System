from __future__ import annotations

import argparse
from src.filter_v1plus import load_yaml
from src.raster_main_plan_cleanup import cleanup_main_plan_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Nettoie un rendered_clean.png existant sans rerendre le DWG")
    parser.add_argument("input_png", help="Ex: output/rendered_clean.png")
    parser.add_argument("--rules", default="config/default_rules_v1plus.yaml")
    parser.add_argument("--out", default="output/rendered_clean_v1plus_from_existing.png")
    args = parser.parse_args()
    rules = load_yaml(args.rules, default={})
    cleanup_main_plan_image(args.input_png, args.out, rules)
    print(f"OK - image nettoyée: {args.out}")


if __name__ == "__main__":
    main()
