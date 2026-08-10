from __future__ import annotations

import argparse
from src.selection_v8 import load_yaml
from src.shadow225_v8 import render_shadow225_from_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Applique seulement le rendu shadow-only 2.25D sur un PNG existant")
    parser.add_argument("input_png", help="Ex: output/rendered_clean_v1plus_from_existing.png")
    parser.add_argument("--rules", default="config/default_rules_shadow225_v8.yaml")
    parser.add_argument("--out", default="output/rendered_clean_shadow225_v8_from_existing.png")
    args = parser.parse_args()
    rules = load_yaml(args.rules, default={})
    render_shadow225_from_image(args.input_png, args.out, rules.get("shadow_225d", {}) or {})
    print(f"OK - shadow-only 2.25D: {args.out}")


if __name__ == "__main__":
    main()
