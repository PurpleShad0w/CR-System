from __future__ import annotations

import argparse
from src.yaml_utils import load_yaml
from src.nonwall_line_cleanup import clean_nonwall_lines
from src.shadow225_v10 import render_shadow225_from_image


def main() -> None:
    parser = argparse.ArgumentParser(description='Nettoie les lignes non-mur puis applique le shadow-only 2.25D')
    parser.add_argument('input_png', help='Rendu 2D source, ex: output/rendered_clean_v8_2d.png')
    parser.add_argument('--rules', default='config/default_rules_shadow225_v10.yaml')
    parser.add_argument('--out-2d', default='output/rendered_clean_v10_2d_nonwall_cleaned.png')
    parser.add_argument('--out-225d', default='output/rendered_clean_v10_shadow225.png')
    args = parser.parse_args()
    rules = load_yaml(args.rules, default={})
    cleaned = clean_nonwall_lines(args.input_png, args.out_2d, rules.get('nonwall_cleanup', {}) or {})
    render_shadow225_from_image(cleaned, args.out_225d, rules.get('shadow_225d', {}) or {})
    print(f'OK - 2D nettoyé: {args.out_2d}')
    print(f'OK - shadow-only 2.25D: {args.out_225d}')


if __name__ == '__main__':
    main()
