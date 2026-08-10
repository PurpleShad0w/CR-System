from __future__ import annotations

import argparse
from src.yaml_utils import load_yaml
from src.nonwall_line_cleanup import clean_nonwall_lines


def main() -> None:
    parser = argparse.ArgumentParser(description='Nettoie les lignes non-mur d’un rendu 2D')
    parser.add_argument('input_png', help='Rendu 2D source, ex: output/rendered_clean_v8_2d.png')
    parser.add_argument('--rules', default='config/default_rules_shadow225_v10.yaml')
    parser.add_argument('--out', default='output/rendered_clean_v10_2d_nonwall_cleaned.png')
    args = parser.parse_args()
    rules = load_yaml(args.rules, default={})
    clean_nonwall_lines(args.input_png, args.out, rules.get('nonwall_cleanup', {}) or {})
    print(f'OK - 2D nettoyé: {args.out}')


if __name__ == '__main__':
    main()
