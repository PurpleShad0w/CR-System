from __future__ import annotations

import argparse
from pathlib import Path

from src.dwg_entities_v8 import extract_entities_df
from src.selection_v8 import load_yaml
from src.render_2d_v8 import render_2d_v8
from src.shadow225_v8 import render_shadow225_from_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Sujet III v8 - rendu précis + shadow-only 2.25D")
    parser.add_argument("input", help="Fichier .dwg ou .dxf")
    parser.add_argument("--rules", default="config/default_rules_shadow225_v8.yaml")
    parser.add_argument("--decisions", default="data/work/layer_review/layer_decisions.yaml")
    parser.add_argument("--out-2d", default="output/rendered_clean_v8_2d.png")
    parser.add_argument("--out-225d", default="output/rendered_clean_v8_shadow225.png")
    parser.add_argument("--entities-csv", default="output/entities_df_v8.csv")
    parser.add_argument("--selected-csv", default="output/selected_entities_v8.csv")
    parser.add_argument("--debug-csv", default="output/debug_selection_v8.csv")
    args = parser.parse_args()

    rules = load_yaml(args.rules, default={})
    decisions = load_yaml(args.decisions, default={"keep": [], "drop": [], "undecided": []})
    df = extract_entities_df(args.input, converter=(rules.get("converter") or {}), include_blocks=True)
    Path(args.entities_csv).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["points"], errors="ignore").to_csv(args.entities_csv, index=False)

    out_2d, selected, debug = render_2d_v8(df, args.out_2d, rules=rules, decisions=decisions)
    selected.drop(columns=["points"], errors="ignore").to_csv(args.selected_csv, index=False)
    debug.drop(columns=["points"], errors="ignore").to_csv(args.debug_csv, index=False)
    render_shadow225_from_image(out_2d, args.out_225d, rules.get("shadow_225d", {}) or {})

    print(f"OK - rendu 2D précis: {args.out_2d}")
    print(f"OK - rendu shadow-only 2.25D: {args.out_225d}")
    print(f"OK - selected CSV: {args.selected_csv}")
    print(f"OK - debug CSV: {args.debug_csv}")


if __name__ == "__main__":
    main()
