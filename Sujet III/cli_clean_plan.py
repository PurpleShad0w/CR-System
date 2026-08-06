from __future__ import annotations

import argparse
from pathlib import Path

from src.dwg_entities import extract_entities_df
from src.layer_classifier import load_yaml
from src.render_clean_plan import render_clean_plan
from src.render_25d import export_25d_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Sujet III - DWG/DXF vers plan propre + 2.5D")
    parser.add_argument("input", help="Fichier .dxf ou .dwg")
    parser.add_argument("--rules", default="config/default_rules.yaml")
    parser.add_argument("--decisions", default="data/work/layer_review/layer_decisions.yaml")
    parser.add_argument("--out", default="output/rendered_clean.png")
    parser.add_argument("--entities-csv", default="output/entities_df.csv")
    parser.add_argument("--html-25d", default="output/rendered_25d.html")
    args = parser.parse_args()

    rules = load_yaml(args.rules, default={})
    decisions = load_yaml(args.decisions, default={"keep": [], "drop": [], "undecided": []})
    df = extract_entities_df(args.input, converter=(rules.get("converter") or {}))
    Path(args.entities_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.entities_csv, index=False)
    render_clean_plan(df, args.out, rules=rules, decisions=decisions)
    export_25d_html(df, args.html_25d, rules=rules, decisions=decisions)
    print(f"OK - entities: {args.entities_csv}")
    print(f"OK - plan 2D: {args.out}")
    print(f"OK - vue 2.5D: {args.html_25d}")


if __name__ == "__main__":
    main()
