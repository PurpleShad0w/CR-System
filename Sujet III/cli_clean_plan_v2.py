from __future__ import annotations

import argparse
from pathlib import Path

from src.dwg_entities import extract_entities_df
from src.feature_classifier import load_yaml
from src.render_clean_plan import render_clean_plan
from src.render_25d import render_aerial_25d


def main() -> None:
    parser = argparse.ArgumentParser(description="Sujet III v2 - DWG/DXF vers plan propre + vue aérienne 2.5D")
    parser.add_argument("input", help="Fichier .dxf ou .dwg")
    parser.add_argument("--rules", default="config/default_rules_aerial_25d.yaml")
    parser.add_argument("--decisions", default="data/work/layer_review/layer_decisions.yaml")
    parser.add_argument("--out", default="output/rendered_clean_v2.png")
    parser.add_argument("--aerial", default="output/rendered_aerial_25d.png")
    parser.add_argument("--entities-csv", default="output/entities_df_v2.csv")
    args = parser.parse_args()

    rules = load_yaml(args.rules, default={})
    decisions = load_yaml(args.decisions, default={"keep": [], "drop": [], "undecided": []})
    df = extract_entities_df(args.input, converter=(rules.get("converter") or {}))
    Path(args.entities_csv).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["points"], errors="ignore").to_csv(args.entities_csv, index=False)
    render_clean_plan(df, args.out, rules=rules, decisions=decisions)
    render_aerial_25d(df, args.aerial, rules=rules, decisions=decisions)
    print(f"OK - entities: {args.entities_csv}")
    print(f"OK - plan 2D propre: {args.out}")
    print(f"OK - vue aérienne 2.5D: {args.aerial}")


if __name__ == "__main__":
    main()
