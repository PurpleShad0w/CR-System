from __future__ import annotations

import argparse
from pathlib import Path

from src.classify_physical import load_yaml
from src.dwg_entities import extract_entities_df
from src.noise_filter_balanced import clean_balanced_entities
from src.render_clean_plan_v4 import render_clean_plan_v4


def main() -> None:
    parser = argparse.ArgumentParser(description="Sujet III v4 - rendu équilibré, moins bruité mais pas trop strict")
    parser.add_argument("input", help="Fichier .dwg ou .dxf")
    parser.add_argument("--rules", default="config/default_rules_balanced_v4.yaml")
    parser.add_argument("--decisions", default="data/work/layer_review/layer_decisions.yaml")
    parser.add_argument("--out", default="output/rendered_clean_v4.png")
    parser.add_argument("--entities-csv", default="output/entities_df_v4.csv")
    parser.add_argument("--clean-csv", default="output/clean_balanced_entities_v4.csv")
    args = parser.parse_args()

    rules = load_yaml(args.rules, default={})
    decisions = load_yaml(args.decisions, default={"keep": [], "drop": [], "undecided": []})
    df = extract_entities_df(args.input, converter=(rules.get("converter") or {}), include_blocks=True)
    Path(args.entities_csv).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["points"], errors="ignore").to_csv(args.entities_csv, index=False)
    clean = clean_balanced_entities(df, rules, decisions)
    clean.drop(columns=["points"], errors="ignore").to_csv(args.clean_csv, index=False)
    render_clean_plan_v4(df, args.out, rules=rules, decisions=decisions)
    print(f"OK - entities brutes: {args.entities_csv}")
    print(f"OK - entités retenues: {args.clean_csv}")
    print(f"OK - rendu v4: {args.out}")


if __name__ == "__main__":
    main()
