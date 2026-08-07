from __future__ import annotations

import argparse
from pathlib import Path

from src.dwg_entities_v1plus import extract_entities_df
from src.filter_v1plus import load_yaml
from src.render_v1plus import render_v1plus


def main() -> None:
    parser = argparse.ArgumentParser(description="Sujet III v6/v1+ - rendu v1 conservé + nettoyage des artefacts éloignés")
    parser.add_argument("input", help="Fichier .dwg ou .dxf")
    parser.add_argument("--rules", default="config/default_rules_v1plus.yaml")
    parser.add_argument("--decisions", default="data/work/layer_review/layer_decisions.yaml")
    parser.add_argument("--out", default="output/rendered_clean_v1plus.png")
    parser.add_argument("--entities-csv", default="output/entities_df_v1plus.csv")
    args = parser.parse_args()

    rules = load_yaml(args.rules, default={})
    decisions = load_yaml(args.decisions, default={"keep": [], "drop": [], "undecided": []})
    df = extract_entities_df(args.input, converter=(rules.get("converter") or {}), include_blocks=True)
    Path(args.entities_csv).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["points"], errors="ignore").to_csv(args.entities_csv, index=False)
    render_v1plus(df, args.out, rules=rules, decisions=decisions)
    print(f"OK - rendu v1plus: {args.out}")
    print(f"OK - entities: {args.entities_csv}")


if __name__ == "__main__":
    main()
