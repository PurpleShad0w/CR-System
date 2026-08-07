from __future__ import annotations
import argparse
from pathlib import Path
from src.classify_physical import load_yaml
from src.dwg_entities import extract_entities_df
from src.component_cleanup import clean_component_entities
from src.render_clean_plan_v5 import render_clean_plan_v5

def main():
    p=argparse.ArgumentParser(description="Sujet III v5 - nettoyage par composant spatial principal")
    p.add_argument("input"); p.add_argument("--rules",default="config/default_rules_component_v5.yaml"); p.add_argument("--decisions",default="data/work/layer_review/layer_decisions.yaml"); p.add_argument("--out",default="output/rendered_clean_v5.png"); p.add_argument("--entities-csv",default="output/entities_df_v5.csv"); p.add_argument("--clean-csv",default="output/clean_component_entities_v5.csv")
    a=p.parse_args(); rules=load_yaml(a.rules,{}); decisions=load_yaml(a.decisions,{"keep":[],"drop":[],"undecided":[]})
    df=extract_entities_df(a.input,converter=(rules.get("converter") or {}),include_blocks=True); Path(a.entities_csv).parent.mkdir(parents=True,exist_ok=True); df.drop(columns=["points"],errors="ignore").to_csv(a.entities_csv,index=False)
    clean=clean_component_entities(df,rules,decisions); clean.drop(columns=["points"],errors="ignore").to_csv(a.clean_csv,index=False)
    render_clean_plan_v5(df,a.out,rules,decisions); print(f"OK - rendu v5: {a.out}"); print(f"OK - entités retenues: {a.clean_csv}")
if __name__=="__main__": main()
