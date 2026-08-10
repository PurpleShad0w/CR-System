from src.yaml_utils import load_yaml
from src.nonwall_line_cleanup import clean_nonwall_lines
import argparse
def main():
    p=argparse.ArgumentParser(description="v10 fastfix: nettoie les lignes non-mur d'un rendu 2D")
    p.add_argument("input_png"); p.add_argument("--rules",default="config/default_rules_shadow225_v10_fast.yaml"); p.add_argument("--out",default="output/rendered_clean_v10_fast_2d.png")
    a=p.parse_args(); rules=load_yaml(a.rules,{})
    clean_nonwall_lines(a.input_png,a.out,rules.get("nonwall_cleanup",{}) or {})
    print(f"OK - 2D nettoyé: {a.out}")
if __name__=="__main__": main()
