from src.yaml_utils import load_yaml
from src.nonwall_line_cleanup import clean_nonwall_lines
from src.shadow225_v10 import render_shadow225_from_image
import argparse
def main():
    p=argparse.ArgumentParser(description="v10 fastfix: nettoie les lignes non-mur puis applique le shadow-only 2.25D")
    p.add_argument("input_png"); p.add_argument("--rules",default="config/default_rules_shadow225_v10_fast.yaml"); p.add_argument("--out-2d",default="output/rendered_clean_v10_fast_2d.png"); p.add_argument("--out-225d",default="output/rendered_clean_v10_fast_shadow225.png")
    a=p.parse_args(); rules=load_yaml(a.rules,{})
    cleaned=clean_nonwall_lines(a.input_png,a.out_2d,rules.get("nonwall_cleanup",{}) or {})
    render_shadow225_from_image(cleaned,a.out_225d,rules.get("shadow_225d",{}) or {})
    print(f"OK - 2D nettoyé: {a.out_2d}"); print(f"OK - shadow-only 2.25D: {a.out_225d}")
if __name__=="__main__": main()
