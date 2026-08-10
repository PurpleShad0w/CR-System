import argparse
from src.yaml_utils import load_yaml
from src.projection_cleanup import clean_projection_lines
from src.shadow225 import render_shadow225
def main():
    p=argparse.ArgumentParser(description='v11 projection cleanup puis shadow-only 2.25D')
    p.add_argument('input_png')
    p.add_argument('--rules',default='config/default_rules_shadow225_v11.yaml')
    p.add_argument('--out-2d',default='output/rendered_clean_v11_projection_2d.png')
    p.add_argument('--out-225d',default='output/rendered_clean_v11_projection_shadow225.png')
    a=p.parse_args(); rules=load_yaml(a.rules,{})
    cleaned=clean_projection_lines(a.input_png,a.out_2d,rules.get('projection_cleanup',{}) or {})
    render_shadow225(cleaned,a.out_225d,rules.get('shadow_225d',{}) or {})
    print(f'OK - 2D nettoye: {a.out_2d}')
    print(f'OK - shadow-only 2.25D: {a.out_225d}')
if __name__=='__main__': main()
