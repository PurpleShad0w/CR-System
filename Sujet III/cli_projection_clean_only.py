import argparse
from src.yaml_utils import load_yaml
from src.projection_cleanup import clean_projection_lines
def main():
    p=argparse.ArgumentParser(description='v11 projection cleanup seulement')
    p.add_argument('input_png')
    p.add_argument('--rules',default='config/default_rules_shadow225_v11.yaml')
    p.add_argument('--out',default='output/rendered_clean_v11_projection_2d.png')
    a=p.parse_args(); rules=load_yaml(a.rules,{})
    clean_projection_lines(a.input_png,a.out,rules.get('projection_cleanup',{}) or {})
    print(f'OK - 2D nettoye: {a.out}')
if __name__=='__main__': main()
