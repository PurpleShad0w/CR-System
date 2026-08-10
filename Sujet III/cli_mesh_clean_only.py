import argparse
from src.yaml_utils import load_yaml
from src.mesh_cleanup import clean_meshes
def main():
    p=argparse.ArgumentParser(description='v12 mesh cleanup seulement')
    p.add_argument('input_png')
    p.add_argument('--rules',default='config/default_rules_shadow225_v12.yaml')
    p.add_argument('--out',default='output/rendered_clean_v12_mesh_2d.png')
    a=p.parse_args(); rules=load_yaml(a.rules,{})
    clean_meshes(a.input_png,a.out,rules.get('mesh_cleanup',{}) or {})
    print(f'OK - 2D nettoye: {a.out}')
if __name__=='__main__': main()
