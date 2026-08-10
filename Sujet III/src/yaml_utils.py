from pathlib import Path
import yaml
def load_yaml(path, default=None):
    p=Path(path)
    if not p.exists(): return default
    with p.open('r', encoding='utf-8') as f: return yaml.safe_load(f) or default
