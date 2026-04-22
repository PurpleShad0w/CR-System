from __future__ import annotations
from pathlib import Path
import pandas as pd


def read_semicolon_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=';', quotechar='"', encoding='utf-8', engine='python')


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
