from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    raw: dict


def load_config(path: str | Path) -> Config:
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
