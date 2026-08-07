from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or default


def normalize_decisions(decisions: dict[str, Any] | None) -> dict[str, list[str]]:
    decisions = decisions or {}
    return {
        "keep": sorted(set(map(str, decisions.get("keep", []) or []))),
        "drop": sorted(set(map(str, decisions.get("drop", []) or []))),
        "undecided": sorted(set(map(str, decisions.get("undecided", []) or []))),
    }


def _contains_any(value: str, needles: list[str]) -> bool:
    value_u = str(value or "").upper()
    return any(str(n).upper() in value_u for n in needles)


def filter_v1plus(df: pd.DataFrame, rules: dict[str, Any] | None, decisions: dict[str, Any] | None) -> pd.DataFrame:
    """Filtrage proche du v1.

    Le but est de préserver le rendu v1 : pas de classification agressive, pas de sélection physique stricte.
    On applique seulement les drops sûrs + les décisions manuelles, puis le nettoyage spatial se fait après rendu.
    """
    rules = rules or {}
    cfg = rules.get("filtering", {}) or {}
    decisions = normalize_decisions(decisions)
    out = df.copy()
    if out.empty:
        return out

    out = out[out["points"].apply(lambda p: isinstance(p, list) and len(p) >= 2)].copy()
    entity_type = out["entity_type"].fillna("").astype(str).str.upper()
    layer = out["layer"].fillna("").astype(str)

    drop_types = set(map(str.upper, cfg.get("drop_entity_types", []) or []))
    drop_contains = cfg.get("drop_layers_contains", []) or []

    keep_mask = pd.Series(False, index=out.index)
    drop_mask = pd.Series(False, index=out.index)

    keep_mask |= layer.isin(decisions["keep"])
    if bool(cfg.get("respect_manual_drop_layers", True)):
        drop_mask |= layer.isin(decisions["drop"])

    drop_mask |= entity_type.isin(drop_types)
    drop_mask |= layer.apply(lambda x: _contains_any(x, drop_contains))

    small_cfg = cfg.get("drop_small_entities", {}) or {}
    if bool(small_cfg.get("enabled", True)):
        min_len = float(small_cfg.get("min_length", 1.0))
        min_diag = float(small_cfg.get("min_bbox_diag", 0.5))
        drop_mask |= (out["length"].fillna(0) < min_len) & (out["bbox_diag"].fillna(0) < min_diag)

    if bool(cfg.get("render_undecided_layers", True)):
        render_mask = ~drop_mask
    else:
        render_mask = keep_mask & ~drop_mask

    return out[render_mask].copy()
