from __future__ import annotations

from typing import Any
import pandas as pd

ROLE_TO_CANONICAL = {
    "elec_total": "elecTotalKwh",
    "elec_aggregated": "elecAggregatedKwh",
    "elec_total_from_drift": "elecTotalFromDriftKwh",
    "elec_total_accurate": "elecTotalAccurateKwh",
    "elec_total_no_bve": "elecTotalNoBveKwh",
    "water_total": "waterM3",
    "indoor_temp": "indoorTempDegC",
    "pred_total_elec": "totalKwh",
    "pred_total_water": "totalWater",
}

USAGE_ROLE_TO_CANONICAL = {
    "bve": "elecBveKwh",
    "cvc": "elecCvcKwh",
    "force": "elecForceKwh",
    "lighting": "elecLightingKwh",
}


def _cfg_vars(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    return cfg.get("variables", {}) or {}


def configured_column(cfg: dict[str, Any] | None, role: str, default: str | None = None) -> str | None:
    variables = _cfg_vars(cfg)
    cols = variables.get("columns", {}) or {}
    if role in cols and cols[role]:
        return str(cols[role])
    if default is not None:
        return default
    return ROLE_TO_CANONICAL.get(role)


def configured_legacy_usage_columns(cfg: dict[str, Any] | None) -> dict[str, str]:
    variables = _cfg_vars(cfg)
    cols = variables.get("columns", {}) or {}
    configured = cols.get("legacy_elec_usages", {}) or {}
    out = {}
    for role, canonical in USAGE_ROLE_TO_CANONICAL.items():
        out[role] = str(configured.get(role, canonical))
    return out


def normalize_input_columns(df: pd.DataFrame, cfg: dict[str, Any] | None) -> pd.DataFrame:
    """Rename configured source CSV columns to the canonical names expected by the pipeline."""
    if df is None or len(df.columns) == 0:
        return df

    out = df.copy()
    out.columns = [str(c).replace("﻿", "").strip() for c in out.columns]
    rename_map: dict[str, str] = {}

    for role, canonical in ROLE_TO_CANONICAL.items():
        source = configured_column(cfg, role, canonical)
        if source and source in out.columns and canonical not in out.columns:
            rename_map[source] = canonical

    for role, canonical in USAGE_ROLE_TO_CANONICAL.items():
        source = configured_legacy_usage_columns(cfg).get(role, canonical)
        if source and source in out.columns and canonical not in out.columns:
            rename_map[source] = canonical

    if rename_map:
        out = out.rename(columns=rename_map)
    return out
