from __future__ import annotations

import numpy as np
import pandas as pd


ELECTRIC_PREFIX = "elec"
ELECTRIC_SUFFIX = "Kwh"

ELEC_EXCLUDE = {
    "elecTotalKwh",
    "elecAggregatedKwh",
    "elecTotalNoBveKwh",
    "elecTotalFromDriftKwh",
    "elecTotalAccurateKwh",
}


def discover_elec_usage_targets(hist: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in hist.columns:
        if not (isinstance(c, str) and c.startswith(ELECTRIC_PREFIX) and c.endswith(ELECTRIC_SUFFIX)):
            continue
        if c in ELEC_EXCLUDE:
            continue
        if c.endswith("_drift"):
            continue
        cols.append(c)
    return sorted(cols)


def drop_groups_with_no_signal(df: pd.DataFrame, id_cols: list[str], target: str) -> pd.DataFrame:
    """
    Keep only groups (siteId or siteId+zoneId) where target has at least one strictly positive value.
    SAFE: uses boolean mask (no merge), so it never drops columns.
    """
    if target not in df.columns:
        return df

    x = pd.to_numeric(df[target], errors="coerce")
    sig = x.fillna(0.0) > 0
    if not sig.any():
        return df.iloc[0:0].copy()

    # groupby on the actual key columns (as Series) and broadcast back to rows
    keep = sig.groupby([df[c] for c in id_cols], dropna=False).transform("any")
    return df.loc[keep.to_numpy()].copy()


def add_elec_total_accurate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create elecTotalAccurateKwh:
      - prefer elecTotalFromDriftKwh if present
      - else fallback elecAggregatedKwh
      - else fallback elecTotalKwh
    Clamp >=0.
    """
    if "elecTotalAccurateKwh" in df.columns:
        return df

    out = df.copy()

    drift = pd.to_numeric(out["elecTotalFromDriftKwh"], errors="coerce") if "elecTotalFromDriftKwh" in out.columns else None
    agg = pd.to_numeric(out["elecAggregatedKwh"], errors="coerce") if "elecAggregatedKwh" in out.columns else None
    tot = pd.to_numeric(out["elecTotalKwh"], errors="coerce") if "elecTotalKwh" in out.columns else None

    if drift is not None and drift.notna().any():
        out["elecTotalAccurateKwh"] = drift
    elif agg is not None and agg.notna().any():
        out["elecTotalAccurateKwh"] = agg
    else:
        out["elecTotalAccurateKwh"] = tot

    out["elecTotalAccurateKwh"] = np.maximum(pd.to_numeric(out["elecTotalAccurateKwh"], errors="coerce"), 0.0)
    return out


def discover_dynamic_consumption_targets(hist: pd.DataFrame) -> list[str]:
    """
    Discover all dynamic consumption targets generated from enriched history.

    Includes:
      - elec*...Kwh dynamic usages
      - water*...M3 dynamic usages
      - ec* hot-water family columns
      - eg* chilled-water family columns

    Excludes structural totals and derived totals that are handled explicitly in BASE_TARGETS.
    """
    excluded = {
        "elecTotalKwh",
        "elecAggregatedKwh",
        "elecTotalFromDriftKwh",
        "elecTotalAccurateKwh",
        "elecTotalNoBveKwh",
        "waterM3",
        "waterTotalFromDriftM3",
        "ecTotalFromDrift",
        "egTotalFromDrift",
    }

    cols: list[str] = []
    for c in hist.columns:
        if not isinstance(c, str):
            continue
        if c in excluded:
            continue
        if c.endswith("_drift"):
            continue

        is_elec_usage = c.startswith("elec") and c.endswith("Kwh")
        is_water_usage = c.startswith("water") and c.endswith("M3")
        is_hot_water_usage = c.startswith("ec")
        is_cold_water_usage = c.startswith("eg")

        if is_elec_usage or is_water_usage or is_hot_water_usage or is_cold_water_usage:
            cols.append(c)

    return sorted(cols)

