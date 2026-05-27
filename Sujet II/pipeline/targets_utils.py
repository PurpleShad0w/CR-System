from __future__ import annotations

import numpy as np
import pandas as pd


ELECTRIC_PREFIX = "elec"
ELECTRIC_SUFFIX = "Kwh"

# Colonnes à exclure de la liste “usages”
ELEC_EXCLUDE = {
    "elecTotalKwh",
    "elecAggregatedKwh",
    "elecTotalNoBveKwh",
    "elecTotalFromDriftKwh",
    "elecTotalAccurateKwh",
}

def discover_elec_usage_targets(hist: pd.DataFrame) -> list[str]:
    """
    Return the list of electric usage columns present in hist:
      - startswith('elec') and endswith('Kwh')
      - excludes totals/aggregates
      - excludes *_drift audit columns
    """
    cols = []
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
    Removes groups (siteId or siteId+zoneId) for which target is entirely missing/zero.
    This matches the user's requirement: if an usage is absent for a site, no need to care.
    """
    if target not in df.columns:
        return df
    x = pd.to_numeric(df[target], errors="coerce")
    # define “signal” as any strictly positive value
    sig = x.fillna(0.0) > 0
    if not sig.any():
        return df.iloc[0:0].copy()

    # groups that have at least one positive value
    grp_has = sig.groupby([df[c] for c in id_cols]).any()
    grp_has = grp_has[grp_has].reset_index()
    grp_has["__keep__"] = True

    out = df.merge(grp_has, on=id_cols, how="inner")
    return out.drop(columns=["__keep__"])


def add_elec_total_accurate(df: pd.DataFrame) -> pd.DataFrame:
    if "elecTotalAccurateKwh" in df.columns:
        return df
    out = df.copy()

    c_drift = pd.to_numeric(out["elecTotalFromDriftKwh"], errors="coerce") if "elecTotalFromDriftKwh" in out.columns else None
    c_agg = pd.to_numeric(out["elecAggregatedKwh"], errors="coerce") if "elecAggregatedKwh" in out.columns else None
    c_tot = pd.to_numeric(out["elecTotalKwh"], errors="coerce") if "elecTotalKwh" in out.columns else None

    if c_drift is not None:
        out["elecTotalAccurateKwh"] = c_drift
    elif c_agg is not None:
        out["elecTotalAccurateKwh"] = c_agg
    else:
        out["elecTotalAccurateKwh"] = c_tot

    # clamp negatives
    out["elecTotalAccurateKwh"] = np.maximum(pd.to_numeric(out["elecTotalAccurateKwh"], errors="coerce"), 0.0)
    return out
