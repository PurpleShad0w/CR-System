from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .dataset import load_level_tables
from .io_utils import ensure_dir
from .cleaning import (
    apply_missing_sentinels,
    expected_range_by_group,
    drop_local_spikes_v12,
    spread_cumul_spikes_v3,
    cap_point_outliers_v1,
)
from .targets_utils import discover_elec_usage_targets, add_elec_total_accurate
from .variable_config import normalize_input_columns


ELEC_TOTAL_NOBVE = "elecTotalNoBveKwh"


def add_elec_total_no_bve(df: pd.DataFrame) -> pd.DataFrame:
    """
    elecTotalNoBveKwh = elecTotalAccurateKwh - elecBveKwh (fillna 0), clamp >= 0.
    """
    if ELEC_TOTAL_NOBVE in df.columns:
        return df
    if "elecTotalAccurateKwh" not in df.columns or "elecBveKwh" not in df.columns:
        return df
    out = df.copy()
    total = pd.to_numeric(out["elecTotalAccurateKwh"], errors="coerce")
    bve = pd.to_numeric(out["elecBveKwh"], errors="coerce").fillna(0.0)
    out[ELEC_TOTAL_NOBVE] = np.maximum(total - bve, 0.0)
    return out


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    elif "dtUpdate" in df.columns:
        df["date"] = pd.to_datetime(df["dtUpdate"], errors="coerce").dt.floor("D")
    return df


def discover_consumption_targets(hist: pd.DataFrame) -> list[str]:
    cols = []

    for c in hist.columns:
        if not isinstance(c, str):
            continue

        is_elec = c.startswith("elec") and c.endswith("Kwh")
        is_water = c.startswith("water") and c.endswith("M3")
        is_hot = c.startswith("ec")      # hot water family from metertypes name
        is_cold = c.startswith("eg")     # chilled water family from metertypes name

        if not (is_elec or is_water or is_hot or is_cold):
            continue

        if c.endswith("_drift"):
            continue

        cols.append(c)

    return sorted(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site")  # site | zone | all
    args = ap.parse_args()

    LEVELS = ["site", "zone"]

    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))

    def clean_one(level: str):
        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]

        hist, pred, _ = load_level_tables(db_dir, level_cfg)
        hist = normalize_input_columns(hist, cfg)
        pred = normalize_input_columns(pred, cfg)

        # always normalize dates explicitly
        hist = _ensure_date(hist)
        pred = _ensure_date(pred)

        # build derived totals from the enriched DB before cleaning
        hist = add_elec_total_accurate(hist)
        hist = add_elec_total_no_bve(hist)

        # discover all electric usage columns dynamically
        dyn_elec_usages = discover_elec_usage_targets(hist)

        # sources that should be cleaned directly
        total_source_cols = [c for c in ["elecTotalKwh", "elecAggregatedKwh", "elecTotalFromDriftKwh"] if c in hist.columns]
        usage_cols = [c for c in dyn_elec_usages if c in hist.columns]
        water_cols = [c for c in ["waterM3"] if c in hist.columns]

        # final derived cols will be rebuilt later
        derived_cols = [c for c in ["elecTotalAccurateKwh", ELEC_TOTAL_NOBVE] if c in hist.columns]

        source_measure_cols = total_source_cols + usage_cols + water_cols
        final_measure_cols = total_source_cols + usage_cols + water_cols + derived_cols
        all_consumption_cols = discover_consumption_targets(hist)

        # source cols = tout ce qu'on nettoie directement avant recalculs dérivés
        measure_cols = [c for c in all_consumption_cols if c in hist.columns]

        # 0 / négatif => missing pour énergie / eau
        zero_map = {}
        for c in source_measure_cols + derived_cols:
            zero_map[c] = True

        # température : borne simple
        if "indoorTempDegC" in hist.columns:
            x = pd.to_numeric(hist["indoorTempDegC"], errors="coerce")
            x = x.mask(x <= 0, np.nan)
            x = x.mask((x < 5) | (x > 40), np.nan)
            hist["indoorTempDegC"] = x

        hist = apply_missing_sentinels(hist, source_measure_cols + derived_cols, zero_map)

        pred_cols = [c for c in ["totalKwh", "totalWater"] if c in pred.columns]
        pred = apply_missing_sentinels(pred, pred_cols, {c: True for c in pred_cols})

        exp = expected_range_by_group(hist, pred, id_cols, "date")
        CLEAN_LOGS: dict[str, pd.DataFrame] = {}

        # ---------------------------------------------------------
        # 1) Local spikes on totals/water
        # ---------------------------------------------------------
        if "elecTotalKwh" in hist.columns and "totalKwh" in pred.columns:
            hist, log = drop_local_spikes_v12(
                hist, pred, id_cols, "date", "elecTotalKwh", "totalKwh", exp, factor=8.0
            )
            if len(log):
                CLEAN_LOGS[f"{level}_local_spike_elecTotalKwh"] = log

        # drift total: no external predictor available, but still clean local spikes
        if "elecTotalFromDriftKwh" in hist.columns:
            hist, log = drop_local_spikes_v12(
                hist, None, id_cols, "date", "elecTotalFromDriftKwh", None, exp, factor=8.0
            )
            if len(log):
                CLEAN_LOGS[f"{level}_local_spike_elecTotalFromDriftKwh"] = log

        if "waterM3" in hist.columns and "totalWater" in pred.columns:
            hist, log = drop_local_spikes_v12(
                hist, pred, id_cols, "date", "waterM3", "totalWater", exp, factor=6.0
            )
            if len(log):
                CLEAN_LOGS[f"{level}_local_spike_waterM3"] = log

        # ---------------------------------------------------------
        # 2) Local spikes on ALL electric usages (legacy + new)
        # ---------------------------------------------------------
        for col in usage_cols:
            hist, log = drop_local_spikes_v12(
                hist, None, id_cols, "date", col, None, exp, factor=8.0
            )
            if len(log):
                CLEAN_LOGS[f"{level}_local_spike_{col}"] = log

        for col in measure_cols:
            if col in {"elecTotalKwh", "elecAggregatedKwh", "elecTotalAccurateKwh", "elecTotalFromDriftKwh", "waterM3"}:
                continue
            hist, log = drop_local_spikes_v12(hist, None, id_cols, "date", col, None, exp, factor=8.0)
            if len(log):
                CLEAN_LOGS[f"{level}_local_spike_{col}"] = log

        # ---------------------------------------------------------
        # 3) Cumulative spike spreading on totals + water
        # ---------------------------------------------------------
        cfg_cumul = {
            "min_missing_run": 3,
            "spike_factor": 20.0,
            "strategy": "spread",
            "baseline_points": 30,
            "max_spread_days": 370,
        }

        for col in [c for c in ["elecTotalKwh", "elecAggregatedKwh", "elecTotalFromDriftKwh", "waterM3"] if c in hist.columns]:
            hist, log = spread_cumul_spikes_v3(hist, id_cols, "date", col, cfg_cumul, exp)
            if len(log):
                CLEAN_LOGS[f"{level}_cumul_{col}"] = log

        # ---------------------------------------------------------
        # 4) Cap final on SOURCE cols first
        # ---------------------------------------------------------
        for col in source_measure_cols:
            cap = 8.0 if col.startswith("elec") else 6.0
            hist, log = cap_point_outliers_v1(hist, id_cols, "date", col, window=30, cap_factor=cap)
            if len(log):
                CLEAN_LOGS[f"{level}_cap_{col}"] = log

        # ---------------------------------------------------------
        # 5) Recompute derived totals from already-cleaned source cols
        # ---------------------------------------------------------
        if "elecTotalAccurateKwh" in hist.columns:
            hist = hist.drop(columns=["elecTotalAccurateKwh"])
        if ELEC_TOTAL_NOBVE in hist.columns:
            hist = hist.drop(columns=[ELEC_TOTAL_NOBVE])

        hist = add_elec_total_accurate(hist)
        hist = add_elec_total_no_bve(hist)

        # ---------------------------------------------------------
        # 6) Cap derived totals too
        # ---------------------------------------------------------
        for col in [c for c in ["elecTotalAccurateKwh", ELEC_TOTAL_NOBVE] if c in hist.columns]:
            hist, log = cap_point_outliers_v1(hist, id_cols, "date", col, window=30, cap_factor=8.0)
            if len(log):
                CLEAN_LOGS[f"{level}_cap_{col}"] = log

        # write outputs
        hist.to_csv(out_dir / f"{level}hist_cleaned.csv", index=False)
        for k, df_log in CLEAN_LOGS.items():
            df_log.to_csv(out_dir / f"cleanlog_{k}.csv", index=False)

        print(f"[{level}] cleaned rows:", len(hist))
        print(f"[{level}] dyn usages:", usage_cols)
        print(f"[{level}] logs:", {k: len(v) for k, v in CLEAN_LOGS.items()})

    levels = LEVELS if args.level == "all" else [args.level]
    for lvl in levels:
        if lvl not in LEVELS:
            raise ValueError("Unknown level. Use site|zone|all")
        clean_one(lvl)


if __name__ == "__main__":
    main()
