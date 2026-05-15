from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

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

ELECTRIC_USES = ["elecBveKwh", "elecCvcKwh", "elecForceKwh", "elecLightingKwh"]


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

        # 0 / négatif => missing pour énergie/eau (inclut usages)
        zero_map = {
            "elecTotalKwh": True,
            "waterM3": True,
            "totalKwh": True,
            "totalWater": True,
        }
        for c in ELECTRIC_USES:
            zero_map[c] = True

        measure_cols = [c for c in (["elecTotalKwh"] + ELECTRIC_USES + ["waterM3"]) if c in hist.columns]
        pred_cols = [c for c in ["totalKwh", "totalWater"] if c in pred.columns]

        # température : borne simple (comme avant)
        if "indoorTempDegC" in hist.columns:
            x = pd.to_numeric(hist["indoorTempDegC"], errors="coerce")
            x = x.mask(x <= 0, np.nan)
            x = x.mask((x < 5) | (x > 40), np.nan)
            hist["indoorTempDegC"] = x

        hist = apply_missing_sentinels(hist, measure_cols, zero_map)
        pred = apply_missing_sentinels(pred, pred_cols, zero_map)

        exp = expected_range_by_group(hist, pred, id_cols, "date")
        CLEAN_LOGS = {}

        # total elec / water : spikes locaux + hard-check vs algo pred
        if "elecTotalKwh" in hist.columns and "totalKwh" in pred.columns:
            hist, log = drop_local_spikes_v12(hist, pred, id_cols, "date", "elecTotalKwh", "totalKwh", exp, factor=8.0)
            if len(log):
                CLEAN_LOGS[f"{level}_local_spike_elecTotalKwh"] = log

        if "waterM3" in hist.columns and "totalWater" in pred.columns:
            hist, log = drop_local_spikes_v12(hist, pred, id_cols, "date", "waterM3", "totalWater", exp, factor=6.0)
            if len(log):
                CLEAN_LOGS[f"{level}_local_spike_waterM3"] = log

        # usages : spikes locaux (sans algo pred)
        for col in ELECTRIC_USES:
            if col in hist.columns:
                hist, log = drop_local_spikes_v12(hist, None, id_cols, "date", col, None, exp, factor=8.0)
                if len(log):
                    CLEAN_LOGS[f"{level}_local_spike_{col}"] = log

        # cumul spread : on garde uniquement sur total elec + water (logique existante)
        cfg_cumul = {
            "min_missing_run": 3,
            "spike_factor": 20.0,
            "strategy": "spread",
            "baseline_points": 30,
            "max_spread_days": 370,
        }
        for col in [c for c in ["elecTotalKwh", "waterM3"] if c in hist.columns]:
            hist, log = spread_cumul_spikes_v3(hist, id_cols, "date", col, cfg_cumul, exp)
            if len(log):
                CLEAN_LOGS[f"{level}_cumul_{col}"] = log

        # cap final : total + usages + eau
        for col in measure_cols:
            cap = 8.0 if col.startswith("elec") else 6.0
            hist, log = cap_point_outliers_v1(hist, id_cols, "date", col, window=30, cap_factor=cap)
            if len(log):
                CLEAN_LOGS[f"{level}_cap_{col}"] = log

        hist.to_csv(out_dir / f"{level}hist_cleaned.csv", index=False)
        for k, df_log in CLEAN_LOGS.items():
            df_log.to_csv(out_dir / f"cleanlog_{k}.csv", index=False)

        print(f"[{level}] cleaned rows:", len(hist))
        print(f"[{level}] logs:", {k: len(v) for k, v in CLEAN_LOGS.items()})

    levels = LEVELS if args.level == "all" else [args.level]
    for lvl in levels:
        if lvl not in LEVELS:
            raise ValueError("Unknown level. Use site|zone|all")
        clean_one(lvl)


if __name__ == "__main__":
    main()