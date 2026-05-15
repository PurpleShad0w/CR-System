from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .io_utils import ensure_dir
from .dataset import load_level_tables
from .features import add_calendar_features
from .modeling import load_model
from .site_infos import load_site_infos


ELECTRIC_USES = ["elecBveKwh", "elecCvcKwh", "elecForceKwh", "elecLightingKwh"]
BASE_TARGETS = ["elecTotalKwh", "waterM3", "indoorTempDegC"]
TARGETS_ALL = BASE_TARGETS + ELECTRIC_USES


def _infer_horizon_days(last_hist_date: pd.Timestamp, weath: pd.DataFrame, max_days: int | None, siteId: int) -> int:
    w = weath[weath["siteId"] == siteId]
    w = w[w["date"] > last_hist_date]
    if len(w) == 0:
        return 0
    max_date = w["date"].max()
    days = int((max_date - last_hist_date).days)
    if max_days is not None:
        days = min(days, int(max_days))
    return max(days, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site")  # site | zone | all
    ap.add_argument(
        "--target",
        required=True,
        # elecTotalKwh | waterM3 | indoorTempDegC | elecBveKwh | elecCvcKwh | elecForceKwh | elecLightingKwh | elecUses | all
    )
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()

    LEVELS = ["site", "zone"]
    BASE_TARGETS_BY_LEVEL = {"site": BASE_TARGETS, "zone": BASE_TARGETS}
    TARGETS_BY_LEVEL = {"site": TARGETS_ALL, "zone": TARGETS_ALL}

    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))

    def expand_targets(level: str, target: str) -> list[str]:
        if target == "elecUses":
            return ELECTRIC_USES[:]
        if target == "all":
            return BASE_TARGETS_BY_LEVEL[level][:]
        return [target]

    def predict_one(level: str, target: str, days: int | None):
        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]

        cleaned_path = out_dir / f"{level}hist_cleaned.csv"
        hist = pd.read_csv(cleaned_path)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")

        info_path = Path(args.config).resolve().parent / cfg.get("paths", {}).get("site_infos_file", "Sites_Shyrka_Infos.xlsx")
        site_infos = load_site_infos(info_path)
        if len(site_infos) and "siteId" in hist.columns:
            hist = hist.merge(site_infos, on="siteId", how="left")

        _, _, weath = load_level_tables(db_dir, level_cfg)
        if len(weath) == 0:
            raise RuntimeError("Missing siteweath.csv")
        weath["date"] = pd.to_datetime(weath["date"], errors="coerce").dt.floor("D")

        model_dir = out_dir / "models"
        meta = json.loads((model_dir / f"{level}_{target}.meta.json").read_text(encoding="utf-8"))
        feat_cols = meta["feature_columns"]
        log1p_target = bool(meta.get("log1p_target", False))
        model = load_model(model_dir / f"{level}_{target}.joblib")

        hist = hist.dropna(subset=id_cols + ["date"]).sort_values(id_cols + ["date"])
        last_hist_date = hist["date"].max()

        weather_cols = cfg["features"].get("weather_cols", [])
        if target == "indoorTempDegC":
            weather_cols = [c for c in weather_cols if c != "tempAmb"]

        max_days = days if days is not None else cfg.get("prediction", {}).get("days", None)

        future_rows = []
        for keys, g in hist.groupby(id_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base_feat = {c: int(v) for c, v in zip(id_cols, keys)}
            site_id = base_feat.get("siteId", None)
            if site_id is None:
                continue

            horizon = _infer_horizon_days(last_hist_date, weath, max_days, site_id)
            if horizon <= 0:
                continue

            state = g[["date", target]].copy()
            state[target] = pd.to_numeric(state[target], errors="coerce")
            state = state.dropna().sort_values("date")
            if state.empty:
                continue

            dates = pd.date_range(last_hist_date + pd.Timedelta(days=1), last_hist_date + pd.Timedelta(days=horizon), freq="D")
            wsite = weath[weath["siteId"] == site_id].drop_duplicates(subset=["siteId", "date"], keep="last").set_index("date")

            for d in dates:
                row = {**base_feat, "date": d}

                if d in wsite.index:
                    ww = wsite.loc[d]
                    for c in weather_cols:
                        if c in wsite.columns:
                            row[c] = float(ww[c]) if pd.notna(ww[c]) else np.nan

                if cfg["features"].get("add_calendar", True):
                    tmp = add_calendar_features(pd.DataFrame([row]), "date")
                    for col in ["dow", "month", "dayofyear", "is_weekend"]:
                        row[col] = int(tmp.iloc[0][col])

                s = state.set_index("date")[target]
                for k in cfg["features"]["lags"]:
                    row[f"lag_{k}"] = float(s.get(d - pd.Timedelta(days=k), np.nan))

                for wdw in cfg["features"]["rolling_windows"]:
                    window = pd.date_range(d - pd.Timedelta(days=wdw), d - pd.Timedelta(days=1), freq="D")
                    vals = s.reindex(window).to_numpy(dtype=float)
                    row[f"roll_med_{wdw}"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
                    row[f"roll_mean_{wdw}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

                X = pd.DataFrame([row])
                for c in feat_cols:
                    if c not in X.columns:
                        X[c] = np.nan
                X = X[feat_cols].copy()

                pred = model.predict(X)
                if log1p_target:
                    yhat = float(np.expm1(pred[0]))
                    yhat = max(yhat, 0.0)
                else:
                    yhat = float(pred[0])

                future_rows.append({**base_feat, "date": d, "yhat": yhat})
                state = pd.concat([state, pd.DataFrame([{"date": d, target: yhat}])], ignore_index=True)

        out = pd.DataFrame(future_rows)
        out_path = out_dir / f"pred_{level}_{target}.csv"
        out.to_csv(out_path, index=False)
        print("wrote", out_path)

    levels = LEVELS if args.level == "all" else [args.level]
    for lvl in levels:
        if lvl not in LEVELS:
            raise ValueError("Unknown level. Use site|zone|all")

        targets = expand_targets(lvl, args.target)
        for tgt in targets:
            if tgt == "elecAggregatedKwh":
                raise ValueError("elecAggregatedKwh est exclu (pas un usage).")
            if tgt not in TARGETS_BY_LEVEL[lvl]:
                raise ValueError(f"Target {tgt} not supported for level {lvl}. Use {TARGETS_BY_LEVEL[lvl]} or elecUses or all")
            predict_one(lvl, tgt, args.days)


if __name__ == "__main__":
    main()