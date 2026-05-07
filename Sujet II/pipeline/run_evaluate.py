from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .io_utils import ensure_dir
from .dataset import load_level_tables
from .features import add_calendar_features, build_lag_features, build_rolling_features
from .modeling import load_model, mae, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site")   # site | zone | all
    ap.add_argument("--target", required=True)   # elecTotalKwh | waterM3 | indoorTempDegC | all
    args = ap.parse_args()

    LEVELS = ["site", "zone"]
    TARGETS_BY_LEVEL = {
        "site": ["elecTotalKwh", "waterM3", "indoorTempDegC"],
        "zone": ["elecTotalKwh", "waterM3", "indoorTempDegC"],
    }

    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))

    def evaluate_one(level : str, target : str):
        cleaned_path = out_dir / f"{level}hist_cleaned.csv"
        if not cleaned_path.exists():
            raise RuntimeError("Run cleaning first")

        hist = pd.read_csv(cleaned_path)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")

        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]

        model_dir = out_dir / "models"
        meta_path = model_dir / f"{level}_{target}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feat_cols = meta["feature_columns"]
        valid_days = int(meta.get("valid_days", cfg["training"].get("valid_days", 60)))
        log1p_target = bool(meta.get("log1p_target", False))

        df = hist[id_cols + ["date", target]].copy()
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.dropna(subset=id_cols + ["date", target])

        # Merge weather
        _, _, weath = load_level_tables(db_dir, level_cfg)
        if len(weath) and "date" in weath.columns:
            weath["date"] = pd.to_datetime(weath["date"], errors="coerce").dt.floor("D")

        weather_cols = cfg["features"].get("weather_cols", [])
        if target == "indoorTempDegC":
            weather_cols = [c for c in weather_cols if c != "tempAmb"]

        if len(weath) and "siteId" in weath.columns and "date" in weath.columns:
            keep = ["siteId", "date"] + [c for c in weather_cols if c in weath.columns]
            w = weath[keep].drop_duplicates(subset=["siteId", "date"], keep="last")
            df = df.merge(w, on=["siteId", "date"], how="left")

        if cfg["features"].get("add_calendar", True):
            df = add_calendar_features(df, "date")

        df = build_lag_features(df, id_cols, "date", target, cfg["features"]["lags"])
        df = build_rolling_features(df, id_cols, "date", target, cfg["features"]["rolling_windows"])

        # même filtre que train (recommandé)
        lag_cols = [f"lag_{k}" for k in cfg["features"]["lags"]]
        roll_cols = []
        for w in cfg["features"]["rolling_windows"]:
            roll_cols += [f"roll_med_{w}", f"roll_mean_{w}"]
        must_have = [c for c in (lag_cols + roll_cols) if c in df.columns]
        if must_have:
            df = df.dropna(subset=must_have)

        cutoff = df["date"].max() - pd.Timedelta(days=valid_days)
        valid = df[df["date"] > cutoff].copy()

        X = valid[feat_cols].copy()
        y = valid[target].to_numpy(dtype=float)

        if target in ["elecTotalKwh", "waterM3"]:
            m = np.isfinite(y) & (y >= 0)
            valid = valid.loc[m].copy()
            y = y[m]
            X = valid[feat_cols].copy()

        model = load_model(model_dir / f"{level}_{target}.joblib")
        pred = model.predict(X)
        if log1p_target:
            yhat = np.expm1(pred)
            yhat = np.maximum(yhat, 0.0)
        else:
            yhat = pred

        pred_df = valid[id_cols + ["date"]].copy()
        pred_df["y_true"] = y
        pred_df["y_pred"] = yhat
        pred_df.to_csv(out_dir / f"eval_preds_{level}_{target}.csv", index=False)

        # Per-group metrics (siteId for site-level, (siteId, zoneId) for zone-level)
        rows = []
        for keys, g in pred_df.groupby(id_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            yy = g["y_true"].to_numpy(dtype=float)
            pp = g["y_pred"].to_numpy(dtype=float)

            if len(yy) == 0:
                continue

            denom_g = float(np.sum(np.abs(yy)))
            wape_g = float(np.sum(np.abs(yy - pp)) / denom_g) if (denom_g > 0 and target in ["elecTotalKwh", "waterM3"]) else np.nan
            bias_g = float(np.mean(pp - yy))

            sst_g = float(np.sum((yy - np.mean(yy)) ** 2))
            sse_g = float(np.sum((yy - pp) ** 2))
            r2_g = float(1.0 - sse_g / sst_g) if sst_g > 0 else np.nan

            row = {c: v for c, v in zip(id_cols, keys)}
            row.update({
                "n": int(len(yy)),
                "MAE": mae(yy, pp),
                "RMSE": rmse(yy, pp),
                "WAPE": wape_g,
                "Bias": bias_g,
                "R2": r2_g,
            })
            rows.append(row)

        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"eval_{level}_{target}_by_group.csv", index=False)

        if target in ["elecTotalKwh", "waterM3"]:
            denom = float(np.sum(np.abs(y)))
            wape = float(np.sum(np.abs(y - yhat)) / denom) if denom > 0 else np.nan
        else:
            wape = np.nan

        # Extra metrics
        bias = float(np.mean(yhat - y)) if len(y) else np.nan
        medae = float(np.median(np.abs(yhat - y))) if len(y) else np.nan

        # R2 (guard)
        sst = float(np.sum((y - np.mean(y)) ** 2)) if len(y) else 0.0
        sse = float(np.sum((y - yhat) ** 2)) if len(y) else 0.0
        r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

        def _smape(y_true, y_pred):
            denom = np.abs(y_true) + np.abs(y_pred)
            m = denom > 0
            return float(200.0 * np.mean(np.abs(y_pred[m] - y_true[m]) / denom[m])) if np.any(m) else np.nan

        smape = _smape(y, yhat) if target in ["elecTotalKwh", "waterM3"] else np.nan

        rep = pd.DataFrame([{
            "rows": int(len(valid)),
            "MAE": mae(y, yhat),
            "RMSE": rmse(y, yhat),
            "WAPE": wape,
            "Bias": bias,
            "MedAE": medae,
            "sMAPE": smape,
            "R2": r2,
        }])

        rep.to_csv(out_dir / f"eval_{level}_{target}.csv", index=False)
        print(rep)

    levels = LEVELS if args.level == "all" else [args.level]
    for lvl in levels:
        if lvl not in LEVELS:
            raise ValueError(f"Unknown level: {lvl}. Use site|zone|all")
        targets = TARGETS_BY_LEVEL[lvl] if args.target == "all" else [args.target]
        for tgt in targets:
            if tgt not in TARGETS_BY_LEVEL[lvl]:
                raise ValueError(f"Target {tgt} not supported for level {lvl}. Use {TARGETS_BY_LEVEL[lvl]} or all")
            evaluate_one(lvl, tgt)


if __name__ == "__main__":
    main()
