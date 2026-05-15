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
from .site_infos import load_site_infos


ELECTRIC_USES = ["elecBveKwh", "elecCvcKwh", "elecForceKwh", "elecLightingKwh"]
ELECTRIC_ALL = ["elecTotalKwh"] + ELECTRIC_USES
BASE_TARGETS = ["elecTotalKwh", "waterM3", "indoorTempDegC"]
TARGETS_ALL = BASE_TARGETS + ELECTRIC_USES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site")  # site | zone | all
    ap.add_argument(
        "--target",
        required=True,
        # elecTotalKwh | waterM3 | indoorTempDegC | elecBveKwh | elecCvcKwh | elecForceKwh | elecLightingKwh | elecUses | all
    )
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

    def evaluate_one(level: str, target: str):
        cleaned_path = out_dir / f"{level}hist_cleaned.csv"
        if not cleaned_path.exists():
            raise RuntimeError("Run cleaning first")

        hist = pd.read_csv(cleaned_path)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")

        info_path = Path(args.config).resolve().parent / cfg.get("paths", {}).get("site_infos_file", "Sites_Shyrka_Infos.xlsx")
        site_infos = load_site_infos(info_path)
        if len(site_infos) and "siteId" in hist.columns:
            hist = hist.merge(site_infos, on="siteId", how="left")

        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]

        model_dir = out_dir / "models"
        meta_path = model_dir / f"{level}_{target}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feat_cols = meta["feature_columns"]
        valid_days = int(meta.get("valid_days", cfg["training"].get("valid_days", 60)))
        log1p_target = bool(meta.get("log1p_target", False))

        static_cols = cfg.get("features", {}).get("static_cols", [])
        extra_cols = [c for c in static_cols if c in hist.columns]

        df = hist[id_cols + ["date", target] + extra_cols].copy()
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.dropna(subset=id_cols + ["date", target] + extra_cols)

        # weather
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

        lag_cols = [f"lag_{k}" for k in cfg["features"]["lags"]]
        roll_cols = []
        for wdw in cfg["features"]["rolling_windows"]:
            roll_cols += [f"roll_med_{wdw}", f"roll_mean_{wdw}"]
        must_have = [c for c in (lag_cols + roll_cols) if c in df.columns]
        if must_have:
            df = df.dropna(subset=must_have)

        cutoff = df["date"].max() - pd.Timedelta(days=valid_days)
        valid = df[df["date"] > cutoff].copy()

        X = valid[feat_cols].copy()
        y = valid[target].to_numpy(dtype=float)

        # énergie/eau: filtre négatifs
        if target in (ELECTRIC_ALL + ["waterM3"]):
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

        # métriques
        if target in (ELECTRIC_ALL + ["waterM3"]):
            denom = float(np.sum(np.abs(y)))
            wape = float(np.sum(np.abs(y - yhat)) / denom) if denom > 0 else np.nan
        else:
            wape = np.nan

        bias = float(np.mean(yhat - y)) if len(y) else np.nan
        medae = float(np.median(np.abs(yhat - y))) if len(y) else np.nan

        sst = float(np.sum((y - np.mean(y)) ** 2)) if len(y) else 0.0
        sse = float(np.sum((y - yhat) ** 2)) if len(y) else 0.0
        r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

        def _smape(y_true, y_pred):
            denom2 = np.abs(y_true) + np.abs(y_pred)
            m2 = denom2 > 0
            return float(200.0 * np.mean(np.abs(y_pred[m2] - y_true[m2]) / denom2[m2])) if np.any(m2) else np.nan

        smape = _smape(y, yhat) if target in (ELECTRIC_ALL + ["waterM3"]) else np.nan

        mae_m2 = rmse_m2 = wape_m2 = np.nan
        if target in (ELECTRIC_ALL + ["waterM3"]) and "surface_m2" in valid.columns:
            surf = pd.to_numeric(valid["surface_m2"], errors="coerce").to_numpy(dtype=float)
            m2 = np.isfinite(surf) & (surf > 0) & np.isfinite(y) & np.isfinite(yhat)
            if np.any(m2):
                y_m2 = y[m2] / surf[m2]
                yhat_m2 = yhat[m2] / surf[m2]
                mae_m2 = float(np.mean(np.abs(yhat_m2 - y_m2)))
                rmse_m2 = float(np.sqrt(np.mean((yhat_m2 - y_m2) ** 2)))
                denom_m2 = float(np.sum(np.abs(y_m2)))
                wape_m2 = float(np.sum(np.abs(yhat_m2 - y_m2)) / denom_m2) if denom_m2 > 0 else np.nan

        # per-group
        rows = []
        for keys, g in pred_df.groupby(id_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            yy = g["y_true"].to_numpy(dtype=float)
            pp = g["y_pred"].to_numpy(dtype=float)
            if len(yy) == 0:
                continue
            denom_g = float(np.sum(np.abs(yy)))
            wape_g = float(np.sum(np.abs(yy - pp)) / denom_g) if (denom_g > 0 and target in (ELECTRIC_ALL + ["waterM3"])) else np.nan
            bias_g = float(np.mean(pp - yy))
            sst_g = float(np.sum((yy - np.mean(yy)) ** 2))
            sse_g = float(np.sum((yy - pp) ** 2))
            r2_g = float(1.0 - sse_g / sst_g) if sst_g > 0 else np.nan
            row = {c: v for c, v in zip(id_cols, keys)}
            row.update(
                {
                    "n": int(len(yy)),
                    "MAE": mae(yy, pp),
                    "RMSE": rmse(yy, pp),
                    "WAPE": wape_g,
                    "Bias": bias_g,
                    "R2": r2_g,
                }
            )
            rows.append(row)

        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"eval_{level}_{target}_by_group.csv", index=False)

        rep = pd.DataFrame(
            [
                {
                    "rows": int(len(valid)),
                    "MAE": mae(y, yhat),
                    "RMSE": rmse(y, yhat),
                    "WAPE": wape,
                    "Bias": bias,
                    "MedAE": medae,
                    "sMAPE": smape,
                    "R2": r2,
                    "MAE_m2": mae_m2,
                    "RMSE_m2": rmse_m2,
                    "WAPE_m2": wape_m2,
                }
            ]
        )
        rep.to_csv(out_dir / f"eval_{level}_{target}.csv", index=False)
        print(rep)

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
            evaluate_one(lvl, tgt)


if __name__ == "__main__":
    main()