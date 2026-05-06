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
from .modeling import load_model
from .reporting import parity_linear_95, parity_linear_99, parity_log, residual_hist, ts_train_valid_site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site")   # site | zone | all
    ap.add_argument("--target", required=True)   # elecTotalKwh | waterM3 | indoorTempDegC | all
    ap.add_argument("--site", default="170", help='siteId pour timeseries, ou "all"')
    args = ap.parse_args()

    LEVELS = ["site", "zone"]
    TARGETS_BY_LEVEL = {
        "site": ["elecTotalKwh", "waterM3"],
        "zone": ["elecTotalKwh", "waterM3", "indoorTempDegC"],
    }

    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))
    fig_dir = ensure_dir(out_dir / "figures")

    def report_one(level : str, target : str, site : str):
        cleaned_path = out_dir / f"{level}hist_cleaned.csv"
        hist = pd.read_csv(cleaned_path)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")

        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]

        model_dir = out_dir / "models"
        meta = json.loads((model_dir / f"{level}_{target}.meta.json").read_text(encoding="utf-8"))
        feat_cols = meta["feature_columns"]
        valid_days = int(meta.get("valid_days", cfg["training"].get("valid_days", 60)))
        log1p_target = bool(meta.get("log1p_target", False))

        df = hist[id_cols + ["date", target]].copy()
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.dropna(subset=id_cols + ["date", target])

        # weather
        _, _, weath = load_level_tables(db_dir, level_cfg)
        if len(weath) and "date" in weath.columns:
            weath["date"] = pd.to_datetime(weath["date"], errors="coerce").dt.floor("D")
        weather_cols = cfg["features"].get("weather_cols", [])

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
        for w in cfg["features"]["rolling_windows"]:
            roll_cols += [f"roll_med_{w}", f"roll_mean_{w}"]
        must_have = [c for c in (lag_cols + roll_cols) if c in df.columns]
        if must_have:
            df = df.dropna(subset=must_have)

        cutoff = df["date"].max() - pd.Timedelta(days=valid_days)
        valid = df[df["date"] > cutoff].copy()

        X = valid[feat_cols].copy()
        y = valid[target].to_numpy(dtype=float)

        model = load_model(model_dir / f"{level}_{target}.joblib")
        pred = model.predict(X)
        if log1p_target:
            yhat = np.expm1(pred)
            yhat = np.maximum(yhat, 0.0)
        else:
            yhat = pred

        site_arg = str(site).lower()
        if site_arg == "all":
            site_ids = sorted([int(x) for x in df["siteId"].dropna().unique().tolist()]) if "siteId" in df.columns else []
        else:
            site_ids = [int(site)]

        for sid in site_ids:
            dsite = df[df["siteId"] == sid].copy().sort_values("date")
            train_site = dsite[dsite["date"] <= cutoff][["date", target]].copy()
            valid_site = dsite[dsite["date"] > cutoff].copy()

            if len(valid_site):
                Xs = valid_site[feat_cols].copy()
                ps = model.predict(Xs)
                if log1p_target:
                    yhat_s = np.expm1(ps)
                    yhat_s = np.maximum(yhat_s, 0.0)
                else:
                    yhat_s = ps
                valid_site["yhat"] = yhat_s
            else:
                valid_site["yhat"] = np.nan

            ts_train_valid_site(
                train_df=train_site,
                valid_df=valid_site[["date", target, "yhat"]],
                date_col="date",
                y_true_col=target,
                y_pred_col="yhat",
                cutoff=cutoff,
                title=f"{level} {target} (train: vérité / valid: vérité+prédiction)",
                out=fig_dir / f"ts_site{sid}_{target}_train_valid.png",
                site_id=sid,
            )

        parity_linear_99(y, yhat, f"Parity — {level} {target} (valid)",
                fig_dir / f"parity_{level}_{target}_p99.png")
        parity_linear_95(y, yhat, f"Parity — {level} {target} (valid)",
                fig_dir / f"parity_{level}_{target}_p95.png")
        parity_log(y, yhat, f"Parity log — {level} {target} (valid)",
                fig_dir / f"parity_{level}_{target}_log.png")
        residual_hist(y, yhat, f"Residuals — {level} {target} (valid)",
                fig_dir / f"resid_{level}_{target}.png")

        print("wrote figures to", fig_dir)

    levels = LEVELS if args.level == "all" else [args.level]
    for lvl in levels:
        if lvl not in LEVELS:
            raise ValueError(f"Unknown level: {lvl}. Use site|zone|all")
        targets = TARGETS_BY_LEVEL[lvl] if args.target == "all" else [args.target]
        for tgt in targets:
            if tgt not in TARGETS_BY_LEVEL[lvl]:
                raise ValueError(f"Target {tgt} not supported for level {lvl}. Use {TARGETS_BY_LEVEL[lvl]} or all")
            report_one(lvl, tgt, args.site)


if __name__ == "__main__":
    main()
