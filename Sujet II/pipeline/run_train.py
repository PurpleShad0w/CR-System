from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from .config import load_config
from .dataset import load_level_tables
from .io_utils import ensure_dir
from .features import (
    add_calendar_features,
    build_lag_features,
    build_rolling_features,
    select_feature_columns,
)
from .modeling import make_model, save_model


def _make_ohe():
    # compat sklearn (sparse_output récent, sparse ancien)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site", choices=["site", "zone"])
    ap.add_argument("--target", required=True, choices=["elecTotalKwh", "waterM3", "indoorTempDegC"])
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))

    level_cfg = cfg["level_defaults"][args.level]
    id_cols = level_cfg["id_cols"]

    # Load cleaned hist if present
    cleaned_path = out_dir / f"{args.level}hist_cleaned.csv"
    if cleaned_path.exists():
        hist = pd.read_csv(cleaned_path)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")
    else:
        hist, _, _ = load_level_tables(db_dir, level_cfg)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")

    # Weather (historical)
    _, _, weath = load_level_tables(db_dir, level_cfg)
    if len(weath) and "date" in weath.columns:
        weath["date"] = pd.to_datetime(weath["date"], errors="coerce").dt.floor("D")

    # Base frame
    df = hist[id_cols + ["date", args.target]].copy()
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df.dropna(subset=id_cols + ["date", args.target])

    # Merge weather
    weather_cols = cfg["features"].get("weather_cols", [])
    if len(weath) and "siteId" in weath.columns and "date" in weath.columns:
        keep = ["siteId", "date"] + [c for c in weather_cols if c in weath.columns]
        w = weath[keep].drop_duplicates(subset=["siteId", "date"], keep="last")
        df = df.merge(w, on=["siteId", "date"], how="left")

    # Calendar + lags + rolling
    if cfg["features"].get("add_calendar", True):
        df = add_calendar_features(df, "date")

    df = build_lag_features(df, id_cols, "date", args.target, cfg["features"]["lags"])
    df = build_rolling_features(df, id_cols, "date", args.target, cfg["features"]["rolling_windows"])

    feat_cols = select_feature_columns(
        df, id_cols if cfg["features"].get("add_site_id", True) else [], weather_cols
    )

    # Drop rows where lags/rollings are missing (stabilise fortement la perf)
    lag_cols = [f"lag_{k}" for k in cfg["features"]["lags"]]
    roll_cols = []
    for w in cfg["features"]["rolling_windows"]:
        roll_cols += [f"roll_med_{w}", f"roll_mean_{w}"]
    must_have = [c for c in (lag_cols + roll_cols) if c in df.columns]
    if must_have:
        df = df.dropna(subset=must_have)

    # Time split
    valid_days = int(cfg["training"].get("valid_days", 60))
    cutoff = df["date"].max() - pd.Timedelta(days=valid_days)
    train = df[df["date"] <= cutoff].copy()
    valid = df[df["date"] > cutoff].copy()

    X_train = train[feat_cols].copy()
    y_train = train[args.target].to_numpy(dtype=float)
    X_valid = valid[feat_cols].copy()
    y_valid = valid[args.target].to_numpy(dtype=float)

    # log1p uniquement pour énergie/eau (pas pour la température)
    use_log1p = args.target in ["elecTotalKwh", "waterM3"]
    y_train_t = np.log1p(y_train) if use_log1p else y_train

    # OneHot siteId (et calendaires discrets)
    candidate_cat = ["siteId", "dow", "month", "is_weekend"]
    cat_cols = [c for c in candidate_cat if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    ohe = _make_ohe()

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )

    model = make_model(cfg)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train_t)

    # Save model + meta
    model_dir = out_dir / "models"
    model_path = model_dir / f"{args.level}_{args.target}.joblib"
    save_model(pipe, model_path)

    meta = {
        "level": args.level,
        "target": args.target,
        "feature_columns": feat_cols,
        "cat_columns": cat_cols,
        "log1p_target": use_log1p,
        "valid_days": valid_days,
    }
    (model_dir / f"{args.level}_{args.target}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Validation (inverse transform si log1p)
    pred_t = pipe.predict(X_valid)
    if use_log1p:
        yhat = np.expm1(pred_t)
        yhat = np.maximum(yhat, 0.0)
    else:
        yhat = pred_t

    from .modeling import mae, rmse, mape
    print("valid rows:", len(valid))
    print("MAE", mae(y_valid, yhat))
    print("RMSE", rmse(y_valid, yhat))
    print("MAPE", mape(y_valid, yhat))
    print("saved", model_path)


if __name__ == "__main__":
    main()
