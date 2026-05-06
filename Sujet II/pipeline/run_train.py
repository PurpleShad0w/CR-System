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
    ap.add_argument("--level", default="site")         # site | zone | all
    ap.add_argument("--target", required=True)         # elecTotalKwh | waterM3 | indoorTempDegC | all
    args = ap.parse_args()

    LEVELS = ["site", "zone"]
    TARGETS_BY_LEVEL = {
        "site": ["elecTotalKwh", "waterM3", "indoorTempDegC"],
        "zone": ["elecTotalKwh", "waterM3", "indoorTempDegC"],
    }

    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))

    def train_one(cfg, db_dir: Path, out_dir: Path, level: str, target: str):
        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]

        cleaned_path = out_dir / f"{level}hist_cleaned.csv"
        if cleaned_path.exists():
            hist = pd.read_csv(cleaned_path)
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")
        else:
            hist, _, _ = load_level_tables(db_dir, level_cfg)
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")

        _, _, weath = load_level_tables(db_dir, level_cfg)
        if len(weath) and "date" in weath.columns:
            weath["date"] = pd.to_datetime(weath["date"], errors="coerce").dt.floor("D")

        df = hist[id_cols + ["date", target]].copy()
        df[target] = pd.to_numeric(df[target], errors="coerce")

        # sécurité spécifique température (si pas déjà fait au clean)
        if target == "indoorTempDegC":
            df[target] = df[target].mask(df[target] <= 0, np.nan)

        df = df.dropna(subset=id_cols + ["date", target])

        weather_cols = cfg["features"].get("weather_cols", [])
        if len(weath) and "siteId" in weath.columns and "date" in weath.columns:
            keep = ["siteId", "date"] + [c for c in weather_cols if c in weath.columns]
            w = weath[keep].drop_duplicates(subset=["siteId", "date"], keep="last")
            df = df.merge(w, on=["siteId", "date"], how="left")

        if cfg["features"].get("add_calendar", True):
            df = add_calendar_features(df, "date")

        df = build_lag_features(df, id_cols, "date", target, cfg["features"]["lags"])
        df = build_rolling_features(df, id_cols, "date", target, cfg["features"]["rolling_windows"])

        feat_cols = select_feature_columns(df, id_cols if cfg["features"].get("add_site_id", True) else [], weather_cols)

        # filtre lags/rollings
        lag_cols = [f"lag_{k}" for k in cfg["features"]["lags"]]
        roll_cols = []
        for w in cfg["features"]["rolling_windows"]:
            roll_cols += [f"roll_med_{w}", f"roll_mean_{w}"]
        must_have = [c for c in (lag_cols + roll_cols) if c in df.columns]
        if must_have:
            df = df.dropna(subset=must_have)

        valid_days = int(cfg["training"].get("valid_days", 60))
        cutoff = df["date"].max() - pd.Timedelta(days=valid_days)
        train = df[df["date"] <= cutoff].copy()
        valid = df[df["date"] > cutoff].copy()

        X_train = train[feat_cols].copy()
        y_train = train[target].to_numpy(dtype=float)
        X_valid = valid[feat_cols].copy()
        y_valid = valid[target].to_numpy(dtype=float)

        # log1p uniquement pour énergie/eau
        use_log1p = target in ["elecTotalKwh", "waterM3"]
        y_train_t = np.log1p(y_train) if use_log1p else y_train

        # --- preprocessing actuel (imputer + OHE) ---
        candidate_cat = ["siteId", "dow", "month", "is_weekend"]
        cat_cols = [c for c in candidate_cat if c in X_train.columns]
        num_cols = [c for c in X_train.columns if c not in cat_cols]

        ohe = _make_ohe()
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])

        pre = ColumnTransformer(
            transformers=[("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)],
            remainder="drop",
        )

        model = make_model(cfg)
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train_t)

        model_dir = out_dir / "models"
        model_path = model_dir / f"{level}_{target}.joblib"
        save_model(pipe, model_path)

        meta = {
            "level": level,
            "target": target,
            "feature_columns": feat_cols,
            "cat_columns": cat_cols,
            "log1p_target": use_log1p,
            "valid_days": valid_days,
        }
        (model_dir / f"{level}_{target}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        pred_t = pipe.predict(X_valid)
        yhat = np.expm1(pred_t) if use_log1p else pred_t
        if use_log1p:
            yhat = np.maximum(yhat, 0.0)

        from .modeling import mae, rmse, mape
        print(f"[TRAIN] level={level} target={target} valid_rows={len(valid)} "
          f"MAE={mae(y_valid, yhat):.4f} RMSE={rmse(y_valid, yhat):.4f} MAPE={mape(y_valid, yhat):.4f}")
        print("saved", model_path)

    # levels to run
    levels = LEVELS if args.level == "all" else [args.level]
    for level in levels:
        if level not in LEVELS:
            raise ValueError(f"Unknown level: {level}. Use site|zone|all")

        # targets to run
        targets = TARGETS_BY_LEVEL[level] if args.target == "all" else [args.target]
        for target in targets:
            if target not in TARGETS_BY_LEVEL[level]:
                raise ValueError(f"Target {target} not supported for level {level}. "
                             f"Use {TARGETS_BY_LEVEL[level]} or all")
            train_one(cfg, db_dir, out_dir, level, target)


if __name__ == "__main__":
    main()
