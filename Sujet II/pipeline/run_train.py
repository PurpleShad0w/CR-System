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
from .site_infos import load_site_infos
from .targets_utils import discover_elec_usage_targets, discover_dynamic_consumption_targets, drop_groups_with_no_signal, add_elec_total_accurate
from .variable_config import normalize_input_columns


ELECTRIC_USES = ["elecBveKwh", "elecCvcKwh", "elecForceKwh", "elecLightingKwh"]
ELEC_TOTAL_NOBVE = "elecTotalNoBveKwh"
ELEC_TOTAL_ACCURATE = "elecTotalAccurateKwh"
ELEC_AGGREGATED = "elecAggregatedKwh"

# énergie = total + accurate + usages + noBVE (pour log1p)
ELECTRIC_ALL = ["elecTotalKwh", ELEC_AGGREGATED, ELEC_TOTAL_ACCURATE] + ELECTRIC_USES + [ELEC_TOTAL_NOBVE]



def _is_consumption_target(target: str) -> bool:
    """True for non-negative consumption-like targets, including dynamic drift targets."""
    return (
        target in (ELECTRIC_ALL + ["waterM3"])
        or (isinstance(target, str) and target.startswith("elec") and target.endswith("Kwh"))
        or (isinstance(target, str) and target.startswith("water") and target.endswith("M3"))
        or (isinstance(target, str) and target.startswith("ec"))
        or (isinstance(target, str) and target.startswith("eg"))
    )

def add_elec_total_no_bve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived target: elecTotalNoBveKwh = elecTotalKwh - elecBveKwh (fillna 0), clamp >= 0.
    If elecTotalKwh is NaN => output is NaN.
    """
    if ELEC_TOTAL_NOBVE in df.columns:
        return df
    if "elecTotalKwh" not in df.columns or "elecBveKwh" not in df.columns:
        return df
    out = df.copy()
    total = pd.to_numeric(out["elecTotalKwh"], errors="coerce")
    bve = pd.to_numeric(out["elecBveKwh"], errors="coerce").fillna(0.0)
    out[ELEC_TOTAL_NOBVE] = np.maximum(total - bve, 0.0)
    return out


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site")  # site | zone | all
    ap.add_argument(
        "--target",
        required=True,
        # elecTotalKwh | elecTotalAccurateKwh | elecTotalNoBveKwh | waterM3 | indoorTempDegC |
        # elecBveKwh | elecCvcKwh | elecForceKwh | elecLightingKwh | elecUses | all
    )
    args = ap.parse_args()

    LEVELS = ["site", "zone"]

    # ✅ all inclut le total "accurate" (drift) + noBVE + base
    BASE_TARGETS_BY_LEVEL = {
        "site": ["elecTotalKwh", ELEC_AGGREGATED, ELEC_TOTAL_ACCURATE, ELEC_TOTAL_NOBVE, "waterM3", "indoorTempDegC"],
        "zone": ["elecTotalKwh", ELEC_AGGREGATED, ELEC_TOTAL_ACCURATE, ELEC_TOTAL_NOBVE, "waterM3", "indoorTempDegC"],
    }

    # ✅ targets “supportés” (validation)
    TARGETS_BY_LEVEL = {
        "site": BASE_TARGETS_BY_LEVEL["site"] + ELECTRIC_USES,
        "zone": BASE_TARGETS_BY_LEVEL["zone"] + ELECTRIC_USES,
    }

    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))

    def expand_targets(level: str, target: str) -> list[str]:
        if target == "elecUses":
            return ELECTRIC_USES[:]
        if target == "all":
            # base + tous les usages élec présents
            cleaned = out_dir / f"{level}hist_cleaned.csv"
            if cleaned.exists():
                hist = pd.read_csv(cleaned)
            else:
                hist, _, _ = load_level_tables(db_dir, cfg["level_defaults"][level])
            hist = normalize_input_columns(hist, cfg)

            hist = add_elec_total_accurate(hist)
            elec_usages = discover_dynamic_consumption_targets(hist)

            # base “all” + usages détectés (sans doublons)
            base = BASE_TARGETS_BY_LEVEL[level][:]
            out = base + [c for c in elec_usages if c not in base]
            return out
        return [target]

    def train_one(level: str, target: str):
        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]

        cleaned_path = out_dir / f"{level}hist_cleaned.csv"
        if cleaned_path.exists():
            hist = pd.read_csv(cleaned_path)
            hist = normalize_input_columns(hist, cfg)
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")
        else:
            hist, _, _ = load_level_tables(db_dir, level_cfg)
            hist = normalize_input_columns(hist, cfg)
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")

        # --- site static infos (surface etc.) ---
        info_path = Path(args.config).resolve().parent / cfg.get("paths", {}).get(
            "site_infos_file", "Sites_Shyrka_Infos.xlsx"
        )
        site_infos = load_site_infos(info_path)
        if len(site_infos) and "siteId" in hist.columns:
            hist = hist.merge(site_infos, on="siteId", how="left")

        # ✅ ajoute total accurate + noBVE
        hist = add_elec_total_accurate(hist)
        hist = add_elec_total_no_bve(hist)

        _, _, weath = load_level_tables(db_dir, level_cfg)
        weath = normalize_input_columns(weath, cfg)
        if len(weath) and "date" in weath.columns:
            weath["date"] = pd.to_datetime(weath["date"], errors="coerce").dt.floor("D")

        static_cols = cfg.get("features", {}).get("static_cols", [])
        extra_cols = [c for c in static_cols if c in hist.columns]


        if target not in hist.columns:
            raise ValueError(f"Target '{target}' absent de {level}hist.")

        df = hist[id_cols + ["date", target] + extra_cols].copy()
        df[target] = pd.to_numeric(df[target], errors="coerce")

        # require enough positive points for usage targets
        if target in discover_dynamic_consumption_targets(hist):
            pos = (df[target].fillna(0.0) > 0).sum()
            if pos < 50:
                print(f"[WARN] level={level} target={target}: only {pos} positive points. Skipping.")
                return

        # ✅ si usage absent d’un site/zone, on le drop
        if target in discover_dynamic_consumption_targets(hist):
            df = drop_groups_with_no_signal(df, id_cols, target)

        # sécurité spécifique température
        if target == "indoorTempDegC":
            df[target] = df[target].mask(df[target] <= 0, np.nan)

        df = df.dropna(subset=id_cols + ["date", target])

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

        feat_cols = select_feature_columns(
            df,
            id_cols if cfg["features"].get("add_site_id", True) else [],
            weather_cols,
        )
        for c in extra_cols:
            if c not in feat_cols:
                feat_cols.append(c)

        # filtre lags/rollings
        lag_cols = [f"lag_{k}" for k in cfg["features"]["lags"]]
        roll_cols = []
        for wdw in cfg["features"]["rolling_windows"]:
            roll_cols += [f"roll_med_{wdw}", f"roll_mean_{wdw}"]
        must_have = [c for c in (lag_cols + roll_cols) if c in df.columns]
        if must_have:
            df = df.dropna(subset=must_have)

        valid_days = int(cfg["training"].get("valid_days", 60))
        cutoff = df["date"].max() - pd.Timedelta(days=valid_days)
        train = df[df["date"] <= cutoff].copy()
        valid = df[df["date"] > cutoff].copy()

        # ---- SAFETY: skip targets with insufficient data ----
        if len(df) < 30:
            print(f"[WARN] level={level} target={target}: too few rows after feature engineering ({len(df)}). Skipping.")
            return

        if len(train) == 0:
            print(f"[WARN] level={level} target={target}: no training rows (valid_days={valid_days}). Skipping.")
            return

        if len(valid) == 0:
            print(f"[WARN] level={level} target={target}: no validation rows (valid_days={valid_days}). Skipping.")
            return

        X_train = train[feat_cols].copy()
        y_train = train[target].to_numpy(dtype=float)
        X_valid = valid[feat_cols].copy()
        y_valid = valid[target].to_numpy(dtype=float)

        # ✅ log1p pour énergie (inclut accurate) + eau
        use_log1p = _is_consumption_target(target)
        y_train_t = np.log1p(y_train) if use_log1p else y_train

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

        if X_train.shape[0] == 0:
            print(f"[WARN] level={level} target={target}: X_train empty after selecting features. Skipping.")
            return

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
        print(
            f"[TRAIN] level={level} target={target} valid_rows={len(valid)} "
            f"MAE={mae(y_valid, yhat):.4f} RMSE={rmse(y_valid, yhat):.4f} MAPE={mape(y_valid, yhat):.4f}"
        )
        print("saved", model_path)

    levels = LEVELS if args.level == "all" else [args.level]
    for level in levels:
        if level not in LEVELS:
            raise ValueError("Unknown level. Use site|zone|all")

        # build allowed targets dynamically from hist columns
        cleaned = out_dir / f"{level}hist_cleaned.csv"
        if cleaned.exists():
            hist0 = pd.read_csv(cleaned)
        else:
            hist0, _, _ = load_level_tables(db_dir, cfg["level_defaults"][level])

        hist0 = add_elec_total_accurate(hist0)
        dyn_targets = discover_dynamic_consumption_targets(hist0)

        allowed = set(TARGETS_BY_LEVEL[level]) | set(dyn_targets) | {ELEC_TOTAL_ACCURATE}

        targets = expand_targets(level, args.target)
        for target in targets:
            if target == "elecUses":
                # expand_targets ne retourne pas elecUses ici normalement, mais safe
                continue
            if target not in allowed:
                raise ValueError(
                    f"Target {target} not supported for level {level}. "
                    f"Allowed base={TARGETS_BY_LEVEL[level]} + dyn_usages({len(dyn_targets)})"
                )
            train_one(level, target)


if __name__ == "__main__":
    main()
