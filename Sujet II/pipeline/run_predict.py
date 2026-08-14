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
from .targets_utils import discover_elec_usage_targets, drop_groups_with_no_signal, add_elec_total_accurate


ELECTRIC_USES_LEGACY = ["elecBveKwh", "elecCvcKwh", "elecForceKwh", "elecLightingKwh"]
ELEC_TOTAL_ACCURATE = "elecTotalAccurateKwh"
ELEC_AGGREGATED = "elecAggregatedKwh"
ELEC_TOTAL_NOBVE = "elecTotalNoBveKwh"

BASE_TARGETS = ["elecTotalKwh", ELEC_AGGREGATED, ELEC_TOTAL_ACCURATE, ELEC_TOTAL_NOBVE, "waterM3", "indoorTempDegC"]


def add_elec_total_no_bve(df: pd.DataFrame) -> pd.DataFrame:
    """
    elecTotalNoBveKwh = elecTotalAccurateKwh - elecBveKwh (fillna 0), clamp >=0.
    """
    if ELEC_TOTAL_NOBVE in df.columns:
        return df
    if ELEC_TOTAL_ACCURATE not in df.columns or "elecBveKwh" not in df.columns:
        return df
    out = df.copy()
    total = pd.to_numeric(out[ELEC_TOTAL_ACCURATE], errors="coerce")
    bve = pd.to_numeric(out["elecBveKwh"], errors="coerce").fillna(0.0)
    out[ELEC_TOTAL_NOBVE] = np.maximum(total - bve, 0.0)
    return out


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
    ap.add_argument("--target", required=True)  # ... | elecUses | all
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()

    LEVELS = ["site", "zone"]
    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))

    def _load_hist(level: str) -> pd.DataFrame:
        cleaned_path = out_dir / f"{level}hist_cleaned.csv"
        if cleaned_path.exists():
            hist = pd.read_csv(cleaned_path)
        else:
            hist, _, _ = load_level_tables(db_dir, cfg["level_defaults"][level])
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")
        hist = add_elec_total_accurate(hist)
        hist = add_elec_total_no_bve(hist)
        return hist

    def expand_targets(level: str, target: str) -> list[str]:
        if target == "elecUses":
            # legacy shortcut (still useful)
            return ELECTRIC_USES_LEGACY[:]
        if target == "all":
            hist = _load_hist(level)
            dyn_usages = discover_elec_usage_targets(hist)
            base = BASE_TARGETS[:]
            return base + [c for c in dyn_usages if c not in base]
        return [target]

    def allowed_targets(level: str) -> set[str]:
        hist = _load_hist(level)
        dyn_usages = discover_elec_usage_targets(hist)
        return set(BASE_TARGETS) | set(dyn_usages)

    def predict_one(level: str, target: str, days: int | None):
        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]

        hist = _load_hist(level)

        # site infos
        info_path = Path(args.config).resolve().parent / cfg.get("paths", {}).get(
            "site_infos_file", "Sites_Shyrka_Infos.xlsx"
        )
        site_infos = load_site_infos(info_path)
        if len(site_infos) and "siteId" in hist.columns:
            hist = hist.merge(site_infos, on="siteId", how="left")

        # Restrict groups if this is a usage: skip sites/zones where fully absent
        dyn_usages = set(discover_elec_usage_targets(hist))
        if target in dyn_usages:
            hist = drop_groups_with_no_signal(hist, id_cols, target)

        # weather
        _, _, weath = load_level_tables(db_dir, level_cfg)
        if len(weath) == 0:
            raise RuntimeError("Missing siteweath.csv")
        weath["date"] = pd.to_datetime(weath["date"], errors="coerce").dt.floor("D")

        model_dir = out_dir / "models"
        meta_path = model_dir / f"{level}_{target}.meta.json"
        model_path = model_dir / f"{level}_{target}.joblib"
        if not meta_path.exists() or not model_path.exists():
            print(f"[WARN] Missing meta/model for {level}_{target}. Skipping.")
            return

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feat_cols = meta["feature_columns"]
        cat_cols = meta.get("cat_columns", [])
        num_feat_cols = [c for c in feat_cols if c not in cat_cols]
        log1p_target = bool(meta.get("log1p_target", False))
        model = load_model(model_path)

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
            state[target] = state[target].replace([np.inf, -np.inf], np.nan)
            state.loc[state[target].abs() > 1e100, target] = np.nan

            state = state.dropna().sort_values("date")
            if state.empty:
                continue

            dates = pd.date_range(last_hist_date + pd.Timedelta(days=1),
                                  last_hist_date + pd.Timedelta(days=horizon), freq="D")
            wsite = weath[weath["siteId"] == site_id].drop_duplicates(
                subset=["siteId", "date"], keep="last"
            ).set_index("date")

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

                # Sanitize numeric features before sklearn.
                # SimpleImputer handles NaN, but not inf / -inf / absurd values.
                for c in num_feat_cols:
                    if c in X.columns:
                        X[c] = pd.to_numeric(X[c], errors="coerce")

                X = X.replace([np.inf, -np.inf], np.nan)

                for c in num_feat_cols:
                    if c in X.columns:
                        X.loc[X[c].abs() > 1e100, c] = np.nan

                pred = model.predict(X)
                raw_pred = float(pred[0])

                if not np.isfinite(raw_pred):
                    print(f"[WARN] Non-finite prediction skipped: level={level} target={target} keys={keys} date={d}")
                    continue

                if log1p_target:
                    # Prevent expm1 overflow in autoregressive prediction.
                    raw_pred = float(np.clip(raw_pred, 0.0, 30.0))
                    yhat = float(np.expm1(raw_pred))
                    yhat = max(yhat, 0.0)
                else:
                    yhat = raw_pred

                if not np.isfinite(yhat) or abs(yhat) > 1e100:
                    print(f"[WARN] Invalid yhat skipped: level={level} target={target} keys={keys} date={d} yhat={yhat}")
                    continue

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

        allowed = allowed_targets(lvl)
        targets = expand_targets(lvl, args.target)

        for tgt in targets:
            if tgt not in allowed:
                raise ValueError(f"Target {tgt} not supported for level {lvl}. Allowed base + dyn usages.")
            predict_one(lvl, tgt, args.days)


if __name__ == "__main__":
    main()
