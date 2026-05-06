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


def _expand_daily(df_in: pd.DataFrame, group_cols: list[str], date_col: str) -> pd.DataFrame:
    """
    Ensure there is one row per day per group between min(date) and max(date).
    Missing days become rows with NaN values -> we can still predict (imputer handles features).
    """
    df = df_in.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.floor("D")
    parts = []

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.dropna(subset=[date_col]).sort_values(date_col)
        if g.empty:
            continue

        idx = pd.date_range(g[date_col].min(), g[date_col].max(), freq="D")
        g2 = g.set_index(date_col).reindex(idx)
        g2.index.name = date_col

        for c, v in zip(group_cols, keys):
            g2[c] = v

        parts.append(g2.reset_index())

    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site")   # site | zone | all
    ap.add_argument("--target", required=True)   # elecTotalKwh | waterM3 | indoorTempDegC | all
    ap.add_argument("--site", default="170", help='siteId pour timeseries, ou "all"')
    args = ap.parse_args()

    LEVELS = ["site", "zone"]
    TARGETS_BY_LEVEL = {
        "site": ["elecTotalKwh", "waterM3", "indoorTempDegC"],
        "zone": ["elecTotalKwh", "waterM3", "indoorTempDegC"],
    }

    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))
    fig_dir = ensure_dir(out_dir / "figures")

    def report_one(level: str, target: str, site: str):
        cleaned_path = out_dir / f"{level}hist_cleaned.csv"
        hist = pd.read_csv(cleaned_path)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")

        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]

        if level == "zone" and "zoneId" not in level_cfg["id_cols"]:
            raise RuntimeError("Config error: zone level must include zoneId in id_cols")

        if level == "zone" and "zoneId" not in hist.columns:
            raise RuntimeError("zonehist missing zoneId column")

        model_dir = out_dir / "models"
        meta = json.loads((model_dir / f"{level}_{target}.meta.json").read_text(encoding="utf-8"))
        feat_cols = meta["feature_columns"]
        valid_days = int(meta.get("valid_days", cfg["training"].get("valid_days", 60)))
        log1p_target = bool(meta.get("log1p_target", False))

        # -------------------------------
        # Build base DF WITHOUT dropping target:
        # we want to predict even when truth is missing.
        # -------------------------------
        df0 = hist[id_cols + ["date", target]].copy()
        df0[target] = pd.to_numeric(df0[target], errors="coerce")
        df0 = df0.dropna(subset=id_cols + ["date"])  # keep rows even if target is NaN

        # Expand to daily grid per group to create missing dates
        df0 = _expand_daily(df0, id_cols, "date")

        # weather
        _, _, weath = load_level_tables(db_dir, level_cfg)
        if len(weath) and "date" in weath.columns:
            weath["date"] = pd.to_datetime(weath["date"], errors="coerce").dt.floor("D")
        weather_cols = cfg["features"].get("weather_cols", [])

        if len(weath) and "siteId" in weath.columns and "date" in weath.columns:
            keep = ["siteId", "date"] + [c for c in weather_cols if c in weath.columns]
            w = weath[keep].drop_duplicates(subset=["siteId", "date"], keep="last")
            df0 = df0.merge(w, on=["siteId", "date"], how="left")

        if cfg["features"].get("add_calendar", True):
            df0 = add_calendar_features(df0, "date")

        df0 = build_lag_features(df0, id_cols, "date", target, cfg["features"]["lags"])
        df0 = build_rolling_features(df0, id_cols, "date", target, cfg["features"]["rolling_windows"])

        # must_have (lags/rollings) -> used for eval only; NOT used to drop timeseries prediction rows
        lag_cols = [f"lag_{k}" for k in cfg["features"]["lags"]]
        roll_cols = []
        for wdw in cfg["features"]["rolling_windows"]:
            roll_cols += [f"roll_med_{wdw}", f"roll_mean_{wdw}"]
        must_have = [c for c in (lag_cols + roll_cols) if c in df0.columns]

        # df_pred: used for timeseries predictions (keep everything)
        df_pred = df0

        # df_eval: used for parity/residuals (need truth + usable features)
        df_eval = df0.dropna(subset=[target])
        if must_have:
            df_eval = df_eval.dropna(subset=must_have)

        if len(df_eval) == 0:
            print(f"[WARN] No evaluable rows for level={level}, target={target}. Skipping parity/residual.")
            return

        # cutoff based on evaluable truth (stable split)
        cutoff = df_eval["date"].max() - pd.Timedelta(days=valid_days)

        model = load_model(model_dir / f"{level}_{target}.joblib")

        # -------------------------------
        # PARITY / RESIDUALS (evaluation only)
        # -------------------------------
        valid_eval = df_eval[df_eval["date"] > cutoff].copy()
        X_eval = valid_eval[feat_cols].copy()
        y_eval = valid_eval[target].to_numpy(dtype=float)

        pred_eval = model.predict(X_eval)
        if log1p_target:
            yhat_eval = np.expm1(pred_eval)
            yhat_eval = np.maximum(yhat_eval, 0.0)
        else:
            yhat_eval = pred_eval

        parity_linear_99(y_eval, yhat_eval, f"Parity — {level} {target} (valid)",
                         fig_dir / f"parity_{level}_{target}_p99.png")
        parity_linear_95(y_eval, yhat_eval, f"Parity — {level} {target} (valid)",
                         fig_dir / f"parity_{level}_{target}_p95.png")
        parity_log(y_eval, yhat_eval, f"Parity log — {level} {target} (valid)",
                   fig_dir / f"parity_{level}_{target}_log.png")
        residual_hist(y_eval, yhat_eval, f"Residuals — {level} {target} (valid)",
                      fig_dir / f"resid_{level}_{target}.png")

        # -------------------------------
        # TIMESERIES (predict on ALL valid dates, even where truth is missing)
        # -------------------------------
        site_arg = str(site).lower()
        if site_arg == "all":
            site_ids = sorted([int(x) for x in df_pred["siteId"].dropna().unique().tolist()]) if "siteId" in df_pred.columns else []
        else:
            site_ids = [int(site)]

        # Sous-dossier dédié aux graphes zone-level
        zone_fig_dir = ensure_dir(fig_dir / "zones")

        for sid in site_ids:
            if level == "zone" and "zoneId" in df_pred.columns:
                # toutes les zones du site (sid)
                zone_ids = sorted([int(z) for z in df_pred.loc[df_pred["siteId"] == sid, "zoneId"].dropna().unique().tolist()])

                for zid in zone_ids:
                    dzone = df_pred[(df_pred["siteId"] == sid) & (df_pred["zoneId"] == zid)].copy().sort_values("date")

                    train_zone = dzone[dzone["date"] <= cutoff][["date", target]].copy()
                    valid_zone = dzone[dzone["date"] > cutoff].copy()

                    # Prédire sur toutes les dates valid présentes dans dzone (même si target est NaN)
                    valid_zone["yhat"] = np.nan
                    if len(valid_zone):
                        Xs = valid_zone[feat_cols].copy()
                        ps = model.predict(Xs)
                        if log1p_target:
                            yhat_z = np.expm1(ps)
                            yhat_z = np.maximum(yhat_z, 0.0)
                        else:
                            yhat_z = ps
                        valid_zone["yhat"] = yhat_z

                    ts_train_valid_site(
                        train_df=train_zone,
                        valid_df=valid_zone[["date", target, "yhat"]],
                        date_col="date",
                        y_true_col=target,
                        y_pred_col="yhat",
                        cutoff=cutoff,
                        title=f"{level} {target} (train: vérité / valid: vérité+prédiction) — zone {zid}",
                        out=zone_fig_dir / f"ts_site{sid}_zone{zid}_{target}_train_valid.png",
                        site_id=sid,
                    )

            else:
                # site-level (inchangé)
                dsite = df_pred[df_pred["siteId"] == sid].copy().sort_values("date")
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
