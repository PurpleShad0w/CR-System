from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import load_config
from .io_utils import ensure_dir
from .dataset import load_level_tables
from .features import add_calendar_features, build_lag_features, build_rolling_features
from .modeling import load_model
from .reporting import parity_linear_95, parity_linear_99, parity_log, residual_hist, ts_train_valid_site
from .site_infos import load_site_infos


ELECTRIC_USES = ["elecBveKwh", "elecCvcKwh", "elecForceKwh", "elecLightingKwh"]
ELEC_TOTAL_NOBVE = "elecTotalNoBveKwh"
ELECTRIC_ALL = ["elecTotalKwh", ELEC_TOTAL_NOBVE] + ELECTRIC_USES
ENERGY_TARGETS = ELECTRIC_ALL + ["waterM3"]  # tout ce qui est “positif / skew”


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


# ----------------------------
# metric helpers for compare
# ----------------------------
def _rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    return float(np.mean(np.abs(a[m] - b[m])))


def _wape(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    denom = float(np.sum(np.abs(a[m])))
    return float(np.sum(np.abs(a[m] - b[m])) / denom) if denom > 0 else np.nan


def _smape(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    denom = np.abs(a[m]) + np.abs(b[m])
    mm = denom > 0
    return float(200.0 * np.mean(np.abs(a[m][mm] - b[m][mm]) / denom[mm])) if np.any(mm) else np.nan


def _imp_pct(base, new):
    if not (np.isfinite(base) and np.isfinite(new)) or base == 0:
        return np.nan
    return float((base - new) / base * 100.0)


def _save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site")  # site | zone | all
    ap.add_argument(
        "--target",
        required=True,
        # elecTotalKwh | elecTotalNoBveKwh | waterM3 | indoorTempDegC | elecBveKwh | elecCvcKwh | elecForceKwh | elecLightingKwh | elecUses | all
    )
    ap.add_argument("--site", default="170", help='siteId pour timeseries, ou "all"')
    args = ap.parse_args()

    LEVELS = ["site", "zone"]

    # ✅ all inclut elecTotalNoBveKwh
    BASE_TARGETS_BY_LEVEL = {
        "site": ["elecTotalKwh", ELEC_TOTAL_NOBVE, "waterM3", "indoorTempDegC"],
        "zone": ["elecTotalKwh", ELEC_TOTAL_NOBVE, "waterM3", "indoorTempDegC"],
    }
    TARGETS_BY_LEVEL = {
        "site": BASE_TARGETS_BY_LEVEL["site"] + ELECTRIC_USES,
        "zone": BASE_TARGETS_BY_LEVEL["zone"] + ELECTRIC_USES,
    }

    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))
    fig_dir = ensure_dir(out_dir / "figures")

    def expand_targets(level: str, target: str) -> list[str]:
        if target == "elecUses":
            return ELECTRIC_USES[:]
        if target == "all":
            return BASE_TARGETS_BY_LEVEL[level][:]
        return [target]

    def report_one(level: str, target: str, site: str):
        cleaned_path = out_dir / f"{level}hist_cleaned.csv"
        hist = pd.read_csv(cleaned_path)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")

        info_path = Path(args.config).resolve().parent / cfg.get("paths", {}).get(
            "site_infos_file", "Sites_Shyrka_Infos.xlsx"
        )
        site_infos = load_site_infos(info_path)
        if len(site_infos) and "siteId" in hist.columns:
            hist = hist.merge(site_infos, on="siteId", how="left")

        # ✅ ensure derived exists even if clean wasn't rerun
        hist = add_elec_total_no_bve(hist)

        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]

        if level == "zone" and "zoneId" not in level_cfg["id_cols"]:
            raise RuntimeError("Config error: zone level must include zoneId in id_cols")
        if level == "zone" and "zoneId" not in hist.columns:
            raise RuntimeError("zonehist missing zoneId column")

        model_dir = out_dir / "models"
        meta_path = model_dir / f"{level}_{target}.meta.json"
        if not meta_path.exists():
            print(f"[WARN] Missing meta for {level}_{target} (train not run?). Skipping.")
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feat_cols = meta["feature_columns"]
        valid_days = int(meta.get("valid_days", cfg["training"].get("valid_days", 60)))
        log1p_target = bool(meta.get("log1p_target", False))

        static_cols = cfg.get("features", {}).get("static_cols", [])
        extra_cols = [c for c in static_cols if c in hist.columns]

        if target not in hist.columns:
            print(f"[WARN] Target {target} not in {level}hist_cleaned.csv. Skipping.")
            return None

        df0 = hist[id_cols + ["date", target] + extra_cols].copy()
        df0[target] = pd.to_numeric(df0[target], errors="coerce")
        df0 = df0.dropna(subset=id_cols + ["date"] + extra_cols)  # keep rows even if target is NaN
        df0 = _expand_daily(df0, id_cols, "date")

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
            df0 = df0.merge(w, on=["siteId", "date"], how="left")

        if cfg["features"].get("add_calendar", True):
            df0 = add_calendar_features(df0, "date")

        df0 = build_lag_features(df0, id_cols, "date", target, cfg["features"]["lags"])
        df0 = build_rolling_features(df0, id_cols, "date", target, cfg["features"]["rolling_windows"])

        lag_cols = [f"lag_{k}" for k in cfg["features"]["lags"]]
        roll_cols = []
        for wdw in cfg["features"]["rolling_windows"]:
            roll_cols += [f"roll_med_{wdw}", f"roll_mean_{wdw}"]
        must_have = [c for c in (lag_cols + roll_cols) if c in df0.columns]

        df_pred = df0
        df_eval = df0.dropna(subset=[target])
        if must_have:
            df_eval = df_eval.dropna(subset=must_have)

        if len(df_eval) == 0:
            print(f"[WARN] No evaluable rows for level={level}, target={target}. Skipping parity/residual.")
            return None

        cutoff = df_eval["date"].max() - pd.Timedelta(days=valid_days)
        model_path = model_dir / f"{level}_{target}.joblib"
        if not model_path.exists():
            print(f"[WARN] Missing model for {level}_{target} (train not run?). Skipping.")
            return None
        model = load_model(model_path)

        # PARITY / RESIDUALS (eval only)
        valid_eval = df_eval[df_eval["date"] > cutoff].copy()
        X_eval = valid_eval[feat_cols].copy()
        y_eval = valid_eval[target].to_numpy(dtype=float)

        pred_eval = model.predict(X_eval)
        if log1p_target:
            yhat_eval = np.expm1(pred_eval)
            yhat_eval = np.maximum(yhat_eval, 0.0)
        else:
            yhat_eval = pred_eval

        parity_linear_99(
            y_eval, yhat_eval,
            f"Parity — {level} {target} (valid)",
            fig_dir / f"parity_{level}_{target}_p99.png"
        )
        parity_linear_95(
            y_eval, yhat_eval,
            f"Parity — {level} {target} (valid)",
            fig_dir / f"parity_{level}_{target}_p95.png"
        )
        parity_log(
            y_eval, yhat_eval,
            f"Parity log — {level} {target} (valid)",
            fig_dir / f"parity_{level}_{target}_log.png"
        )
        residual_hist(
            y_eval, yhat_eval,
            f"Residuals — {level} {target} (valid)",
            fig_dir / f"resid_{level}_{target}.png"
        )

        # diagnostic payload
        pred_df = valid_eval[id_cols + ["date"]].copy()
        pred_df["y_true"] = y_eval
        pred_df["y_pred"] = yhat_eval

        # TIMESERIES (predict even where truth is missing)
        site_arg = str(site).lower()
        if site_arg == "all":
            site_ids = sorted([int(x) for x in df_pred["siteId"].dropna().unique().tolist()]) if "siteId" in df_pred.columns else []
        else:
            site_ids = [int(site)]

        zone_fig_dir = ensure_dir(fig_dir / "zones")

        for sid in site_ids:
            if level == "zone" and "zoneId" in df_pred.columns:
                zone_ids = sorted([int(z) for z in df_pred.loc[df_pred["siteId"] == sid, "zoneId"].dropna().unique().tolist()])
                for zid in zone_ids:
                    dzone = df_pred[(df_pred["siteId"] == sid) & (df_pred["zoneId"] == zid)].copy().sort_values("date")
                    train_zone = dzone[dzone["date"] <= cutoff][["date", target]].copy()
                    valid_zone = dzone[dzone["date"] > cutoff].copy()
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
        return {"level": level, "target": target, "id_cols": id_cols, "pred_df": pred_df}

    # -------------------------------
    # run all requested targets
    # -------------------------------
    reports = []
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

            r = report_one(lvl, tgt, args.site)
            if r is not None:
                reports.append(r)

    # ------------------------------------------------------------
    # COMPARISON + VISUALS: elecTotalKwh vs elecTotalNoBveKwh
    # ------------------------------------------------------------
    cmp_dir = ensure_dir(fig_dir / "compare")
    TOTAL = "elecTotalKwh"
    NOBVE = ELEC_TOTAL_NOBVE

    by_lt = {(r["level"], r["target"]): r for r in reports}

    for lvl in ["site", "zone"]:
        if (lvl, TOTAL) not in by_lt or (lvl, NOBVE) not in by_lt:
            continue

        r_tot = by_lt[(lvl, TOTAL)]
        r_nb = by_lt[(lvl, NOBVE)]
        id_cols = r_tot["id_cols"]

        a = r_tot["pred_df"].copy()
        b = r_nb["pred_df"].copy()

        m = a.merge(b, on=id_cols + ["date"], how="inner", suffixes=("_total", "_nobve"))
        if len(m) == 0:
            continue

        yt_tot = m["y_true_total"].to_numpy(dtype=float)
        yp_tot = m["y_pred_total"].to_numpy(dtype=float)
        yt_nb = m["y_true_nobve"].to_numpy(dtype=float)
        yp_nb = m["y_pred_nobve"].to_numpy(dtype=float)

        glob = {
            "level": lvl,
            "rows_common": int(len(m)),
            "MAE_total": _mae(yt_tot, yp_tot),
            "RMSE_total": _rmse(yt_tot, yp_tot),
            "WAPE_total": _wape(yt_tot, yp_tot),
            "sMAPE_total": _smape(yt_tot, yp_tot),
            "MAE_nobve": _mae(yt_nb, yp_nb),
            "RMSE_nobve": _rmse(yt_nb, yp_nb),
            "WAPE_nobve": _wape(yt_nb, yp_nb),
            "sMAPE_nobve": _smape(yt_nb, yp_nb),
        }
        glob["MAE_improvement_%"] = _imp_pct(glob["MAE_total"], glob["MAE_nobve"])
        glob["RMSE_improvement_%"] = _imp_pct(glob["RMSE_total"], glob["RMSE_nobve"])
        glob["WAPE_improvement_%"] = _imp_pct(glob["WAPE_total"], glob["WAPE_nobve"])

        pd.DataFrame([glob]).to_csv(cmp_dir / f"compare_elec_total_vs_nobve_{lvl}_global.csv", index=False)

        rows = []
        for keys, g in m.groupby(id_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            yt1 = g["y_true_total"].to_numpy(dtype=float)
            yp1 = g["y_pred_total"].to_numpy(dtype=float)
            yt2 = g["y_true_nobve"].to_numpy(dtype=float)
            yp2 = g["y_pred_nobve"].to_numpy(dtype=float)

            row = {c: v for c, v in zip(id_cols, keys)}
            row.update({
                "n": int(len(g)),
                "MAE_total": _mae(yt1, yp1),
                "RMSE_total": _rmse(yt1, yp1),
                "WAPE_total": _wape(yt1, yp1),
                "MAE_nobve": _mae(yt2, yp2),
                "RMSE_nobve": _rmse(yt2, yp2),
                "WAPE_nobve": _wape(yt2, yp2),
            })
            row["MAE_improvement_%"] = _imp_pct(row["MAE_total"], row["MAE_nobve"])
            row["RMSE_improvement_%"] = _imp_pct(row["RMSE_total"], row["RMSE_nobve"])
            row["WAPE_improvement_%"] = _imp_pct(row["WAPE_total"], row["WAPE_nobve"])
            rows.append(row)

        df_rows = pd.DataFrame(rows)
        df_rows.to_csv(cmp_dir / f"compare_elec_total_vs_nobve_{lvl}_by_group.csv", index=False)

        # ----------------
        # VISUAL 1: bar chart global
        # ----------------
        labels = ["MAE", "RMSE", "WAPE"]
        total_vals = [glob["MAE_total"], glob["RMSE_total"], glob["WAPE_total"]]
        nobve_vals = [glob["MAE_nobve"], glob["RMSE_nobve"], glob["WAPE_nobve"]]

        x = np.arange(len(labels))
        w = 0.38

        plt.figure(figsize=(9, 4.5))
        plt.bar(x - w/2, total_vals, width=w, label="elecTotalKwh", color="#1f77b4")
        plt.bar(x + w/2, nobve_vals, width=w, label="elecTotalNoBveKwh", color="#2ca02c")
        plt.xticks(x, labels)
        plt.ylabel("Erreur (plus bas = mieux)")
        plt.title(
            f"Comparaison global — {lvl} — total vs noBVE\n"
            f"ΔRMSE={glob['RMSE_improvement_%']:.1f}%  ΔWAPE={glob['WAPE_improvement_%']:.1f}%  ΔMAE={glob['MAE_improvement_%']:.1f}%"
        )
        plt.legend(loc="upper right")
        _save_fig(cmp_dir / f"compare_elec_total_vs_nobve_{lvl}_bars.png")

        # ----------------
        # VISUAL 2: histogram improvements per group
        # ----------------
        plt.figure(figsize=(9, 4.5))
        imp_rmse = pd.to_numeric(df_rows["RMSE_improvement_%"], errors="coerce").dropna().to_numpy(dtype=float)
        imp_wape = pd.to_numeric(df_rows["WAPE_improvement_%"], errors="coerce").dropna().to_numpy(dtype=float)

        bins = 30
        if len(imp_rmse):
            plt.hist(imp_rmse, bins=bins, alpha=0.6, label="RMSE improvement %", color="#ff7f0e")
        if len(imp_wape):
            plt.hist(imp_wape, bins=bins, alpha=0.6, label="WAPE improvement %", color="#9467bd")
        plt.axvline(0, color="k", linewidth=1)
        plt.xlabel("Amélioration (%)  (positif = noBVE meilleur)")
        plt.ylabel("Nombre de groupes")
        plt.title(f"Distribution des gains — {lvl} (par groupe)")
        plt.legend(loc="upper right")
        _save_fig(cmp_dir / f"compare_elec_total_vs_nobve_{lvl}_improvements_hist.png")

        # ----------------
        # VISUAL 3: scatter WAPE_total vs WAPE_noBVE (per group)
        # ----------------
        df_sc = df_rows.copy()
        df_sc["WAPE_total"] = pd.to_numeric(df_sc["WAPE_total"], errors="coerce")
        df_sc["WAPE_nobve"] = pd.to_numeric(df_sc["WAPE_nobve"], errors="coerce")
        df_sc["WAPE_improvement_%"] = pd.to_numeric(df_sc["WAPE_improvement_%"], errors="coerce")
        df_sc = df_sc.dropna(subset=["WAPE_total", "WAPE_nobve"])

        if len(df_sc):
            xw = df_sc["WAPE_total"].to_numpy(dtype=float)
            yw = df_sc["WAPE_nobve"].to_numpy(dtype=float)
            imp = df_sc["WAPE_improvement_%"].to_numpy(dtype=float)

            # axes limits robustes (p99) pour éviter qu’un outlier écrase tout
            lim = float(np.nanpercentile(np.concatenate([xw, yw]), 99))
            lim = max(lim, 1e-6)

            # color scale robust (p95 abs)
            v = imp[np.isfinite(imp)]
            vlim = float(np.nanpercentile(np.abs(v), 95)) if len(v) else 10.0
            vlim = max(vlim, 1e-6)

            improved = float(np.mean(yw < xw) * 100.0) if len(xw) else 0.0

            plt.figure(figsize=(7.2, 6.2))
            sc = plt.scatter(
                xw, yw,
                c=np.clip(imp, -vlim, vlim),
                cmap="coolwarm",
                s=28,
                alpha=0.75,
                edgecolors="none",
            )
            plt.plot([0, lim], [0, lim], color="k", linewidth=1, alpha=0.7)
            plt.xlim(0, lim)
            plt.ylim(0, lim)
            plt.xlabel("WAPE total (elecTotalKwh)")
            plt.ylabel("WAPE noBVE (elecTotalNoBveKwh)")
            plt.title(f"WAPE par groupe — {lvl}\n{improved:.1f}% des groupes améliorés (noBVE < total)")
            cb = plt.colorbar(sc)
            cb.set_label("Gain WAPE (%)  (positif = noBVE meilleur)")
            _save_fig(cmp_dir / f"compare_elec_total_vs_nobve_{lvl}_scatter_wape.png")

        print(f"[COMPARE] wrote comparison CSV+PNG for {lvl} under {cmp_dir}")


if __name__ == "__main__":
    main()