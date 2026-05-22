from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
ENERGY_TARGETS = ELECTRIC_ALL + ["waterM3"]


def add_elec_total_no_bve(df: pd.DataFrame) -> pd.DataFrame:
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
# metrics
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


def _imp_pct(base, new):
    if not (np.isfinite(base) and np.isfinite(new)) or base == 0:
        return np.nan
    return float((base - new) / base * 100.0)


def _save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _compute_caps(x, y, p_full=99, p_zoom=95):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return 1.0, 1.0

    v = np.concatenate([x[m], y[m]])
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 1.0, 1.0

    # percentile caps
    lim_full = float(np.nanpercentile(v, p_full))
    lim_zoom = float(np.nanpercentile(v, p_zoom))

    # IQR cap (anti-outliers quand N petit)
    q1 = float(np.nanpercentile(v, 25))
    q3 = float(np.nanpercentile(v, 75))
    iqr = max(q3 - q1, 1e-12)
    lim_iqr = q3 + 10.0 * iqr

    lim_full = min(lim_full, lim_iqr)
    lim_zoom = min(lim_zoom, lim_iqr)

    lim_full = max(lim_full, 1e-6)
    lim_zoom = max(lim_zoom, 1e-6)
    lim_zoom = min(lim_zoom, lim_full)
    return lim_full, lim_zoom


def _scatter_gain(x, y, gain_pct, lim, title, xlabel, ylabel, cblabel, out_path: Path):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(gain_pct, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = np.clip(x[m], 0, lim)
    y = np.clip(y[m], 0, lim)
    g = g[m]

    plt.figure(figsize=(7.2, 6.2))
    if len(x) == 0:
        plt.axis("off")
        plt.title(title + "\n(aucun point)")
        _save_fig(out_path)
        return

    vg = g[np.isfinite(g)]
    vlim = float(np.nanpercentile(np.abs(vg), 95)) if len(vg) else 10.0
    vlim = max(vlim, 1e-6)

    nan_gain = ~np.isfinite(g)
    if np.any(nan_gain):
        plt.scatter(x[nan_gain], y[nan_gain], s=28, alpha=0.55, c="#9e9e9e", edgecolors="none", label="gain NaN")

    fin = np.isfinite(g)
    if np.any(fin):
        sc = plt.scatter(
            x[fin], y[fin],
            c=np.clip(g[fin], -vlim, vlim),
            cmap="coolwarm",
            s=28,
            alpha=0.80,
            edgecolors="none",
        )
        cb = plt.colorbar(sc)
        cb.set_label(cblabel)

    plt.plot([0, lim], [0, lim], color="k", linewidth=1, alpha=0.7)
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (n={len(x)})")
    if np.any(nan_gain):
        plt.legend(loc="upper right")
    _save_fig(out_path)


def _hexbin_safe(x, y, lim, title, xlabel, ylabel, out_path: Path):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = np.clip(x[m], 0, lim)
    y = np.clip(y[m], 0, lim)

    plt.figure(figsize=(7.2, 6.2))
    if len(x) == 0:
        plt.axis("off")
        plt.title(title + "\n(aucun point)")
        _save_fig(out_path)
        return

    hb = plt.hexbin(
        x, y,
        gridsize=55,
        extent=(0, lim, 0, lim),
        mincnt=1,
        cmap="viridis",
    )
    counts = hb.get_array()
    vmax = float(np.nanmax(counts)) if counts is not None and len(counts) else 0.0
    if np.isfinite(vmax) and vmax >= 2:
        hb.set_norm(LogNorm(vmin=1, vmax=vmax))
        cb = plt.colorbar(hb)
        cb.set_label("densité (log count)")
    else:
        cb = plt.colorbar(hb)
        cb.set_label("densité (count)")

    plt.plot([0, lim], [0, lim], color="k", linewidth=1, alpha=0.7)
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + f"\n(n={len(x)})")
    _save_fig(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site")  # site | zone | all
    ap.add_argument("--target", required=True)
    ap.add_argument("--site", default="170", help='siteId pour timeseries, ou \"all\"')
    args = ap.parse_args()

    LEVELS = ["site", "zone"]

    # all inclut NoBVE
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
            # pour que le reporting "all" regénère aussi les usages + comparaisons
            return BASE_TARGETS_BY_LEVEL[level][:] + ELECTRIC_USES
        return [target]

    def report_one(level: str, target: str, site: str):
        cleaned_path = out_dir / f"{level}hist_cleaned.csv"
        hist = pd.read_csv(cleaned_path)
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")

        info_path = Path(args.config).resolve().parent / cfg.get("paths", {}).get("site_infos_file", "Sites_Shyrka_Infos.xlsx")
        site_infos = load_site_infos(info_path)
        if len(site_infos) and "siteId" in hist.columns:
            hist = hist.merge(site_infos, on="siteId", how="left")

        hist = add_elec_total_no_bve(hist)

        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]

        model_dir = out_dir / "models"
        meta_path = model_dir / f"{level}_{target}.meta.json"
        model_path = model_dir / f"{level}_{target}.joblib"
        if not meta_path.exists() or not model_path.exists():
            print(f"[WARN] Missing meta/model for {level}_{target}. Skipping.")
            return None

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feat_cols = meta["feature_columns"]
        valid_days = int(meta.get("valid_days", cfg["training"].get("valid_days", 60)))
        log1p_target = bool(meta.get("log1p_target", False))

        if target not in hist.columns:
            print(f"[WARN] Target {target} not in {level}hist_cleaned.csv. Skipping.")
            return None

        static_cols = cfg.get("features", {}).get("static_cols", [])
        extra_cols = [c for c in static_cols if c in hist.columns]

        df0 = hist[id_cols + ["date", target] + extra_cols].copy()
        df0[target] = pd.to_numeric(df0[target], errors="coerce")
        df0 = df0.dropna(subset=id_cols + ["date"] + extra_cols)
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

        cutoff = df0["date"].max() - pd.Timedelta(days=valid_days)

        model = load_model(model_path)

        # eval truth window
        df_eval = df0.dropna(subset=[target]).copy()
        valid_eval = df_eval[df_eval["date"] > cutoff].copy()
        if len(valid_eval) == 0:
            return {"level": level, "target": target, "id_cols": id_cols, "eval_df": None, "win_pred": None}

        X_eval = valid_eval[feat_cols].copy()
        y_eval = pd.to_numeric(valid_eval[target], errors="coerce").to_numpy(dtype=float)
        pe = model.predict(X_eval)
        if log1p_target:
            yhat_eval = np.expm1(pe)
            yhat_eval = np.maximum(yhat_eval, 0.0)
        else:
            yhat_eval = pe

        # core reporting (parity/residual/ts)
        parity_linear_99(y_eval, yhat_eval, f"Parity — {level} {target} (valid)", fig_dir / f"parity_{level}_{target}_p99.png")
        parity_linear_95(y_eval, yhat_eval, f"Parity — {level} {target} (valid)", fig_dir / f"parity_{level}_{target}_p95.png")
        parity_log(y_eval, yhat_eval, f"Parity log — {level} {target} (valid)", fig_dir / f"parity_{level}_{target}_log.png")
        residual_hist(y_eval, yhat_eval, f"Residuals — {level} {target} (valid)", fig_dir / f"resid_{level}_{target}.png")

        eval_df = valid_eval[id_cols + ["date"]].copy()
        eval_df["y_true"] = y_eval
        eval_df["y_pred"] = yhat_eval

        # prediction window (for comparisons): same keys as total valid window
        win = df0[df0["date"] > cutoff].copy()
        if len(win):
            Xw = win[feat_cols].copy()
            pw = model.predict(Xw)
            if log1p_target:
                yhat_w = np.expm1(pw)
                yhat_w = np.maximum(yhat_w, 0.0)
            else:
                yhat_w = pw
            win_pred = win[id_cols + ["date"]].copy()
            win_pred["y_pred"] = yhat_w
        else:
            win_pred = win[id_cols + ["date"]].copy()
            win_pred["y_pred"] = np.nan

        # TS (simple : only site level, zone level graphs handled elsewhere if needed)
        site_arg = str(site).lower()
        if site_arg == "all":
            site_ids = sorted([int(x) for x in df0["siteId"].dropna().unique().tolist()]) if "siteId" in df0.columns else []
        else:
            site_ids = [int(site)]

        zone_fig_dir = ensure_dir(fig_dir / "zones")
        for sid in site_ids:
            if level == "zone" and "zoneId" in df0.columns:
                zone_ids = sorted([int(z) for z in df0.loc[df0["siteId"] == sid, "zoneId"].dropna().unique().tolist()])
                for zid in zone_ids:
                    dzone = df0[(df0["siteId"] == sid) & (df0["zoneId"] == zid)].copy().sort_values("date")
                    train_zone = dzone[dzone["date"] <= cutoff][["date", target]].copy()
                    valid_zone = dzone[dzone["date"] > cutoff][["date", "siteId", "zoneId", target]].copy()
                    wp = win_pred[(win_pred["siteId"] == sid) & (win_pred["zoneId"] == zid)][["date", "siteId", "zoneId", "y_pred"]].copy()
                    valid_zone = valid_zone.merge(wp, on=["date", "siteId", "zoneId"], how="left").rename(columns={"y_pred": "yhat"})
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
                dsite = df0[df0["siteId"] == sid].copy().sort_values("date")
                train_site = dsite[dsite["date"] <= cutoff][["date", target]].copy()
                valid_site = dsite[dsite["date"] > cutoff][["date", "siteId", target]].copy()
                wp = win_pred[win_pred["siteId"] == sid][["date", "siteId", "y_pred"]].copy()
                valid_site = valid_site.merge(wp, on=["date", "siteId"], how="left").rename(columns={"y_pred": "yhat"})
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

        return {"level": level, "target": target, "id_cols": id_cols, "eval_df": eval_df, "win_pred": win_pred}

    # run
    reports = []
    levels = LEVELS if args.level == "all" else [args.level]
    for lvl in levels:
        targets = expand_targets(lvl, args.target)
        for tgt in targets:
            if tgt == "elecAggregatedKwh":
                raise ValueError("elecAggregatedKwh est exclu.")
            if tgt not in TARGETS_BY_LEVEL[lvl]:
                raise ValueError(f"Target {tgt} not supported for level {lvl}.")
            r = report_one(lvl, tgt, args.site)
            if r is not None:
                reports.append(r)

    # compare direct vs sum uses (on la fenêtre du total)
    cmp_dir = ensure_dir(fig_dir / "compare")
    by_lt = {(r["level"], r["target"]): r for r in reports}

    TOTAL = "elecTotalKwh"
    USES = ELECTRIC_USES[:]

    for lvl in ["site", "zone"]:
        if (lvl, TOTAL) not in by_lt:
            continue
        if any((lvl, u) not in by_lt for u in USES):
            continue

        rtot = by_lt[(lvl, TOTAL)]
        id_cols = rtot["id_cols"]
        eval_total = rtot["eval_df"]
        if eval_total is None or len(eval_total) == 0:
            continue

        base = eval_total[id_cols + ["date"]].copy()
        base["y_true_total"] = pd.to_numeric(eval_total["y_true"], errors="coerce")
        base["y_pred_total_direct"] = pd.to_numeric(eval_total["y_pred"], errors="coerce")

        for u in USES:
            wp = by_lt[(lvl, u)]["win_pred"].copy().rename(columns={"y_pred": f"y_pred_{u}"})
            base = base.merge(wp, on=id_cols + ["date"], how="left")

        base["y_pred_total_from_uses"] = (
            pd.to_numeric(base["y_pred_elecBveKwh"], errors="coerce").fillna(0.0)
            + pd.to_numeric(base["y_pred_elecCvcKwh"], errors="coerce").fillna(0.0)
            + pd.to_numeric(base["y_pred_elecForceKwh"], errors="coerce").fillna(0.0)
            + pd.to_numeric(base["y_pred_elecLightingKwh"], errors="coerce").fillna(0.0)
        )
        base["y_pred_total_from_uses"] = np.maximum(pd.to_numeric(base["y_pred_total_from_uses"], errors="coerce"), 0.0)

        # per-group metrics
        rows = []
        for keys, g in base.groupby(id_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            yt_g = pd.to_numeric(g["y_true_total"], errors="coerce").to_numpy(dtype=float)
            yd_g = pd.to_numeric(g["y_pred_total_direct"], errors="coerce").to_numpy(dtype=float)
            ys_g = pd.to_numeric(g["y_pred_total_from_uses"], errors="coerce").to_numpy(dtype=float)
            row = {c: v for c, v in zip(id_cols, keys)}
            row.update({
                "n": int(len(g)),
                "RMSE_direct": _rmse(yt_g, yd_g),
                "RMSE_sumUses": _rmse(yt_g, ys_g),
                "WAPE_direct": _wape(yt_g, yd_g),
                "WAPE_sumUses": _wape(yt_g, ys_g),
            })
            row["RMSE_improvement_%"] = _imp_pct(row["RMSE_direct"], row["RMSE_sumUses"])
            row["WAPE_improvement_%"] = _imp_pct(row["WAPE_direct"], row["WAPE_sumUses"])
            rows.append(row)

        df_rows = pd.DataFrame(rows)
        df_rows.to_csv(cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_by_group.csv", index=False)

        # outliers dump (diagnostic)
        for col in ["RMSE_direct", "RMSE_sumUses", "WAPE_direct", "WAPE_sumUses"]:
            tmp = df_rows.copy()
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            tmp = tmp.sort_values(col, ascending=False).head(50)
            tmp.to_csv(cmp_dir / f"outliers_{lvl}_{col}.csv", index=False)

        # ---- plots with caps (full + zoom)
        # WAPE
        xw = pd.to_numeric(df_rows["WAPE_direct"], errors="coerce").to_numpy(dtype=float)
        yw = pd.to_numeric(df_rows["WAPE_sumUses"], errors="coerce").to_numpy(dtype=float)
        gw = pd.to_numeric(df_rows["WAPE_improvement_%"], errors="coerce").to_numpy(dtype=float)
        lim_full_w, lim_zoom_w = _compute_caps(xw, yw)

        _scatter_gain(
            xw, yw, gw, lim_full_w,
            f"WAPE — {lvl} — direct vs somme usages (FULL cap)",
            "WAPE direct (modèle total)",
            "WAPE somme usages (4 modèles)",
            "Gain WAPE (%) (positif = somme usages meilleur)",
            cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_wape_scatter_full.png",
        )
        _hexbin_safe(
            xw, yw, lim_full_w,
            f"WAPE — {lvl} — densité (FULL cap)",
            "WAPE direct (modèle total)",
            "WAPE somme usages (4 modèles)",
            cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_wape_hexbin_full.png",
        )

        _scatter_gain(
            xw, yw, gw, lim_zoom_w,
            f"WAPE — {lvl} — direct vs somme usages (ZOOM cap)",
            "WAPE direct (modèle total)",
            "WAPE somme usages (4 modèles)",
            "Gain WAPE (%) (positif = somme usages meilleur)",
            cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_wape_scatter_zoom.png",
        )
        _hexbin_safe(
            xw, yw, lim_zoom_w,
            f"WAPE — {lvl} — densité (ZOOM cap)",
            "WAPE direct (modèle total)",
            "WAPE somme usages (4 modèles)",
            cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_wape_hexbin_zoom.png",
        )

        # RMSE
        xr = pd.to_numeric(df_rows["RMSE_direct"], errors="coerce").to_numpy(dtype=float)
        yr = pd.to_numeric(df_rows["RMSE_sumUses"], errors="coerce").to_numpy(dtype=float)
        gr = pd.to_numeric(df_rows["RMSE_improvement_%"], errors="coerce").to_numpy(dtype=float)
        lim_full_r, lim_zoom_r = _compute_caps(xr, yr)

        _scatter_gain(
            xr, yr, gr, lim_full_r,
            f"RMSE — {lvl} — direct vs somme usages (FULL cap)",
            "RMSE direct (modèle total)",
            "RMSE somme usages (4 modèles)",
            "Gain RMSE (%) (positif = somme usages meilleur)",
            cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_rmse_scatter_full.png",
        )
        _hexbin_safe(
            xr, yr, lim_full_r,
            f"RMSE — {lvl} — densité (FULL cap)",
            "RMSE direct (modèle total)",
            "RMSE somme usages (4 modèles)",
            cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_rmse_hexbin_full.png",
        )

        _scatter_gain(
            xr, yr, gr, lim_zoom_r,
            f"RMSE — {lvl} — direct vs somme usages (ZOOM cap)",
            "RMSE direct (modèle total)",
            "RMSE somme usages (4 modèles)",
            "Gain RMSE (%) (positif = somme usages meilleur)",
            cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_rmse_scatter_zoom.png",
        )
        _hexbin_safe(
            xr, yr, lim_zoom_r,
            f"RMSE — {lvl} — densité (ZOOM cap)",
            "RMSE direct (modèle total)",
            "RMSE somme usages (4 modèles)",
            cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_rmse_hexbin_zoom.png",
        )

        print(f"[COMPARE] wrote capped FULL+ZOOM graphs for {lvl} under {cmp_dir}")


if __name__ == "__main__":
    main()