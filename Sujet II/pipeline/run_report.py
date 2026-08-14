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
from .reporting import parity_linear_95, parity_linear_99, parity_log, residual_hist
from .targets_utils import discover_elec_usage_targets, add_elec_total_accurate


ELECTRIC_USES = ["elecBveKwh", "elecCvcKwh", "elecForceKwh", "elecLightingKwh"]
ELEC_TOTAL_ACCURATE = "elecTotalAccurateKwh"
ELEC_TOTAL_NOBVE = "elecTotalNoBveKwh"
BASE_TARGETS = ["elecTotalKwh", ELEC_TOTAL_ACCURATE, ELEC_TOTAL_NOBVE, "waterM3", "indoorTempDegC"]


def add_elec_total_no_bve(df: pd.DataFrame) -> pd.DataFrame:
    if ELEC_TOTAL_NOBVE in df.columns:
        return df
    if ELEC_TOTAL_ACCURATE not in df.columns or "elecBveKwh" not in df.columns:
        return df
    out = df.copy()
    total = pd.to_numeric(out[ELEC_TOTAL_ACCURATE], errors="coerce")
    bve = pd.to_numeric(out["elecBveKwh"], errors="coerce").fillna(0.0)
    out[ELEC_TOTAL_NOBVE] = np.maximum(total - bve, 0.0)
    return out


def _read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        na_values=["NULL", "null", ""],
    )


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
    lim_full = float(np.nanpercentile(v, p_full))
    lim_zoom = float(np.nanpercentile(v, p_zoom))
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
            x[fin],
            y[fin],
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
    hb = plt.hexbin(x, y, gridsize=55, extent=(0, lim, 0, lim), mincnt=1, cmap="viridis")
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


def _plot_train_valid_ts(train_df: pd.DataFrame, valid_df: pd.DataFrame, target: str, title: str, out: Path):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    if "date" in train_df.columns:
        train_df["date"] = pd.to_datetime(train_df["date"], errors="coerce")
    if "date" in valid_df.columns:
        valid_df["date"] = pd.to_datetime(valid_df["date"], errors="coerce")
    if target in train_df.columns:
        train_df[target] = pd.to_numeric(train_df[target], errors="coerce")
    if "y_true" in valid_df.columns:
        valid_df["y_true"] = pd.to_numeric(valid_df["y_true"], errors="coerce")
    if "y_pred" in valid_df.columns:
        valid_df["y_pred"] = pd.to_numeric(valid_df["y_pred"], errors="coerce")
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    valid_df = valid_df.replace([np.inf, -np.inf], np.nan)
    has_train = len(train_df.dropna(subset=["date", target])) > 0 if target in train_df.columns else False
    has_valid = len(valid_df.dropna(subset=["date", "y_true", "y_pred"])) > 0
    if not has_train and not has_valid:
        return
    plt.figure(figsize=(13, 4.5))
    if has_train:
        t = train_df.dropna(subset=["date", target]).sort_values("date")
        plt.plot(t["date"], t[target], label="train truth", linewidth=1.2, alpha=0.75)
    if has_valid:
        v = valid_df.dropna(subset=["date", "y_true", "y_pred"]).sort_values("date")
        plt.plot(v["date"], v["y_true"], label="valid truth", linewidth=1.5)
        plt.plot(v["date"], v["y_pred"], label="valid prediction", linewidth=1.5)
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel(target)
    plt.legend()
    _save_fig(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site")  # site | zone | all
    ap.add_argument("--target", required=True)
    ap.add_argument("--site", default="all", help='siteId pour timeseries, ou "all"')
    args = ap.parse_args()

    LEVELS = ["site", "zone"]
    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))
    fig_dir = ensure_dir(out_dir / "figures")

    def _load_hist(level: str) -> pd.DataFrame:
        cleaned_path = out_dir / f"{level}hist_cleaned.csv"
        if cleaned_path.exists():
            hist = _read_csv_safe(cleaned_path)
        else:
            hist, _, _ = load_level_tables(db_dir, cfg["level_defaults"][level])
        if "date" in hist.columns:
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")
        elif "dtUpdate" in hist.columns:
            hist["date"] = pd.to_datetime(hist["dtUpdate"], errors="coerce").dt.floor("D")
        hist = add_elec_total_accurate(hist)
        hist = add_elec_total_no_bve(hist)
        return hist

    def _targets_from_eval_preds(level: str) -> list[str]:
        prefix = f"eval_preds_{level}_"
        targets = []
        for p in out_dir.glob(f"{prefix}*.csv"):
            target = p.stem.replace(prefix, "")
            if target:
                targets.append(target)
        return sorted(set(targets))

    def expand_targets(level: str, target: str) -> list[str]:
        if target == "all":
            eval_targets = _targets_from_eval_preds(level)
            if eval_targets:
                return eval_targets
            hist = _load_hist(level)
            dyn_usages = discover_elec_usage_targets(hist)
            return BASE_TARGETS + [c for c in dyn_usages if c not in BASE_TARGETS]
        if target == "elecUses":
            hist = _load_hist(level)
            dyn_usages = discover_elec_usage_targets(hist)
            return dyn_usages if dyn_usages else ELECTRIC_USES[:]
        return [target]

    def report_from_eval_preds(level: str, target: str):
        level_cfg = cfg["level_defaults"][level]
        id_cols = level_cfg["id_cols"]
        pred_path = out_dir / f"eval_preds_{level}_{target}.csv"
        if not pred_path.exists():
            print(f"[REPORT][SKIP] missing {pred_path.name}")
            return None
        df = _read_csv_safe(pred_path)
        required = set(id_cols + ["date", "y_true", "y_pred"])
        missing = required - set(df.columns)
        if missing:
            print(f"[WARN] {pred_path.name}: colonnes manquantes {sorted(missing)}. Skip.")
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=id_cols + ["date", "y_true", "y_pred"])
        if df.empty:
            print(f"[WARN] {pred_path.name}: aucune ligne exploitable après nettoyage. Skip.")
            return None

        y = df["y_true"].to_numpy(dtype=float)
        yhat = df["y_pred"].to_numpy(dtype=float)
        target_fig_dir = ensure_dir(fig_dir / level / target)

        parity_linear_99(y, yhat, f"Parity — {level} {target} (valid)", target_fig_dir / f"parity_{level}_{target}_p99.png")
        parity_linear_95(y, yhat, f"Parity — {level} {target} (valid)", target_fig_dir / f"parity_{level}_{target}_p95.png")
        parity_log(y, yhat, f"Parity log — {level} {target} (valid)", target_fig_dir / f"parity_{level}_{target}_log.png")
        residual_hist(y, yhat, f"Residuals — {level} {target} (valid)", target_fig_dir / f"resid_{level}_{target}.png")

        rep = pd.DataFrame([{
            "rows": int(len(df)),
            "MAE": _mae(y, yhat),
            "RMSE": _rmse(y, yhat),
            "WAPE": _wape(y, yhat),
            "Bias": float(np.nanmean(yhat - y)) if len(y) else np.nan,
        }])
        rep.to_csv(out_dir / f"report_metrics_{level}_{target}.csv", index=False)

        try:
            hist = _load_hist(level)
        except Exception as e:
            print(f"[WARN] Could not load hist for TS: level={level} target={target}: {type(e).__name__}: {e}")
            hist = pd.DataFrame()

        if len(hist) and target in hist.columns:
            hist = hist.copy()
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")
            hist[target] = pd.to_numeric(hist[target], errors="coerce")
            hist = hist.replace([np.inf, -np.inf], np.nan)
            hist = hist.dropna(subset=id_cols + ["date"])
            eval_start = df["date"].min()
            ts_dir = ensure_dir(target_fig_dir / "timeseries")
            site_arg = str(args.site).lower()

            for keys, g_eval in df.groupby(id_cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                key_map = {c: v for c, v in zip(id_cols, keys)}
                if "siteId" in key_map and site_arg != "all":
                    try:
                        if int(key_map["siteId"]) != int(args.site):
                            continue
                    except Exception:
                        continue
                mask = np.ones(len(hist), dtype=bool)
                for c, v in key_map.items():
                    mask &= hist[c].eq(v)
                g_hist = hist.loc[mask, id_cols + ["date", target]].copy()
                train_hist = g_hist[g_hist["date"] < eval_start].copy()
                valid_eval = g_eval[id_cols + ["date", "y_true", "y_pred"]].copy()
                if level == "zone" and "zoneId" in key_map:
                    out_name = f"ts_site{int(key_map['siteId'])}_zone{int(key_map['zoneId'])}_{target}_train_valid.png"
                    title = f"{level} {target} — site {int(key_map['siteId'])} zone {int(key_map['zoneId'])}"
                elif "siteId" in key_map:
                    out_name = f"ts_site{int(key_map['siteId'])}_{target}_train_valid.png"
                    title = f"{level} {target} — site {int(key_map['siteId'])}"
                else:
                    safe_key = "_".join([str(v) for v in keys])
                    out_name = f"ts_{safe_key}_{target}_train_valid.png"
                    title = f"{level} {target} — {safe_key}"
                _plot_train_valid_ts(train_hist, valid_eval, target, title, ts_dir / out_name)
        else:
            print(f"[REPORT][TS-SKIP] {level}_{target}: target absent de l'historique nettoyé")

        print(f"[REPORT] generated from eval_preds: level={level} target={target} rows={len(df)}")
        return {
            "level": level,
            "target": target,
            "id_cols": id_cols,
            "eval_df": df.copy(),
            "win_pred": df[id_cols + ["date", "y_pred"]].copy(),
        }

    def report_one(level: str, target: str, site: str):
        return report_from_eval_preds(level, target)

    reports = []
    levels = LEVELS if args.level == "all" else [args.level]
    for lvl in levels:
        if lvl not in LEVELS:
            raise ValueError("Unknown level. Use site|zone|all")
        targets = expand_targets(lvl, args.target)
        print(f"[REPORT] level={lvl}: {len(targets)} target(s) detected")
        for tgt in targets:
            if tgt == "elecAggregatedKwh":
                print(f"[REPORT][SKIP] {lvl}_{tgt}: excluded target")
                continue
            try:
                r = report_one(lvl, tgt, args.site)
                if r is not None:
                    reports.append(r)
                else:
                    print(f"[REPORT][SKIP] {lvl}_{tgt}: no report generated")
            except Exception as e:
                print(f"[REPORT][ERROR] {lvl}_{tgt}: {type(e).__name__}: {e}")

    print(f"[REPORT] total generated reports: {len(reports)}")

    cmp_dir = ensure_dir(fig_dir / "compare")
    by_lt = {(r["level"], r["target"]): r for r in reports}
    total_candidates = [ELEC_TOTAL_ACCURATE, "elecTotalKwh"]

    for lvl in ["site", "zone"]:
        total = next((t for t in total_candidates if (lvl, t) in by_lt), None)
        if total is None:
            continue
        if any((lvl, u) not in by_lt for u in ELECTRIC_USES):
            continue
        rtot = by_lt[(lvl, total)]
        id_cols = rtot["id_cols"]
        eval_total = rtot["eval_df"]
        if eval_total is None or len(eval_total) == 0:
            continue
        base = eval_total[id_cols + ["date"]].copy()
        base["y_true_total"] = pd.to_numeric(eval_total["y_true"], errors="coerce")
        base["y_pred_total_direct"] = pd.to_numeric(eval_total["y_pred"], errors="coerce")
        for u in ELECTRIC_USES:
            wp = by_lt[(lvl, u)]["win_pred"].copy().rename(columns={"y_pred": f"y_pred_{u}"})
            base = base.merge(wp, on=id_cols + ["date"], how="left")
        base["y_pred_total_from_uses"] = sum(
            pd.to_numeric(base[f"y_pred_{u}"], errors="coerce").fillna(0.0)
            for u in ELECTRIC_USES
        )
        base["y_pred_total_from_uses"] = np.maximum(pd.to_numeric(base["y_pred_total_from_uses"], errors="coerce"), 0.0)

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
        for col in ["RMSE_direct", "RMSE_sumUses", "WAPE_direct", "WAPE_sumUses"]:
            tmp = df_rows.copy()
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            tmp = tmp.sort_values(col, ascending=False).head(50)
            tmp.to_csv(cmp_dir / f"outliers_{lvl}_{col}.csv", index=False)

        xw = pd.to_numeric(df_rows["WAPE_direct"], errors="coerce").to_numpy(dtype=float)
        yw = pd.to_numeric(df_rows["WAPE_sumUses"], errors="coerce").to_numpy(dtype=float)
        gw = pd.to_numeric(df_rows["WAPE_improvement_%"], errors="coerce").to_numpy(dtype=float)
        lim_full_w, lim_zoom_w = _compute_caps(xw, yw)
        _scatter_gain(xw, yw, gw, lim_full_w, f"WAPE — {lvl} — direct vs somme usages (FULL cap)", "WAPE direct", "WAPE somme usages", "Gain WAPE (%)", cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_wape_scatter_full.png")
        _hexbin_safe(xw, yw, lim_full_w, f"WAPE — {lvl} — densité (FULL cap)", "WAPE direct", "WAPE somme usages", cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_wape_hexbin_full.png")
        _scatter_gain(xw, yw, gw, lim_zoom_w, f"WAPE — {lvl} — direct vs somme usages (ZOOM cap)", "WAPE direct", "WAPE somme usages", "Gain WAPE (%)", cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_wape_scatter_zoom.png")
        _hexbin_safe(xw, yw, lim_zoom_w, f"WAPE — {lvl} — densité (ZOOM cap)", "WAPE direct", "WAPE somme usages", cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_wape_hexbin_zoom.png")

        xr = pd.to_numeric(df_rows["RMSE_direct"], errors="coerce").to_numpy(dtype=float)
        yr = pd.to_numeric(df_rows["RMSE_sumUses"], errors="coerce").to_numpy(dtype=float)
        gr = pd.to_numeric(df_rows["RMSE_improvement_%"], errors="coerce").to_numpy(dtype=float)
        lim_full_r, lim_zoom_r = _compute_caps(xr, yr)
        _scatter_gain(xr, yr, gr, lim_full_r, f"RMSE — {lvl} — direct vs somme usages (FULL cap)", "RMSE direct", "RMSE somme usages", "Gain RMSE (%)", cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_rmse_scatter_full.png")
        _hexbin_safe(xr, yr, lim_full_r, f"RMSE — {lvl} — densité (FULL cap)", "RMSE direct", "RMSE somme usages", cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_rmse_hexbin_full.png")
        _scatter_gain(xr, yr, gr, lim_zoom_r, f"RMSE — {lvl} — direct vs somme usages (ZOOM cap)", "RMSE direct", "RMSE somme usages", "Gain RMSE (%)", cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_rmse_scatter_zoom.png")
        _hexbin_safe(xr, yr, lim_zoom_r, f"RMSE — {lvl} — densité (ZOOM cap)", "RMSE direct", "RMSE somme usages", cmp_dir / f"compare_total_direct_vs_sumuses_{lvl}_rmse_hexbin_zoom.png")
        print(f"[COMPARE] wrote graphs for {lvl} under {cmp_dir}")


if __name__ == "__main__":
    main()
