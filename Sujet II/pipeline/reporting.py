from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm


def _save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)


def parity_linear_99(y_true, y_pred, title: str, out: Path):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if len(y_true) == 0:
        return

    lim = float(np.nanpercentile(np.concatenate([y_true, y_pred]), 99))
    lim = max(lim, 1e-9)

    bins = 80 if len(y_true) > 5000 else 60

    plt.figure(figsize=(5.5, 5.5))
    plt.hist2d(
        y_true, y_pred,
        bins=bins,
        range=[[0, lim], [0, lim]],
        norm=LogNorm()
    )
    plt.plot([0, lim], [0, lim], color="r", linewidth=1)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel("Réel")
    plt.ylabel("Prédit")
    plt.title(title)
    cb = plt.colorbar()
    cb.set_label("count (log)")
    _save(out)
    plt.close()


def parity_linear_95(y_true, y_pred, title: str, out: Path):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if len(y_true) == 0:
        return

    lim = float(np.nanpercentile(np.concatenate([y_true, y_pred]), 95))
    lim = max(lim, 1e-9)

    bins = 80 if len(y_true) > 5000 else 60

    plt.figure(figsize=(5.5, 5.5))
    plt.hist2d(
        y_true, y_pred,
        bins=bins,
        range=[[0, lim], [0, lim]],
        norm=LogNorm()
    )
    plt.plot([0, lim], [0, lim], color="r", linewidth=1)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel("Réel")
    plt.ylabel("Prédit")
    plt.title(title)
    cb = plt.colorbar()
    cb.set_label("count (log)")
    _save(out)
    plt.close()



def parity_log(y_true, y_pred, title: str, out: Path):
    """
    Parity en log: binning fait en log10(espace), puis affichage propre.
    Évite l'artefact 'bloc' causé par hexbin en linéaire + axes log après coup.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if len(y_true) == 0:
        return

    lx = np.log10(y_true)
    ly = np.log10(y_pred)

    # bornes robustes en log (p1–p99)
    lo = float(np.nanpercentile(np.concatenate([lx, ly]), 1))
    hi = float(np.nanpercentile(np.concatenate([lx, ly]), 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return

    bins = 80 if len(lx) > 5000 else 60

    plt.figure(figsize=(5.5, 5.5))
    plt.hist2d(
        lx, ly,
        bins=bins,
        range=[[lo, hi], [lo, hi]],
        norm=LogNorm()
    )

    # diagonale y=x en espace log (donc droite)
    plt.plot([lo, hi], [lo, hi], color="r", linewidth=1)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    # ticks lisibles en 10^k
    ticks = np.arange(np.floor(lo), np.ceil(hi) + 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([rf"$10^{{{int(t)}}}$" for t in ticks])
    ax.set_yticklabels([rf"$10^{{{int(t)}}}$" for t in ticks])

    plt.xlabel("Réel (log)")
    plt.ylabel("Prédit (log)")
    plt.title(title)

    cb = plt.colorbar()
    cb.set_label("count (log)")

    _save(out)
    plt.close()


def residual_hist(y_true, y_pred, title: str, out: Path):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    res = y_pred[m] - y_true[m]
    if len(res) == 0:
        return

    p1, p99 = np.nanpercentile(res, [1, 99])
    plt.figure(figsize=(10, 4))
    plt.hist(res, bins=80, range=(p1, p99))
    plt.axvline(0, color="k", linewidth=1)
    plt.xlabel("Résidu (pred - true)")
    plt.ylabel("Count")
    plt.title(title)
    _save(out)
    plt.close()


def ts_train_valid_site(
    train_df,
    valid_df,
    date_col: str,
    y_true_col: str,
    y_pred_col: str,
    cutoff,
    title: str,
    out: Path,
    site_id: int | None = None,
):
    """
    Plot: train truth only (blue) + valid truth (blue) and prediction (orange).
    Fix diagonals: reindex to a daily date range so missing days become NaN
    and matplotlib does not draw connecting lines across gaps.
    Assumes train_df has columns [date_col, y_true_col]
            valid_df has columns [date_col, y_true_col, y_pred_col]
    """
    import numpy as np

    # Defensive copy + normalize dates
    t = train_df.copy()
    v = valid_df.copy()

    t[date_col] = pd.to_datetime(t[date_col], errors="coerce").dt.floor("D")
    v[date_col] = pd.to_datetime(v[date_col], errors="coerce").dt.floor("D")

    t = t.dropna(subset=[date_col]).sort_values(date_col)
    v = v.dropna(subset=[date_col]).sort_values(date_col)

    # Nothing to plot
    if len(t) == 0 and len(v) == 0:
        return

    # Build a global daily index to avoid gaps => NaN => no diagonals
    min_date = None
    max_date = None
    if len(t):
        min_date = t[date_col].min() if min_date is None else min(min_date, t[date_col].min())
        max_date = t[date_col].max() if max_date is None else max(max_date, t[date_col].max())
    if len(v):
        min_date = v[date_col].min() if min_date is None else min(min_date, v[date_col].min())
        max_date = v[date_col].max() if max_date is None else max(max_date, v[date_col].max())

    idx = pd.date_range(min_date, max_date, freq="D")

    def _daily_series(df, value_col: str):
        if len(df) == 0:
            return pd.Series(index=idx, dtype=float)
        s = pd.to_numeric(df[value_col], errors="coerce")
        s = pd.Series(s.to_numpy(dtype=float), index=df[date_col])
        # If duplicates exist for same day, keep last (or mean if you prefer)
        s = s[~s.index.duplicated(keep="last")]
        return s.reindex(idx)

    # Build series (daily, aligned)
    t_true = _daily_series(t, y_true_col)
    v_true = _daily_series(v, y_true_col)
    v_pred = _daily_series(v, y_pred_col)

    plt.figure(figsize=(12, 4))

    # Train truth only (blue)
    if np.isfinite(t_true.to_numpy(dtype=float)).any():
        plt.plot(
            t_true.index,
            t_true.to_numpy(dtype=float),
            color="#1f77b4",
            linewidth=1.6,
            label="vérité (train)",
        )

    # Valid truth (blue) + prediction (orange)
    if np.isfinite(v_true.to_numpy(dtype=float)).any():
        plt.plot(
            v_true.index,
            v_true.to_numpy(dtype=float),
            color="#1f77b4",
            linewidth=1.6,
            label="vérité (valid)",
        )

    if np.isfinite(v_pred.to_numpy(dtype=float)).any():
        plt.plot(
            v_pred.index,
            v_pred.to_numpy(dtype=float),
            color="#ff7f0e",
            linewidth=1.3,
            label="prédiction (valid)",
        )

    plt.axvline(pd.to_datetime(cutoff), color="k", linewidth=1, alpha=0.35)

    if site_id is not None:
        plt.title(f"{title} — site {site_id}")
    else:
        plt.title(title)

    plt.xlabel("Date")
    plt.ylabel(y_true_col)
    plt.legend(loc="upper right")

    _save(out)
    plt.close()


def ts_truth_vs_algo_pred(
    truth_df,
    pred_df,
    date_col: str,
    truth_col: str,
    pred_col: str,
    title: str,
    out: Path,
):
    """
    Plot full-period truth (blue) vs algorithmic prediction (orange), no cleaning.
    Reindex daily to avoid misleading diagonals across missing days (NaN breaks lines).
    """
    # Defensive copy + normalize dates
    t = truth_df.copy()
    p = pred_df.copy()

    t[date_col] = pd.to_datetime(t[date_col], errors="coerce").dt.floor("D")
    p[date_col] = pd.to_datetime(p[date_col], errors="coerce").dt.floor("D")

    t = t.dropna(subset=[date_col]).sort_values(date_col)
    p = p.dropna(subset=[date_col]).sort_values(date_col)

    if len(t) == 0 and len(p) == 0:
        return

    min_date = None
    max_date = None
    if len(t):
        min_date = t[date_col].min() if min_date is None else min(min_date, t[date_col].min())
        max_date = t[date_col].max() if max_date is None else max(max_date, t[date_col].max())
    if len(p):
        min_date = p[date_col].min() if min_date is None else min(min_date, p[date_col].min())
        max_date = p[date_col].max() if max_date is None else max(max_date, p[date_col].max())

    idx = pd.date_range(min_date, max_date, freq="D")

    def _series(df, col):
        if len(df) == 0:
            return pd.Series(index=idx, dtype=float)
        s = pd.to_numeric(df[col], errors="coerce")
        s = pd.Series(s.to_numpy(dtype=float), index=df[date_col])
        s = s[~s.index.duplicated(keep="last")]
        return s.reindex(idx)

    s_truth = _series(t, truth_col)
    s_pred = _series(p, pred_col)

    plt.figure(figsize=(12, 4))

    if np.isfinite(s_truth.to_numpy(dtype=float)).any():
        plt.plot(s_truth.index, s_truth.to_numpy(dtype=float),
                 color="#1f77b4", linewidth=1.6, label="vérité (hist)")

    if np.isfinite(s_pred.to_numpy(dtype=float)).any():
        plt.plot(s_pred.index, s_pred.to_numpy(dtype=float),
                 color="#ff7f0e", linewidth=1.3, label="prédiction algo")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(truth_col)
    plt.legend(loc="upper right")

    _save(out)
    plt.close()
