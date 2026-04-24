from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)


def parity(y_true, y_pred, title: str, out: Path):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    if len(y_true) == 0:
        return
    lim = float(np.nanpercentile(np.concatenate([y_true, y_pred]), 99))
    lim = max(lim, 1e-9)
    plt.figure(figsize=(5.5, 5.5))
    plt.hexbin(y_true, y_pred, gridsize=60, bins='log', mincnt=1)
    plt.plot([0, lim], [0, lim], color='r', linewidth=1)
    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.xlabel('Réel')
    plt.ylabel('Prédit')
    plt.title(title)
    cb = plt.colorbar(); cb.set_label('log10(count)')
    _save(out)
    plt.close()


def residual_hist(y_true, y_pred, title: str, out: Path):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    res = (y_pred[m] - y_true[m])
    if len(res) == 0:
        return
    p1, p99 = np.nanpercentile(res, [1, 99])
    plt.figure(figsize=(10, 4))
    plt.hist(res, bins=80, range=(p1, p99))
    plt.axvline(0, color='k', linewidth=1)
    plt.xlabel('Résidu (pred - true)')
    plt.ylabel('Count')
    plt.title(title)
    _save(out)
    plt.close()


def ts_actual_vs_pred(df: pd.DataFrame, date_col: str, y_col: str, yhat_col: str, title: str, out: Path):
    d = df.dropna(subset=[date_col]).sort_values(date_col)
    plt.figure(figsize=(12, 4))
    plt.plot(d[date_col], d[y_col], label='réel', linewidth=1.5)
    plt.plot(d[date_col], d[yhat_col], label='prédit', linewidth=1.2)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(y_col)
    plt.legend(loc='upper right')
    _save(out)
    plt.close()


def top_sites_rmse(errors: pd.DataFrame, title: str, out: Path, top=20):
    if errors.empty:
        return
    d = errors.sort_values('RMSE', ascending=False).head(top)
    plt.figure(figsize=(10, 5))
    plt.barh(d['siteId'].astype(str), d['RMSE'].to_numpy())
    plt.xscale('log')
    plt.xlabel('RMSE (log)')
    plt.title(title)
    _save(out)
    plt.close()
