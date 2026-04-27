from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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

    plt.figure(figsize=(5.5, 5.5))
    plt.hexbin(y_true, y_pred, gridsize=60, bins="log", mincnt=1)
    plt.plot([0, lim], [0, lim], color="r", linewidth=1)
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel("Réel")
    plt.ylabel("Prédit")
    plt.title(title)
    cb = plt.colorbar()
    cb.set_label("log10(count)")
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

    plt.figure(figsize=(5.5, 5.5))
    plt.hexbin(y_true, y_pred, gridsize=60, bins="log", mincnt=1)
    plt.plot([0, lim], [0, lim], color="r", linewidth=1)
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel("Réel")
    plt.ylabel("Prédit")
    plt.title(title)
    cb = plt.colorbar()
    cb.set_label("log10(count)")
    _save(out)
    plt.close()


def parity_log(y_true, y_pred, title: str, out: Path):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if len(y_true) == 0:
        return

    lo = max(float(np.nanpercentile(np.concatenate([y_true, y_pred]), 1)), 1e-6)
    hi = float(np.nanpercentile(np.concatenate([y_true, y_pred]), 99))

    plt.figure(figsize=(5.5, 5.5))
    plt.hexbin(y_true, y_pred, gridsize=60, bins="log", mincnt=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot([lo, hi], [lo, hi], color="r", linewidth=1)
    plt.xlabel("Réel (log)")
    plt.ylabel("Prédit (log)")
    plt.title(title)
    cb = plt.colorbar()
    cb.set_label("log10(count)")
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
