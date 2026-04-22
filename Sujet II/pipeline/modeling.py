from __future__ import annotations
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


def make_model(cfg: dict):
    mtype = cfg['model']['type']
    if mtype == 'ridge':
        alpha = float(cfg['model']['ridge'].get('alpha', 1.0))
        return Ridge(alpha=alpha, random_state=0)
    if mtype == 'hist_gbdt':
        p = cfg['model']['hist_gbdt']
        return HistGradientBoostingRegressor(
            max_depth=int(p.get('max_depth', 8)),
            learning_rate=float(p.get('learning_rate', 0.08)),
            max_iter=int(p.get('max_iter', 350)),
            l2_regularization=float(p.get('l2_regularization', 0.0)),
            random_state=0
        )
    raise ValueError(f'Unknown model type: {mtype}')


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    return joblib.load(path)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[m] - y_pred[m]))) if m.any() else np.nan


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2))) if m.any() else np.nan


def mape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    return float(100.0 * np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m]))) if m.any() else np.nan
