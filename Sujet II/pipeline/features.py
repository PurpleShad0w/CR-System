from __future__ import annotations
import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col], errors='coerce')
    df['dow'] = dt.dt.dayofweek.astype('Int64')
    df['month'] = dt.dt.month.astype('Int64')
    df['dayofyear'] = dt.dt.dayofyear.astype('Int64')
    df['is_weekend'] = (df['dow'] >= 5).astype('Int64')
    return df


def build_lag_features(df: pd.DataFrame, id_cols, date_col: str, target_col: str, lags: list[int]) -> pd.DataFrame:
    df = df.copy().sort_values(id_cols + [date_col])
    for k in lags:
        df[f'lag_{k}'] = df.groupby(id_cols)[target_col].shift(k)
    return df


def build_rolling_features(df: pd.DataFrame, id_cols, date_col: str, target_col: str, windows: list[int]) -> pd.DataFrame:
    df = df.copy().sort_values(id_cols + [date_col])

    g = df.groupby(id_cols, dropna=False)[target_col]

    for w in windows:
        minp = max(2, w // 3)

        df[f'roll_med_{w}'] = (
            g.apply(lambda x: x.shift(1).rolling(w, min_periods=minp).median())
             .reset_index(level=id_cols, drop=True)
        )

        df[f'roll_mean_{w}'] = (
            g.apply(lambda x: x.shift(1).rolling(w, min_periods=minp).mean())
             .reset_index(level=id_cols, drop=True)
        )

    return df


def select_feature_columns(df: pd.DataFrame, id_cols, weather_cols: list[str]) -> list[str]:
    cols = []
    for c in ['dow', 'month', 'dayofyear', 'is_weekend']:
        if c in df.columns:
            cols.append(c)
    for c in df.columns:
        if c.startswith('lag_') or c.startswith('roll_'):
            cols.append(c)
    for c in weather_cols:
        if c in df.columns:
            cols.append(c)
    for c in id_cols:
        if c in df.columns:
            cols.append(c)
    return cols
