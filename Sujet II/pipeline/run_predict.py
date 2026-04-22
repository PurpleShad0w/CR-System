from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .config import load_config
from .dataset import load_level_tables
from .io_utils import ensure_dir
from .features import add_calendar_features, select_feature_columns
from .modeling import load_model


def _infer_horizon_days(last_hist_date: pd.Timestamp, weath: pd.DataFrame, max_days: int | None, siteId: int) -> int:
    w = weath[weath['siteId'] == siteId]
    w = w[w['date'] > last_hist_date]
    if len(w) == 0:
        return 0
    max_date = w['date'].max()
    days = int((max_date - last_hist_date).days)
    if max_days is not None:
        days = min(days, int(max_days))
    return max(days, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--level', default='site', choices=['site','zone'])
    ap.add_argument('--target', required=True, choices=['elecTotalKwh','waterM3'])
    ap.add_argument('--days', type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    db_dir = Path(cfg['paths']['db_dir'])
    out_dir = ensure_dir(Path(cfg['paths']['out_dir']))

    level_cfg = cfg['level_defaults'][args.level]
    id_cols = level_cfg['id_cols']

    cleaned_path = out_dir / f'{args.level}hist_cleaned.csv'
    if not cleaned_path.exists():
        raise RuntimeError('Run cleaning first: python -m pipeline.run_clean --config config.yaml')

    hist = pd.read_csv(cleaned_path)
    hist['date'] = pd.to_datetime(hist['date'], errors='coerce').dt.floor('D')

    # load siteweath (contains both historical and future rows in live DB)
    _, _, weath = load_level_tables(db_dir, level_cfg)
    if len(weath) == 0:
        raise RuntimeError('Missing siteweath.csv (weather data)')
    weath['date'] = pd.to_datetime(weath['date'], errors='coerce').dt.floor('D')

    model_path = out_dir / 'models' / f'{args.level}_{args.target}.joblib'
    model = load_model(model_path)

    hist = hist.dropna(subset=id_cols + ['date']).sort_values(id_cols + ['date'])
    last_hist_date = hist['date'].max()

    weather_cols = cfg['features'].get('weather_cols', [])

    future_rows = []

    # if not provided, use config
    max_days = args.days
    if max_days is None:
        max_days = cfg.get('prediction', {}).get('days', None)

    for keys, g in hist.groupby(id_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        state = g[['date', args.target]].copy()
        state[args.target] = pd.to_numeric(state[args.target], errors='coerce')
        state = state.dropna().sort_values('date')
        if state.empty:
            continue

        base_feat = {c: int(v) for c, v in zip(id_cols, keys)}
        site_id = base_feat.get('siteId', None)
        if site_id is None:
            continue

        horizon = _infer_horizon_days(last_hist_date, weath, max_days, site_id)
        if horizon <= 0:
            continue

        dates = pd.date_range(last_hist_date + pd.Timedelta(days=1), last_hist_date + pd.Timedelta(days=horizon), freq='D')

        # subset weather for this site
        wsite = weath[weath['siteId'] == site_id].drop_duplicates(subset=['siteId','date'], keep='last')
        wsite = wsite.set_index('date')

        for d in dates:
            row = {**base_feat, 'date': d}

            # attach weather for that day if available
            if d in wsite.index:
                ww = wsite.loc[d]
                for c in weather_cols:
                    if c in wsite.columns:
                        row[c] = float(ww[c]) if pd.notna(ww[c]) else np.nan

            # calendar
            tmp = pd.DataFrame([row])
            if cfg['features'].get('add_calendar', True):
                tmp = add_calendar_features(tmp, 'date')
                for col in ['dow','month','dayofyear','is_weekend']:
                    row[col] = tmp.iloc[0][col]

            # lags from state
            s = state.set_index('date')[args.target]
            for k in cfg['features']['lags']:
                row[f'lag_{k}'] = float(s.get(d - pd.Timedelta(days=k), np.nan))

            for w in cfg['features']['rolling_windows']:
                window = pd.date_range(d - pd.Timedelta(days=w), d - pd.Timedelta(days=1), freq='D')
                vals = s.reindex(window)
                arr = vals.to_numpy(dtype=float)
                row[f'roll_med_{w}'] = float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan
                row[f'roll_mean_{w}'] = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan

            X = pd.DataFrame([row])
            feat_cols = select_feature_columns(X, id_cols if cfg['features'].get('add_site_id', True) else [], weather_cols)
            X = X[feat_cols]

            for c in id_cols:
                if c in X.columns:
                    X[c] = pd.to_numeric(X[c], errors='coerce')

            yhat = float(model.predict(X)[0])
            future_rows.append({**base_feat, 'date': d, 'yhat': yhat})

            state = pd.concat([state, pd.DataFrame([{'date': d, args.target: yhat}])], ignore_index=True)

    out = pd.DataFrame(future_rows)
    out.to_csv(out_dir / f'pred_{args.level}_{args.target}.csv', index=False)
    print('wrote', out_dir / f'pred_{args.level}_{args.target}.csv')


if __name__ == '__main__':
    main()
