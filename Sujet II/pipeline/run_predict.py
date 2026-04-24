from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import load_config
from .dataset import load_level_tables
from .io_utils import ensure_dir
from .features import add_calendar_features
from .modeling import load_model
from .encoding import encode_id_columns


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
    ap.add_argument('--level', default='site', choices=['site', 'zone'])
    ap.add_argument('--target', required=True, choices=['elecTotalKwh', 'waterM3'])
    ap.add_argument('--days', type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    db_dir = Path(cfg['paths']['db_dir'])
    out_dir = ensure_dir(Path(cfg['paths']['out_dir']))

    level_cfg = cfg['level_defaults'][args.level]
    id_cols = level_cfg['id_cols']

    cleaned_path = out_dir / f'{args.level}hist_cleaned.csv'
    if not cleaned_path.exists():
        raise RuntimeError('Run cleaning first')

    hist = pd.read_csv(cleaned_path)
    hist['date'] = pd.to_datetime(hist['date'], errors='coerce').dt.floor('D')

    _, _, weath = load_level_tables(db_dir, level_cfg)
    if len(weath) == 0:
        raise RuntimeError('Missing siteweath.csv')
    weath['date'] = pd.to_datetime(weath['date'], errors='coerce').dt.floor('D')

    model_dir = out_dir / 'models'
    meta = json.loads((model_dir / f'{args.level}_{args.target}.meta.json').read_text(encoding='utf-8'))
    feat_cols = meta['feature_columns']
    cats = {c: pd.Index(meta['id_categories'][c]) for c in meta.get('id_categories', {})}

    model = load_model(model_dir / f'{args.level}_{args.target}.joblib')

    hist = hist.dropna(subset=id_cols + ['date']).sort_values(id_cols + ['date'])
    last_hist_date = hist['date'].max()

    weather_cols = cfg['features'].get('weather_cols', [])

    max_days = args.days
    if max_days is None:
        max_days = cfg.get('prediction', {}).get('days', None)

    future_rows = []

    for keys, g in hist.groupby(id_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base_feat = {c: int(v) for c, v in zip(id_cols, keys)}
        site_id = base_feat.get('siteId', None)
        if site_id is None:
            continue

        horizon = _infer_horizon_days(last_hist_date, weath, max_days, site_id)
        if horizon <= 0:
            continue

        state = g[['date', args.target]].copy()
        state[args.target] = pd.to_numeric(state[args.target], errors='coerce')
        state = state.dropna().sort_values('date')
        if state.empty:
            continue

        dates = pd.date_range(last_hist_date + pd.Timedelta(days=1), last_hist_date + pd.Timedelta(days=horizon), freq='D')
        wsite = weath[weath['siteId'] == site_id].drop_duplicates(subset=['siteId','date'], keep='last').set_index('date')

        for d in dates:
            row = {**base_feat, 'date': d}

            if d in wsite.index:
                ww = wsite.loc[d]
                for c in weather_cols:
                    if c in wsite.columns:
                        row[c] = float(ww[c]) if pd.notna(ww[c]) else np.nan

            if cfg['features'].get('add_calendar', True):
                tmp = add_calendar_features(pd.DataFrame([row]), 'date')
                for col in ['dow', 'month', 'dayofyear', 'is_weekend']:
                    row[col] = int(tmp.iloc[0][col])

            s = state.set_index('date')[args.target]
            for k in cfg['features']['lags']:
                row[f'lag_{k}'] = float(s.get(d - pd.Timedelta(days=k), np.nan))

            for w in cfg['features']['rolling_windows']:
                window = pd.date_range(d - pd.Timedelta(days=w), d - pd.Timedelta(days=1), freq='D')
                vals = s.reindex(window).to_numpy(dtype=float)
                row[f'roll_med_{w}'] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
                row[f'roll_mean_{w}'] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

            X = pd.DataFrame([row])[feat_cols].copy()
            X = encode_id_columns(X, id_cols, cats)

            yhat = float(model.predict(X)[0])
            future_rows.append({**base_feat, 'date': d, 'yhat': yhat})
            state = pd.concat([state, pd.DataFrame([{'date': d, args.target: yhat}])], ignore_index=True)

    out = pd.DataFrame(future_rows)
    out.to_csv(out_dir / f'pred_{args.level}_{args.target}.csv', index=False)
    print('wrote', out_dir / f'pred_{args.level}_{args.target}.csv')


if __name__ == '__main__':
    main()
