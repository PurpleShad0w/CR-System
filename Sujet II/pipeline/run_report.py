from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import load_config
from .io_utils import ensure_dir
from .features import add_calendar_features, build_lag_features, build_rolling_features
from .encoding import encode_id_columns
from .modeling import load_model
from .reporting import parity, residual_hist, ts_actual_vs_pred, top_sites_rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--level', default='site', choices=['site', 'zone'])
    ap.add_argument('--target', required=True, choices=['elecTotalKwh', 'waterM3'])
    ap.add_argument('--site', type=int, default=130)
    ap.add_argument('--top', type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    out_dir = ensure_dir(Path(cfg['paths']['out_dir']))
    fig_dir = ensure_dir(out_dir / 'figures')

    cleaned_path = out_dir / f'{args.level}hist_cleaned.csv'
    if not cleaned_path.exists():
        raise RuntimeError('Run cleaning first')

    hist = pd.read_csv(cleaned_path)
    hist['date'] = pd.to_datetime(hist['date'], errors='coerce').dt.floor('D')

    level_cfg = cfg['level_defaults'][args.level]
    id_cols = level_cfg['id_cols']

    model_dir = out_dir / 'models'
    meta = json.loads((model_dir / f'{args.level}_{args.target}.meta.json').read_text(encoding='utf-8'))
    feat_cols = meta['feature_columns']
    cats = {c: pd.Index(meta['id_categories'][c]) for c in meta.get('id_categories', {})}

    df = hist[id_cols + ['date', args.target]].copy()
    df[args.target] = pd.to_numeric(df[args.target], errors='coerce')
    df = df.dropna(subset=id_cols + ['date', args.target])

    # merge weather
    from .dataset import load_level_tables
    db_dir = Path(cfg['paths']['db_dir'])
    _, _, weath = load_level_tables(db_dir, level_cfg)
    if len(weath) and 'date' in weath.columns:
        weath['date'] = pd.to_datetime(weath['date'], errors='coerce').dt.floor('D')

    weather_cols = cfg['features'].get('weather_cols', [])
    if len(weath) and 'siteId' in weath.columns and 'date' in weath.columns:
        keep = ['siteId', 'date'] + [c for c in weather_cols if c in weath.columns]
        w = weath[keep].drop_duplicates(subset=['siteId','date'], keep='last')
        df = df.merge(w, on=['siteId','date'], how='left')

    if cfg['features'].get('add_calendar', True):
        df = add_calendar_features(df, 'date')

    df = build_lag_features(df, id_cols, 'date', args.target, cfg['features']['lags'])
    df = build_rolling_features(df, id_cols, 'date', args.target, cfg['features']['rolling_windows'])

    valid_days = int(cfg['training'].get('valid_days', 60))
    cutoff = df['date'].max() - pd.Timedelta(days=valid_days)
    valid = df[df['date'] > cutoff].copy()

    X = valid[feat_cols].copy()
    X = encode_id_columns(X, id_cols, cats)
    y = valid[args.target].to_numpy(dtype=float)

    model = load_model(model_dir / f'{args.level}_{args.target}.joblib')
    yhat = model.predict(X)

    parity(y, yhat, f'Parity — {args.level} {args.target} (valid)', fig_dir / f'parity_{args.level}_{args.target}.png')
    residual_hist(y, yhat, f'Residuals — {args.level} {args.target} (valid)', fig_dir / f'resid_{args.level}_{args.target}.png')

    if 'siteId' in valid.columns:
        mask = (valid['siteId'].to_numpy() == args.site)
        if mask.any():
            vsite = valid.loc[mask, ['date', args.target]].copy()
            vsite['yhat'] = yhat[mask]
            ts_actual_vs_pred(vsite, 'date', args.target, 'yhat', f'Site {args.site} — {args.target} (valid)', fig_dir / f'ts_site{args.site}_{args.target}.png')

    # top sites
    if 'siteId' in valid.columns:
        tmp = pd.DataFrame({'siteId': valid['siteId'].to_numpy(), 'y': y, 'yhat': yhat})
        rows = []
        for sid, g in tmp.groupby('siteId'):
            a = g['y'].to_numpy(dtype=float)
            b = g['yhat'].to_numpy(dtype=float)
            m = np.isfinite(a) & np.isfinite(b)
            rmse = float(np.sqrt(np.mean((a[m]-b[m])**2))) if m.any() else np.nan
            rows.append({'siteId': int(sid), 'RMSE': rmse})
        errors = pd.DataFrame(rows)
        top_sites_rmse(errors, f'Top sites RMSE — {args.target} (valid)', fig_dir / f'top_sites_rmse_{args.target}.png', top=args.top)

    print('wrote figures to', fig_dir)


if __name__ == '__main__':
    main()
