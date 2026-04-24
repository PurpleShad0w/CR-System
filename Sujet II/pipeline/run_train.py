from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd

from .config import load_config
from .dataset import load_level_tables
from .io_utils import ensure_dir
from .features import add_calendar_features, build_lag_features, build_rolling_features, select_feature_columns
from .modeling import make_model, save_model
from .encoding import build_id_categories, encode_id_columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--level', default='site', choices=['site', 'zone'])
    ap.add_argument('--target', required=True, choices=['elecTotalKwh', 'waterM3'])
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    db_dir = Path(cfg['paths']['db_dir'])
    out_dir = ensure_dir(Path(cfg['paths']['out_dir']))

    level_cfg = cfg['level_defaults'][args.level]
    id_cols = level_cfg['id_cols']

    cleaned_path = out_dir / f'{args.level}hist_cleaned.csv'
    if cleaned_path.exists():
        hist = pd.read_csv(cleaned_path)
        hist['date'] = pd.to_datetime(hist['date'], errors='coerce').dt.floor('D')
    else:
        hist, _, _ = load_level_tables(db_dir, level_cfg)
        hist['date'] = pd.to_datetime(hist['date'], errors='coerce').dt.floor('D')

    _, _, weath = load_level_tables(db_dir, level_cfg)
    if len(weath) and 'date' in weath.columns:
        weath['date'] = pd.to_datetime(weath['date'], errors='coerce').dt.floor('D')

    df = hist[id_cols + ['date', args.target]].copy()
    df[args.target] = pd.to_numeric(df[args.target], errors='coerce')
    df = df.dropna(subset=id_cols + ['date', args.target])

    weather_cols = cfg['features'].get('weather_cols', [])
    if len(weath) and 'siteId' in weath.columns and 'date' in weath.columns:
        keep = ['siteId', 'date'] + [c for c in weather_cols if c in weath.columns]
        w = weath[keep].drop_duplicates(subset=['siteId', 'date'], keep='last')
        df = df.merge(w, on=['siteId', 'date'], how='left')

    if cfg['features'].get('add_calendar', True):
        df = add_calendar_features(df, 'date')

    df = build_lag_features(df, id_cols, 'date', args.target, cfg['features']['lags'])
    df = build_rolling_features(df, id_cols, 'date', args.target, cfg['features']['rolling_windows'])

    df = df.dropna(subset=[f'lag_{k}' for k in cfg['features']['lags']] + [f'roll_med_{w}' for w in cfg['features']['rolling_windows']])

    feat_cols = select_feature_columns(df, id_cols if cfg['features'].get('add_site_id', True) else [], weather_cols)

    valid_days = int(cfg['training'].get('valid_days', 60))
    cutoff = df['date'].max() - pd.Timedelta(days=valid_days)

    train = df[df['date'] <= cutoff].copy()
    valid = df[df['date'] > cutoff].copy()

    X_train = train[feat_cols].copy()
    y_train = train[args.target]
    X_valid = valid[feat_cols].copy()
    y_valid = valid[args.target]

    cats = build_id_categories(df, id_cols)
    X_train = encode_id_columns(X_train, id_cols, cats)
    X_valid = encode_id_columns(X_valid, id_cols, cats)

    model = make_model(cfg)
    model.fit(X_train, y_train)

    model_dir = out_dir / 'models'
    model_path = model_dir / f'{args.level}_{args.target}.joblib'
    save_model(model, model_path)

    meta = {
        'id_categories': {c: [int(v) for v in cats[c].tolist()] for c in cats},
        'feature_columns': feat_cols,
        'level': args.level,
        'target': args.target
    }
    (model_dir / f'{args.level}_{args.target}.meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    from .modeling import mae, rmse, mape
    pred_valid = model.predict(X_valid)
    print('valid rows:', len(valid))
    print('MAE', mae(y_valid, pred_valid))
    print('RMSE', rmse(y_valid, pred_valid))
    print('MAPE', mape(y_valid, pred_valid))
    print('saved', model_path)


if __name__ == '__main__':
    main()
