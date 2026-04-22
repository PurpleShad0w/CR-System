from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from .config import load_config
from .io_utils import ensure_dir
from .features import add_calendar_features, build_lag_features, build_rolling_features, select_feature_columns
from .modeling import load_model, mae, rmse, mape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--level', default='site', choices=['site','zone'])
    ap.add_argument('--target', required=True, choices=['elecTotalKwh','waterM3'])
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    out_dir = ensure_dir(Path(cfg['paths']['out_dir']))

    cleaned_path = out_dir / f'{args.level}hist_cleaned.csv'
    if not cleaned_path.exists():
        raise RuntimeError('Run cleaning first')

    hist = pd.read_csv(cleaned_path)
    hist['date'] = pd.to_datetime(hist['date'], errors='coerce').dt.floor('D')

    level_cfg = cfg['level_defaults'][args.level]
    id_cols = level_cfg['id_cols']

    df = hist[id_cols + ['date', args.target]].copy()
    df[args.target] = pd.to_numeric(df[args.target], errors='coerce')
    df = df.dropna(subset=id_cols + ['date', args.target])

    if cfg['features'].get('add_calendar', True):
        df = add_calendar_features(df, 'date')

    df = build_lag_features(df, id_cols, 'date', args.target, cfg['features']['lags'])
    df = build_rolling_features(df, id_cols, 'date', args.target, cfg['features']['rolling_windows'])

    weather_cols = cfg['features'].get('weather_cols', [])
    feat_cols = select_feature_columns(df, id_cols if cfg['features'].get('add_site_id', True) else [], weather_cols)

    valid_days = int(cfg['training'].get('valid_days', 60))
    cutoff = df['date'].max() - pd.Timedelta(days=valid_days)
    valid = df[df['date'] > cutoff].copy()

    X = valid[feat_cols].copy()
    y = valid[args.target].to_numpy(dtype=float)

    model_path = out_dir / 'models' / f'{args.level}_{args.target}.joblib'
    model = load_model(model_path)
    yhat = model.predict(X)

    rep = pd.DataFrame([{
        'rows': int(len(valid)),
        'MAE': mae(y, yhat),
        'RMSE': rmse(y, yhat),
        'MAPE': mape(y, yhat)
    }])

    rep.to_csv(out_dir / f'eval_{args.level}_{args.target}.csv', index=False)
    print(rep)


if __name__ == '__main__':
    main()
