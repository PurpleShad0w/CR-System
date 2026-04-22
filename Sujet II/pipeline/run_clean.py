from __future__ import annotations
import argparse
from pathlib import Path

from .config import load_config
from .dataset import load_level_tables
from .io_utils import ensure_dir
from .cleaning import apply_missing_sentinels, expected_range_by_group, drop_local_spikes_v12, spread_cumul_spikes_v3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--level', default='site', choices=['site','zone'])
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    db_dir = Path(cfg['paths']['db_dir'])
    out_dir = ensure_dir(Path(cfg['paths']['out_dir']))

    level_cfg = cfg['level_defaults'][args.level]
    id_cols = level_cfg['id_cols']

    hist, pred, weath = load_level_tables(db_dir, level_cfg)

    zero_map = {'elecTotalKwh': True, 'totalKwh': True, 'waterM3': True, 'totalWater': True}
    measure_cols = [c for c in ['elecTotalKwh','waterM3'] if c in hist.columns]
    pred_cols = [c for c in ['totalKwh','totalWater'] if c in pred.columns]

    hist = apply_missing_sentinels(hist, measure_cols, zero_map)
    pred = apply_missing_sentinels(pred, pred_cols, zero_map)

    exp = expected_range_by_group(hist, pred, id_cols, 'date')

    CLEAN_LOGS = {}

    if 'elecTotalKwh' in hist.columns and 'totalKwh' in pred.columns:
        hist, log = drop_local_spikes_v12(hist, pred, id_cols, 'date', 'elecTotalKwh', 'totalKwh', exp, factor=8.0)
        if len(log):
            CLEAN_LOGS[f'{args.level}_local_spike_elecTotalKwh'] = log

    if 'waterM3' in hist.columns and 'totalWater' in pred.columns:
        hist, log = drop_local_spikes_v12(hist, pred, id_cols, 'date', 'waterM3', 'totalWater', exp, factor=6.0)
        if len(log):
            CLEAN_LOGS[f'{args.level}_local_spike_waterM3'] = log

    cfg_cumul = {'min_missing_run': 3, 'spike_factor': 20.0, 'strategy': 'spread', 'baseline_points': 30, 'max_spread_days': 370}
    for col in measure_cols:
        hist, log = spread_cumul_spikes_v3(hist, id_cols, 'date', col, cfg_cumul, exp)
        if len(log):
            CLEAN_LOGS[f'{args.level}_cumul_{col}'] = log

    hist.to_csv(out_dir / f'{args.level}hist_cleaned.csv', index=False)
    for k, df in CLEAN_LOGS.items():
        df.to_csv(out_dir / f'cleanlog_{k}.csv', index=False)

    print('cleaned rows:', len(hist))
    print('logs:', {k: len(v) for k, v in CLEAN_LOGS.items()})


if __name__ == '__main__':
    main()
