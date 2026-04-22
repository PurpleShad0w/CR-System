from __future__ import annotations
from pathlib import Path
import pandas as pd

from .io_utils import read_semicolon_csv


def load_level_tables(db_dir: Path, level_cfg: dict):
    hist = read_semicolon_csv(db_dir / level_cfg['hist_file'])
    pred = read_semicolon_csv(db_dir / level_cfg['pred_file'])

    weath_path = db_dir / level_cfg['weath_file']
    weath = read_semicolon_csv(weath_path) if weath_path.exists() else pd.DataFrame()

    # ids
    for c in level_cfg['id_cols']:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors='coerce').astype('Int64')
        if c in pred.columns:
            pred[c] = pd.to_numeric(pred[c], errors='coerce').astype('Int64')
        if c in weath.columns:
            weath[c] = pd.to_numeric(weath[c], errors='coerce').astype('Int64')

    # dates
    if 'dtUpdate' in hist.columns and 'date' not in hist.columns:
        hist['date'] = pd.to_datetime(hist['dtUpdate'], errors='coerce').dt.floor('D')
    if 'date' in pred.columns:
        pred['date'] = pd.to_datetime(pred['date'], errors='coerce').dt.floor('D')

    if len(weath):
        if 'dtUpdate' in weath.columns and 'date' not in weath.columns:
            weath['date'] = pd.to_datetime(weath['dtUpdate'], errors='coerce').dt.floor('D')
        if 'date' in weath.columns:
            weath['date'] = pd.to_datetime(weath['date'], errors='coerce').dt.floor('D')

    return hist, pred, weath
