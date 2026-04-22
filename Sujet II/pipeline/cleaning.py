from __future__ import annotations
import numpy as np
import pandas as pd


def _num(s):
    return pd.to_numeric(s, errors='coerce')


def apply_missing_sentinels(df: pd.DataFrame, cols, zero_or_neg_is_missing: dict) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        x = _num(df[c])
        if zero_or_neg_is_missing.get(c, False):
            x = x.mask(x <= 0, np.nan)
        else:
            x = x.mask(x < 0, np.nan)
        df[c] = x
    return df


def expected_range_by_group(hist_df: pd.DataFrame, pred_df: pd.DataFrame, group_cols, date_col: str):
    cols = list(group_cols) + [date_col]
    h = hist_df[cols].dropna(subset=cols).drop_duplicates()
    p = pred_df[cols].dropna(subset=cols).drop_duplicates()
    if len(h) == 0 and len(p) == 0:
        return {}

    out = {}
    hmin = h.groupby(group_cols)[date_col].min() if len(h) else pd.Series(dtype='datetime64[ns]')
    hmax = h.groupby(group_cols)[date_col].max() if len(h) else pd.Series(dtype='datetime64[ns]')
    pmin = p.groupby(group_cols)[date_col].min() if len(p) else pd.Series(dtype='datetime64[ns]')
    pmax = p.groupby(group_cols)[date_col].max() if len(p) else pd.Series(dtype='datetime64[ns]')

    keys = set(hmin.index.tolist()) | set(pmin.index.tolist())
    for k in keys:
        a, b = [], []
        if k in hmin.index:
            a.append(hmin.loc[k]); b.append(hmax.loc[k])
        if k in pmin.index:
            a.append(pmin.loc[k]); b.append(pmax.loc[k])
        out[k if isinstance(k, tuple) else (k,)] = (pd.to_datetime(min(a)).floor('D'), pd.to_datetime(max(b)).floor('D'))
    return out


def drop_local_spikes_v12(
    hist_df: pd.DataFrame,
    pred_df: pd.DataFrame | None,
    group_cols,
    date_col: str,
    value_col: str,
    pred_col: str | None,
    expected_range: dict | None,
    window_days=2,
    factor=8.0,
    post_gap_days=30,
    post_gap_factor=0.6,
    pred_hard_factor=10.0
):
    hist_df = hist_df.copy()
    logs = []

    hist_df[date_col] = pd.to_datetime(hist_df[date_col], errors='coerce').dt.floor('D')
    if pred_df is not None and date_col in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df[date_col] = pd.to_datetime(pred_df[date_col], errors='coerce').dt.floor('D')

    for keys, g in hist_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.dropna(subset=[date_col]).sort_values(date_col)
        if g.empty:
            continue

        if expected_range is not None and keys in expected_range:
            start, end = expected_range[keys]
        else:
            start, end = g[date_col].min(), g[date_col].max()
        if pd.isna(start) or pd.isna(end):
            continue

        idx = pd.date_range(pd.to_datetime(start).floor('D'), pd.to_datetime(end).floor('D'), freq='D')
        s = g.set_index(date_col)[value_col].reindex(idx)
        s = pd.to_numeric(s, errors='coerce')

        if pred_df is not None and pred_col is not None and pred_col in pred_df.columns:
            p = pred_df
            for c, v in zip(group_cols, keys):
                p = p[p[c] == v]
            p = p.dropna(subset=[date_col]).drop_duplicates(subset=[date_col])
            p = p.set_index(date_col)[pred_col].reindex(idx)
            p = pd.to_numeric(p, errors='coerce')
        else:
            p = pd.Series(index=idx, dtype=float)

        missing = s.isna() | (s <= 0)
        gap_len = missing.groupby((~missing).cumsum()).transform('sum')
        post_gap = gap_len.shift(1).fillna(0) >= post_gap_days

        for i, d in enumerate(idx):
            v = s.iloc[i]
            if not np.isfinite(v) or v <= 0:
                continue

            lo = max(0, i - window_days)
            hi = min(len(s), i + window_days + 1)
            neigh = s.iloc[lo:hi].drop(index=d, errors='ignore')
            neigh = neigh[(neigh > 0) & np.isfinite(neigh)]
            if len(neigh) < 2:
                continue

            med = float(np.nanmedian(neigh))
            if not np.isfinite(med) or med <= 0:
                continue

            f = factor * (post_gap_factor if post_gap.iloc[i] else 1.0)
            is_spike = v > med * f
            if np.isfinite(p.iloc[i]) and p.iloc[i] > 0:
                is_spike = is_spike or (v > p.iloc[i] * pred_hard_factor)

            if is_spike:
                s.iloc[i] = np.nan
                logs.append({
                    'group': keys,
                    'value_col': value_col,
                    'date': str(pd.to_datetime(d).date()),
                    'action': 'local_spike_drop',
                    'old': float(v),
                    'local_median': float(med),
                    'pred': float(p.iloc[i]) if np.isfinite(p.iloc[i]) else np.nan
                })

        update_map = s
        mask_group = np.ones(len(hist_df), dtype=bool)
        for c, v in zip(group_cols, keys):
            mask_group &= (hist_df[c] == v)
        hist_df.loc[mask_group, value_col] = hist_df.loc[mask_group, date_col].map(update_map).to_numpy()

    return hist_df, pd.DataFrame(logs)


def spread_cumul_spikes_v3(df: pd.DataFrame, group_cols, date_col: str, value_col: str, cfg: dict, expected_range: dict | None):
    df = df.copy()
    logs = []
    if value_col not in df.columns or date_col not in df.columns:
        return df, pd.DataFrame(logs)

    df[value_col] = _num(df[value_col])

    min_run = int(cfg.get('min_missing_run', 3))
    spike_factor = float(cfg.get('spike_factor', 20.0))
    strategy = cfg.get('strategy', 'spread')
    baseline_points = int(cfg.get('baseline_points', 30))
    max_spread_days = int(cfg.get('max_spread_days', 370))

    out_parts = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.sort_values(date_col)
        if g[date_col].isna().all():
            out_parts.append(g)
            continue

        if expected_range is not None and keys in expected_range:
            start, end = expected_range[keys]
        else:
            start, end = g[date_col].min(), g[date_col].max()
        if pd.isna(start) or pd.isna(end):
            out_parts.append(g)
            continue

        idx = pd.date_range(pd.to_datetime(start).floor('D'), pd.to_datetime(end).floor('D'), freq='D')
        g2 = g.set_index(date_col).reindex(idx)
        g2.index.name = date_col
        for c, v in zip(group_cols, keys):
            g2[c] = v

        x = g2[value_col].copy()
        valid = x.notna() & (x > 0)
        is_missing = ~valid

        overall_med = float(np.nanmedian(x[valid].to_numpy())) if valid.any() else np.nan

        last_valid_pos = np.full(len(x), -1, dtype=int)
        last = -1
        for i in range(len(x)):
            last_valid_pos[i] = last
            if valid.iloc[i]:
                last = i

        for i in range(len(x)):
            v = x.iloc[i]
            if not np.isfinite(v) or v <= 0:
                continue
            prev = last_valid_pos[i]
            if prev < 0:
                continue
            gap = i - prev - 1
            if gap < min_run:
                continue

            prev_vals = x.iloc[:prev + 1]
            prev_valid = prev_vals[prev_vals.notna() & (prev_vals > 0)]
            baseline = float(np.nanmedian(prev_valid.tail(baseline_points).to_numpy())) if len(prev_valid) else np.nan
            if not np.isfinite(baseline) or baseline <= 0:
                baseline = overall_med
            if not np.isfinite(baseline) or baseline <= 0:
                continue

            if v <= spike_factor * baseline:
                continue

            span = gap + 1
            action = strategy
            if span > max_spread_days and strategy == 'spread':
                action = 'drop'

            if action == 'drop':
                x.iloc[i] = np.nan
                logs.append({'group': keys, 'value_col': value_col, 'date': str(g2.index[i].date()), 'action': 'drop', 'old': float(v), 'gap_days': int(gap), 'span_days': int(span), 'baseline': float(baseline)})
            else:
                per_day = v / span
                start_i = i - gap
                for j in range(start_i, i + 1):
                    if j == i or is_missing.iloc[j]:
                        x.iloc[j] = per_day
                logs.append({'group': keys, 'value_col': value_col, 'date': str(g2.index[i].date()), 'action': 'spread', 'old': float(v), 'new_per_day': float(per_day), 'gap_days': int(gap), 'span_days': int(span), 'baseline': float(baseline)})

        g2[value_col] = x
        out_parts.append(g2.reset_index())

    out = pd.concat(out_parts, ignore_index=True)
    out = out[[c for c in df.columns if c in out.columns]]
    return out, pd.DataFrame(logs)
