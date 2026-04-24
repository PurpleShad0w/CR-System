from __future__ import annotations
import pandas as pd


def build_id_categories(df: pd.DataFrame, id_cols: list[str]) -> dict[str, pd.Index]:
    cats = {}
    for c in id_cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors='coerce')
            uniq = pd.Index(sorted([v for v in vals.dropna().unique().tolist()]))
            cats[c] = uniq
    return cats


def encode_id_columns(df: pd.DataFrame, id_cols: list[str], categories: dict[str, pd.Index]) -> pd.DataFrame:
    df = df.copy()
    for c in id_cols:
        if c in df.columns and c in categories:
            df[c] = pd.Categorical(df[c], categories=categories[c]).codes
    return df
