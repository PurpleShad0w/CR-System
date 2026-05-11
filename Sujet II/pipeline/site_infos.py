from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_site_infos(xlsx_path: Path) -> pd.DataFrame:
    """
    Load site-level static info (surface, activity, context) from Sites_Shyrka_Infos.xlsx.

    Returns columns:
      - siteId (int)
      - surface_m2 (float)
      - activity (str|None)
      - context (str|None)

    The input file contains multiple rows per site (one per indicator). We deduplicate by siteId.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        return pd.DataFrame(columns=["siteId", "surface_m2", "activity", "context"])

    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    def find_col(candidates):
        # exact
        for c in candidates:
            if c in df.columns:
                return c
        # case-insensitive
        low = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    # IMPORTANT: in your file, the ID column may be "ID" (not "ID Site")
    c_id = find_col(["ID Site", "ID", "siteId", "SiteId", "id_site"])
    c_surface = find_col(["Surface", "surface", "Surface (m2)", "Surface (m²)", "m2"])
    c_act = find_col(["Activité", "Activite", "Activity"])
    c_ctx = find_col(["Info Contexte", "Contexte", "Context"])

    if c_id is None or c_surface is None:
        return pd.DataFrame(columns=["siteId", "surface_m2", "activity", "context"])

    out = pd.DataFrame()
    out["siteId"] = pd.to_numeric(df[c_id], errors="coerce")
    out["surface_m2"] = pd.to_numeric(df[c_surface], errors="coerce")

    out["activity"] = df[c_act].astype(str) if c_act is not None else None
    out["context"] = df[c_ctx].astype(str) if c_ctx is not None else None

    out = out.dropna(subset=["siteId"]).copy()
    out["siteId"] = out["siteId"].astype(int)

    def first_non_null(s):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else None

    agg = out.groupby("siteId", as_index=False).agg({
        "surface_m2": first_non_null,
        "activity": first_non_null,
        "context": first_non_null,
    })

    return agg
