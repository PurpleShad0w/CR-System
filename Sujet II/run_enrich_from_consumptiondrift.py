from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Existing 4 columns in historical db
EXISTING_4 = {"elecBveKwh", "elecCvcKwh", "elecForceKwh", "elecLightingKwh"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _read_lookup_csv(path: Path) -> pd.DataFrame:
    """
    Read lookup CSV robustly without sep autodetection.
    """
    last_err = None

    for sep in [",", ";"]:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                encoding="utf-8-sig",
                on_bad_lines="skip",
            )
            df = _normalize_columns(df)
            if {"id", "name"}.issubset(df.columns):
                return df
        except Exception as e:
            last_err = e

    raise ValueError(f"Unable to parse lookup CSV {path}. Last error: {last_err}")


def _to_camel(s: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", str(s))
    parts = [p for p in parts if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) if parts else ""


def drift_col_name(meter_name: str, usage_name: str) -> str:
    u = _to_camel(usage_name)

    if meter_name == "elec":
        base = f"elec{u}Kwh"
        return base + "_drift" if base in EXISTING_4 else base

    if meter_name == "water":
        return f"water{u}M3"

    if meter_name == "eg":
        # chilled water meter
        return f"eg{u}"

    if meter_name == "ec":
        # hot water meter
        return f"ec{u}"

    return f"{meter_name}{u}"


def add_elec_total_no_bve(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived elecTotalNoBveKwh = elecTotalKwh - elecBveKwh (fillna 0), clamp >= 0."""
    if "elecTotalNoBveKwh" in df.columns:
        return df
    if "elecTotalKwh" not in df.columns or "elecBveKwh" not in df.columns:
        return df
    out = df.copy()
    total = pd.to_numeric(out["elecTotalKwh"], errors="coerce")
    bve = pd.to_numeric(out["elecBveKwh"], errors="coerce").fillna(0.0)
    out["elecTotalNoBveKwh"] = np.maximum(total - bve, 0.0)
    return out


def enrich_histories(
    db_dir: Path,
    perimeter: str = "zone",
    fill_existing_4_if_zero: bool = True,
) -> tuple[Path, Path, Path]:
    """Build site/zone enriched history CSVs from consumptiondrift + mapping tables.

    Inputs expected in db_dir:
      - sitehist.csv (semicolon)
      - zonehist.csv (semicolon)
      - consumptiondrift.csv (comma)
      - usages.csv (comma)
      - metertypes.csv (comma)

    Strategy:
      - use electricity rows only (meterTypeId==1)
      - use perimeter='zone' (has siteId + zoneId) as canonical decomposition
      - create wide daily usage columns at zone-level, then aggregate to site-level
      - merge into historical tables on (siteId, zoneId, date) / (siteId, date)
      - keep *_drift columns for the 4 existing uses to avoid collisions
      - optionally fill existing 4 columns when missing/0 from their *_drift value
      - add elecTotalFromDriftKwh = sum(all drift usage columns)
      - write enriched CSVs back under db_dir
    """

    site_path = db_dir / "sitehist.csv"
    zone_path = db_dir / "zonehist.csv"
    drift_path = db_dir / "consumptiondrift.csv"
    usages_path = db_dir / "usages.csv"
    meters_path = db_dir / "metertypes.csv"

    site = pd.read_csv(site_path, sep=";", engine="python")
    zone = pd.read_csv(zone_path, sep=";", engine="python")

    # date key from dtUpdate
    site["date"] = pd.to_datetime(site["dtUpdate"], errors="coerce").dt.floor("D")
    zone["date"] = pd.to_datetime(zone["dtUpdate"], errors="coerce").dt.floor("D")

    cdr = pd.read_csv(drift_path)
    cdr["date"] = pd.to_datetime(cdr["date"], errors="coerce").dt.floor("D")

    usg = _read_lookup_csv(usages_path)
    mt = _read_lookup_csv(meters_path)

    print("[DEBUG] usages columns:", usg.columns.tolist())
    print("[DEBUG] metertypes columns:", mt.columns.tolist())

    required_cols = {"id", "name"}

    if not required_cols.issubset(set(usg.columns)):
        raise ValueError(f"usages.csv missing columns {required_cols}. Found: {usg.columns.tolist()}")

    if not required_cols.issubset(set(mt.columns)):
        raise ValueError(f"metertypes.csv missing columns {required_cols}. Found: {mt.columns.tolist()}")

    usg = usg[["id", "name"]].copy().rename(columns={"name": "usage_name"})
    mt = mt[["id", "name"]].copy().rename(columns={"name": "meter_name"})

    usg["id"] = pd.to_numeric(usg["id"], errors="coerce")
    mt["id"] = pd.to_numeric(mt["id"], errors="coerce")

    usg = usg.dropna(subset=["id"]).copy()
    mt = mt.dropna(subset=["id"]).copy()

    usg["id"] = usg["id"].astype(int)
    mt["id"] = mt["id"].astype(int)

    cdr = cdr.merge(usg, left_on="usageId", right_on="id", how="left")
    cdr = cdr.merge(mt, left_on="meterTypeId", right_on="id", how="left", suffixes=("", "_meter"))

    cdr["usage_col"] = cdr.apply(
        lambda r: drift_col_name(r["meter_name"], r["usage_name"]),
        axis=1,
    )

    # canonical perimeter for enrichment
    # zone is the safest for zonehist, then aggregate to sitehist
    cdr_base = cdr[cdr["perimeter"] == "zone"].copy()

    cdr_base = cdr_base.dropna(subset=["siteId", "zoneId", "date", "consumption", "usage_col"]).copy()
    cdr_base["siteId"] = pd.to_numeric(cdr_base["siteId"], errors="coerce")
    cdr_base["zoneId"] = pd.to_numeric(cdr_base["zoneId"], errors="coerce")
    cdr_base["consumption"] = pd.to_numeric(cdr_base["consumption"], errors="coerce")

    cdr_base = cdr_base.dropna(subset=["siteId", "zoneId", "consumption"]).copy()
    cdr_base["siteId"] = cdr_base["siteId"].astype(int)
    cdr_base["zoneId"] = cdr_base["zoneId"].astype(int)

    # zone-level wide
    agg_zone = (
    cdr_base
    .groupby(["siteId", "zoneId", "date", "usage_col"], as_index=False)["consumption"]
    .sum()
)

    wide_zone = (
        agg_zone
        .pivot_table(
            index=["siteId", "zoneId", "date"],
            columns="usage_col",
            values="consumption",
            aggfunc="sum",
        ).reset_index()
    )

    wide_zone.columns.name = None

    # site-level wide
    agg_site = (
    cdr_base
    .groupby(["siteId", "date", "usage_col"], as_index=False)["consumption"]
    .sum()
)

    wide_site = (
        agg_site
        .pivot_table(
            index=["siteId", "date"],
            columns="usage_col",
            values="consumption",
            aggfunc="sum",
        ).reset_index()
    )

    wide_site.columns.name = None

    def electric_usage_cols(df: pd.DataFrame) -> list[str]:
        return sorted([
            c for c in df.columns
            if isinstance(c, str)
            and c.startswith("elec")
            and c.endswith("Kwh")
            and c not in {"elecTotalKwh", "elecAggregatedKwh", "elecTotalFromDriftKwh", "elecTotalAccurateKwh"}
            and not c.endswith("_drift")
        ])

    def water_usage_cols(df: pd.DataFrame) -> list[str]:
        return sorted([
            c for c in df.columns
            if isinstance(c, str)
            and c.startswith("water")
            and c.endswith("M3")
        ])

    def ec_usage_cols(df: pd.DataFrame) -> list[str]:
        return sorted([
            c for c in df.columns
            if isinstance(c, str)
            and c.startswith("ec")
        ])

    def eg_usage_cols(df: pd.DataFrame) -> list[str]:
        return sorted([
            c for c in df.columns
            if isinstance(c, str)
            and c.startswith("eg")
        ])

    elec_cols_zone = electric_usage_cols(wide_zone)
    elec_cols_site = electric_usage_cols(wide_site)

    wide_zone["elecTotalFromDriftKwh"] = wide_zone[elec_cols_zone].sum(axis=1, min_count=1)
    wide_site["elecTotalFromDriftKwh"] = wide_site[elec_cols_site].sum(axis=1, min_count=1)

    water_cols_zone = water_usage_cols(wide_zone)
    water_cols_site = water_usage_cols(wide_site)
    if water_cols_zone:
        wide_zone["waterTotalFromDriftM3"] = wide_zone[water_cols_zone].sum(axis=1, min_count=1)
    if water_cols_site:
        wide_site["waterTotalFromDriftM3"] = wide_site[water_cols_site].sum(axis=1, min_count=1)

    ec_cols_zone = ec_usage_cols(wide_zone)
    ec_cols_site = ec_usage_cols(wide_site)
    if ec_cols_zone:
        wide_zone["ecTotalFromDrift"] = wide_zone[ec_cols_zone].sum(axis=1, min_count=1)
    if ec_cols_site:
        wide_site["ecTotalFromDrift"] = wide_site[ec_cols_site].sum(axis=1, min_count=1)

    eg_cols_zone = eg_usage_cols(wide_zone)
    eg_cols_site = eg_usage_cols(wide_site)
    if eg_cols_zone:
        wide_zone["egTotalFromDrift"] = wide_zone[eg_cols_zone].sum(axis=1, min_count=1)
    if eg_cols_site:
        wide_site["egTotalFromDrift"] = wide_site[eg_cols_site].sum(axis=1, min_count=1)

    created_site_cols = [c for c in wide_site.columns if c not in {"siteId", "date"}]
    created_zone_cols = [c for c in wide_zone.columns if c not in {"siteId", "zoneId", "date"}]

    print("[ENRICH] site cols created:", created_site_cols)
    print("[ENRICH] zone cols created:", created_zone_cols)

    # Merge
    zone_en = zone.merge(wide_zone, on=["siteId", "zoneId", "date"], how="left")
    site_en = site.merge(wide_site, on=["siteId", "date"], how="left")

    # Optionally fill existing 4 from *_drift
    if fill_existing_4_if_zero:
        for base in sorted(EXISTING_4):
            drift_col = base + "_drift"
            if drift_col in zone_en.columns:
                cur = pd.to_numeric(zone_en[base], errors="coerce")
                drift = pd.to_numeric(zone_en[drift_col], errors="coerce")
                m = cur.isna() | (cur == 0)
                zone_en.loc[m, base] = drift.loc[m]
            if drift_col in site_en.columns:
                cur = pd.to_numeric(site_en[base], errors="coerce")
                drift = pd.to_numeric(site_en[drift_col], errors="coerce")
                m = cur.isna() | (cur == 0)
                site_en.loc[m, base] = drift.loc[m]

    # Keep derived noBVE in enriched outputs too
    site_en = add_elec_total_no_bve(site_en)
    zone_en = add_elec_total_no_bve(zone_en)

    # Write
    site_out = db_dir / "sitehist_enriched.csv"
    zone_out = db_dir / "zonehist_enriched.csv"
    map_out = db_dir / "drift_usage_mapping.csv"

    site_en.to_csv(site_out, sep=";", index=False)
    zone_en.to_csv(zone_out, sep=";", index=False)

    mapping = cdr[["usageId", "usage_name", "meterTypeId", "meter_name", "usage_col"]].drop_duplicates().copy()
    mapping = mapping.sort_values(["meterTypeId", "usageId"])
    mapping.to_csv(map_out, index=False)

    return site_out, zone_out, map_out


def main():
    ap = argparse.ArgumentParser(description="Enrich sitehist/zonehist with usage breakdown from consumptiondrift.")
    ap.add_argument("--db-dir", required=True, help="Path to Sujet II/db/ folder")
    ap.add_argument("--perimeter", default="zone", help="Which perimeter to use as canonical decomposition (default: zone)")
    ap.add_argument("--no-fill-existing-4", action="store_true", help="Do not fill existing 4 usage cols from *_drift when missing/0")
    args = ap.parse_args()

    db_dir = Path(args.db_dir)
    site_out, zone_out, map_out = enrich_histories(
        db_dir=db_dir,
        perimeter=args.perimeter,
        fill_existing_4_if_zero=not args.no_fill_existing_4,
    )
    print("Wrote:")
    print(" -", site_out)
    print(" -", zone_out)
    print(" -", map_out)


if __name__ == "__main__":
    main()
