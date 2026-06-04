from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import re

# Existing 4 columns in historical db
EXISTING_4 = {"elecBveKwh", "elecCvcKwh", "elecForceKwh", "elecLightingKwh"}


def _to_camel(s: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", str(s))
    parts = [p for p in parts if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) if parts else ""


def drift_col_name(meter_name: str, usage_name: str) -> str:
    u = _to_camel(usage_name)

    if meter_name == "elec":
        return f"elec{u}Kwh"

    if meter_name == "water":
        # same convention as existing historical DB
        return f"water{u}M3"

    if meter_name == "eg":
        # chilled water meter — keep neutral naming if your legacy DB has no established suffix
        return f"eg{u}"

    if meter_name == "ec":
        # hot water meter — keep neutral naming if your legacy DB has no established suffix
        return f"ec{u}"

    return f"{meter_name}{u}"


def _to_camel(s: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", str(s))
    parts = [p for p in parts if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) if parts else ""


def _usage_to_base_col(usage_name: str) -> str:
    return f"elec{_to_camel(usage_name)}Kwh"


def _usage_to_col(usage_name: str) -> str:
    base = _usage_to_base_col(usage_name)
    # Avoid column collision with existing 4 by keeping a *_drift version
    return base + "_drift" if base in EXISTING_4 else base


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

    usg = pd.read_csv(usages_path)[["id", "name"]].rename(columns={"name": "usage_name"})
    mt = pd.read_csv(meters_path)[["id", "name"]].rename(columns={"name": "meter_name"})

    cdr = cdr.merge(usg, left_on="usageId", right_on="id", how="left")
    cdr = cdr.merge(mt, left_on="meterTypeId", right_on="id", how="left")

    # meter type mapping from metertypes.csv
    # source-supported IDs:
    # 1 = elec, 2 = water, 4 = eg (chilled water), 5 = ec (hot water)
    meter_name_map = (
        mt[["id", "name"]]
        .drop_duplicates()
        .rename(columns={"id": "meterTypeId", "name": "meter_name"})
    )

    cdr = cdr.merge(meter_name_map, on="meterTypeId", how="left")
    cdr = cdr.merge(
        usg[["id", "name"]].drop_duplicates().rename(columns={"id": "usageId", "name": "usage_name"}),
        on="usageId",
        how="left",
    )
    cdr["usage_col"] = cdr.apply(
    lambda r: drift_col_name(r["meter_name"], r["usage_name"]),
    axis=1,
)

    # canonical perimeter for enrichment
    # zone is the safest for zonehist, then aggregate to sitehist
    cdr_base = cdr[cdr["perimeter"] == "zone"].copy()

    cdr_base = cdr_base.dropna(subset=["siteId", "zoneId", "date", "consumption", "usage_col"]).copy()
    cdr_base["siteId"] = cdr_base["siteId"].astype(int)
    cdr_base["zoneId"] = cdr_base["zoneId"].astype(int)
    cdr_base["consumption"] = pd.to_numeric(cdr_base["consumption"], errors="coerce")
    cdr_base = cdr_base.dropna(subset=["consumption"])

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

    usage_cols_zone = [c for c in wide_zone.columns if c not in ("siteId", "zoneId", "date")]
    wide_zone["elecTotalFromDriftKwh"] = wide_zone[usage_cols_zone].sum(axis=1, min_count=1)

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

    usage_cols_site = [c for c in wide_site.columns if c not in ("siteId", "date")]
    wide_site["elecTotalFromDriftKwh"] = wide_site[usage_cols_site].sum(axis=1, min_count=1)

    def electric_usage_cols(df: pd.DataFrame) -> list[str]:
        return sorted([
            c for c in df.columns
            if isinstance(c, str)
            and c.startswith("elec")
            and c.endswith("Kwh")
            and c not in {"elecTotalKwh", "elecAggregatedKwh", "elecTotalFromDriftKwh", "elecTotalAccurateKwh"}
            and not c.endswith("_drift")
        ])

    elec_cols_zone = electric_usage_cols(wide_zone)
    elec_cols_site = electric_usage_cols(wide_site)

    wide_zone["elecTotalFromDriftKwh"] = wide_zone[elec_cols_zone].sum(axis=1, min_count=1)
    wide_site["elecTotalFromDriftKwh"] = wide_site[elec_cols_site].sum(axis=1, min_count=1)

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

    mapping = cdr[["usageId", "usage_name"]].drop_duplicates().copy()
    mapping["column"] = mapping["usage_name"].map(_usage_to_base_col)
    mapping["column_in_enriched"] = mapping["usage_name"].map(_usage_to_col)
    mapping = mapping.sort_values("usageId")
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
