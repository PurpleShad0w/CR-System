from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from .config import load_config
from .io_utils import ensure_dir
from .dataset import load_level_tables
from .reporting import ts_truth_vs_algo_pred


PRED_COL_BY_TARGET = {
    "elecTotalKwh": "totalKwh",
    "waterM3": "totalWater",
    "indoorTempDegC": "tempAmb",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", default="site")   # site | zone | all
    ap.add_argument("--target", required=True)   # elecTotalKwh | waterM3 | indoorTempDegC | all
    ap.add_argument("--site", default="all", help='siteId, ou "all"')
    ap.add_argument("--zone", default="all", help='zoneId (zone-level), ou "all"')
    args = ap.parse_args()

    LEVELS = ["site", "zone"]
    TARGETS_BY_LEVEL = {
        "site": ["elecTotalKwh", "waterM3", "indoorTempDegC"],
        "zone": ["elecTotalKwh", "waterM3", "indoorTempDegC"],
    }

    cfg = load_config(args.config).raw
    db_dir = Path(cfg["paths"]["db_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]))
    fig_dir = ensure_dir(out_dir / "figures")

    base_dir = ensure_dir(fig_dir / "baseline")
    site_dir = ensure_dir(base_dir / "sites")
    zone_dir = ensure_dir(base_dir / "zones")

    def report_one(level: str, target: str):
        level_cfg = cfg["level_defaults"][level]
        hist, pred, _ = load_level_tables(db_dir, level_cfg)

        # Normalize dates (dataset loader usually provides date, but keep safe)
        if "date" in hist.columns:
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce").dt.floor("D")
        if "date" in pred.columns:
            pred["date"] = pd.to_datetime(pred["date"], errors="coerce").dt.floor("D")

        # Determine mapping truth vs algo-pred columns
        if target not in PRED_COL_BY_TARGET:
            # no algo prediction available for this target (ex: indoorTempDegC)
            print(f"[SKIP] No algorithmic prediction column for target={target} at level={level}")
            return

        pred_col = PRED_COL_BY_TARGET[target]

        # Choose which sites
        site_arg = str(args.site).lower()
        if site_arg == "all":
            site_ids = sorted([int(x) for x in hist["siteId"].dropna().unique().tolist()]) if "siteId" in hist.columns else []
        else:
            site_ids = [int(args.site)]

        if level == "site":
            for sid in site_ids:
                h = hist[hist["siteId"] == sid].copy()
                p = pred[pred["siteId"] == sid].copy()

                if target not in h.columns or pred_col not in p.columns:
                    continue

                ts_truth_vs_algo_pred(
                    truth_df=h[["date", target]],
                    pred_df=p[["date", pred_col]],
                    date_col="date",
                    truth_col=target,
                    pred_col=pred_col,
                    title=f"BASELINE algo vs vérité — site {sid} — {target}",
                    out=site_dir / f"baseline_site{sid}_{target}.png",
                )

        else:
            # zone-level: one file per (siteId, zoneId)
            zone_arg = str(args.zone).lower()

            for sid in site_ids:
                hz = hist[hist["siteId"] == sid].copy()
                pz = pred[pred["siteId"] == sid].copy()

                if "zoneId" not in hz.columns or "zoneId" not in pz.columns:
                    continue
                if target not in hz.columns or pred_col not in pz.columns:
                    continue

                if zone_arg == "all":
                    zone_ids = sorted([int(z) for z in hz["zoneId"].dropna().unique().tolist()])
                else:
                    zone_ids = [int(args.zone)]

                # optional per-site folder to avoid too many files in one folder
                z_site_dir = ensure_dir(zone_dir / f"site{sid}")

                for zid in zone_ids:
                    h = hz[hz["zoneId"] == zid].copy()
                    p = pz[pz["zoneId"] == zid].copy()
                    if len(h) == 0 and len(p) == 0:
                        continue

                    ts_truth_vs_algo_pred(
                        truth_df=h[["date", target]],
                        pred_df=p[["date", pred_col]],
                        date_col="date",
                        truth_col=target,
                        pred_col=pred_col,
                        title=f"BASELINE algo vs vérité — site {sid} zone {zid} — {target}",
                        out=z_site_dir / f"baseline_site{sid}_zone{zid}_{target}.png",
                    )

        print(f"[OK] baseline graphs written under {base_dir}")

    # levels/targets batch
    levels = LEVELS if args.level == "all" else [args.level]
    for lvl in levels:
        if lvl not in LEVELS:
            raise ValueError(f"Unknown level: {lvl}. Use site|zone|all")
        targets = TARGETS_BY_LEVEL[lvl] if args.target == "all" else [args.target]
        for tgt in targets:
            if tgt not in TARGETS_BY_LEVEL[lvl]:
                raise ValueError(f"Target {tgt} not supported for level {lvl}. Use {TARGETS_BY_LEVEL[lvl]} or all")
            report_one(lvl, tgt)


if __name__ == "__main__":
    main()