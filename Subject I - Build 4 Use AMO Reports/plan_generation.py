#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plan_generation.py

Purpose
-------
Create a per-case generation plan that links:
- report_type macro-parts (4 total, generate only 1-3)
- candidate section buckets (variable, selected from OneNote context)
- routed OneNote pages (evidence) per selected bucket

Inputs
------
input/config/report_types.json
process/learning/skeletons/*.json
process/onenote/<notebook>/pages/*.json

Outputs
-------
process/plans/<case_id>.json

HARDENING (small addition)
--------------------------
If planning is restricted to a OneNote section (--onenote-section), the plan now includes
`section_context` (canonical identity object) in addition to the legacy `onenote_section` field.

This allows downstream steps (drafts, assembled report, PPTX) to always know which OneNote
section the report corresponds to.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

from section_context import build_section_context


# ----------------------------
# Helpers: text extraction
# ----------------------------

def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def collect_onenote_text(page: Dict[str, Any]) -> str:
    """Concatenate title + headings + paragraphs + OCR + audio transcripts."""
    out = [page.get("title", "") or ""]
    for b in page.get("blocks", []) or []:
        t = b.get("type")
        if t in ("heading", "paragraph", "image_ocr"):
            out.append(b.get("text", "") or "")
        elif t == "audio":
            out.append(b.get("transcript", "") or "")
    return norm("\n".join(out))


def load_onenote_pages(onenote_root: Path) -> List[Dict[str, Any]]:
    pages_dir = onenote_root / "pages"
    if not pages_dir.exists():
        raise SystemExit(f"pages/ not found under {onenote_root} (expected {pages_dir})")
    pages: List[Dict[str, Any]] = []
    for p in sorted(pages_dir.glob("*.json")):
        try:
            pages.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return pages


def filter_pages_by_section(pages: List[Dict[str, Any]], onenote_section: str) -> List[Dict[str, Any]]:
    if not onenote_section:
        return pages
    return [
        p
        for p in pages
        if (p.get("section") or (p.get("metadata", {}) or {}).get("section")) == onenote_section
    ]


# ----------------------------
# Skeleton catalog loading
# ----------------------------

def load_skeleton_catalogs(skeletons_dir: Path) -> Dict[str, Dict[str, Any]]:
    catalogs: Dict[str, Dict[str, Any]] = {}
    if not skeletons_dir.exists():
        return catalogs
    for p in sorted(skeletons_dir.glob("*.json")):
        try:
            catalogs[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return catalogs


# ----------------------------
# Report type detection
# ----------------------------

def detect_report_type(config: Dict[str, Any], corpus_text: str) -> Tuple[str, Dict[str, Any]]:
    best = None
    best_score = -10**9
    best_meta = None
    for rt_name, rt in config["report_types"].items():
        td = rt.get("type_detection", {})
        pos = [norm(k) for k in td.get("positive_keywords", [])]
        neg = [norm(k) for k in td.get("negative_keywords", [])]
        score = 0
        for k in pos:
            if k and k in corpus_text:
                score += 2
        for k in neg:
            if k and k in corpus_text:
                score -= 1
        if score > best_score:
            best_score = score
            best = rt_name
            best_meta = {
                "score": score,
                "positive_hits": [k for k in pos if k in corpus_text],
                "negative_hits": [k for k in neg if k in corpus_text],
            }
    return best or "ETAT_GTB_AUDIT", best_meta or {"score": 0, "positive_hits": [], "negative_hits": []}


# ----------------------------
# Section bucket scoring & routing
# ----------------------------

def score_bucket(bucket_keywords: List[str], corpus_text: str) -> float:
    s = 0.0
    for kw in bucket_keywords:
        kw = norm(kw)
        if kw and kw in corpus_text:
            s += 1.0
    return s


def rank_pages_for_bucket(pages: List[Dict[str, Any]], bucket_keywords: List[str]) -> List[Tuple[str, float]]:
    kws = [norm(k) for k in bucket_keywords if k]
    ranked: List[Tuple[str, float]] = []
    for p in pages:
        txt = collect_onenote_text(p)
        hit = 0.0
        for kw in kws:
            if kw and kw in txt:
                hit += 1.0
        if hit > 0:
            ranked.append((p.get("page_id") or p.get("title") or "unknown", hit))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ----------------------------
# Main planning
# ----------------------------

def build_plan(config: Dict[str, Any], skeleton_catalogs: Dict[str, Any], pages: List[Dict[str, Any]], case_id: str) -> Dict[str, Any]:
    corpus_text = norm("\n".join(collect_onenote_text(p) for p in pages))
    report_type, rt_debug = detect_report_type(config, corpus_text)
    rt_cfg = config["report_types"][report_type]

    defaults = config.get("defaults", {})
    gen_parts = defaults.get("generate_macro_parts", [1, 2, 3])

    sel_cfg = defaults.get("selection", {})
    min_section_score = float(sel_cfg.get("min_section_score", 2.0))
    max_sections_per_part = int(sel_cfg.get("max_sections_per_macro_part", 8))
    max_pages_per_section = int(sel_cfg.get("max_pages_per_section", 5))

    bucket_defs = rt_cfg.get("section_buckets", {})

    bucket_scores: List[Tuple[str, float]] = []
    for bucket_id, bd in bucket_defs.items():
        kws = bd.get("keywords", [])
        s = score_bucket(kws, corpus_text)
        bucket_scores.append((bucket_id, s))
    bucket_scores.sort(key=lambda x: x[1], reverse=True)

    selected_by_part: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for bucket_id, s in bucket_scores:
        bd = bucket_defs[bucket_id]
        mp = int(bd.get("macro_part"))
        if mp not in gen_parts:
            continue
        if s >= min_section_score or mp in (1, 2):
            selected_by_part[mp].append({"bucket_id": bucket_id, "score": s})

    for mp in list(selected_by_part.keys()):
        selected_by_part[mp] = selected_by_part[mp][:max_sections_per_part]

    routing: Dict[str, Any] = {}
    for mp, items in selected_by_part.items():
        for it in items:
            bid = it["bucket_id"]
            kws = bucket_defs[bid].get("keywords", [])
            ranked_pages = rank_pages_for_bucket(pages, kws)[:max_pages_per_section]
            routing[bid] = {
                "macro_part": mp,
                "bucket_label": bid,
                "keywords": kws,
                "top_pages": [{"page_id": pid, "score": sc} for pid, sc in ranked_pages],
            }

    skeleton_summary: Dict[str, Any] = {}
    for name, cat in (skeleton_catalogs or {}).items():
        skeleton_summary[name] = {
            "family": cat.get("family"),
            "cases_used": cat.get("cases_used"),
            "has_selected_skeleton": bool(cat.get("selected_skeleton")),
            "num_candidates": len((cat.get("section_scores") or {}).keys()),
        }

    return {
        "case_id": case_id,
        "report_type": report_type,
        "report_type_detection": rt_debug,
        "generate_macro_parts": gen_parts,
        "macro_parts": rt_cfg.get("macro_parts"),
        "selected_buckets_by_macro_part": dict(selected_by_part),
        "routing": routing,
        "skeleton_catalogs_loaded": skeleton_summary,
        "onenote_section": None,  # populated in main when filtering is used
        "notes": {
            "principle": "Buckets are selected from OneNote context; macro-part 4 is never generated.",
            "skeletons_role": "Skeleton JSONs are treated as candidate catalogs / priors, not a fixed outline.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="input/config/report_types.json")
    ap.add_argument("--skeletons", default="process/learning/skeletons")
    ap.add_argument("--onenote", required=True, help="Path to process/onenote/<notebook> folder (contains pages/)")
    ap.add_argument("--onenote-section", default="", help="Restrict planning to a OneNote section (e.g., 'Oseraie - OSNY').")
    ap.add_argument("--case-id", default="", help="Case identifier used for output plan filename.")
    ap.add_argument("--out", default="process/plans", help="Output folder for plans (under process/).")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    skeleton_catalogs = load_skeleton_catalogs(Path(args.skeletons))

    onenote_root = Path(args.onenote)
    pages = load_onenote_pages(onenote_root)

    onenote_section = args.onenote_section.strip()
    if onenote_section:
        pages = filter_pages_by_section(pages, onenote_section)

    # Default case_id falls back to explicit arg or the folder name.
    case_id = args.case_id.strip() or onenote_root.name

    plan = build_plan(cfg, skeleton_catalogs, pages, case_id)
    plan["onenote_section"] = onenote_section if onenote_section else None

    # HARDENING: add section_context when section filtering is in use
    if onenote_section:
        plan["section_context"] = build_section_context(onenote_root.name, onenote_section)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.json"
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote plan: {out_path}")


if __name__ == "__main__":
    main()
