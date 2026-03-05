#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_draft.py

Purpose
-------
Convert a per-case plan into LLM-ready section writing jobs:
- macro-parts 1-3 only
- per selected bucket: evidence from OneNote + prompt text

NEW FOLDER CONVENTION (project reorg)
-------------------------------------
- Inputs live under:   input/
- Pipeline artifacts:  process/
- Final reports:       output/

Inputs
------
process/plans/<case_id>.json
process/onenote/<notebook>/pages/*.json
input/config/prompt_templates.json
process/onenote_aggregates/<notebook>/<section_slug>.json (optional, recommended)

Outputs
-------
process/drafts/<case_id>/
  draft_bundle.json
  prompts.txt
  prompts/<bucket_id>.txt

Usage Example
-------------
python generate_draft.py --plan process/plans/P050011.json --onenote process/onenote/test --templates input/config/prompt_templates.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

# --- BACX INTENT RULES ------------------------------------
BACX_INTENT_BY_MACRO = {
    "État des lieux GTB": "DESCRIBE_EXISTING",
    "Scoring GTB actuel": "SCORE_EXISTING",
    "Scoring projeté": "PROJECT_FUTURE",
}

INTENT_RULES = {
    "DESCRIBE_EXISTING": {
        "forbidden": [
            "classe", "conforme", "non conforme",
            "iso", "52120",
            "objectif", "à atteindre",
            "projeté", "futur", "travaux",
        ],
        "note": (
            "Objectif : décrire strictement l’existant à partir des preuves OneNote. "
            "Aucune interprétation normative ni projection."
        ),
    },
    "SCORE_EXISTING": {
        "forbidden": [
            "travaux", "mise en œuvre",
            "objectif", "futur", "projeté",
        ],
        "note": (
            "Objectif : expliquer le scoring actuel selon l’ISO 52120‑1 "
            "à partir des capacités observées."
        ),
    },
    "PROJECT_FUTURE": {
        "forbidden": [],
        "note": (
            "Objectif : décrire une cible normative (classe B a minima). "
            "L’inférence est autorisée."
        ),
    },
}


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_onenote_pages(onenote_root: Path) -> List[Dict[str, Any]]:
    pages_dir = onenote_root / "pages"
    if not pages_dir.exists():
        raise SystemExit(f"pages/ not found under {onenote_root} (expected {pages_dir})")
    pages = []
    for p in sorted(pages_dir.glob("*.json")):
        try:
            pages.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return pages


def load_section_aggregate(agg_path: Path) -> Dict[str, Any]:
    if not agg_path or not agg_path.exists():
        return {}
    try:
        return json.loads(agg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_section_context_block(agg: Dict[str, Any], keywords: List[str], max_titles: int = 12) -> str:
    """
    Build a compact synthesis of the whole OneNote section:
    - inventory by equipment family
    - relevant titles for this bucket (keyword match)
    - reminders to reproduce inventories when evidence is list-like
    """
    if not agg:
        return ""
    sec_name = agg.get("onenote_section") or ""
    page_count = agg.get("page_count") or 0
    inv = agg.get("inventory") or []

    kw_norm = [norm(k) for k in (keywords or []) if k]

    lines = []
    lines.append(f"## Synthèse OneNote (section: {sec_name}, pages: {page_count})")
    lines.append("")
    lines.append("### Inventaire (par familles)")
    for it in inv[:12]:
        fam = it.get('family')
        cnt = it.get('count')
        lines.append(f"- {fam}: {cnt} élément(s)")
    lines.append("")

    # bucket-relevant titles
    lines.append("### Pages probablement pertinentes pour ce bucket (titres)")
    picked = []
    for it in inv:
        for t in (it.get("titles") or []):
            nt = norm(t)
            if not kw_norm or any(k in nt for k in kw_norm):
                picked.append(t)
            if len(picked) >= max_titles:
                break
        if len(picked) >= max_titles:
            break
    if not picked:
        lines.append("(Aucun titre détecté via mots-clés; se baser sur les preuves ci-dessous.)")
    else:
        for t in picked:
            lines.append(f"- {t}")
    lines.append("")

    lines.append("### Consigne de rédaction (synthèse → rapport)")
    lines.append("- Si l’évidence ressemble à un inventaire matériel/équipement, PRODUIS un inventaire similaire (liste structurée) dans la section.")
    lines.append("- Regroupe les observations répétées sur plusieurs pages en paragraphes de synthèse (éviter 'page par page').")
    lines.append("")
    return "\n".join(lines).strip()


def page_text_blocks(page: Dict[str, Any]) -> List[Tuple[str, str]]:
    out = []
    title = page.get("title") or ""
    if title:
        out.append(("title", title))
    for b in page.get("blocks", []) or []:
        t = b.get("type")
        if t in ("heading", "paragraph", "image_ocr"):
            txt = (b.get("text") or "").strip()
            if txt:
                out.append((t, txt))
        elif t == "audio":
            txt = (b.get("transcript") or "").strip()
            if txt:
                out.append(("audio_transcript", txt))
    return out


def extract_snippets(page: Dict[str, Any], keywords: List[str], max_snips: int = 8) -> List[str]:
    kws = [norm(k) for k in keywords if k]
    snips = []
    for t, txt in page_text_blocks(page):
        nt = norm(txt)
        hit = sum(1 for k in kws if k and k in nt)
        if hit > 0:
            snips.append(f"[{t}] {txt[:500]}")
            if len(snips) >= max_snips:
                break
    return snips


def build_evidence_block(case_id: str,
                         bucket_id: str,
                         keywords: List[str],
                         routed_pages: List[Dict[str, Any]],
                         pages_by_id: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    lines.append(f"- Bucket: {bucket_id}")
    lines.append(f"- Mots-clés: {', '.join(keywords)}")
    lines.append("")

    for rp in routed_pages:
        pid = rp.get("page_id")
        score = rp.get("score")
        page = pages_by_id.get(pid)
        if not page:
            continue
        title = page.get("title") or pid
        lines.append(f"## Page: {title} (page_id={pid}, score={score})")
        snips = extract_snippets(page, keywords, max_snips=10)
        if not snips:
            lines.append("(Aucun extrait direct trouvé; vérifier la page complète.)")
        else:
            for s in snips:
                lines.append(f"- {s}")
        lines.append("")

    return "\n".join(lines).strip()


def render_prompt(template_cfg: Dict[str, Any],
                  case_id: str,
                  report_type: str,
                  macro_part_num: int,
                  macro_part_name: str,
                  bucket_id: str,
                  evidence_block: str) -> str:
    fmt_lines = template_cfg["section_prompt_format"]
    txt = "\n".join(fmt_lines)
    return txt.format(
        case_id=case_id,
        report_type=report_type,
        macro_part_num=macro_part_num,
        macro_part_name=macro_part_name,
        bucket_id=bucket_id,
        evidence_block=evidence_block,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="Path to process/plans/<case_id>.json")
    ap.add_argument("--onenote", required=True, help="Path to process/onenote/<notebook> folder (contains pages/)")
    ap.add_argument("--templates", default="input/config/prompt_templates.json", help="Path to input/config/prompt_templates.json")
    ap.add_argument("--out", default="process/drafts", help="Output root under process/ (default: process/drafts)")
    ap.add_argument("--section-aggregate", default="", help="Optional path to process/onenote_aggregates/<notebook>/<section_slug>.json")
    args = ap.parse_args()

    plan = load_json(Path(args.plan))
    # Load optional section synthesis (recommended when report unit = OneNote section)
    agg = load_section_aggregate(Path(args.section_aggregate)) if args.section_aggregate else {}
    tpl = load_json(Path(args.templates))
    pages = load_onenote_pages(Path(args.onenote))

    case_id = plan.get("case_id") or Path(args.plan).stem
    report_type = plan.get("report_type")
    macro_parts = plan.get("macro_parts") or {}
    gen_parts = plan.get("generate_macro_parts") or [1, 2, 3]

    # Index pages by page_id for routing lookup
    pages_by_id = {}
    for p in pages:
        pid = p.get("page_id") or p.get("title")
        if pid:
            pages_by_id[pid] = p

    # Prepare output dirs
    out_root = Path(args.out) / case_id
    prompts_dir = out_root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "case_id": case_id,
        "report_type": report_type,
        "generate_macro_parts": gen_parts,
        "macro_parts": macro_parts,
        "sections": [],
    }

    all_prompts = []
    rt_templates = tpl.get("templates", {}).get(report_type, {})
    routing = plan.get("routing") or {}
    selected_by_part = plan.get("selected_buckets_by_macro_part") or {}

    # Iterate macro-parts 1..3 and create prompts per bucket
    for mp_str, mp_cfg in (macro_parts or {}).items():
        mp = int(mp_str)
        if mp not in gen_parts:
            continue

        mp_name = mp_cfg.get("name") if isinstance(mp_cfg, dict) else str(mp_cfg)

        mp_template = rt_templates.get(f"macro_part_{mp}", {})
        mp_instructions = mp_template.get("instructions", [])

        bucket_items = selected_by_part.get(str(mp)) or selected_by_part.get(mp) or []

        for bi in bucket_items:
            bucket_id = bi.get("bucket_id")
            if not bucket_id:
                continue

            r = routing.get(bucket_id) or {}
            keywords = r.get("keywords") or []
            top_pages = r.get("top_pages") or []
            # NEW: section-level context block (inventory + grouping guidance)
            section_ctx = build_section_context_block(agg, keywords)

            evidence = build_evidence_block(case_id, bucket_id, keywords, top_pages, pages_by_id)

            # prepend section synthesis before per-page evidence
            if section_ctx:
                evidence = section_ctx + "\n\n" + evidence

            intent = BACX_INTENT_BY_MACRO.get(mp_name)
            rules = INTENT_RULES.get(intent, {})
            forbidden = rules.get("forbidden", [])

            prompt = render_prompt(
                tpl,
                case_id=case_id,
                report_type=report_type,
                macro_part_num=mp,
                macro_part_name=mp_name,
                bucket_id=bucket_id,
                evidence_block=evidence,
            )

            # Prepend macro-part specific instructions
            if mp_instructions:
                header = "### Instructions macro-partie\n- " + "\n- ".join(mp_instructions) + "\n\n"
                prompt = header + prompt

            section_obj = {
                "macro_part": mp,
                "macro_part_name": mp_name,
                "bucket_id": bucket_id,
                "bucket_score": bi.get("score"),
                "keywords": keywords,
                "top_pages": top_pages,
                "evidence": evidence,
                "prompt": prompt,
            }
            bundle["sections"].append(section_obj)

            p_out = prompts_dir / f"{bucket_id}.txt"
            p_out.write_text(prompt, encoding="utf-8")
            all_prompts.append(f"\n\n===== {bucket_id} (Macro {mp}: {mp_name}) =====\n\n{prompt}")

    # Write outputs
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "draft_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "prompts.txt").write_text("\n".join(all_prompts).strip() + "\n", encoding="utf-8")

    print(f"Wrote: {out_root / 'draft_bundle.json'}")
    print(f"Wrote: {out_root / 'prompts.txt'}")
    print(f"Prompts directory: {prompts_dir}")


if __name__ == "__main__":
    main()
