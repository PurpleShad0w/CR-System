#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_llm_jobs.py

Implements the report assembly and post-processing steps.

Key additions (client quality fixes)
-----------------------------------
1) CLIENT vs INTERNAL TRACE separation
   - Removes any raw OneNote identifiers from the client deck output.
   - Extracts lines starting with "Preuve:" or containing "page_id=" into an evidence trace.
   - Writes a separate internal trace artifact (internal_trace.json).

2) Slide schema generation (slides[])
   - Produces a structured slide list ("slides") so PPTX rendering is not pure pagination.
   - Keeps renderer "dumb": render_report_pptx.py simply transcribes slides[] -> PPTX.

3) Images plumbing
   - Optionally, when --onenote is provided, resolves 1-3 images from OneNote processed pages
     for each section and attaches them to slide specs.

4) Deterministic Context slides
   - Adds 1-2 context slides at start of macro-part 1.

5) BACS Part 2/3 client formatting
   - Builds a scorecard slide + per-group slides for Part 2.
   - Builds a prioritized action plan for Part 3.

6) Quality guardrails
   - The quality_score module is extended to hard-fail if any {{TOKEN}} or page_id= leaks
     into the client deck text.

"""

import argparse
import json
import random
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from llm_client import make_client
from quality_score import evaluate_quality
from section_context import require_section_context
from evidence_pack import build_evidence_pack
from bacs_scoring_from_bundle import (
    load_rules_json,
    build_digest_from_bundle,
    build_level_inference_prompt_from_digest,
    compute_group_scores_from_table6,
    parse_json_safely,
)

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def save_json(p: Path, obj: Dict[str, Any]):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


# -----------------------------------------------------------------------------
# LLM helpers
# -----------------------------------------------------------------------------

def build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "Tu es un rédacteur technique Build 4 Use. Respecte les contraintes et n'invente rien."},
        {"role": "user", "content": prompt},
    ]


def safe_chat(client, messages, *, temperature: float, max_tokens: int, top_p: float = 1.0,
              retries: int = 4, base_sleep: float = 1.2) -> Tuple[Optional[Any], Optional[str]]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat(messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p, stream=False)
            return resp, None
        except Exception as e:
            msg = str(e)
            last_err = msg
            transient = (
                "HF error 500" in msg
                or "Internal Server Error" in msg
                or "Unknown error" in msg
                or "Model too busy" in msg
                or "unable to get response" in msg
            )
            if attempt >= retries or not transient:
                break
            time.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.4))
    return None, last_err


def prompt_write_part1_from_pack(section: Dict[str, Any], pack: Dict[str, Any]) -> str:
    base_prompt = (section.get('prompt') or '').strip()
    payload = json.dumps(pack, ensure_ascii=False, indent=2)[:12000]
    return (
        f"{base_prompt}\n\n"
        "---\n"
        "EVIDENCE_PACK (JSON) = source UNIQUE\n"
        "Tu dois utiliser UNIQUEMENT ce EVIDENCE_PACK pour rédiger la PARTIE 1 (État des lieux).\n"
        "Règles strictes:\n"
        "- Tu ne fais PAS de scoring (pas de classes A/B/C/D).\n"
        "- Tu ne proposes PAS de travaux.\n"
        "- Tu listes l'existant de manière structurée, en regroupant les faits.\n"
        "- Pour chaque constat important, ajoute une ligne 'Preuve:' (avec page_id + extrait).\n"
        "- Les questions/demandes vont dans '#### Points à confirmer'.\n\n"
        "Format attendu:\n"
        "- Une phrase d'introduction (2-3 lignes) synthétisant l'état global.\n"
        "- Ensuite, pour chaque rubrique #### du plan: 3 à 12 puces 'Constat:' (si data), sinon 'À confirmer'.\n"
        "- Ajouter une sous-section '#### Inventaire / équipements mentionnés' si equipment_facts n'est pas vide.\n\n"
        f"{payload}\n"
        "---\n"
    )


# -----------------------------------------------------------------------------
# Evidence trace extraction (CLIENT vs INTERNAL)
# -----------------------------------------------------------------------------

RE_EVIDENCE_LINE = re.compile(r"^(\s*[-•]?\s*)?(preuve\s*:|.*\bpage_id\s*=)", re.IGNORECASE)
RE_PAGE_ID = re.compile(r"\bpage_id\s*=\s*([^\s,;]+)", re.IGNORECASE)


def extract_client_text_and_trace(raw_text: str, *, origin: str, trace_start_index: int = 1) -> Tuple[str, List[Dict[str, Any]]]:
    """Remove evidence lines from raw_text and return (client_text, trace_items).

    Heuristic:
    - Any line starting with "Preuve:" OR containing "page_id=" is moved to trace.
    - If an evidence line follows a bullet/Constat line, we append a short ref token [P#] to that previous line.
    """
    if not raw_text:
        return "", []

    lines = raw_text.splitlines()
    out_lines: List[str] = []
    trace: List[Dict[str, Any]] = []
    p_idx = trace_start_index

    for ln in lines:
        if RE_EVIDENCE_LINE.search(ln.strip()):
            ref = f"P{p_idx}"
            trace.append({
                "ref": ref,
                "origin": origin,
                "raw": ln.strip(),
            })
            # Attach ref to previous visible line when possible
            if out_lines:
                prev = out_lines[-1]
                if prev.strip():
                    # Avoid duplicating if already has a ref
                    if f"[{ref}]" not in prev:
                        out_lines[-1] = prev.rstrip() + f" [{ref}]"
            p_idx += 1
            continue
        out_lines.append(ln)

    # Remove any residual page_id fragments that could have leaked inline
    client_text = "\n".join(out_lines)
    client_text = re.sub(r"\s*Preuve\s*:\s*.*", "", client_text, flags=re.IGNORECASE)
    client_text = RE_PAGE_ID.sub("", client_text)

    # Clean extra blank lines
    client_text = re.sub(r"\n{3,}", "\n\n", client_text).strip()
    return client_text, trace


# -----------------------------------------------------------------------------
# Images resolution from processed OneNote pages
# -----------------------------------------------------------------------------


def load_page_index(onenote_root: Path) -> Dict[str, Dict[str, Any]]:
    """Build a map page_id -> page_json for quick lookup."""
    pages_dir = onenote_root / "pages"
    idx: Dict[str, Dict[str, Any]] = {}
    if not pages_dir.exists():
        return idx
    for p in pages_dir.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = obj.get("page_id") or obj.get("metadata", {}).get("page_id")
        if pid:
            idx[str(pid)] = obj
    return idx


def resolve_images_for_page_ids(page_index: Dict[str, Dict[str, Any]], page_ids: List[str], *, max_images: int = 3) -> List[str]:
    """Return up to max_images image paths from the given page_ids."""
    imgs: List[str] = []
    for pid in page_ids:
        obj = page_index.get(pid)
        if not obj:
            continue
        assets = obj.get("assets") or {}
        images = assets.get("images") or []
        for im in images:
            # processed pages may store images as strings or dicts
            if isinstance(im, str):
                path = im
            elif isinstance(im, dict):
                path = im.get("path") or im.get("file") or im.get("name")
            else:
                path = None
            if path:
                imgs.append(path)
            if len(imgs) >= max_images:
                return imgs
    return imgs[:max_images]


# -----------------------------------------------------------------------------
# Slide schema generation helpers
# -----------------------------------------------------------------------------


def normalize_ws(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def md_to_bullets(md: str, *, max_bullets: int = 12) -> str:
    """Extract bullet-ish lines from markdown-ish text."""
    md = normalize_ws(md)
    if not md:
        return ""
    bullets: List[str] = []
    for ln in md.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("####"):
            continue
        # keep bullets or 'Constat:' lines
        if s.startswith("-"):
            bullets.append(s if s.startswith("- ") else "- " + s.lstrip("-").strip())
        elif s.lower().startswith("constat"):
            bullets.append("- " + s)
        elif s.lower().startswith("à confirmer") or s.lower().startswith("a confirmer"):
            bullets.append("- À confirmer")
    if not bullets:
        # fallback: keep first lines as bullets
        for ln in md.splitlines()[:max_bullets]:
            s = ln.strip()
            if s and not s.startswith("#"):
                bullets.append("- " + s)
    return "\n".join(bullets[:max_bullets]).strip()


def split_by_h4(md: str) -> List[Tuple[str, str]]:
    """Return list of (title, body) split on #### headings."""
    md = normalize_ws(md)
    if not md:
        return []
    cur_title = None
    cur_lines: List[str] = []
    out: List[Tuple[str, str]] = []
    for ln in md.splitlines():
        s = ln.strip()
        if s.startswith("#### "):
            if cur_title is not None:
                out.append((cur_title, "\n".join(cur_lines).strip()))
            cur_title = s.replace("#### ", "").strip()
            cur_lines = []
        else:
            cur_lines.append(ln)
    if cur_title is not None:
        out.append((cur_title, "\n".join(cur_lines).strip()))
    return out


def worst_class(group_scores: Dict[str, Any]) -> str:
    order = {'D': 0, 'C': 1, 'B': 2, 'A': 3}
    cls = None
    for g, v in (group_scores or {}).items():
        c = (v or {}).get('achieved_class')
        if not c:
            continue
        if cls is None or order.get(c, -1) < order.get(cls, -1):
            cls = c
    return cls or 'D'


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True, help='Path to process/drafts/<case_id>/draft_bundle.json')
    ap.add_argument('--out', default='', help='Output folder (defaults to bundle folder)')
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--max_tokens', type=int, default=1500)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--mode', choices=['single','multistep'], default='multistep')
    ap.add_argument('--min_quality', type=float, default=0.0)
    ap.add_argument('--quality', default='')
    ap.add_argument('--retries', type=int, default=4)
    ap.add_argument('--retry_sleep', type=float, default=1.2)

    # Optional OneNote processed root for images (process/onenote/<notebook>)
    ap.add_argument('--onenote', default='', help='Optional path to process/onenote/<notebook> to resolve images')

    # BACS scoring inputs
    ap.add_argument('--bacs_rules', default='', help='Path to Tableau 6 rules JSON')
    ap.add_argument('--bacs_building_scope', default='Non résidentiel', choices=['Résidentiel','Non résidentiel'])
    ap.add_argument('--bacs_targets', default='', help='Optional JSON mapping group->target class for Part 3')
    ap.add_argument('--bacs_part2_slides', action='store_true', help='(legacy flag)')

    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    out_dir = Path(args.out) if args.out else bundle_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_json(bundle_path)
    if bundle.get('section_context'):
        require_section_context(bundle, 'draft_bundle.json')

    client = make_client()
    generated = dict(bundle)

    # Optional OneNote page index for images
    page_index: Dict[str, Dict[str, Any]] = {}
    if args.onenote:
        page_index = load_page_index(Path(args.onenote))

    # ------------------------------------------------------------------
    # Precompute BACS scoring from bundle if rules provided
    # ------------------------------------------------------------------
    bacs_meta = None
    rules_doc = None
    observed_levels: Dict[str, Any] = {}
    group_scores: Dict[str, Any] = {}
    target_by_group: Dict[str, str] = {}

    if (generated.get('report_type') == 'BACS_SCORING') and args.bacs_rules:
        try:
            rules_doc = load_rules_json(args.bacs_rules)
            digest = build_digest_from_bundle(generated)
            prompt = build_level_inference_prompt_from_digest(digest, rules_doc, args.bacs_building_scope)
            resp, err = safe_chat(client, build_messages(prompt), temperature=0.0, max_tokens=1900, top_p=1.0,
                                  retries=args.retries, base_sleep=args.retry_sleep)
            if resp is None:
                inferred = {'levels': {}, 'evidence': {}, 'unknowns': [f'LLM error: {err}']}
            else:
                obj, perr = parse_json_safely(resp.text or '')
                inferred = obj if obj is not None else {'levels': {}, 'evidence': {}, 'unknowns': [f'Parse error: {perr}']}
            observed_levels = inferred.get('levels') or {}
            group_scores = compute_group_scores_from_table6(rules_doc, args.bacs_building_scope, observed_levels)

            if args.bacs_targets:
                try:
                    target_by_group = load_json(Path(args.bacs_targets))
                except Exception:
                    target_by_group = {}

            bacs_meta = {
                'building_scope': args.bacs_building_scope,
                'inferred': inferred,
                'group_scores': group_scores,
                'targets': target_by_group,
            }
        except Exception as e:
            bacs_meta = {'error': str(e)}

    # ------------------------------------------------------------------
    # 1) Generate sections (LLM) – primarily Part 1
    # ------------------------------------------------------------------
    for sec in generated.get('sections', []):
        mp = sec.get('macro_part')
        base_prompt = (sec.get('prompt') or '').strip()
        if not base_prompt:
            sec['generated_text'] = ''
            sec['final_text'] = ''
            continue

        # Skip LLM generation for BACS Part 2/3 (we generate deterministic slides instead)
        if generated.get('report_type') == 'BACS_SCORING' and mp in (2, 3) and rules_doc is not None:
            sec['generated_text'] = ''
            sec['final_text'] = ''
            sec['skipped_reason'] = 'deterministic_table6_slides'
            continue

        # Build evidence pack (rich)
        pack = build_evidence_pack(sec, max_facts_per_topic=14)
        sec['evidence_pack'] = pack

        # Part 1 prompt
        if mp == 1:
            write_prompt = prompt_write_part1_from_pack(sec, pack)
        else:
            payload = json.dumps(pack, ensure_ascii=False, indent=2)[:12000]
            write_prompt = (
                f"{base_prompt}\n\n---\nEVIDENCE_PACK (JSON) = source UNIQUE\n"
                "Utilise uniquement ce pack. Respecte le plan ####.\n"
                f"{payload}\n---\n"
            )

        resp, err = safe_chat(client, build_messages(write_prompt), temperature=args.temperature,
                              max_tokens=args.max_tokens, top_p=args.top_p, retries=args.retries,
                              base_sleep=args.retry_sleep)
        if resp is None:
            sec['generated_text'] = ''
            sec['final_text'] = ''
            sec['llm_error'] = err
        else:
            text = (resp.text or '').strip()
            sec['generated_text'] = text
            sec['final_text'] = text

    gen_path = out_dir / 'generated_bundle.json'
    save_json(gen_path, generated)

    # ------------------------------------------------------------------
    # 2) Assemble report (client text + internal trace)
    # ------------------------------------------------------------------
    assembled: Dict[str, Any] = {
        'case_id': generated.get('case_id'),
        'report_type': generated.get('report_type'),
        'section_context': generated.get('section_context'),
        'render_mode': 'CLIENT_DECK',
        'macro_parts': [],
        'slides': [],
        'evidence_trace': [],
    }

    internal_trace: Dict[str, Any] = {
        'case_id': generated.get('case_id'),
        'report_type': generated.get('report_type'),
        'section_context': generated.get('section_context'),
        'created_from': str(bundle_path),
        'evidence_trace': [],
        'full_text_by_section': [],
    }

    # Group sections by macro_part
    by_mp: Dict[int, List[Dict[str, Any]]] = {}
    for sec in generated.get('sections', []):
        by_mp.setdefault(sec.get('macro_part'), []).append(sec)

    trace_cursor = 1
    for mp in sorted(by_mp.keys()):
        mp_name = by_mp[mp][0].get('macro_part_name')
        mp_obj = {
            'macro_part': mp,
            'macro_part_name': mp_name,
            'sections': [],
        }
        for s in by_mp[mp]:
            raw = (s.get('final_text') or s.get('generated_text') or '').strip()
            origin = f"mp{mp}:{s.get('bucket_id') or 'SECTION'}"
            client_text, trace_items = extract_client_text_and_trace(raw, origin=origin, trace_start_index=trace_cursor)
            trace_cursor += len(trace_items)

            mp_obj['sections'].append({
                'bucket_id': s.get('bucket_id'),
                'text': client_text,
            })

            assembled['evidence_trace'].extend(trace_items)
            internal_trace['evidence_trace'].extend(trace_items)
            internal_trace['full_text_by_section'].append({
                'macro_part': mp,
                'bucket_id': s.get('bucket_id'),
                'raw_text': raw,
            })

        assembled['macro_parts'].append(mp_obj)

    # Attach BACS meta if any
    if generated.get('report_type') == 'BACS_SCORING' and bacs_meta:
        assembled['bacs_table6'] = bacs_meta
        internal_trace['bacs_table6'] = bacs_meta

    # ------------------------------------------------------------------
    # 3) Build slides[] schema (client deck)
    # ------------------------------------------------------------------
    slides: List[Dict[str, Any]] = []

    def add_part_divider(part_num: int, title: str):
        slides.append({
            'type': 'PART_DIVIDER',
            'part': part_num,
            'title': title,
        })

    # Macro-part titles
    part_titles = {1: 'Partie 1', 2: 'Partie 2', 3: 'Partie 3'}
    for mp in assembled.get('macro_parts', []):
        try:
            part_titles[int(mp.get('macro_part'))] = mp.get('macro_part_name') or part_titles.get(int(mp.get('macro_part')))
        except Exception:
            pass

    # ---- Part 1: context slides + content ----
    mp1 = next((x for x in assembled.get('macro_parts', []) if int(x.get('macro_part', -1)) == 1), None)
    if mp1:
        add_part_divider(1, part_titles.get(1, 'Partie 1'))
        ctx = assembled.get('section_context') or {}
        site = ctx.get('onenote_section_name') or assembled.get('case_id') or 'N/A'
        notebook = ctx.get('onenote_notebook') or 'N/A'

        # Context slide 1
        slides.append({
            'type': 'CONTENT_TEXT',
            'part': 1,
            'title': 'Contexte',
            'bullets': "\n".join([
                f"- Site / section OneNote : {site}",
                f"- Notebook : {notebook}",
                "- Mission / type de rapport : " + (assembled.get('report_type') or 'N/A'),
                "- Date de génération : (voir page de garde)",
            ]),
            'images': [],
        })

        # Context slide 2 (objectifs/perimetre) – best effort
        obj_lines = [
            "- Objectifs : À confirmer (non trouvé de manière déterministe dans les artefacts)",
            "- Périmètre : À confirmer (installations auditées à lister depuis les notes terrain)",
        ]
        # If BACS scoring, we can at least mention target classes if provided
        if isinstance(bacs_meta, dict) and bacs_meta.get('targets'):
            tmap = bacs_meta.get('targets') or {}
            if isinstance(tmap, dict) and tmap:
                obj_lines[0] = "- Objectifs : Cibles de classe par groupe (selon configuration)"
                for g, t in list(tmap.items())[:6]:
                    obj_lines.append(f"  - {g}: classe {t}")
        slides.append({
            'type': 'CONTENT_TEXT',
            'part': 1,
            'title': 'Objectifs & périmètre',
            'bullets': "\n".join(obj_lines),
            'images': [],
        })

        # Content slides from mp1 sections
        for sec in (mp1.get('sections') or []):
            bucket = sec.get('bucket_id') or 'SECTION'
            text = sec.get('text') or ''
            # If no useful content, skip creating dedicated empty slide
            if not text or text.strip().lower() in ("aucune information disponible.", "aucune information disponible", "n/a"):
                continue

            # Split by #### headings
            blocks = split_by_h4(text)
            if not blocks:
                bullets = md_to_bullets(text)
                imgs = []
                # attach images from top_pages if possible
                # Find original generated section by bucket to get top_pages
                orig = next((s for s in (generated.get('sections') or []) if s.get('macro_part') == 1 and s.get('bucket_id') == bucket), None)
                if orig and page_index:
                    pids = [p.get('page_id') for p in (orig.get('top_pages') or []) if p.get('page_id')]
                    imgs = resolve_images_for_page_ids(page_index, pids, max_images=3)
                slides.append({
                    'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT',
                    'part': 1,
                    'title': bucket,
                    'bullets': bullets,
                    'images': imgs,
                })
            else:
                # put images on first block
                orig = next((s for s in (generated.get('sections') or []) if s.get('macro_part') == 1 and s.get('bucket_id') == bucket), None)
                imgs0 = []
                if orig and page_index:
                    pids = [p.get('page_id') for p in (orig.get('top_pages') or []) if p.get('page_id')]
                    imgs0 = resolve_images_for_page_ids(page_index, pids, max_images=3)
                for i, (h, body) in enumerate(blocks):
                    bullets = md_to_bullets(body)
                    if not bullets:
                        continue
                    imgs = imgs0 if i == 0 else []
                    slides.append({
                        'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT',
                        'part': 1,
                        'title': f"{bucket} — {h}",
                        'bullets': bullets,
                        'images': imgs,
                    })

    # ---- Part 2: scorecard + per-group ----
    if generated.get('report_type') == 'BACS_SCORING' and group_scores:
        add_part_divider(2, part_titles.get(2, 'Partie 2'))

        # Scorecard slide
        score_lines = []
        for grp, s in group_scores.items():
            score_lines.append(f"- {grp}: classe {s.get('achieved_class')}")
        slides.append({
            'type': 'CONTENT_TEXT',
            'part': 2,
            'title': 'Scorecard (synthèse)',
            'bullets': "\n".join(score_lines[:12]) or "- À confirmer",
            'images': [],
        })

        # Per-group slides (Top blockers + OK)
        if rules_doc is not None:
            rules_list = rules_doc.get('rules', [])

            # helper: requirement level
            def req_level(rule: Dict[str, Any], target_class: str) -> Optional[int]:
                req = (rule.get('class_requirements') or {}).get(args.bacs_building_scope) or {}
                v = req.get(target_class)
                return int(v) if v is not None else None

            def meets(rule: Dict[str, Any], target_class: str, lvl: Optional[int]) -> bool:
                r = req_level(rule, target_class)
                if r is None:
                    return True
                if lvl is None:
                    return False
                return int(lvl) >= r

            for grp, s in group_scores.items():
                blockers_b = ((s.get('blockers') or {}).get('B') or [])
                top_block = blockers_b[:3]

                # OK points: choose 2-3 rules in group that meet B
                ok_points = []
                for r in rules_list:
                    if (r.get('group') or 'Autres') != grp:
                        continue
                    rid = r.get('rule_id')
                    lvl = observed_levels.get(rid)
                    lvl = int(lvl) if isinstance(lvl, (int, float)) else None
                    if meets(r, 'B', lvl):
                        ok_points.append((rid, r.get('title')))
                    if len(ok_points) >= 3:
                        break

                bullets = [f"- Classe atteinte : {s.get('achieved_class')}" ]
                if top_block:
                    bullets.append("- Top 3 manquants bloquants (vers classe B):")
                    for b in top_block:
                        bullets.append(f"  - {b.get('rule_id')} — {b.get('title')}")
                if ok_points:
                    bullets.append("- Déjà en place (extraits):")
                    for rid, title in ok_points:
                        bullets.append(f"  - {rid} — {title}")

                slides.append({
                    'type': 'CONTENT_TEXT',
                    'part': 2,
                    'title': grp,
                    'bullets': "\n".join(bullets),
                    'images': [],
                })

    # ---- Part 3: action plan prioritized ----
    if generated.get('report_type') == 'BACS_SCORING' and group_scores and rules_doc is not None:
        add_part_divider(3, part_titles.get(3, 'Partie 3'))

        rules_list = rules_doc.get('rules', [])

        def req_level(rule: Dict[str, Any], target_class: str) -> Optional[int]:
            req = (rule.get('class_requirements') or {}).get(args.bacs_building_scope) or {}
            v = req.get(target_class)
            return int(v) if v is not None else None

        def meets(rule: Dict[str, Any], target_class: str, lvl: Optional[int]) -> bool:
            r = req_level(rule, target_class)
            if r is None:
                return True
            if lvl is None:
                return False
            return int(lvl) >= r

        # Synthesis slide
        synth = []
        overall = worst_class(group_scores)
        synth.append(f"- Classe globale (pire par groupe) : {overall}")
        synth.append("- Chemin critique (nombre d'écarts vers la cible):")
        for grp, s in group_scores.items():
            tgt = (target_by_group or {}).get(grp, 'B')
            blockers = ((s.get('blockers') or {}).get(tgt) or [])
            synth.append(f"  - {grp}: {len(blockers)} écart(s) vers classe {tgt}")
        slides.append({
            'type': 'CONTENT_TEXT',
            'part': 3,
            'title': 'Synthèse (avant / cible / chemin critique)',
            'bullets': "\n".join(synth),
            'images': [],
        })

        # Per-group action plan
        for grp in sorted({r.get('group') or 'Autres' for r in rules_list}):
            tgt = (target_by_group or {}).get(grp, 'B')
            missing = []
            for r in rules_list:
                if (r.get('group') or 'Autres') != grp:
                    continue
                rid = r.get('rule_id')
                lvl = observed_levels.get(rid)
                lvl = int(lvl) if isinstance(lvl, (int, float)) else None
                if not meets(r, tgt, lvl):
                    missing.append((rid, r.get('title')))

            if not missing:
                continue

            prio1 = missing[:6]
            prio2 = missing[6:10]
            bullets = [f"- Objectif : classe {tgt}", "- Priorité 1 (bloquants):"]
            for rid, title in prio1:
                bullets.append(f"  - {rid} — {title}")
            if prio2:
                bullets.append("- Priorité 2 (optimisation):")
                for rid, title in prio2:
                    bullets.append(f"  - {rid} — {title}")
            bullets.append("- Dépendances / points à confirmer : À confirmer")

            slides.append({
                'type': 'CONTENT_TEXT',
                'part': 3,
                'title': f"Plan d'action — {grp}",
                'bullets': "\n".join(bullets),
                'images': [],
            })

        # Decision-like conclusion slide (client)
        slides.append({
            'type': 'CONTENT_TEXT',
            'part': 3,
            'title': 'Conclusion (décisionnelle)',
            'bullets': "\n".join([
                "- Avant travaux : classe globale = " + worst_class(group_scores) + " (à interpréter selon périmètre)",
                "- Après travaux (cible) : classe(s) cible(s) = " + ("; ".join(sorted({(target_by_group or {}).get(g, 'B') for g in group_scores.keys()})) or 'B'),
                "- Risques / conditions : À confirmer (exceptions, périmètre décret, éclairage, etc.)",
                "- Top actions (extraits):",
            ]),
            'images': [],
        })

    assembled['slides'] = slides

    assembled_path = out_dir / 'assembled_report.json'
    save_json(assembled_path, assembled)

    internal_path = out_dir / 'internal_trace.json'
    save_json(internal_path, internal_trace)

    # ------------------------------------------------------------------
    # 4) Quality scoring
    # ------------------------------------------------------------------
    quality = evaluate_quality(generated, assembled)
    q_path = Path(args.quality) if args.quality else (out_dir / 'quality_report.json')
    save_json(q_path, quality)

    print(f"Wrote: {gen_path}")
    print(f"Wrote: {assembled_path}")
    print(f"Wrote: {internal_path}")
    print(f"Wrote: {q_path}")
    print(f"Quality total: {quality.get('total')} / 100")

    if args.min_quality and (quality.get('total', 0) < args.min_quality):
        raise SystemExit(2)

    # Hard fail in case quality added a hard_fail flag
    if quality.get('hard_fail'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
