#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_llm_jobs.py

Adds a "humanization" editorial pass so slides stop looking like copy/paste.

What it does
------------
- Keeps your 4 axes (A/B/C/D) behavior (themes + deterministic inventory + diversity + Part2/3 enrichment).
- After slide bullets are built, rewrites them in a human tone:
  * paraphrase + contextualize
  * remove list-like duplication
  * keep facts (no new numbers/equipment)
- Anti-copy guard: if rewritten text overlaps too much with source (token overlap), it retries once with stronger constraints.
- If LLM is unavailable/fails, falls back to deterministic paraphrase templates.

CLI
---
--humanize/--no-humanize (default: humanize)

Safety
------
- Client deck never contains page_id= nor "Preuve:" lines.
- Internal trace keeps exhaustive evidence.

"""

from __future__ import annotations

import argparse
import json
import random
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# -----------------------------
# IO
# -----------------------------

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


# -----------------------------
# LLM helpers
# -----------------------------

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
        f"{base_prompt}\n\n---\n"
        "EVIDENCE_PACK (JSON) = source UNIQUE\n"
        "Tu dois utiliser UNIQUEMENT ce EVIDENCE_PACK pour rédiger la PARTIE 1 (État des lieux).\n"
        "Règles strictes:\n"
        "- Tu ne fais PAS de scoring (pas de classes A/B/C/D).\n"
        "- Tu ne proposes PAS de travaux.\n"
        "- Tu listes l'existant de manière structurée, en regroupant les faits.\n"
        "- Pour chaque constat important, ajoute une ligne 'Preuve:' (avec page_id + extrait).\n"
        "- Les questions/demandes vont dans '#### Points à confirmer'.\n\n"
        f"{payload}\n---\n"
    )


# -----------------------------
# Evidence extraction (client vs internal)
# -----------------------------

RE_EVIDENCE_LINE = re.compile(r"^(\s*[-•]?\s*)?(preuve\s*:|.*\bpage_id\s*=)", re.IGNORECASE)
RE_PAGE_ID = re.compile(r"\bpage_id\s*=\s*([^\s,;]+)", re.IGNORECASE)


def extract_client_text_and_trace(raw_text: str, *, origin: str, trace_start_index: int = 1) -> Tuple[str, List[Dict[str, Any]]]:
    if not raw_text:
        return '', []

    lines = raw_text.splitlines()
    out_lines: List[str] = []
    trace: List[Dict[str, Any]] = []
    p_idx = trace_start_index

    for ln in lines:
        s = ln.strip()
        if RE_EVIDENCE_LINE.search(s):
            ref = f"P{p_idx}"
            trace.append({"ref": ref, "origin": origin, "raw": s})
            p_idx += 1
            continue
        out_lines.append(ln)

    client_text = "\n".join(out_lines)
    client_text = re.sub(r"\s*Preuve\s*:\s*.*", "", client_text, flags=re.IGNORECASE)
    client_text = RE_PAGE_ID.sub("", client_text)
    client_text = re.sub(r"\[P\d+\]", "", client_text)
    client_text = re.sub(r"\n{3,}", "\n\n", client_text).strip()
    return client_text, trace


def strip_client_noise(text: str) -> str:
    if not text:
        return ''
    t = re.sub(r"\bConstat\s*:\s*", "", text, flags=re.IGNORECASE)
    t = re.sub(r"\[P\d+\]", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# -----------------------------
# OneNote aggregate reader
# -----------------------------

def _norm(s: str) -> str:
    s = (s or '').lower().strip()
    s = s.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('ë', 'e')
    s = s.replace('à', 'a').replace('â', 'a').replace('ä', 'a')
    s = s.replace('î', 'i').replace('ï', 'i')
    s = s.replace('ô', 'o').replace('ö', 'o')
    s = s.replace('û', 'u').replace('ü', 'u')
    s = re.sub(r"\s+", " ", s)
    return s


def locate_section_aggregate(bundle_path: Path, bundle: Dict[str, Any]) -> Optional[Path]:
    ctx = bundle.get('section_context') or {}
    notebook = ctx.get('onenote_notebook')
    slug = ctx.get('section_slug')
    if not notebook or not slug:
        return None
    # bundle_path is process/drafts/<case>/draft_bundle.json => project root is 4 parents up
    project_root = bundle_path.parent.parent.parent.parent
    cand = project_root / 'process' / 'onenote_aggregates' / str(notebook) / f"{slug}.json"
    if cand.exists():
        return cand
    return None


def aggregate_pages(agg: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = agg.get('pages') or []
    out = []
    for p in pages:
        pid = p.get('page_id') or p.get('id')
        title = p.get('title') or ''
        blocks = p.get('text_blocks') or []
        lines = []
        for b in blocks:
            txt = (b.get('text') or '').strip()
            if not txt:
                continue
            for ln in txt.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
        out.append({'page_id': pid, 'title': title, 'lines': lines, 'num_images': int(p.get('num_images') or 0)})
    return out


def extract_equipment_list(pages: List[Dict[str, Any]]) -> List[str]:
    best = None
    for p in pages:
        if 'liste des equipements' in _norm(p.get('title')):
            best = p
            break
    if best is None:
        for p in pages:
            blob = _norm(' '.join(p.get('lines') or []))
            if 'equipement' in blob or 'equipements' in blob:
                best = p
                break
    if best is None:
        return []

    items: List[str] = []
    for ln in best.get('lines') or []:
        s = ln.strip()
        if not s:
            continue
        if s.startswith(('-', '•')):
            s = s.lstrip('-•').strip()
        s = re.sub(r"^[0-9]+[)\.\-]\s*", "", s)
        # keep short-ish lines
        if 2 <= len(s) <= 120:
            items.append(s)
    seen = set()
    out = []
    for it in items:
        k = _norm(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def theme_definitions() -> List[Dict[str, Any]]:
    return [
        {'key': 'contexte', 'title': 'Contexte & objectifs', 'keywords': ['contexte', 'objectif', 'mission', 'decret', 'bacs', 'classe', 'emeis', 'idex', 'schneider', 'integrateur']},
        {'key': 'architecture', 'title': 'Architecture GTB / supervision', 'keywords': ['gtb', 'supervision', 'ebo', 'tac', 'vista', 'workstation', 'automate', 'bacnet', 'lon']},
        {'key': 'comptage', 'title': 'Comptage & suivi des consommations', 'keywords': ['comptage', 'diris', 'tgbt', 'kwh', 'compteur', 'energie', 'eau', 'calorie', 'frigorie']},
        {'key': 'alarmes', 'title': 'Alarmes & historisation', 'keywords': ['alarme', 'defaut', 'histor', 'derive', 'reporting', 'diagnostic']},
    ]


def extract_theme_bullets(pages: List[Dict[str, Any]], theme: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    kws = [_norm(k) for k in (theme.get('keywords') or [])]
    bullets: List[str] = []
    used_pages: List[str] = []
    for p in pages:
        pid = p.get('page_id')
        for ln in (p.get('lines') or []):
            n = _norm(ln)
            if any(k in n for k in kws):
                b = ln.strip()
                if len(b) > 170:
                    b = b[:167].rstrip() + '…'
                bullets.append(b)
                if pid and pid not in used_pages:
                    used_pages.append(pid)
    seen = set()
    out = []
    for b in bullets:
        k = _norm(b)
        if k in seen:
            continue
        seen.add(k)
        out.append(b)
    return out, used_pages


# -----------------------------
# Images (diversity)
# -----------------------------

def load_page_index(onenote_root: Path) -> Dict[str, Dict[str, Any]]:
    pages_dir = onenote_root / 'pages'
    idx: Dict[str, Dict[str, Any]] = {}
    if not pages_dir.exists():
        return idx
    for p in pages_dir.glob('*.json'):
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        pid = obj.get('page_id') or obj.get('metadata', {}).get('page_id')
        if pid:
            idx[str(pid)] = obj
    return idx


def short_caption(text: str) -> str:
    t = (text or '').strip()
    if not t:
        return ''
    t = t.split('\n', 1)[0].strip()
    if len(t) > 60:
        t = t[:57].rstrip() + '…'
    return t


def resolve_images_for_page_ids(page_index: Dict[str, Dict[str, Any]], page_ids: List[str], *, max_images: int = 6,
    per_page_cap: int = 2) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for pid in page_ids:
        obj = page_index.get(pid)
        if not obj:
            continue
        page_title = (obj.get('title') or obj.get('metadata', {}).get('title') or '').strip()
        assets = obj.get('assets') or {}
        images = assets.get('images') or []
        for im in images:
            if len(out) >= max_images:
                return out
            if counts.get(pid, 0) >= per_page_cap:
                break
            path = None
            cap = None
            if isinstance(im, str):
                path = im
            elif isinstance(im, dict):
                path = im.get('path') or im.get('file') or im.get('name')
                cap = im.get('caption') or im.get('alt') or im.get('title')
            if not cap:
                cap = page_title
            if path:
                out.append({'path': path, 'caption': short_caption(cap), 'page_id': pid, 'page_title': page_title})
                counts[pid] = counts.get(pid, 0) + 1
    return out


def diverse_page_order(pages: List[Dict[str, Any]], preferred: List[str]) -> List[str]:
    all_ids = [p.get('page_id') for p in pages if p.get('page_id')]
    seen = set()
    out = []
    for pid in preferred + all_ids:
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


# -----------------------------
# Slide utilities
# -----------------------------

def chunk_bullets(bullets: List[str], max_bullets: int = 10) -> List[str]:
    if not bullets:
        return ['']
    chunks = []
    for i in range(0, len(bullets), max_bullets):
        chunks.append('\n'.join(bullets[i:i + max_bullets]))
    return chunks


# -----------------------------
# Humanization layer
# -----------------------------

def tokenize_simple(s: str) -> List[str]:
    s = (s or '').lower()
    s = re.sub(r"[^a-z0-9àâäéèêëîïôöûüç\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split(' ') if t]


def overlap_ratio(src: str, out: str) -> float:
    """Jaccard-like overlap on tokens (rough)."""
    a = set(tokenize_simple(src))
    b = set(tokenize_simple(out))
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a))


def deterministic_rephrase_lines(lines: List[str]) -> List[str]:
    """Cheap paraphrase without LLM (keeps facts, adds minimal context)."""
    out = []
    for ln in lines:
        s = ln.strip().lstrip('-•').strip()
        if not s:
            continue
        # common transforms
        s = re.sub(r"^Le decret BACS", "Conformément au décret BACS,", s, flags=re.IGNORECASE)
        s = re.sub(r"^La GTB", "La GTB du site", s, flags=re.IGNORECASE)
        s = re.sub(r"^Aucune\s+", "À ce stade, aucune ", s, flags=re.IGNORECASE)
        out.append(s)
    return out


def llm_humanize_slide(client, *, title: str, bullets_text: str, section_ctx: Dict[str, Any],
    max_bullets: int = 8, temperature: float = 0.2, max_tokens: int = 300,
    retries: int = 2, base_sleep: float = 0.8) -> str:
    """Return rewritten bullet block (lines starting with '-')"""
    site = (section_ctx or {}).get('onenote_section_name') or ''
    prompt = (
        "Réécris ce contenu pour qu'il paraisse rédigé par un humain, sans copier-coller.\n"
        "Contraintes STRICTES:\n"
        "- N'invente aucun fait, aucune valeur, aucun équipement.\n"
        "- Tu peux reformuler et contextualiser (phrases de liaison).\n"
        "- Ne reprends pas plus de 3 mots consécutifs identiques à l'entrée.\n"
        "- Sortie: uniquement des puces commençant par '- '.\n"
        f"- Maximum {max_bullets} puces, chacune <= 160 caractères.\n\n"
        f"Contexte: site='{site}', slide='{title}'.\n\n"
        "Entrée (puces brutes):\n"
        f"{bullets_text.strip()}\n"
    )
    msg = build_messages(prompt)
    for attempt in range(retries + 1):
        resp, err = safe_chat(client, msg, temperature=temperature, max_tokens=max_tokens, top_p=1.0,
            retries=0, base_sleep=base_sleep)
        if resp is None:
            break
        out = (resp.text or '').strip()
        # keep only lines starting with '- '
        lines = [ln.strip() for ln in out.split('\n') if ln.strip().startswith('- ')]
        if not lines:
            continue
        out2 = '\n'.join(lines[:max_bullets])
        # anti-copy check
        if overlap_ratio(bullets_text, out2) <= 0.65:
            return out2
        # stronger re-prompt once
        msg = build_messages(prompt + "\nRappel: reformule davantage, sans reprendre les formulations exactes.\n")
        time.sleep(base_sleep)
    # fallback deterministic
    lines_in = [ln.strip() for ln in bullets_text.split('\n') if ln.strip()]
    ref = deterministic_rephrase_lines(lines_in)
    ref = [('- ' + r) if not r.startswith('- ') else r for r in ref]
    return '\n'.join(ref[:max_bullets])


def humanize_slides_inplace(client, slides: List[Dict[str, Any]], section_ctx: Dict[str, Any], *, enabled: bool) -> None:
    if not enabled:
        return
    for s in slides:
        # only humanize textual slides
        bul = (s.get('bullets') or s.get('body') or '').strip()
        if not bul:
            continue
        # keep sub-bullets indentation, but ensure it still begins with '-'
        lines = [ln.strip() for ln in bul.split('\n') if ln.strip()]
        # if not bullets, wrap
        if not any(ln.startswith('- ') for ln in lines):
            lines = [('- ' + ln) for ln in lines]
        bul_in = '\n'.join(lines)
        title = (s.get('title') or '').strip()
        bul_out = llm_humanize_slide(client, title=title, bullets_text=bul_in, section_ctx=section_ctx)
        s['bullets'] = bul_out


# -----------------------------
# Part 2/3 helpers
# -----------------------------

def group_keywords() -> Dict[str, List[str]]:
    return {
        'Chauffage': ['chaufferie', 'chaudiere', 'radiateur', 'pac', 'loi d', 'pompe'],
        'ECS': ['ecs', 'eau chaude', 'ballon', 'bouclage'],
        'Refroidissement': ['refroid', 'groupe froid', 'eau glacee', 'pac'],
        'Ventilation': ['cta', 'ventilation', 'vmc', 'soufflage', 'reprise', 'co2'],
        'Éclairage': ['eclairage', 'luminaire', 'detecteur de presence'],
        'Stores': ['store'],
        'GTB': ['gtb', 'supervision', 'ebo', 'tac', 'vista', 'automate', 'bacnet', 'lon'],
    }


def choose_group_images(pages: List[Dict[str, Any]], page_index: Dict[str, Dict[str, Any]], group: str, *, max_images: int = 2) -> List[Dict[str, Any]]:
    kws = [_norm(k) for k in group_keywords().get(group, [])]
    preferred = []
    for p in pages:
        pid = p.get('page_id')
        if not pid:
            continue
        blob = _norm(p.get('title', '') + ' ' + ' '.join(p.get('lines') or []))
        if any(k in blob for k in kws):
            preferred.append(pid)
    order = diverse_page_order(pages, preferred)
    return resolve_images_for_page_ids(page_index, order, max_images=max_images, per_page_cap=1)


def group_explanation(group: str, group_score: Dict[str, Any]) -> str:
    ach = (group_score or {}).get('achieved_class') or ''
    block_b = ((group_score or {}).get('blockers') or {}).get('B') or []
    nb = len(block_b)
    if not ach:
        return ''
    if nb:
        return f"Classe {ach} : {nb} exigence(s) minimales vers la classe B ne sont pas démontrées dans les preuves disponibles."
    return f"Classe {ach} : les exigences minimales vers la classe B semblent satisfaites sur la base des preuves disponibles."


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--max_tokens', type=int, default=1500)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--mode', choices=['single', 'multistep'], default='multistep')
    ap.add_argument('--min_quality', type=float, default=0.0)
    ap.add_argument('--quality', default='')
    ap.add_argument('--retries', type=int, default=4)
    ap.add_argument('--retry_sleep', type=float, default=1.2)
    ap.add_argument('--onenote', default='', help='Path to process/onenote/<notebook>')
    ap.add_argument('--bacs_rules', default='')
    ap.add_argument('--bacs_building_scope', default='Non résidentiel', choices=['Résidentiel', 'Non résidentiel'])
    ap.add_argument('--bacs_targets', default='')
    ap.add_argument('--bacs_part2_slides', action='store_true')
    ap.add_argument('--humanize', dest='humanize', action='store_true')
    ap.add_argument('--no-humanize', dest='humanize', action='store_false')
    ap.set_defaults(humanize=True)
    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    out_dir = Path(args.out) if args.out else bundle_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_json(bundle_path)
    if bundle.get('section_context'):
        require_section_context(bundle, 'draft_bundle.json')

    client = make_client()
    generated = dict(bundle)
    section_ctx = generated.get('section_context') or {}

    page_index = load_page_index(Path(args.onenote)) if args.onenote else {}

    # Load section aggregate if available
    agg_path = locate_section_aggregate(bundle_path, bundle)
    agg_obj = load_json(agg_path) if agg_path else None
    agg_pages = aggregate_pages(agg_obj) if isinstance(agg_obj, dict) else []

    # BACS scoring (deterministic)
    bacs_meta = None
    rules_doc = None
    observed_levels: Dict[str, Any] = {}
    group_scores: Dict[str, Any] = {}

    if (generated.get('report_type') == 'BACS_SCORING') and args.bacs_rules:
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
        bacs_meta = {'building_scope': args.bacs_building_scope, 'inferred': inferred, 'group_scores': group_scores}

    # LLM generation for Part 1 (kept)
    for sec in generated.get('sections', []):
        mp = sec.get('macro_part')
        base_prompt = (sec.get('prompt') or '').strip()
        if not base_prompt:
            sec['final_text'] = ''
            continue
        if generated.get('report_type') == 'BACS_SCORING' and mp in (2, 3) and rules_doc is not None:
            sec['final_text'] = ''
            sec['skipped_reason'] = 'deterministic_table6'
            continue
        pack = build_evidence_pack(sec, max_facts_per_topic=14)
        sec['evidence_pack'] = pack
        write_prompt = prompt_write_part1_from_pack(sec, pack) if mp == 1 else base_prompt
        resp, err = safe_chat(client, build_messages(write_prompt), temperature=args.temperature, max_tokens=args.max_tokens,
            top_p=args.top_p, retries=args.retries, base_sleep=args.retry_sleep)
        sec['final_text'] = (resp.text or '').strip() if resp else ''
        if err:
            sec['llm_error'] = err

    save_json(out_dir / 'generated_bundle.json', generated)

    # Assemble macro parts + trace
    assembled: Dict[str, Any] = {
        'case_id': generated.get('case_id'),
        'report_type': generated.get('report_type'),
        'section_context': section_ctx,
        'macro_parts': [],
        'slides': [],
        'evidence_trace': [],
    }
    internal_trace: Dict[str, Any] = {
        'case_id': generated.get('case_id'),
        'report_type': generated.get('report_type'),
        'section_context': section_ctx,
        'created_from': str(bundle_path),
        'evidence_trace': [],
        'full_text_by_section': [],
    }

    by_mp: Dict[int, List[Dict[str, Any]]] = {}
    for sec in generated.get('sections', []):
        by_mp.setdefault(int(sec.get('macro_part') or 0), []).append(sec)

    trace_cursor = 1
    for mp in sorted(by_mp.keys()):
        mp_name = by_mp[mp][0].get('macro_part_name')
        mp_obj = {'macro_part': mp, 'macro_part_name': mp_name, 'sections': []}
        for s in by_mp[mp]:
            raw = (s.get('final_text') or '').strip()
            origin = f"mp{mp}:{s.get('bucket_id') or 'SECTION'}"
            client_text, trace_items = extract_client_text_and_trace(raw, origin=origin, trace_start_index=trace_cursor)
            trace_cursor += len(trace_items)
            client_text = strip_client_noise(client_text)
            mp_obj['sections'].append({'bucket_id': s.get('bucket_id'), 'text': client_text})
            assembled['evidence_trace'].extend(trace_items)
            internal_trace['evidence_trace'].extend(trace_items)
            internal_trace['full_text_by_section'].append({'macro_part': mp, 'bucket_id': s.get('bucket_id'), 'raw_text': raw})
        assembled['macro_parts'].append(mp_obj)

    if bacs_meta:
        assembled['bacs_table6'] = bacs_meta
        internal_trace['bacs_table6'] = bacs_meta

    # -----------------------------
    # Build slides[] (4 axes)
    # -----------------------------
    slides: List[Dict[str, Any]] = []

    def add_div(part: int, title: str):
        slides.append({'type': 'PART_DIVIDER', 'part': part, 'title': title})

    ptitle = {1: 'Etat des lieux GTB', 2: 'Scoring GTB actuel', 3: 'Scoring projeté'}
    for mp in assembled['macro_parts']:
        try:
            ptitle[int(mp['macro_part'])] = mp.get('macro_part_name') or ptitle.get(int(mp['macro_part']))
        except Exception:
            pass

    # ---- Part 1 ----
    mp1 = next((x for x in assembled['macro_parts'] if int(x.get('macro_part', -1)) == 1), None)
    if mp1:
        add_div(1, ptitle.get(1, 'Etat des lieux GTB'))

        # B) deterministic inventory slide
        equipment_items = extract_equipment_list(agg_pages) if agg_pages else []
        inv_bullets = ['- ' + it for it in equipment_items[:24]]
        if not inv_bullets:
            inv_bullets = ["- À confirmer (liste d'équipements non trouvée dans les artefacts)"]
        preferred = [p.get('page_id') for p in agg_pages if 'liste des equipements' in _norm(p.get('title')) and p.get('page_id')]
        order = diverse_page_order(agg_pages, preferred) if agg_pages else []
        inv_imgs = resolve_images_for_page_ids(page_index, order, max_images=4, per_page_cap=1) if page_index and order else []
        slides.append({'type': 'CONTENT_TEXT_IMAGES' if inv_imgs else 'CONTENT_TEXT', 'part': 1, 'title': 'Inventaire / équipements mentionnés', 'bullets': '\n'.join(inv_bullets), 'images': inv_imgs})

        # A) theme slides from aggregate
        for th in theme_definitions():
            bullets_raw, used_pids = extract_theme_bullets(agg_pages, th) if agg_pages else ([], [])
            if not bullets_raw:
                continue
            bullets = ['- ' + b.lstrip('-• ').strip() for b in bullets_raw][:30]
            chunks = chunk_bullets(bullets, max_bullets=10)
            pid_order = diverse_page_order(agg_pages, used_pids) if agg_pages else []
            imgs = resolve_images_for_page_ids(page_index, pid_order, max_images=4, per_page_cap=1) if page_index and pid_order else []
            for ci, chunk in enumerate(chunks):
                slides.append({'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT', 'part': 1, 'title': th.get('title') + ('' if ci == 0 else f" ({ci+1})"), 'bullets': chunk, 'images': imgs if ci == 0 else []})

    # ---- Part 2 ----
    if bacs_meta and isinstance(group_scores, dict) and group_scores:
        add_div(2, ptitle.get(2, 'Scoring GTB actuel'))
        lines = [f"- {grp}: classe {sc.get('achieved_class')}" for grp, sc in group_scores.items()]
        slides.append({'type': 'CONTENT_TEXT', 'part': 2, 'title': 'Scorecard (synthèse)', 'bullets': '\n'.join(lines[:12]), 'images': []})
        for grp, sc in group_scores.items():
            expl = group_explanation(grp, sc)
            block_b = ((sc.get('blockers') or {}).get('B') or [])
            bul = []
            if expl:
                bul.append('- ' + expl)
            bul.append(f"- Classe atteinte : {sc.get('achieved_class')}")
            if block_b:
                bul.append('- Principaux écarts vers la classe B:')
                for b in block_b[:3]:
                    bul.append(f"  - {b.get('rule_id')} — {b.get('title')}")
            imgs = choose_group_images(agg_pages, page_index, grp, max_images=2) if (agg_pages and page_index) else []
            slides.append({'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT', 'part': 2, 'title': grp, 'bullets': '\n'.join(bul), 'images': imgs})

    # ---- Part 3 ----
    if bacs_meta and isinstance(group_scores, dict) and group_scores:
        add_div(3, ptitle.get(3, 'Scoring projeté'))
        synth = ['- Synthèse des écarts vers la cible (classe B)']
        for grp, sc in group_scores.items():
            block_b = ((sc.get('blockers') or {}).get('B') or [])
            synth.append(f"- {grp}: {len(block_b)} écart(s) vers classe B")
        slides.append({'type': 'CONTENT_TEXT', 'part': 3, 'title': 'Synthèse (chemin critique)', 'bullets': '\n'.join(synth), 'images': []})
        for grp, sc in group_scores.items():
            block_b = ((sc.get('blockers') or {}).get('B') or [])
            if not block_b:
                continue
            bul = ['- Priorité 1 (bloquants):']
            for b in block_b[:6]:
                bul.append(f"  - {b.get('rule_id')} — {b.get('title')}")
            imgs = choose_group_images(agg_pages, page_index, grp, max_images=2) if (agg_pages and page_index) else []
            slides.append({'type': 'CONTENT_TEXT_IMAGES' if imgs else 'CONTENT_TEXT', 'part': 3, 'title': f"Plan d'action — {grp}", 'bullets': '\n'.join(bul), 'images': imgs})

    # Apply humanization pass (the whole point)
    humanize_slides_inplace(client, slides, section_ctx, enabled=bool(args.humanize))

    assembled['slides'] = slides

    save_json(out_dir / 'assembled_report.json', assembled)
    save_json(out_dir / 'internal_trace.json', internal_trace)

    quality = evaluate_quality(generated, assembled)
    save_json(out_dir / 'quality_report.json', quality)

    print('Wrote:', out_dir / 'assembled_report.json')


if __name__ == '__main__':
    main()
