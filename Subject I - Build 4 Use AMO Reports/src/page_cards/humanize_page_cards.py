#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Like legacy: load .env if present (HF_TOKEN, HF_MODEL, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_client import make_client


# -----------------------------
# IO
# -----------------------------

def load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except UnicodeDecodeError:
        return json.loads(p.read_text(encoding='utf-8-sig', errors='replace'))


def save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def load_style_card(repo_root: Path) -> str:
    p = repo_root / 'input' / 'config' / 'style_card.md'
    if not p.exists():
        return ''
    try:
        return p.read_text(encoding='utf-8').strip()
    except Exception:
        return ''


# -----------------------------
# Normalization & dedup
# -----------------------------

def normalize_whitespace(text: str) -> str:
    t = (text or '').replace('\r\n', '\n').replace('\r', '\n')
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _norm(s: str) -> str:
    s = (s or '').lower().strip()
    s = s.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('ë', 'e')
    s = s.replace('à', 'a').replace('â', 'a').replace('ä', 'a')
    s = s.replace('î', 'i').replace('ï', 'i')
    s = s.replace('ô', 'o').replace('ö', 'o')
    s = s.replace('û', 'u').replace('ü', 'u')
    s = s.replace('ç', 'c')
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def token_set(s: str) -> set:
    return set([t for t in _norm(s).split(' ') if t])


def overlap(a: str, b: str) -> float:
    A = token_set(a)
    B = token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, min(len(A), len(B)))


def dedup_lines(lines: List[str], *, thresh: float = 0.82, max_keep: int = 8) -> List[str]:
    out: List[str] = []
    for ln in lines:
        keep = True
        for prev in out:
            if overlap(prev, ln) >= thresh:
                keep = False
                break
        if keep:
            out.append(ln)
        if len(out) >= max_keep:
            break
    return out


# -----------------------------
# STRICT bullet extraction (prevents "voici les puces..." etc.)
# -----------------------------

LLM_META_PREFIXES = (
    'voici les puces',
    'voici les bullets',
    'puces finales',
    "j'ai respecte",
    "j’ai respecte",
    'contraintes',
    'en sortie',
    'reponse',
    "je n'ai pas",
    "je n’ai pas",
)


def extract_title_and_bullets(model_text: str) -> Tuple[Optional[str], str]:
    """Expected output:
    Titre: ...
    - ...
    - ...
    Everything else is ignored.
    """
    if not model_text:
        return None, ''

    lines = [ln.strip() for ln in normalize_whitespace(model_text).split('\n') if ln.strip()]
    title_out: Optional[str] = None
    bullets: List[str] = []

    for ln in lines:
        low = ln.lower()
        if low.startswith('titre:') or low.startswith('title:'):
            title_out = ln.split(':', 1)[1].strip()
            title_out = re.sub(r"\?{2,}", "", title_out).strip()
            title_out = re.sub(r"\s{2,}", " ", title_out).strip()
            continue

        if ln.startswith('- '):
            payload = ln[2:].strip()
            if payload:
                bullets.append('- ' + payload)
            continue
        if ln.startswith('•'):
            payload = ln.lstrip('•').strip()
            if payload:
                bullets.append('- ' + payload)
            continue

        if any(pfx in low for pfx in LLM_META_PREFIXES):
            continue

    cleaned: List[str] = []
    for b in bullets:
        s = b[2:].strip()
        if not s:
            continue
        # forbid rigid "Field: ..." patterns (avoid incoherence)
        if re.match(r"^[A-Za-zÀ-ÿ '\-]+\s*:\s+", s):
            # keep only if the colon is part of a genuine phrase (rare); safer to drop
            continue
        if len(s) > 170:
            s = s[:169].rstrip() + '…'
        cleaned.append('- ' + s)

    cleaned = dedup_lines(cleaned, thresh=0.82, max_keep=6)
    return title_out, '\n'.join(cleaned)


def fallback_bullets_from_notes(raw_notes: str, image_captions: List[str]) -> str:
    """If little/no text, do NOT invent. Return "Rien à signaler"."""
    n = _norm(raw_notes or '')
    if len(n) < 25:
        return '- Rien à signaler (notes insuffisantes).'

    items: List[str] = []
    for ln in normalize_whitespace(raw_notes).split('\n'):
        s = ln.strip()
        if not s:
            continue
        if 'Texte Détail' in s or 'Info Clé' in s:
            continue
        items.append(s)

    if not items:
        return '- Rien à signaler (notes insuffisantes).'

    out = []
    for it in items[:4]:
        it = it.strip()
        if len(it) > 170:
            it = it[:169].rstrip() + '…'
        out.append('- ' + it)
    out = dedup_lines(out, thresh=0.82, max_keep=4)
    return '\n'.join(out)


# -----------------------------
# LLM helpers
# -----------------------------

def build_messages(prompt: str, *, style_card: str = '') -> List[Dict[str, str]]:
    sys_msg = "Tu es un rédacteur technique Build 4 Use. Style professionnel, neutre et factuel."
    if style_card:
        sys_msg += "\n\nSTYLE CARD (à respecter strictement):\n" + style_card
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]


def safe_chat(client, messages, *, temperature: float, max_tokens: int, top_p: float = 1.0,
    retries: int = 2, base_sleep: float = 1.0) -> Tuple[Optional[str], Optional[str]]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat(messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p, stream=False)
            return (resp.text or '').strip(), None
        except Exception as e:
            last_err = str(e)
            transient = (
                "HF error 500" in last_err
                or "Internal Server Error" in last_err
                or "Model too busy" in last_err
                or "Unknown error" in last_err
            )
            if attempt >= retries or not transient:
                break
            time.sleep(base_sleep * (2 ** attempt))
    return None, last_err


def prompt_slide_etat_des_lieux(*, section_name: str, title: str, raw_notes: str, image_captions: List[str]) -> str:
    """Strict format. No fixed headings (Supervision/Historisation/etc.)."""
    raw_notes = (raw_notes or '').strip()
    section_name = (section_name or '').strip()
    title = (title or '').strip()

    caps = [c.strip() for c in (image_captions or []) if (c or '').strip()]
    caps_block = ''
    if caps:
        caps_block = 'CAPTIONS IMAGES (indices, optionnel):\n' + '\n'.join([f"- {c}" for c in caps[:8]]) + '\n'

    return (
        "Tu rédiges une diapositive d'état des lieux GTB à partir de notes brutes.\n"
        "Objectif: décrire l'existant de manière factuelle et sobre, sans sur-interprétation.\n\n"
        "RÈGLES STRICTES:\n"
        "- Ne pas inventer de faits/valeurs/équipements/sujets absents des notes.\n"
        "- NE PAS créer de rubriques fixes (interdit: 'Supervision:', 'Historisation:', 'Alarmes:', 'Comptage:', etc.).\n"
        "- Si les notes sont trop pauvres: produire UNE SEULE puce: 'Rien à signaler (notes insuffisantes)'.\n"
        "- Interdit: 'voici les puces', 'j'ai respecté', explications, préambules.\n"
        "- Sortie: uniquement des puces commençant par '- '.\n"
        "- 1 à 6 puces maximum, chacune ≤ 170 caractères.\n\n"
        "FORMAT DE SORTIE OBLIGATOIRE:\n"
        "Titre: <titre reformulé, 2 à 7 mots, style rapport>\n"
        "- <puce 1>\n"
        "- <puce 2>\n\n"
        f"Site/section OneNote: {section_name}\n"
        f"Titre brut (note): {title}\n\n"
        "NOTES BRUTES (source):\n"
        f"{raw_notes if raw_notes else '(vide)'}\n\n"
        + caps_block
    )


def humanize_assembled(assembled: Dict[str, Any], *, enabled: bool, style_card: str,
    temperature: float, max_tokens: int, top_p: float, sleep_s: float) -> Dict[str, Any]:
    if not enabled:
        return assembled

    section_name = ((assembled.get('section_context') or {}).get('onenote_section_name') or '').strip()
    slides = assembled.get('slides') or []
    if not isinstance(slides, list):
        return assembled

    client = make_client()

    for s in slides:
        if not isinstance(s, dict):
            continue
        if (s.get('type') or '').strip() == 'PART_DIVIDER':
            continue

        title_raw = (s.get('title') or '').strip()
        raw_notes = (s.get('bullets') or s.get('body') or '').strip()

        imgs = s.get('images') or []
        caps: List[str] = []
        if isinstance(imgs, list):
            for im in imgs:
                if isinstance(im, dict):
                    c = (im.get('caption') or '').strip()
                    if c:
                        caps.append(c)

        # If little/no text, do NOT call LLM, output "Rien à signaler".
        if len(_norm(raw_notes)) < 25:
            s['raw_title'] = title_raw
            s['title'] = re.sub(r"\?{2,}", "", title_raw).strip()
            s['raw_bullets'] = raw_notes
            s['bullets'] = '- Rien à signaler (notes insuffisantes).'
            continue

        prompt = prompt_slide_etat_des_lieux(section_name=section_name, title=title_raw, raw_notes=raw_notes, image_captions=caps)
        msg = build_messages(prompt, style_card=style_card)
        out, err = safe_chat(client, msg, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

        if out:
            t_new, b_new = extract_title_and_bullets(out)
            if t_new and 2 <= len(t_new) <= 80:
                s['raw_title'] = title_raw
                s['title'] = t_new
            if b_new:
                s['raw_bullets'] = raw_notes
                s['bullets'] = b_new
            else:
                s['raw_bullets'] = raw_notes
                s['bullets'] = fallback_bullets_from_notes(raw_notes, caps)
                s['llm_noncompliant'] = True
        else:
            s['llm_error'] = err or 'unknown'
            s['raw_bullets'] = raw_notes
            s['bullets'] = fallback_bullets_from_notes(raw_notes, caps)

        if sleep_s and sleep_s > 0:
            time.sleep(float(sleep_s))

    assembled['slides'] = slides
    return assembled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--assembled', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--no-humanize', dest='humanize', action='store_false')
    ap.add_argument('--humanize', dest='humanize', action='store_true')
    ap.set_defaults(humanize=True)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--max-tokens', type=int, default=380)
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--sleep', type=float, default=0.0)
    args = ap.parse_args()

    inp = Path(args.assembled)
    outp = Path(args.out) if args.out else inp

    assembled = load_json(inp)
    style_card = load_style_card(REPO_ROOT)

    assembled = humanize_assembled(
        assembled,
        enabled=bool(args.humanize),
        style_card=style_card,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        top_p=float(args.top_p),
        sleep_s=float(args.sleep),
    )

    save_json(outp, assembled)
    print('Wrote:', outp)


if __name__ == '__main__':
    main()
