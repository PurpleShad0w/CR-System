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
    s = re.sub(r"[^a-z0-9\s\+\-]", " ", s)
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


# Option A: deterministic title normalization
TITLE_MAP_EXACT = {
    'ge': 'Groupe électrogène',
    'groupes froids': 'Groupes froids',
    'vrv': 'Système VRV',
    'tgbt': 'TGBT',
    'local autocom': 'Local autocom',
    'local eau': 'Local eau',
    'extracteurs': 'Extracteurs',
    'supervision': 'Supervision GTB',
    'coffret gtb': 'Coffret GTB',
}


def normalize_title(raw: str) -> str:
    s = (raw or '').strip()
    if not s:
        return ''
    s = re.sub(r"\?{2,}", "", s).strip()
    s = re.sub(r"\s{2,}", " ", s).strip()
    low = _norm(s)
    if low in TITLE_MAP_EXACT:
        return TITLE_MAP_EXACT[low]
    m = re.match(r"^(cta)\s*([0-9]+)\s*(.*)$", s, flags=re.IGNORECASE)
    if m:
        num = m.group(2)
        tail = (m.group(3) or '').strip()
        head = f"CTA {num}"
        if tail:
            tail = tail.replace(' - ', ' – ').replace('-', ' – ')
            return f"{head} – {tail}" if '–' not in tail else f"{head} {tail}"
        return head
    m = re.match(r"^(td)\s*(.*)$", s, flags=re.IGNORECASE)
    if m:
        tail = (m.group(2) or '').strip()
        tail = re.sub(r"\brdc\b", "RDC", tail, flags=re.IGNORECASE)
        tail = re.sub(r"\bsous\s*sol\b", "Sous-sol", tail, flags=re.IGNORECASE)
        tail = re.sub(r"\bniveau\b", "Niveau", tail, flags=re.IGNORECASE)
        if tail:
            return f"Tableau divisionnaire – {tail}"
        return "Tableau divisionnaire"
    m = re.match(r"^(cta)([0-9]+)$", s, flags=re.IGNORECASE)
    if m:
        return f"CTA {m.group(2)}"
    if len(s) <= 4:
        return s.upper()
    words = s.split(' ')
    out = []
    for w in words:
        wl = w.lower()
        if wl in ('gtb', 'tgbt', 'vrv', 'knx'):
            out.append(w.upper())
        elif re.match(r"^r\+\d+$", wl):
            out.append(w.upper())
        else:
            out.append(w[:1].upper() + w[1:])
    return ' '.join(out).strip()


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


FINAL_MAX_BULLETS = 8
FINAL_MAX_CHARS = 230
FALLBACK_MAX_FACTS = 10

INSUFFICIENT_PHRASES = (
    "rien a signaler",
    "notes insuffisantes",
    "pas assez de sources",
    "source insuffisante",
)

ASR_REPLACEMENTS = [
    # Corrections prudentes, métier, avec faible risque de contresens.
    (r"\bjtb\b", "GTB"),
    (r"\bgtbs\b", "GTB"),
    (r"\bgtb\b", "GTB"),
    (r"\btgbt\b", "TGBT"),
    (r"\bvrv\b", "VRV"),
    (r"\bcta\b", "CTA"),
    (r"\bbacs\b", "BACS"),
    (r"\ba[eé]roterm(?:e|es)?\b", "aérothermes"),
    (r"\ba[eé]rotherm(?:e|es)?\b", "aérothermes"),
    (r"\br-plus-1\b", "R+1"),
    (r"\br\+1\b", "R+1"),
    (r"\brez[- ]?de[- ]?chauss[ée]e\b", "rez-de-chaussée"),
]


def normalize_asr_text(text: str) -> str:
    """Nettoyage léger des transcriptions audio avant extraction de faits.

    Le but n'est pas de réécrire le fond, seulement de stabiliser les termes
    techniques récurrents pour aider la synthèse.
    """
    t = text or ""
    for pattern, repl in ASR_REPLACEMENTS:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def strip_bullet_prefix(s: str) -> str:
    s = (s or "").strip()
    while s.startswith("- "):
        s = s[2:].strip()
    while s.startswith("•"):
        s = s[1:].strip()
    return s.strip()


def is_insufficient_text(text: str) -> bool:
    n = _norm(text or "")
    return any(p in n for p in INSUFFICIENT_PHRASES)


def is_noise_line(s: str) -> bool:
    n = _norm(s or "")
    if not n:
        return True
    if n in {"page", "texte", "texte detail", "info cle", "user flow"}:
        return True
    if "page_id" in n:
        return True
    if n.startswith("preuve"):
        return True
    if is_insufficient_text(n):
        return True
    return False


def is_probably_title_line(line: str, title: str) -> bool:
    if not line or not title:
        return False
    return _norm(strip_bullet_prefix(line)) == _norm(title)


def split_long_line_into_sentences(line: str) -> list:
    """Découpe une longue transcription en phrases sans être trop agressif."""
    line = normalize_asr_text(strip_bullet_prefix(line))
    if not line:
        return []

    # Découpe principale sur ponctuation forte.
    chunks = re.split(r"(?<=[.!?])\s+", line)
    out: list[str] = []

    for c in chunks:
        c = c.strip()
        if not c:
            continue

        # Si une phrase reste très longue, on découpe aussi sur certains connecteurs.
        if len(c) > 320:
            parts = re.split(
                r"\s+(?:et|mais|tandis que|ce qui|donc|dans ce cadre|par conséquent)\s+",
                c,
                flags=re.IGNORECASE,
            )
            for p in parts:
                p = p.strip(" ,;")
                if p:
                    out.append(p)
        else:
            out.append(c)

    return out


def extract_atomic_facts(raw_notes: str, *, title: str = "") -> list:
    """Transforme raw_text/raw_bullets en faits atomiques.
    Règles :
    - ne garde pas le titre seul comme fait,
    - ne garde pas les lignes de bruit,
    - découpe les longues transcriptions audio,
    - déduplique doucement,
    - ne compresse pas encore le contenu.
    """
    raw_notes = raw_notes or ""
    facts: list[str] = []

    for ln in normalize_whitespace(raw_notes).split("\n"):
        s = strip_bullet_prefix(ln)
        if not s:
            continue

        # Les anciennes chaînes peuvent contenir "Note vocale : ...".
        # On garde le contenu, mais pas le marqueur.
        if _norm(s).startswith("note vocale"):
            if ":" in s:
                s = s.split(":", 1)[1].strip()

        if is_noise_line(s):
            continue

        if is_probably_title_line(s, title):
            continue

        for sent in split_long_line_into_sentences(s):
            sent = sent.strip(" -•\t")
            if not sent:
                continue
            if is_noise_line(sent):
                continue
            if is_probably_title_line(sent, title):
                continue
            if len(_norm(sent)) < 12:
                continue
            facts.append(sent)

    # Déduplication souple.
    deduped: list[str] = []
    for f in facts:
        keep = True
        for prev in deduped:
            if overlap(prev, f) >= 0.88:
                keep = False
                break
        if keep:
            deduped.append(f)

    return deduped[:FALLBACK_MAX_FACTS]


def raw_source_is_rich(raw_notes: str, *, title: str = "") -> bool:
    facts = extract_atomic_facts(raw_notes, title=title)
    if len(facts) >= 2:
        return True
    if len(_norm(raw_notes or "")) >= 120:
        return False


def extract_title_and_bullets(model_text: str) -> Tuple[Optional[str], str]:
    if not model_text:
        return None, ''
    lines = [ln.strip() for ln in normalize_whitespace(model_text).split('\n') if ln.strip()]
    title_out: Optional[str] = None
    bullets: List[str] = []
    for ln in lines:
        low = ln.lower()
        if low.startswith('titre:') or low.startswith('title:'):
            title_out = normalize_title(ln.split(':', 1)[1].strip())
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
        if re.match(r"^[A-Za-zÀ-ÿ '\-]+\s*:\s+", s):
            continue
        if len(s) > FINAL_MAX_CHARS:
            s = s[: FINAL_MAX_CHARS - 1].rstrip() + "…"
        cleaned.append("- " + s)
    if len(cleaned) >= 2:
        cleaned = [b for b in cleaned if not is_insufficient_text(b)]

    cleaned = dedup_lines(cleaned, thresh=0.86, max_keep=FINAL_MAX_BULLETS)
    return title_out, "\n".join(cleaned)


def fallback_bullets_from_notes(
    raw_notes: str,
    *,
    title: str = "",
    max_bullets: int = FINAL_MAX_BULLETS,
) -> str:
    """Fallback non destructif.

    Objectif :
    - ne jamais écraser une source riche par "notes insuffisantes",
    - produire des puces lisibles directement depuis les faits atomiques,
    - préserver plus d'information que l'ancien fallback.
    """
    facts = extract_atomic_facts(raw_notes, title=title)

    if not facts:
        return "- Rien à signaler (notes insuffisantes)."

    bullets: list[str] = []

    for fact in facts:
        fact = normalize_asr_text(fact)
        fact = fact.strip(" .")
        if not fact:
            continue

        # On tolère des puces plus longues pour éviter la perte d'information.
        if len(fact) > FINAL_MAX_CHARS:
            fact = fact[: FINAL_MAX_CHARS - 1].rstrip() + "…"

        bullets.append("- " + fact)

        if len(bullets) >= max_bullets:
            break

    if not bullets:
        return "- Rien à signaler (notes insuffisantes)."

    return "\n".join(dedup_lines(bullets, thresh=0.88, max_keep=max_bullets))


def bullet_payloads(bullets: str) -> list[str]:
    out: list[str] = []
    for ln in normalize_whitespace(bullets or "").split("\n"):
        s = strip_bullet_prefix(ln)
        if s:
            out.append(s)
    return out


def content_overlap_score(source_facts: list[str], bullets: str) -> float:
    """Mesure simple de couverture.

    Pour chaque fait source, on regarde si au moins une puce couvre une partie
    raisonnable du vocabulaire. On ne cherche pas la perfection, seulement à
    détecter les sorties catastrophiques.
    """
    if not source_facts:
        return 1.0

    outs = bullet_payloads(bullets)
    if not outs:
        return 0.0

    covered = 0
    for fact in source_facts:
        best = 0.0
        for b in outs:
            best = max(best, overlap(fact, b))
        if best >= 0.28:
            covered += 1

    return covered / max(1, len(source_facts))


def llm_output_is_too_poor(raw_notes: str, model_bullets: str, *, title: str = "") -> bool:
    """Refuse les sorties LLM qui perdent trop d'information."""
    if not model_bullets.strip():
        return True

    if is_insufficient_text(model_bullets) and raw_source_is_rich(raw_notes, title=title):
        return True

    facts = extract_atomic_facts(raw_notes, title=title)
    outs = bullet_payloads(model_bullets)

    # Si la source a plusieurs faits, une seule puce générique est insuffisante.
    if len(facts) >= 3 and len(outs) <= 1:
        return True

    # Couverture minimale.
    # Le seuil est volontairement bas pour ne pas rejeter les bonnes reformulations.
    cov = content_overlap_score(facts, model_bullets)
    if len(facts) >= 3 and cov < 0.35:
        return True

    return False


def choose_best_bullets(raw_notes: str, model_bullets: str, *, title: str = "") -> str:
    """Accepte le LLM uniquement s'il conserve suffisamment la source."""
    if model_bullets and not llm_output_is_too_poor(raw_notes, model_bullets, title=title):
        return model_bullets

    return fallback_bullets_from_notes(raw_notes, title=title)


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


def prompt_slide_etat_des_lieux(*, section_name: str, title: str, raw_notes: str) -> str:
    raw_notes = (raw_notes or "").strip()
    section_name = (section_name or "").strip()
    title = (title or "").strip()

    return (
        "Tu rédiges une diapositive d'état des lieux GTB à partir de notes brutes.\n"
        "Objectif : produire des puces factuelles, cohérentes et complètes.\n\n"

        "PRIORITÉ ABSOLUE : NE PAS PERDRE D'INFORMATION.\n"
        "- Chaque fait utile des notes doit être conservé, soit tel quel, soit fusionné avec un fait proche.\n"
        "- Tu peux reformuler pour clarifier, mais tu ne dois pas supprimer un fait technique utile.\n"
        "- Si plusieurs faits concernent le même sujet, regroupe-les dans une puce cohérente.\n"
        "- Si les notes viennent d'audio et contiennent des erreurs de transcription, corrige prudemment les termes techniques évidents.\n"
        "- Ne crée aucun fait absent des notes.\n\n"

        "QUALITÉ RÉDACTIONNELLE :\n"
        "- Style professionnel, neutre et factuel.\n"
        "- Les puces d'une même slide doivent suivre un fil logique commun.\n"
        "- Évite les phrases incohérentes ou bruitées issues directement de l'ASR.\n"
        "- Harmonise les termes techniques sur toute la slide.\n"
        "- Ne crée pas de rubriques fixes comme 'Supervision:', 'Historisation:', 'Alarmes:', 'Comptage:'.\n\n"

        "RÈGLE SUR LES NOTES INSUFFISANTES :\n"
        "- Tu ne peux écrire 'Rien à signaler (notes insuffisantes)' QUE si les notes ne contiennent aucun fait exploitable.\n"
        "- Si les notes contiennent au moins deux faits, même imparfaits, tu dois produire des puces utiles.\n\n"

        "FORMAT DE SORTIE OBLIGATOIRE :\n"
        "Titre: <titre reformulé, style rapport>\n"
        "- <puce 1>\n"
        "- <puce 2>\n\n"

        "CONTRAINTES :\n"
        f"- 1 à {FINAL_MAX_BULLETS} puces maximum.\n"
        f"- Chaque puce doit faire au maximum {FINAL_MAX_CHARS} caractères.\n"
        "- Sortie uniquement au format demandé, sans explication ni préambule.\n\n"

        f"Site/section OneNote: {section_name}\n"
        f"Titre brut: {title}\n\n"
        "NOTES BRUTES SOURCE :\n"
        f"{raw_notes if raw_notes else '(vide)'}\n"
    )



def ensure_image_captions(slide: Dict[str, Any], fallback: str) -> None:
    """Ensure image captions are non-empty so legends can be rendered.
    If caption is missing/empty, set it to fallback.
    """
    imgs = slide.get('images')
    if not isinstance(imgs, list) or not fallback:
        return
    for im in imgs:
        if isinstance(im, dict):
            cap = (im.get('caption') or '').strip()
            if not cap:
                im['caption'] = fallback


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
        s['raw_title'] = title_raw
        s['title'] = normalize_title(title_raw)
        # ensure captions exist even if we don't call LLM
        ensure_image_captions(s, s['title'] or title_raw)
        raw_notes = (
            s.get("raw_text")
            or s.get("raw_bullets")
            or s.get("bullets")
            or s.get("body")
            or ""
        ).strip()

        raw_notes = normalize_asr_text(raw_notes)
        source_facts = extract_atomic_facts(raw_notes, title=title_raw)
        s["source_fact_count"] = len(source_facts)

        if not source_facts:
            s["raw_bullets"] = raw_notes
            s["bullets"] = "- Rien à signaler (notes insuffisantes)."
            s["final_bullet_count"] = 1
            s["coverage_score"] = 0.0
            continue
        prompt = prompt_slide_etat_des_lieux(section_name=section_name, title=title_raw, raw_notes=raw_notes)
        msg = build_messages(prompt, style_card=style_card)
        out, err = safe_chat(client, msg, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        if out:
            t_new, b_new = extract_title_and_bullets(out)

            if t_new:
                s["title"] = t_new

            ensure_image_captions(s, s["title"] or title_raw)

            final_bullets = choose_best_bullets(raw_notes, b_new, title=title_raw)

            s["raw_bullets"] = raw_notes
            s["bullets"] = final_bullets

            if b_new and final_bullets != b_new:
                s["llm_rejected_reason"] = "coverage_or_insufficient_output"
                s["llm_raw_output"] = out

        else:
            s["llm_error"] = err or "unknown"
            s["raw_bullets"] = raw_notes
            s["bullets"] = fallback_bullets_from_notes(raw_notes, title=title_raw)

        s["final_bullet_count"] = len(bullet_payloads(s.get("bullets") or ""))
        s["coverage_score"] = round(
            content_overlap_score(source_facts, s.get("bullets") or ""),
            3,
        )

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
    ap.add_argument('--max-tokens', type=int, default=700)
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
