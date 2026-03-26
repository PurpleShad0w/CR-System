#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load .env if available (same spirit as legacy run_llm_jobs.py)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

# Ensure SRC is early in sys.path (but we keep script dir too)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_client import make_client  # expects HF_TOKEN set; raises if missing [1](https://crsystem.sharepoint.com/sites/projets-build4use/Plateforme%20DS/Forms/DispForm.aspx?ID=20689&web=1)


# -----------------------------
# IO
# -----------------------------
def load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(p.read_text(encoding="utf-8-sig", errors="replace"))


def save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_style_card(repo_root: Path) -> str:
    p = repo_root / "input" / "config" / "style_card.md"
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


# -----------------------------
# Text helpers
# -----------------------------
def normalize_whitespace(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("é", "e").replace("è", "e").replace("ê", "e").replace("ë", "e")
    s = s.replace("à", "a").replace("â", "a").replace("ä", "a")
    s = s.replace("î", "i").replace("ï", "i")
    s = s.replace("ô", "o").replace("ö", "o")
    s = s.replace("û", "u").replace("ü", "u")
    s = s.replace("ç", "c")
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def token_set(s: str) -> set:
    return set([t for t in _norm(s).split(" ") if t])


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


def clean_bullets(text: str, *, max_lines: int = 8, max_chars: int = 170) -> str:
    if not text:
        return ""
    lines: List[str] = []
    for ln in normalize_whitespace(text).split("\n"):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("- "):
            s = s[2:].strip()
        elif s.startswith("•"):
            s = s.lstrip("•").strip()
        if not s:
            continue
        if len(s) > max_chars:
            s = s[: max_chars - 1].rstrip() + "…"
        lines.append("- " + s)
    lines = dedup_lines(lines, thresh=0.82, max_keep=max_lines)
    return "\n".join(lines)


# -----------------------------
# LLM helpers
# -----------------------------
def build_messages(prompt: str, *, style_card: str = "") -> List[Dict[str, str]]:
    sys_msg = "Tu es un rédacteur technique Build 4 Use. Style professionnel, neutre et factuel."
    if style_card:
        sys_msg += "\n\nSTYLE CARD (à respecter strictement):\n" + style_card
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]


def safe_chat(
    client,
    messages,
    *,
    temperature: float,
    max_tokens: int,
    top_p: float = 1.0,
    retries: int = 2,
    base_sleep: float = 1.0,
) -> Tuple[Optional[str], Optional[str]]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat(messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p, stream=False)
            return (resp.text or "").strip(), None
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
    raw_notes = (raw_notes or "").strip()
    section_name = (section_name or "").strip()
    title = (title or "").strip()

    lines: List[str] = []
    lines.append("Tu dois rédiger le texte d'une diapositive d'état des lieux GTB à partir de notes brutes.")
    lines.append("Objectif: décrire l'existant de façon claire et professionnelle, sans copier-coller les notes.")
    lines.append("")
    lines.append("Contraintes STRICTES:")
    lines.append("- Ne pas inventer de faits, valeurs, équipements ou acteurs absents des notes.")
    lines.append("- Si une info est incertaine/incomplète, ajouter une puce: 'À confirmer: ...'.")
    lines.append("- Sortie: uniquement des puces commençant par '- '.")
    lines.append("- 4 à 8 puces, chacune ≤ 170 caractères.")
    lines.append("- Pas d'emojis, pas de tutoiement, pas de style conversationnel.")
    lines.append("")
    if section_name:
        lines.append(f"Site/section OneNote: {section_name}")
    if title:
        lines.append(f"Titre de slide: {title}")
    lines.append("")
    lines.append("NOTES BRUTES (source):")
    lines.append(raw_notes if raw_notes else "(vide)")
    lines.append("")

    if image_captions:
        caps = [c.strip() for c in image_captions if (c or "").strip()]
        if caps:
            lines.append("CAPTIONS IMAGES (indices, optionnel):")
            for c in caps[:8]:
                lines.append(f"- {c}")
            lines.append("")

    lines.append("Rédige maintenant les puces finales.")
    return "\n".join(lines)


def humanize_assembled(
    assembled: Dict[str, Any],
    *,
    enabled: bool,
    style_card: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    sleep_s: float,
) -> Dict[str, Any]:
    if not enabled:
        return assembled

    section_name = ((assembled.get("section_context") or {}).get("onenote_section_name") or "").strip()
    slides = assembled.get("slides") or []
    if not isinstance(slides, list):
        return assembled

    client = make_client()  # will raise if HF_TOKEN missing [1](https://crsystem.sharepoint.com/sites/projets-build4use/Plateforme%20DS/Forms/DispForm.aspx?ID=20689&web=1)

    for s in slides:
        if not isinstance(s, dict):
            continue
        stype = (s.get("type") or "").strip()
        if stype == "PART_DIVIDER":
            continue

        title = (s.get("title") or "").strip()
        raw = (s.get("bullets") or s.get("body") or "").strip()
        if not raw:
            continue

        imgs = s.get("images") or []
        caps: List[str] = []
        if isinstance(imgs, list):
            for im in imgs:
                if isinstance(im, dict):
                    c = (im.get("caption") or "").strip()
                    if c:
                        caps.append(c)

        prompt = prompt_slide_etat_des_lieux(section_name=section_name, title=title, raw_notes=raw, image_captions=caps)
        messages = build_messages(prompt, style_card=style_card)

        out, err = safe_chat(
            client,
            messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=float(top_p),
            retries=2,
            base_sleep=1.0,
        )

        if out:
            clean = clean_bullets(out, max_lines=8, max_chars=170)
            if clean:
                s["raw_bullets"] = raw
                s["bullets"] = clean
        else:
            s["llm_error"] = err or "unknown"

        if sleep_s and sleep_s > 0:
            time.sleep(float(sleep_s))

    assembled["slides"] = slides
    return assembled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", required=True, help="Input assembled_page_cards.json")
    ap.add_argument("--out", default="", help="Output JSON (default: overwrite input)")
    ap.add_argument("--no-humanize", dest="humanize", action="store_false")
    ap.add_argument("--humanize", dest="humanize", action="store_true")
    ap.set_defaults(humanize=True)

    # HF runtime overrides (optional)
    ap.add_argument("--hf-token", default="", help="Optional: override HF_TOKEN for this run only.")
    ap.add_argument("--hf-model", default="", help="Optional: override HF_MODEL for this run only.")

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=380)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between calls (seconds)")
    args = ap.parse_args()

    # Allow passing token/model via CLI (without changing global env)
    if args.hf_token:
        os.environ["HF_TOKEN"] = str(args.hf_token).strip()
    if args.hf_model:
        os.environ["HF_MODEL"] = str(args.hf_model).strip()

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
    print("Wrote:", outp)


if __name__ == "__main__":
    main()