#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

CAPTION_PREFIXES = (
    'légende:', 'legende:', 'legend:',
    'caption:',
    '#legende', '#légende', '#legend',
    '[legende]', '[légende]', '[legend]',
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or '').strip()).lower()


def _is_caption_line(s: str) -> bool:
    low = _norm(s)
    return any(low.startswith(p) for p in CAPTION_PREFIXES)


def _strip_caption_prefix(s: str) -> str:
    if not s:
        return ''
    low = s.strip()
    for p in CAPTION_PREFIXES:
        if low.lower().startswith(p):
            return low[len(p):].strip(' :-\t')
    return low.strip()


def extract_keywords(title: str, bullets: str) -> List[str]:
    """Lightweight keyword extraction (no dependencies).

    Used only to score images, NOT to build captions.
    """
    text = f"{title or ''} {bullets or ''}".lower()
    text = re.sub(r"[^a-z0-9àâäéèêëîïôöùûüçœ\s-]", " ", text)
    toks = [t.strip('-') for t in re.split(r"\s+", text) if t.strip()]
    stop = {'le','la','les','des','de','du','un','une','et','ou','à','au','aux','dans','sur','pour','avec','sans','en','d','l'}
    out: List[str] = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in stop:
            continue
        if t not in out:
            out.append(t)
    return out[:18]


def _blocks(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    b = page.get('blocks')
    return b if isinstance(b, list) else []


def _find_image_block_id(blocks: List[Dict[str, Any]], img_path: str) -> str:
    img_path = (img_path or '').strip()
    if not img_path:
        return ''
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if (b.get('type') or '').lower() != 'image':
            continue
        if (b.get('path') or '').strip() == img_path:
            return (b.get('block_id') or '').strip()
    return ''


def _image_ocr_text(blocks: List[Dict[str, Any]], image_block_id: str) -> str:
    if not image_block_id:
        return ''
    texts: List[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if (b.get('type') or '').lower() != 'image_ocr':
            continue
        if (b.get('image_block_id') or '').strip() != image_block_id:
            continue
        t = b.get('text')
        if isinstance(t, str) and t.strip():
            texts.append(t.strip())
    return "\n".join(texts).strip()


def _explicit_caption_adjacent(blocks: List[Dict[str, Any]], img_path: str) -> str:
    """Return explicit caption if a caption-tagged text block immediately follows an image."""
    img_path = (img_path or '').strip()
    if not img_path:
        return ''
    for i, b in enumerate(blocks):
        if not isinstance(b, dict):
            continue
        if (b.get('type') or '').lower() != 'image':
            continue
        if (b.get('path') or '').strip() != img_path:
            continue
        # look ahead for the first textual block
        for j in range(i + 1, min(i + 4, len(blocks))):
            nb = blocks[j]
            if not isinstance(nb, dict):
                continue
            nt = (nb.get('type') or '').lower()
            txt = nb.get('text')
            if not isinstance(txt, str) or not txt.strip():
                continue
            if _is_caption_line(txt):
                return _strip_caption_prefix(txt)
            # stop if we hit a normal paragraph: we do NOT want body text to become caption
            if nt in ('paragraph', 'text', 'heading', 'list', 'bullet'):
                return ''
        return ''
    return ''


def caption_from_blocks(page: Dict[str, Any], img_path: str) -> str:
    """Infer caption for an image.

    Rule (to avoid leakage):
    1) Use ONLY explicit caption markers right under the image (Légende:/Caption:).
    2) Else fallback to OCR (image_ocr blocks).
    3) Else empty.
    """
    blocks = _blocks(page)
    cap = _explicit_caption_adjacent(blocks, img_path)
    if cap:
        return cap
    image_block_id = _find_image_block_id(blocks, img_path)
    ocr = _image_ocr_text(blocks, image_block_id)
    if ocr:
        # Keep OCR short-ish; first line is often best.
        first = ocr.splitlines()[0].strip()
        return first[:120]
    return ''


def score_images(images: List[Dict[str, str]], *, keywords: List[str]) -> List[Tuple[int, Dict[str, str]]]:
    """Deterministic scoring by keyword overlap with caption/path."""
    out: List[Tuple[int, Dict[str, str]]] = []
    for im in images:
        cap = _norm(im.get('caption') or '')
        path = _norm(im.get('path') or '')
        score = 0
        for kw in keywords:
            if not kw:
                continue
            if kw in cap:
                score += 3
            elif kw in path:
                score += 1
        out.append((score, im))
    out.sort(key=lambda t: (-t[0], len(t[1].get('path') or '')))
    return out


def select_best_images(page: Dict[str, Any], images: List[Dict[str, str]], *, title: str, bullets: str, max_images: int) -> List[Dict[str, str]]:
    """Select images, filling missing captions safely.

    Important: we do NOT turn bullet text into captions.
    """
    if not images:
        return []

    # Fill missing captions from explicit caption markers or OCR only
    for im in images:
        if not (im.get('caption') or '').strip():
            cap = caption_from_blocks(page, im.get('path') or '')
            if cap:
                im['caption'] = cap

    kws = extract_keywords(title, bullets)
    ranked = score_images(images, keywords=kws)
    chosen = [im for _sc, im in ranked[:max_images]]

    # Ensure stable output format
    out: List[Dict[str, str]] = []
    for im in chosen:
        out.append({'path': (im.get('path') or '').strip(), 'caption': (im.get('caption') or '').strip()})
    return [d for d in out if d.get('path')]
