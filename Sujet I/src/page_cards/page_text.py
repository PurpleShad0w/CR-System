#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Any, Dict, List

CAPTION_PREFIXES = (
    'légende:', 'legende:', 'legend:',
    'caption:',
    '#legende', '#légende', '#legend',
    '[legende]', '[légende]', '[legend]',
)


def _is_caption_line(s: str) -> bool:
    low = (s or '').strip().lower()
    return any(low.startswith(p) for p in CAPTION_PREFIXES)


def normalize_whitespace(text: str) -> str:
    t = (text or '').replace('\r\n', '\n').replace('\r', '\n')
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_markdown(text: str) -> str:
    if not text:
        return ''
    t = text
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"\*(.*?)\*", r"\1", t)
    t = re.sub(r"^\s*#+\s*", "", t, flags=re.MULTILINE)
    t = t.replace('•', '-')
    return t.strip()


def sanitize_client_body(text: str) -> str:
    if not text:
        return ''
    t = normalize_whitespace(strip_markdown(text))
    out: List[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            out.append('')
            continue
        low = s.lower()
        # Remove proof traces from generated text
        if 'page_id=' in low:
            continue
        if low.startswith('preuve:') or 'preuve:' in low:
            continue
        # Remove explicit caption lines from body text to prevent leakage into slide bullets
        if _is_caption_line(s):
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _collect_from_blocks(blocks: Any) -> List[str]:
    """Collect human-readable text from OneNote page JSON blocks.

    - Includes audio transcripts (key: 'transcript') so --transcribe matters downstream.
    - Excludes caption-tagged lines (Légende:/Caption:) so legends don't leak into bullets.
    """
    parts: List[str] = []
    if not isinstance(blocks, list):
        return parts

    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = (b.get('type') or '').lower()

        # Standard textual content
        text = b.get('text')
        if isinstance(text, str) and text.strip():
            # If it's an explicit caption line, do not treat as body text
            if _is_caption_line(text):
                continue
            if btype in ('paragraph', 'text', 'heading', 'list', 'bullet'):
                parts.append(text)
                continue
            parts.append(text)
            continue

        # Audio transcript
        if btype == 'audio':
            tr = b.get('transcript')
            if isinstance(tr, str) and tr.strip():
                parts.append(f"Note vocale : {tr.strip()}")
                continue
            meta = b.get('transcript_meta')
            if isinstance(meta, dict):
                status = (meta.get('status') or '').strip().lower()
                err = (meta.get('error') or '').strip()
                if status and status != 'ok':
                    msg = f"Note vocale : transcription {status}"
                    if err:
                        msg += f" ({err})"
                    parts.append(msg)
                    continue

        # Ignore images (handled elsewhere)
        if btype == 'image':
            continue

    return parts


def collect_text(page: Dict[str, Any]) -> str:
    if not isinstance(page, dict):
        return ''
    blocks = page.get('blocks')
    parts = _collect_from_blocks(blocks)
    return normalize_whitespace('\n'.join([p for p in parts if isinstance(p, str) and p.strip()]))


def to_bullets(text: str, *, max_lines: int = 10) -> str:
    body = sanitize_client_body(text)
    lines = [ln.strip() for ln in normalize_whitespace(strip_markdown(body)).split('\n') if ln.strip()]
    out: List[str] = []
    for ln in lines:
        s = ln
        while s.startswith('- '):
            s = s[2:].strip()
        if len(s) > 190:
            s = s[:187].rstrip() + '…'
        out.append(s)
        if len(out) > max_lines:
            out = out[:max_lines-1] + ['…']
            break
    if not out:
        return ''
    return "\n".join([f"- {x}" for x in out])
