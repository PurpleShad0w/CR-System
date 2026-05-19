#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tif', '.tiff'}


def _is_image_path(p: str) -> bool:
    try:
        return Path(p).suffix.lower() in IMG_EXTS
    except Exception:
        return False


def _as_image_dict(it: Any) -> Dict[str, str]:
    if isinstance(it, str):
        return {'path': it, 'caption': ''}
    if isinstance(it, dict):
        p = it.get('path') or it.get('file') or it.get('relpath') or it.get('name') or ''
        c = it.get('caption') or it.get('legend') or it.get('title') or ''
        return {'path': p, 'caption': c}
    return {'path': '', 'caption': ''}


def _push(out: List[Dict[str, str]], p: str, c: str = '') -> None:
    p = (p or '').strip()
    if not p:
        return
    if Path(p).suffix and not _is_image_path(p):
        return
    out.append({'path': p, 'caption': (c or '')})


def collect_images(page: Dict[str, Any]) -> List[Dict[str, str]]:
    """Collect image references from a OneNote page JSON.

    Sources supported:
    - page['assets']['images'] (process_onenote output)
    - blocks with type 'image' and key 'path'
    - legacy keys: page['images']

    This function intentionally does NOT guess captions from body text.
    Captions are resolved in image_selection.caption_from_blocks, based on explicit
    caption markers or OCR.
    """
    out: List[Dict[str, str]] = []
    if not isinstance(page, dict):
        return out

    assets = page.get('assets')
    if isinstance(assets, dict) and isinstance(assets.get('images'), list):
        for it in assets['images']:
            d = _as_image_dict(it)
            if d.get('path'):
                _push(out, d['path'], d.get('caption', ''))

    imgs = page.get('images')
    if isinstance(imgs, list):
        for it in imgs:
            d = _as_image_dict(it)
            if d.get('path'):
                _push(out, d['path'], d.get('caption', ''))

    blocks = page.get('blocks')
    if isinstance(blocks, list):
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if (b.get('type') or '').lower() != 'image':
                continue
            p = (b.get('path') or '').strip()
            if p:
                _push(out, p, (b.get('caption') or ''))

    # de-dup by path preserving order
    seen = set()
    dedup: List[Dict[str, str]] = []
    for im in out:
        p = (im.get('path') or '').strip()
        if not p or p in seen:
            continue
        seen.add(p)
        dedup.append(im)
    return dedup
