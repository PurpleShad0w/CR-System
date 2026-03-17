#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""render_report_pptx.py (v5)

Fixes the "images still missing" root cause.

Your log shows the renderer *thinks* it embedded images:
  "Images: embedded 38 / requested 38".
But the produced PPTX still has no images.

The reason is structural: the renderer was building a deck (prs_out), adding pictures,
then creating a *second* presentation (prs_final) by cloning slide XML with deepcopy.
When slides contain pictures, copying raw XML does NOT copy the relationship parts
(ppt/media/* + slide rels). The result is a PPTX with PICTURE shapes referencing rId's
that do not exist and an empty ppt/media folder.

Fix:
- Do NOT rebuild a second presentation at the end.
- Save prs_out directly.
- Keep the TOC de-dup logic by preventing duplicate TOC insertion upstream.

Also keeps:
- repo root inference (process+input)
- image resolution (basename fallback)
- image normalization to PNG when needed
- bullet fit to shape (auto-size)

"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.util import Pt

from PIL import Image


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


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
        if 'page_id=' in s:
            continue
        low = s.lower()
        if low.startswith('preuve:') or 'preuve:' in low:
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def clone_slide(prs_out: Presentation, slide_in) -> Any:
    blank_layout = prs_out.slide_layouts[6]
    new_slide = prs_out.slides.add_slide(blank_layout)
    try:
        new_slide._element.get_or_add_bg()._set_bg(slide_in._element.get_or_add_bg())
    except Exception:
        pass
    for shape in slide_in.shapes:
        new_slide.shapes._spTree.insert_element_before(deepcopy(shape._element), 'p:extLst')
    return new_slide


def slide_text(slide) -> str:
    parts: List[str] = []
    for shape in slide.shapes:
        if getattr(shape, 'has_text_frame', False) and shape.has_text_frame:
            parts.append(shape.text_frame.text or '')
    return "\n".join(parts)


def replace_text_in_shape(shape, mapping: Dict[str, str]) -> None:
    if not getattr(shape, 'has_text_frame', False) or not shape.has_text_frame:
        return
    tf = shape.text_frame
    full = tf.text or ''
    replaced = full
    for k, v in mapping.items():
        if k in replaced:
            replaced = replaced.replace(k, v)
    if replaced != full:
        tf.text = replaced


def replace_placeholders(slide, mapping: Dict[str, str]) -> None:
    for shape in slide.shapes:
        replace_text_in_shape(shape, mapping)


def detect_template_slide_types(tpl: Presentation, slide_types_cfg: Dict[str, Any]) -> Dict[str, Any]:
    types = (slide_types_cfg.get('types') or {})
    found: Dict[str, Any] = {}
    for sl in tpl.slides:
        stxt = slide_text(sl)
        for tname, tcfg in types.items():
            toks = tcfg.get('detect_tokens') or []
            if toks and all(tok in stxt for tok in toks):
                found.setdefault(tname, sl)
    return found


def find_slide_by_predicate(tpl: Presentation, pred) -> Optional[Any]:
    for sl in tpl.slides:
        if pred(slide_text(sl)):
            return sl
    return None


def find_shapes_containing(slide, token: str) -> List[Any]:
    out = []
    for shape in slide.shapes:
        if getattr(shape, 'has_text_frame', False) and shape.has_text_frame:
            if token in (shape.text_frame.text or ''):
                out.append(shape)
    return out


def find_shapes_exact(slide, exact: str) -> List[Any]:
    out = []
    for shape in slide.shapes:
        if getattr(shape, 'has_text_frame', False) and shape.has_text_frame:
            if (shape.text_frame.text or '').strip() == exact:
                out.append(shape)
    return out


def infer_repo_root(assembled_path: Path) -> Path:
    start = assembled_path.resolve()
    for cand in [start.parent] + list(start.parents):
        if (cand / 'process').exists() and (cand / 'input').exists():
            return cand
        if cand.name.lower() == 'process' and (cand.parent / 'input').exists():
            return cand.parent
    return assembled_path.parent.parent.parent


_img_cache: Dict[str, Optional[Path]] = {}
_conv_cache: Dict[Path, Path] = {}


def _search_under(repo_root: Path, name: str) -> Optional[Path]:
    for sub in ('process', 'input'):
        base = repo_root / sub
        if not base.exists():
            continue
        try:
            for hit in base.rglob(name):
                if hit.is_file():
                    return hit
        except Exception:
            continue
    return None


def resolve_image_path(img: str, base_dir: Path, repo_root: Path) -> Optional[Path]:
    if not img:
        return None
    if img in _img_cache:
        return _img_cache[img]

    p = Path(img)
    if p.is_absolute() and p.exists():
        _img_cache[img] = p
        return p

    cand = (base_dir / img).resolve()
    if cand.exists():
        _img_cache[img] = cand
        return cand

    bn = p.name
    cand2 = (base_dir / bn).resolve()
    if cand2.exists():
        _img_cache[img] = cand2
        return cand2

    for r in (
        repo_root / 'process' / 'onenote',
        repo_root / 'input' / 'onenote-exporter',
        repo_root / 'input' / 'onenote-exporter' / 'output',
        repo_root / 'input',
    ):
        if not r.exists():
            continue
        c3 = r / img
        if c3.exists():
            _img_cache[img] = c3
            return c3
        c4 = r / bn
        if c4.exists():
            _img_cache[img] = c4
            return c4

    found = _search_under(repo_root, bn)
    _img_cache[img] = found
    return found


def magic_type(p: Path) -> str:
    try:
        b = p.read_bytes()[:16]
    except Exception:
        return 'unknown'
    if len(b) >= 3 and b[0:3] == b'\xFF\xD8\xFF':
        return 'jpeg'
    if len(b) >= 8 and b[0:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if len(b) >= 12 and b[0:4] == b'RIFF' and b[8:12] == b'WEBP':
        return 'webp'
    if len(b) >= 6 and (b[0:6] == b'GIF87a' or b[0:6] == b'GIF89a'):
        return 'gif'
    return 'unknown'


def normalize_image_for_ppt(p: Path, tmp_dir: Path) -> Optional[Path]:
    if not p.exists():
        return None
    if p in _conv_cache:
        return _conv_cache[p]
    kind = magic_type(p)
    ext = p.suffix.lower()
    if kind == 'jpeg' and ext in ('.jpg', '.jpeg'):
        _conv_cache[p] = p
        return p
    if kind == 'png' and ext == '.png':
        _conv_cache[p] = p
        return p
    try:
        im = Image.open(p)
        if im.mode not in ('RGB', 'RGBA'):
            im = im.convert('RGB')
        out = tmp_dir / (p.stem + '.png')
        im.save(out, format='PNG', optimize=True)
        _conv_cache[p] = out
        return out
    except Exception:
        return None


def images_to_pairs(images: List[Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for im in images or []:
        if isinstance(im, str):
            out.append((im, ''))
        elif isinstance(im, dict):
            out.append((im.get('path') or '', im.get('caption') or ''))
    return [(p, c) for p, c in out if p]


def _shape_area(sh) -> int:
    try:
        return int(sh.width) * int(sh.height)
    except Exception:
        return 0


def _pic_has_image_rel(sh) -> bool:
    try:
        _ = sh.image
        return True
    except Exception:
        return False


def _dedup_rects(shapes: List[Any]) -> List[Any]:
    seen = set(); out = []
    for sh in shapes:
        key = (int(sh.left), int(sh.top), int(sh.width), int(sh.height))
        if key in seen:
            continue
        seen.add(key)
        out.append(sh)
    return out


def pick_image_slots(slide, max_slots: int) -> List[Any]:
    pics = [sh for sh in slide.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE and not _pic_has_image_rel(sh)]
    pics = _dedup_rects(pics)
    pics.sort(key=_shape_area, reverse=True)
    return pics[:max_slots]


def overlay_images(slide, images: List[Any], base_dir: Path, repo_root: Path, tmp_dir: Path, stats: Dict[str, Any]) -> None:
    pairs = images_to_pairs(images)
    if not pairs:
        return
    slots = pick_image_slots(slide, max_slots=len(pairs))
    for idx, (path, _) in enumerate(pairs[:len(slots)]):
        stats['requested'] += 1
        ip = resolve_image_path(path, base_dir, repo_root)
        if not ip:
            stats['unresolved'].append(path)
            continue
        normp = normalize_image_for_ppt(ip, tmp_dir)
        if not normp:
            stats['unresolved'].append(path)
            continue
        ph = slots[idx]
        slide.shapes.add_picture(str(normp), ph.left, ph.top, width=ph.width, height=ph.height)
        stats['embedded'] += 1


def fill_legends(slide, images: List[Any]) -> None:
    pairs = images_to_pairs(images)
    legend_shapes = find_shapes_exact(slide, 'Légende Photo') + find_shapes_containing(slide, 'Légende Photo')
    # de-dup by id
    seen = set(); uniq = []
    for sh in legend_shapes:
        if id(sh) in seen:
            continue
        seen.add(id(sh)); uniq.append(sh)
    legend_shapes = uniq
    if not pairs:
        for sh in legend_shapes:
            try:
                sh.text_frame.text = ''
            except Exception:
                pass
        return
    for i, (_, cap) in enumerate(pairs[:len(legend_shapes)]):
        cap = (cap or '').strip()
        if len(cap) > 60:
            cap = cap[:57].rstrip() + '…'
        try:
            legend_shapes[i].text_frame.text = cap
        except Exception:
            pass


def normalize_bullet_lines(body: str, max_lines: int) -> List[str]:
    body = sanitize_client_body(body)
    lines = [ln.strip() for ln in normalize_whitespace(strip_markdown(body)).split('\n') if ln.strip()]
    out = []
    for ln in lines:
        s = ln
        if s.startswith('- '):
            s = s[2:].strip()
        while s.startswith('- '):
            s = s[2:].strip()
        if len(s) > 190:
            s = s[:187].rstrip() + '…'
        out.append(s)
    if len(out) > max_lines:
        out = out[:max_lines-1] + ['…']
    return out


def set_bullets_fit(shape, body: str, *, max_lines: int = 12, min_font_pt: int = 10) -> None:
    if not getattr(shape, 'has_text_frame', False) or not shape.has_text_frame:
        return
    tf = shape.text_frame
    lines = normalize_bullet_lines(body, max_lines=max_lines)
    for p in tf.paragraphs:
        p.text = ''
    if not lines:
        return
    for i, s in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = s
        p.level = 0
        p.alignment = PP_ALIGN.LEFT
    tf.word_wrap = True
    try:
        tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    except Exception:
        pass
    try:
        for p in tf.paragraphs:
            if p.font.size is not None and p.font.size < Pt(min_font_pt):
                p.font.size = Pt(min_font_pt)
    except Exception:
        pass


def is_toc_slide_text(txt: str) -> bool:
    t = txt or ''
    return ('Sommaire' in t) and ('01' in t or '01 –' in t or '01 -' in t)


def build_deck(base_template: Path, assembled_path: Path, out_path: Path, slide_types_path: Path,
    info_template: Optional[Path] = None, repo_root_override: Optional[Path] = None) -> None:
    base_tpl = Presentation(str(base_template))
    data = load_json(assembled_path)
    repo_root = repo_root_override or infer_repo_root(assembled_path)
    base_dir = assembled_path.parent
    tmp_dir = base_dir / '_tmp_img'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stats = {'requested': 0, 'embedded': 0, 'unresolved': []}

    slide_types_cfg = load_json(slide_types_path) if slide_types_path.exists() else {'types': {}, 'defaults': {}}
    catalog = detect_template_slide_types(base_tpl, slide_types_cfg)

    if 'TOC' in catalog and 'PART_DIVIDER' in catalog and catalog['TOC'] == catalog['PART_DIVIDER']:
        alt = find_slide_by_predicate(base_tpl, lambda t: ('{{PART1_TITLE}}' in t) and ('Sommaire' not in t))
        if alt is not None:
            catalog['PART_DIVIDER'] = alt

    info_tpl = Presentation(str(info_template)) if info_template and info_template.exists() else None
    info_slide6 = info_tpl.slides[0] if info_tpl and len(info_tpl.slides) >= 1 else None
    info_slide4 = info_tpl.slides[1] if info_tpl and len(info_tpl.slides) >= 2 else None

    case_id = data.get('case_id') or assembled_path.stem
    report_type = data.get('report_type') or 'AUDIT'
    today = datetime.now().strftime('%d/%m/%Y')
    section_ctx = data.get('section_context') or {}
    site_label = section_ctx.get('onenote_section_name') or case_id

    prs_out = Presentation()
    prs_out.slide_width = base_tpl.slide_width
    prs_out.slide_height = base_tpl.slide_height

    part_titles: Dict[int, str] = {1: 'Partie 1', 2: 'Partie 2', 3: 'Partie 3', 4: '', 5: ''}
    for mp in (data.get('macro_parts') or []):
        try:
            mp_num = int(mp.get('macro_part'))
            part_titles[mp_num] = mp.get('macro_part_name') or part_titles.get(mp_num, f'Partie {mp_num}')
        except Exception:
            pass

    global_map = {
        '{{PART1_TITLE}}': part_titles.get(1, 'Partie 1'),
        '{{PART2_TITLE}}': part_titles.get(2, 'Partie 2'),
        '{{PART3_TITLE}}': part_titles.get(3, 'Partie 3'),
        '{{PART4_TITLE}}': part_titles.get(4, ''),
        '{{PART5_TITLE}}': part_titles.get(5, ''),
        '{{DATE}}': today,
        '{{VERSION}}': '1',
        '{{CONTACT_EMAIL}}': 'contact@build4use.eu',
    }

    cover_map = {
        '{{AUDIT_TYPE}}': report_type,
        '{{CLIENT}}': 'N/A',
        '{{SITE}}': site_label,
        '{{VILLE}}': '',
        '{{CASE_CODE}}': case_id,
        **global_map,
    }

    if 'COVER' in catalog:
        s_cover = clone_slide(prs_out, catalog['COVER'])
        replace_placeholders(s_cover, cover_map)

    # TOC only once
    if 'TOC' in catalog:
        s_toc = clone_slide(prs_out, catalog['TOC'])
        replace_placeholders(s_toc, global_map)

    def apply_global(slide):
        replace_placeholders(slide, global_map)

    def emit_part_divider(part: int, title: str):
        slide_in = catalog.get('PART_DIVIDER')
        if not slide_in:
            return
        s_div = clone_slide(prs_out, slide_in)
        apply_global(s_div)
        replace_placeholders(s_div, {'{{PART1_TITLE}}': title})
        badge = f"{int(part):02d}"
        for sh in s_div.shapes:
            if getattr(sh, 'has_text_frame', False) and sh.has_text_frame:
                if (sh.text_frame.text or '').strip() in {'01', '02', '03', '99'}:
                    sh.text_frame.text = badge

    def emit_content_slide(s: Dict[str, Any], idx_in_part: int):
        stype = (s.get('type') or 'CONTENT_TEXT').strip()
        title = (s.get('title') or '').strip()
        body = (s.get('bullets') or s.get('body') or '')
        images = s.get('images') or []

        if info_tpl and stype in ('CONTENT_TEXT', 'CONTENT_TEXT_IMAGES') and (info_slide6 or info_slide4):
            pairs = images_to_pairs(images)
            chosen = info_slide6 if (len(pairs) > 4 and info_slide6 is not None) else (info_slide4 or info_slide6)
            if chosen is not None:
                slide_out = clone_slide(prs_out, chosen)
                apply_global(slide_out)
                # title
                for sh in slide_out.shapes:
                    if getattr(sh, 'has_text_frame', False) and sh.has_text_frame and 'Titre section' in (sh.text_frame.text or ''):
                        sh.text_frame.text = title
                        break
                # body
                body_shape = None
                for sh in slide_out.shapes:
                    if getattr(sh, 'has_text_frame', False) and sh.has_text_frame and 'Texte Texte' in (sh.text_frame.text or ''):
                        body_shape = sh
                        break
                if body_shape is not None:
                    set_bullets_fit(body_shape, body, max_lines=12 if images else 14)
                fill_legends(slide_out, images)
                overlay_images(slide_out, images, base_dir, repo_root, tmp_dir, stats)
                return

        slide_in = catalog.get(stype) or catalog.get('CONTENT_TEXT')
        if not slide_in:
            return
        slide_out = clone_slide(prs_out, slide_in)
        apply_global(slide_out)
        replace_placeholders(slide_out, {'{{SLIDE_TITLE}}': title})

        for sh in slide_out.shapes:
            if getattr(sh, 'has_text_frame', False) and sh.has_text_frame and '{{TEXTE_BULLETS}}' in (sh.text_frame.text or ''):
                set_bullets_fit(sh, body, max_lines=12 if images else 14)
                break

        fill_legends(slide_out, images)
        overlay_images(slide_out, images, base_dir, repo_root, tmp_dir, stats)

    slides_spec = data.get('slides')
    per_part_counter = {1: 0, 2: 0, 3: 0}
    if isinstance(slides_spec, list) and slides_spec:
        for s in slides_spec:
            stype = (s.get('type') or 'CONTENT_TEXT').strip()
            if stype == 'PART_DIVIDER':
                part = int(s.get('part') or 0)
                emit_part_divider(part, (s.get('title') or '').strip() or f"Partie {part}")
                if part in per_part_counter:
                    per_part_counter[part] = 0
                continue
            part = int(s.get('part') or 0)
            if part in per_part_counter:
                per_part_counter[part] += 1
            emit_content_slide(s, per_part_counter.get(part, 0))

    # SAVE DIRECTLY (no re-clone!)
    prs_out.save(str(out_path))

    print(f"Repo root: {repo_root}")
    print(f"Images: embedded {stats['embedded']} / requested {stats['requested']}")
    if stats['unresolved']:
        print('Unresolved image paths (sample):')
        for u in stats['unresolved'][:12]:
            print(' -', u)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--template', required=True)
    ap.add_argument('--assembled', required=True)
    ap.add_argument('--out', default='Rapport_Audit.pptx')
    ap.add_argument('--slide-types', default='input/config/slide_types.json')
    ap.add_argument('--info-template', default='')
    ap.add_argument('--project-root', default='', help='Override repo root (folder that contains process/ and input/)')
    args = ap.parse_args()

    repo_override = Path(args.project_root) if args.project_root else None
    build_deck(Path(args.template), Path(args.assembled), Path(args.out), Path(args.slide_types),
        Path(args.info_template) if args.info_template else None, repo_root_override=repo_override)


if __name__ == '__main__':
    main()
