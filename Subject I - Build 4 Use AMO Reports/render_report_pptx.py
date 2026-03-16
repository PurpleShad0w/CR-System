#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""render_report_pptx.py

Fixes
-----
1) Robust placeholder replacement at *shape* text level (not run-level), so split-runs are handled.
2) When cloning ANY slide, we now apply a global token map (PART1..PART5 titles + date/version/contact)
   to avoid leftover {{PART2_TITLE}} etc on slides that reuse the TOC block.
3) Legacy tolerance: remove evidence lines ("Preuve:" / "page_id=") from bodies before writing.
4) Images are inserted into IMAGE_1/2/3 placeholder shapes when provided.
5) Guardrails: fail if any {{TOKEN}} or page_id= remains (with offender preview).
"""

import argparse
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from pptx import Presentation
from pptx.util import Pt
from pptx.enum.text import PP_ALIGN


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_markdown(text: str) -> str:
    if not text:
        return ""
    t = text
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"\*(.*?)\*", r"\1", t)
    t = re.sub(r"^\s*#+\s*", "", t, flags=re.MULTILINE)
    t = t.replace("•", "-")
    return t.strip()


def normalize_whitespace(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def sanitize_client_body(text: str) -> str:
    if not text:
        return ""
    t = normalize_whitespace(strip_markdown(text))
    out = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            out.append("")
            continue
        if re.search(r"\bpage_id\s*=", s, flags=re.IGNORECASE):
            continue
        if re.match(r"^\s*preuve\s*:", s, flags=re.IGNORECASE):
            continue
        if "preuve:" in s.lower():
            continue
        out.append(ln)
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# -----------------------------
# PPTX cloning utilities
# -----------------------------

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


# -----------------------------
# Placeholder replacement (shape-level)
# -----------------------------

def replace_text_in_shape(shape, mapping: Dict[str, str]):
    if not getattr(shape, "has_text_frame", False) or not shape.has_text_frame:
        return
    tf = shape.text_frame
    full = tf.text or ""
    replaced = full
    for k, v in mapping.items():
        if k in replaced:
            replaced = replaced.replace(k, v)
    if replaced != full:
        tf.text = replaced


def replace_placeholders(slide, mapping: Dict[str, str]):
    for shape in slide.shapes:
        replace_text_in_shape(shape, mapping)


def find_shape_containing(slide, token: str):
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
            if token in (shape.text_frame.text or ""):
                return shape
    return None


def set_bullets_in_shape(shape, text: str, font_size_pt: int = 16):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True

    text = sanitize_client_body(text)
    text = normalize_whitespace(strip_markdown(text))

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        tf.paragraphs[0].text = ""
        return

    for i, ln in enumerate(lines):
        is_bullet = ln.startswith("- ")
        content = ln[2:].strip() if is_bullet else ln
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = content
        p.level = 0
        p.font.size = Pt(font_size_pt)
        p.font.name = "Calibri"
        p.font.bold = False
        if is_bullet:
            p.space_before = Pt(2)
            p.space_after = Pt(2)
            p.alignment = PP_ALIGN.LEFT


# -----------------------------
# Slide type detection
# -----------------------------

def slide_text(slide) -> str:
    parts = []
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
            parts.append(shape.text_frame.text or "")
    return "\n".join(parts)


def detect_template_slide_types(tpl: Presentation, slide_types_cfg: Dict[str, Any]) -> Dict[str, Any]:
    types = (slide_types_cfg.get("types") or {})
    found: Dict[str, Any] = {}
    for slide in tpl.slides:
        stxt = slide_text(slide)
        for tname, tcfg in types.items():
            tokens = tcfg.get("detect_tokens") or []
            if tokens and all(tok in stxt for tok in tokens):
                if tname not in found:
                    found[tname] = slide
    return found


# -----------------------------
# Image placement
# -----------------------------

def _resolve_image_path(img: str, base_dir: Path) -> Optional[Path]:
    if not img:
        return None
    p = Path(img)
    if p.is_absolute() and p.exists():
        return p
    cand = (base_dir / img).resolve()
    if cand.exists():
        return cand
    if p.exists():
        return p
    return None


def place_images_on_slide(slide, images: List[str], *, base_dir: Path):
    tokens = ["{{IMAGE_1}}", "{{IMAGE_2}}", "{{IMAGE_3}}"]
    for idx, tok in enumerate(tokens):
        shape = find_shape_containing(slide, tok)
        if not shape:
            continue
        try:
            shape.text_frame.text = ""
        except Exception:
            pass
        if idx >= len(images):
            continue
        img_path = _resolve_image_path(images[idx], base_dir)
        if not img_path:
            continue
        left, top, width, height = shape.left, shape.top, shape.width, shape.height
        slide.shapes.add_picture(str(img_path), left, top, width=width, height=height)


# -----------------------------
# Build deck
# -----------------------------

def build_deck(template_path: Path, assembled_path: Path, out_path: Path, slide_types_path: Optional[Path] = None):
    tpl = Presentation(str(template_path))
    data = load_json(assembled_path)

    slide_types_cfg = load_json(slide_types_path) if slide_types_path and slide_types_path.exists() else {"types": {}, "defaults": {}}
    catalog = detect_template_slide_types(tpl, slide_types_cfg)

    case_id = data.get("case_id") or assembled_path.stem
    report_type = data.get("report_type") or "AUDIT"
    today = datetime.now().strftime("%d/%m/%Y")
    section_ctx = data.get("section_context") or {}
    site_label = section_ctx.get("onenote_section_name") or case_id

    prs_out = Presentation()
    prs_out.slide_width = tpl.slide_width
    prs_out.slide_height = tpl.slide_height

    # Part titles for TOC and global replacement
    part_titles = {1: "Partie 1", 2: "Partie 2", 3: "Partie 3", 4: "", 5: ""}
    for mp in (data.get("macro_parts") or []):
        try:
            mp_num = int(mp.get("macro_part"))
            mp_name = mp.get("macro_part_name") or f"Partie {mp_num}"
            part_titles[mp_num] = mp_name
        except Exception:
            pass

    global_map = {
        "{{PART1_TITLE}}": part_titles.get(1, "Partie 1"),
        "{{PART2_TITLE}}": part_titles.get(2, "Partie 2"),
        "{{PART3_TITLE}}": part_titles.get(3, "Partie 3"),
        "{{PART4_TITLE}}": part_titles.get(4, ""),
        "{{PART5_TITLE}}": part_titles.get(5, ""),
        "{{DATE}}": today,
        "{{VERSION}}": "1",
        "{{CONTACT_EMAIL}}": "contact@build4use.eu",
    }

    cover_map = {
        "{{AUDIT_TYPE}}": report_type,
        "{{CLIENT}}": "N/A",
        "{{SITE}}": site_label,
        "{{VILLE}}": "",
        "{{CASE_CODE}}": case_id,
        **global_map,
    }

    # ---- COVER ----
    if "COVER" in catalog:
        s_cover = clone_slide(prs_out, catalog["COVER"])
        replace_placeholders(s_cover, cover_map)

    # ---- TOC ----
    if "TOC" in catalog:
        s_toc = clone_slide(prs_out, catalog["TOC"])
        replace_placeholders(s_toc, global_map)

    slides_spec = data.get("slides")

    def apply_global(slide):
        # Some template slides repeat TOC block; always apply global map
        replace_placeholders(slide, global_map)

    def emit_part_divider(part: int, title: str):
        slide_in = catalog.get("PART_DIVIDER")
        if not slide_in:
            return
        s_div = clone_slide(prs_out, slide_in)
        apply_global(s_div)
        # Divider title token is usually PART1_TITLE, but we want the current part name
        replace_placeholders(s_div, {"{{PART1_TITLE}}": title})
        badge_num = f"{int(part):02d}"
        for sh in s_div.shapes:
            if getattr(sh, "has_text_frame", False) and sh.has_text_frame:
                if (sh.text_frame.text or "").strip() in {"01", "02", "03", "99"}:
                    sh.text_frame.text = badge_num

    def emit_content_slide(stype: str, title: str, body: str, images: Optional[List[str]] = None):
        images = images or []
        slide_in = catalog.get(stype) or catalog.get("CONTENT_TEXT")
        if not slide_in:
            return
        slide_out = clone_slide(prs_out, slide_in)
        apply_global(slide_out)
        replace_placeholders(slide_out, {"{{SLIDE_TITLE}}": title})

        shape = find_shape_containing(slide_out, "{{TEXTE_BULLETS}}")
        if shape:
            set_bullets_in_shape(shape, body, font_size_pt=16)

        place_images_on_slide(slide_out, images, base_dir=assembled_path.parent)

    if isinstance(slides_spec, list) and slides_spec:
        for s in slides_spec:
            stype = (s.get("type") or "CONTENT_TEXT").strip()
            if stype == "PART_DIVIDER":
                emit_part_divider(int(s.get("part") or 0), (s.get("title") or "").strip())
                continue
            emit_content_slide(stype, (s.get("title") or "").strip(), (s.get("bullets") or s.get("body") or ""), s.get("images") or [])
    else:
        # Legacy fallback
        for mp_num in [1, 2, 3]:
            mp = next((x for x in (data.get("macro_parts") or []) if int(x.get("macro_part", -1)) == mp_num), None)
            if not mp:
                continue
            emit_part_divider(mp_num, mp.get("macro_part_name") or f"Partie {mp_num}")
            for sec in (mp.get("sections") or []):
                emit_content_slide(slide_types_cfg.get("defaults", {}).get("content_type", "CONTENT_TEXT"), sec.get("bucket_id") or "SECTION", sec.get("text") or "", [])

    # ---- POST-RENDER GUARDRAILS (with diagnostics) ----
    offenders = []
    for si, slide in enumerate(prs_out.slides, start=1):
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                txt = shape.text_frame.text or ""
                if "{{" in txt or "page_id=" in txt:
                    offenders.append({
                        "slide": si,
                        "text": (txt.strip().replace("\n", " ")[:160] + ("..." if len(txt) > 160 else "")),
                    })
    if offenders:
        preview = " | ".join([f"S{d['slide']}: {d['text']}" for d in offenders[:6]])
        raise SystemExit("❌ Unreplaced template tokens or page_id leak detected in rendered deck. Offenders: " + preview)

    prs_out.save(str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Path to TEMPLATE_AUDIT_BUILD4USE.pptx")
    ap.add_argument("--assembled", required=True, help="Path to assembled_report.json")
    ap.add_argument("--out", default="Rapport_Audit.pptx", help="Output PPTX path")
    ap.add_argument("--slide-types", default="input/config/slide_types.json", help="Path to slide_types.json")
    args = ap.parse_args()

    build_deck(Path(args.template), Path(args.assembled), Path(args.out), Path(args.slide_types))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
