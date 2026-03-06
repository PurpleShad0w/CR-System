#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tableau 6 extractor v5 — LLM-friendly structured JSON (fixes v4 issues).

Fixes vs v4 (based on your v4 output review):
1) Prevent section-title leakage inside levels:
   - After the first line of a level block, drop any line matching ^\d+\s+.* (section headers like "3 Régulation ...").
2) Noise filtering:
   - Drop common table headers inside levels ("Définition des classes", "Résidentiel Non résidentiel", "Tableau 6 (suite)", "D C B A D C B A", etc.).
3) Safe 'x' degluing (do NOT break words like "aux"):
   - Only insert a space before 'x' when it is followed by another x-token (e.g. communicationx x -> communication x x).
4) Robust x token extraction:
   - Extract x/xa/xb tokens from ALL raw lines of the level block, not only the header.
5) Footnotes:
   - Capture multi-line footnotes (a/b/...) into rule.footnotes and remove them from level text.
6) Missing rule chunks:
   - If a rule_id cannot be found in the PDF slice, mark rule['_missing_in_pdf']=true.

Output JSON:
- rules[].function_text.lines
- rules[].levels[lv].text.lines (clean)
- rules[].levels[lv].x.tokens (all tokens from block)
- rules[].footnotes (a/b/...) consolidated and joined

Note: eligible_classes stays null (manual mapping is recommended).
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path


# ------------------------
# Text utilities
# ------------------------

def normalize_text_keep_newlines(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize('NFC', s)
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = s.replace('\u00A0', ' ').replace('\u2007', ' ').replace('\u202F', ' ')
    s = s.replace('\f', '\n')
    return s


def safe_deglue_x(s: str) -> str:
    """Insert a space before 'x' ONLY if it starts a run of x-tokens.

    Examples:
      communicationx x x  -> communication x x x
      fixex x x           -> fixe x x
      aux systèmes        -> unchanged

    We match: <letter> x (?= optional spaces then another x)
    """
    return re.sub(r'([A-Za-zÀ-ÖØ-öø-ÿ])x(?=\s*[xX])', r'\1 x', s)


def is_noise_line(t: str) -> bool:
    if not t:
        return True
    # table headers / footers frequently leaking
    noise_exact = {
        'Définition des classes',
        'Résidentiel Non résidentiel',
        'Tableau 6 (suite)',
    }
    if t in noise_exact:
        return True

    # column header line
    if re.fullmatch(r'(D\s+C\s+B\s+A\s+){1,2}D\s+C\s+B\s+A', t):
        return True

    # generic footer noise
    if any(p in t for p in [
        'AFNOR', '© ISO', 'Tous droits réservés',
        'NF EN ISO', 'EN ISO', 'ISO 52120',
        '@', 'Pour :', 'RUSU',
    ]):
        return True

    # standalone page numbers
    if re.fullmatch(r'\d{1,3}', t):
        return True

    return False


def dehyphenate_lines(lines):
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i+1]
            # "détec -" + "tion" or "tempé -" + "rature"
            if re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]\s*-\s*$', ln) and re.match(r'^\s*[A-Za-zÀ-ÖØ-öø-ÿ]', nxt):
                merged = re.sub(r'\s*-\s*$', '', ln) + re.sub(r'^\s*', '', nxt)
                out.append(merged)
                i += 2
                continue
            # "va-" + "riable"
            if re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]-\s*$', ln) and re.match(r'^\s*[A-Za-zÀ-ÖØ-öø-ÿ]', nxt):
                merged = re.sub(r'-\s*$', '', ln) + re.sub(r'^\s*', '', nxt)
                out.append(merged)
                i += 2
                continue
        out.append(ln)
        i += 1
    return out


def norm_lines(s: str):
    s = normalize_text_keep_newlines(s)
    s = safe_deglue_x(s)
    raw = [re.sub(r'\s+', ' ', ln).strip() for ln in s.split('\n')]
    raw = [ln for ln in raw if ln]
    raw = [ln for ln in raw if not is_noise_line(ln)]
    raw = dehyphenate_lines(raw)
    return raw


# ------------------------
# PDF extraction
# ------------------------

def extract_pdf_text(pdf_path: Path, start_page: int, end_page: int) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(str(pdf_path))
    start = max(1, start_page)
    end = min(len(reader.pages), end_page)
    pages = []
    for i in range(start - 1, end):
        pages.append(reader.pages[i].extract_text() or '')
    return '\n'.join(pages)


def slice_by_rule_ids(blob: str, rule_ids):
    positions = []
    for rid in rule_ids:
        m = re.search(rf"(?m)^\s*{re.escape(rid)}\b", blob)
        if m:
            positions.append((m.start(), rid))
    positions.sort()

    slices = {}
    for idx, (pos, rid) in enumerate(positions):
        endpos = positions[idx + 1][0] if idx + 1 < len(positions) else len(blob)
        slices[rid] = blob[pos:endpos]
    return slices


# ------------------------
# Rule parsing
# ------------------------

def expected_levels_from_skeleton(rule) -> list[int]:
    lv = rule.get('levels')
    if isinstance(lv, dict) and lv:
        out = []
        for k in lv.keys():
            if str(k).isdigit():
                out.append(int(k))
        return sorted(out)
    return []


def split_rule_chunk(chunk: str, expected_levels: list[int]):
    if not expected_levels:
        return chunk, ''
    # split at first expected level occurrence at line start
    exp = '|'.join(map(str, expected_levels))
    m = re.search(rf"(?m)^\s*(?:{exp})\s+", chunk)
    if not m:
        return chunk, ''
    return chunk[:m.start()], chunk[m.start():]


def parse_level_blocks(level_text: str, expected_levels: list[int]):
    if not level_text or not expected_levels:
        return {}

    # Find all expected level starts
    starts = []
    for lv in expected_levels:
        m = re.search(rf"(?m)^\s*{lv}\s+", level_text)
        if m:
            starts.append((m.start(), lv))
    starts.sort()

    blocks = {}
    for i, (pos, lv) in enumerate(starts):
        end = starts[i+1][0] if i+1 < len(starts) else len(level_text)
        blocks[str(lv)] = level_text[pos:end]
    return blocks


def remove_x_marks_from_line(line: str) -> str:
    # remove trailing x/xa/xb tokens
    return re.sub(r"(\b[xX][ab]?\b\s*)+$", "", line).strip()


def collect_footnotes(lines: list[str], expected_levels: list[int]):
    """Extract multi-line footnotes from a list of normalized lines.

    Footnote starts with 'a ' or 'b ' etc.
    Continue until next footnote start, or a new level line, or a section header line.
    """
    foot = {}
    kept = []
    current_key = None
    buf = []

    lvl_pat = None
    if expected_levels:
        exp = '|'.join(map(str, expected_levels))
        lvl_pat = re.compile(rf"^(?:{exp})\s+")

    def flush():
        nonlocal current_key, buf
        if current_key and buf:
            foot[current_key] = ' '.join(buf).strip()
        current_key = None
        buf = []

    for ln in lines:
        # Start of a new footnote
        m = re.match(r'^([a-z])\s+(.+)$', ln)
        if m and m.group(1) in ('a', 'b', 'c', 'd'):
            flush()
            current_key = m.group(1)
            buf = [m.group(0)]
            continue

        # Stop conditions for footnote continuation
        if current_key:
            if lvl_pat and lvl_pat.match(ln):
                flush()
                kept.append(ln)
                continue
            if re.match(r'^\d+\s+.+', ln):
                # section header
                flush()
                kept.append(ln)
                continue
            if is_noise_line(ln):
                continue
            buf.append(ln)
            continue

        kept.append(ln)

    flush()
    # Now remove footnote lines from kept (they were already diverted, but m.group(0) is included in foot).
    kept2 = []
    for ln in kept:
        if re.match(r'^[ab]\s+', ln):
            continue
        kept2.append(ln)
    return kept2, foot


def drop_section_headers_inside_level(cleaned_lines: list[str]):
    """After first line, drop any line that looks like a new section header: '^\d+\s+...'."""
    if not cleaned_lines:
        return cleaned_lines
    out = [cleaned_lines[0]]
    for ln in cleaned_lines[1:]:
        if re.match(r'^\d+\s+.+', ln):
            continue
        if is_noise_line(ln):
            continue
        out.append(ln)
    return out


def extract_x_tokens_from_block(lines: list[str]):
    # Extract x tokens across all lines
    txt = ' '.join(lines)
    return re.findall(r"\b[xX][ab]?\b", txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', required=True)
    ap.add_argument('--skeleton', required=True)
    ap.add_argument('--out', default='bacs_table6_rules_structured.json')
    ap.add_argument('--start-page', type=int, default=43)
    ap.add_argument('--end-page', type=int, default=53)
    args = ap.parse_args()

    skel = json.loads(Path(args.skeleton).read_text(encoding='utf-8'))
    rule_ids = [r['rule_id'] for r in skel.get('rules', [])]

    blob = extract_pdf_text(Path(args.pdf), args.start_page, args.end_page)
    slices = slice_by_rule_ids(blob, rule_ids)

    out = {
        'meta': {
            **(skel.get('meta') or {}),
            'structured': True,
            'extractor': 'v5',
            'start_page': args.start_page,
            'end_page': args.end_page,
        },
        'rules': []
    }

    for rule in skel.get('rules', []):
        rid = rule['rule_id']
        chunk = slices.get(rid, '')
        expected_levels = expected_levels_from_skeleton(rule)

        rule_obj = {
            **rule,
            'raw_text_excerpt': chunk[:8000] if chunk else '',
            'function_text': {'lines': []},
            'levels': {},
            'footnotes': {},
        }

        if not chunk:
            rule_obj['_missing_in_pdf'] = True
            out['rules'].append(rule_obj)
            continue

        fn_text, lvl_text = split_rule_chunk(chunk, expected_levels)

        # Function lines + footnotes (from function text)
        fn_lines = norm_lines(fn_text)
        fn_lines, foot_fn = collect_footnotes(fn_lines, expected_levels)
        rule_obj['function_text']['lines'] = fn_lines
        rule_obj['footnotes'].update(foot_fn)

        # Levels
        lvl_blocks = parse_level_blocks(lvl_text, expected_levels)
        for lv, block in lvl_blocks.items():
            raw_lines = norm_lines(block)
            raw_lines, foot_lv = collect_footnotes(raw_lines, expected_levels)
            rule_obj['footnotes'].update(foot_lv)

            if not raw_lines:
                continue

            # Strip leading numeric prefix on first line
            header = raw_lines[0]
            header_wo_num = re.sub(r'^\s*' + re.escape(str(lv)) + r'\s+', '', header).strip()

            # Extract x tokens from ALL lines (including header_wo_num + subsequent lines)
            all_for_x = [header_wo_num] + raw_lines[1:]
            x_tokens = extract_x_tokens_from_block(all_for_x)

            # Clean header by removing trailing x tokens
            header_clean = remove_x_marks_from_line(header_wo_num)

            cleaned_lines = [header_clean] + raw_lines[1:]
            cleaned_lines = drop_section_headers_inside_level(cleaned_lines)

            rule_obj['levels'][lv] = {
                'text': {
                    'header': header_clean,
                    'lines': cleaned_lines,
                },
                'raw': {
                    'header': header,
                    'lines': raw_lines,
                },
                'x': {
                    'tokens': x_tokens,
                },
                'eligible_classes': None,
            }

        out['rules'].append(rule_obj)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote: {args.out}")


if __name__ == '__main__':
    main()
