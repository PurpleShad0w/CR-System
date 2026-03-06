import json, re

def strip_x_tokens(s: str) -> str:
    # remove standalone x/xa/xb tokens and also ")x x" glued cases
    s = re.sub(r'\b[xX][ab]?\b', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    # fix ")x" glue
    s = re.sub(r'\)\s*$', ')', s)
    return s

def is_section_header(line: str) -> bool:
    return bool(re.match(r'^\d+\s+.+', line) or re.match(r'^\d+\.\d+(\.\d+)?\s+.*', line))

data = json.load(open("bacs_table6_rules_structured.json","r",encoding="utf-8"))

for r in data["rules"]:
    for lv, lvobj in (r.get("levels") or {}).items():
        raw = (lvobj.get("raw") or {}).get("lines") or []
        text = (lvobj.get("text") or {}).get("lines") or []

        # if raw contains a new section header after the level start, cut everything after it
        cut_idx = None
        for i, ln in enumerate(raw[1:], start=1):
            if is_section_header(ln):
                cut_idx = i
                break
        if cut_idx is not None:
            raw = raw[:cut_idx]
            text = text[:cut_idx] if cut_idx < len(text) else text

        # strip x tokens from all text lines
        cleaned = [strip_x_tokens(ln) for ln in text]
        cleaned = [ln for ln in cleaned if ln]  # drop empty

        lvobj["text"]["cleaned_lines"] = cleaned

json.dump(data, open("bacs_table6_rules_structured_clean.json","w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("Wrote bacs_table6_rules_structured_clean.json")