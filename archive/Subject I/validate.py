#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick validator for bacs_table6_rules_structured.json."""
import json, re, sys
from pathlib import Path

NOISE = ['Définition des classes','Résidentiel Non résidentiel','Tableau 6 (suite)']

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def main(path):
    obj = load(path)
    meta = obj.get('meta', {})
    print('extractor:', meta.get('extractor'))

    bad = []
    for r in obj.get('rules', []):
        rid = r.get('rule_id')
        if r.get('_missing_in_pdf'):
            bad.append(f"{rid}: missing_in_pdf")
        for lv, lvobj in (r.get('levels') or {}).items():
            lines = ((lvobj.get('text') or {}).get('lines') or [])
            for ln in lines[1:]:
                if re.match(r'^\d+\s+.+', ln):
                    bad.append(f"{rid}.{lv}: section header leaked -> {ln}")
            for ln in lines:
                if any(n in ln for n in NOISE):
                    bad.append(f"{rid}.{lv}: noise -> {ln}")
                if re.search(r'\bau\s+x\b', ln):
                    bad.append(f"{rid}.{lv}: 'au x' artifact -> {ln}")

    if bad:
        print('FAIL:')
        for b in bad[:80]:
            print(' -', b)
        sys.exit(1)
    print('OK')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: validate.py bacs_table6_rules_structured.json')
        raise SystemExit(2)
    main(sys.argv[1])
