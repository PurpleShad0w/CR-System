
# -*- coding: utf-8 -*-
"""BACS scoring (ISO 52120-1 Table 6) deterministic engine.

This module:
- loads a rules JSON (skeleton or enriched)
- computes achieved class per group (Chauffage, ECS, Refroidissement, Ventilation, Éclairage, Stores, GTB)
- produces:
  - Part 2 text: scoring actuel (classes + blockers)
  - Part 3 text: travaux to reach a target class (default B, per usage configurable)

It does NOT contain ISO verbatim text.
"""

import json
from pathlib import Path

CLASSES = ['D','C','B','A']


def load_rules(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def required_level(rule: dict, building_scope: str, target_class: str) -> int | None:
    # Returns minimum implemented level needed for target_class, or None if not required.
    req = (rule.get('class_requirements') or {}).get(building_scope) or {}
    return req.get(target_class)


def rule_meets(rule: dict, building_scope: str, target_class: str, implemented_level: int | None) -> bool:
    req = required_level(rule, building_scope, target_class)
    if req is None:
        return True
    if implemented_level is None:
        return False
    return int(implemented_level) >= int(req)


def highest_class_for_rule(rule: dict, building_scope: str, implemented_level: int | None) -> str:
    # Determine highest class satisfied for this rule.
    # If C requirement exists and isn't met, return 'D'. Else walk up.
    if not rule_meets(rule, building_scope, 'C', implemented_level):
        return 'D'
    if rule_meets(rule, building_scope, 'A', implemented_level):
        return 'A'
    if rule_meets(rule, building_scope, 'B', implemented_level):
        return 'B'
    return 'C'


def compute_group_scores(rules_doc: dict, building_scope: str, observed_levels: dict) -> dict:
    by_group = {}
    for r in rules_doc.get('rules', []):
        grp = r.get('group')
        rid = r.get('rule_id')
        lvl = observed_levels.get(rid)
        by_group.setdefault(grp, []).append((r, lvl))

    out = {}
    for grp, items in by_group.items():
        # group class is min of rule classes among applicable rules
        rule_classes = []
        blockers = { 'C': [], 'B': [], 'A': [] }
        for r, lvl in items:
            rc = highest_class_for_rule(r, building_scope, lvl)
            rule_classes.append(rc)
            for tgt in ['C','B','A']:
                if not rule_meets(r, building_scope, tgt, lvl):
                    req = required_level(r, building_scope, tgt)
                    blockers[tgt].append({
                        'rule_id': r.get('rule_id'),
                        'title': r.get('title'),
                        'required_level': req,
                        'observed_level': lvl,
                    })

        # Determine group achieved class: the minimum across rules by ordering
        order = {'D':0,'C':1,'B':2,'A':3}
        achieved = min(rule_classes, key=lambda c: order.get(c,0)) if rule_classes else 'D'
        out[grp] = {
            'achieved_class': achieved,
            'blockers': blockers,
            'rules_count': len(items),
        }
    return out


def make_part2_markdown(group_scores: dict) -> str:
    # Report-style text similar structure to reference report: per usage
    lines = []
    lines.append('## Scoring GTB actuel selon ISO 52120-1')
    lines.append('')
    for grp, s in group_scores.items():
        lines.append(f"#### {grp}")
        lines.append(f"Classe atteinte (fonctionnalités BAC/GTB): **{s['achieved_class']}**")
        # list main blockers to reach B/A
        b_block = s['blockers'].get('B') or []
        if s['achieved_class'] in ('C','D') and b_block:
            lines.append('Principaux écarts bloquants vers la classe B:')
            for b in b_block[:8]:
                rid = b['rule_id']
                req = b['required_level']
                obs = b['observed_level']
                title = b['title']
                lines.append(f"- {rid} — {title} (niveau requis: {req}, observé: {obs if obs is not None else 'non établi'})")
        a_block = s['blockers'].get('A') or []
        if s['achieved_class'] in ('B','C','D') and a_block:
            # only show a few A blockers
            lines.append('Écarts vers la classe A (indicatif):')
            for b in a_block[:5]:
                rid = b['rule_id']
                req = b['required_level']
                obs = b['observed_level']
                title = b['title']
                lines.append(f"- {rid} — {title} (niveau requis: {req}, observé: {obs if obs is not None else 'non établi'})")
        lines.append('')
    return '\n'.join(lines).strip() + '\n'


def make_part3_markdown(rules_doc: dict, building_scope: str, observed_levels: dict, target_by_group: dict | None = None) -> str:
    target_by_group = target_by_group or {}
    lines = []
    lines.append('## Futur : Scoring projeté et travaux à mettre en œuvre')
    lines.append('')
    # group missing items
    for grp in sorted({r['group'] for r in rules_doc.get('rules', [])}):
        tgt = target_by_group.get(grp, 'B')
        missing = []
        for r in rules_doc.get('rules', []):
            if r.get('group') != grp:
                continue
            rid = r.get('rule_id')
            lvl = observed_levels.get(rid)
            if not rule_meets(r, building_scope, tgt, lvl):
                missing.append({
                    'rule_id': rid,
                    'title': r.get('title'),
                    'required_level': required_level(r, building_scope, tgt),
                    'observed_level': lvl,
                })
        if not missing:
            continue
        lines.append(f"#### {grp} — Objectif classe {tgt}")
        lines.append('Travaux / évolutions fonctionnelles à prévoir (au strict sens ISO 52120-1):')
        for m in missing:
            rid = m['rule_id']
            title = m['title']
            req = m['required_level']
            obs = m['observed_level']
            lines.append(f"- Mettre en œuvre {rid} — {title} (niveau requis: {req}, observé: {obs if obs is not None else 'non établi'})")
        lines.append('')
    return '\n'.join(lines).strip() + '\n'
