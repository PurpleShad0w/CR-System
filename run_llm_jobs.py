#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from llm_client import make_client
from quality_score import evaluate_quality  # deterministic full-report scorer [1](https://discuss.huggingface.co/t/500-internal-server-error-with-inference-endpoint/89605)
import quality_score as qs  # reuse exact rule lists + helpers to avoid divergence [1](https://discuss.huggingface.co/t/500-internal-server-error-with-inference-endpoint/89605)


# -----------------------------
# IO helpers
# -----------------------------
def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: Dict[str, Any]):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# LLM message builder
# -----------------------------
def build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "Tu es un rédacteur technique Build 4 Use. Respecte les contraintes et n'invente rien."},
        {"role": "user", "content": prompt},
    ]


# -----------------------------
# Robust chat wrapper (retries + backoff)
# -----------------------------
def safe_chat(client, messages, *, temperature: float, max_tokens: int, top_p: float,
              retries: int = 4, base_sleep: float = 1.2) -> Tuple[Optional[Any], Optional[str]]:
    """
    Returns (resp, err). Never raises.
    Retries transient HF failures (500, busy/timeout-like).
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False
            )
            return resp, None
        except Exception as e:
            msg = str(e)
            last_err = msg

            # Common HF transient patterns: 500 internal errors, "model too busy", timeout-like.
            transient = (
                "HF error 500" in msg
                or "Internal Server Error" in msg
                or "Unknown error" in msg
                or "Model too busy" in msg
                or "unable to get response" in msg
            )
            if attempt >= retries or not transient:
                break

            # exponential backoff + jitter
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.4)
            time.sleep(sleep_s)

    return None, last_err


# -----------------------------
# Multistep prompts (internal)
# -----------------------------
def prompt_extract_facts(section: Dict[str, Any]) -> str:
    mp = section.get("macro_part")
    mp_name = section.get("macro_part_name") or f"Macro {mp}"
    bucket = section.get("bucket_id") or "SECTION"
    evidence = (section.get("evidence") or "").strip()

    schema = {
        "observations": ["string"],
        "entities": ["string"],
        "quantities": ["string"],
        "constraints_from_evidence": ["string"],
        "unknowns": ["string"],
        "norm_refs": ["string"],
        "do_not_say": ["string"]
    }

    return (
        "Tu dois produire UNIQUEMENT un objet JSON valide (sans texte autour, sans markdown).\n"
        "Tâche: extraire un 'fact sheet' à partir des preuves OneNote.\n\n"
        f"Contexte:\n- Macro-partie: {mp} - {mp_name}\n- Bucket: {bucket}\n\n"
        "Règles STRICTES:\n"
        "- Ne rédige PAS de section de rapport.\n"
        "- Ne propose PAS de travaux/actions (sauf si c'est explicitement énoncé comme fait/élément dans les preuves).\n"
        "- Ne déduis pas de faits non présents.\n"
        "- Si une information manque, mets-la dans unknowns.\n"
        "- Si une phrase des preuves est une QUESTION/DEMANDE (ex: 'peux-tu...'), ne la transforme pas en fait.\n\n"
        f"Format JSON attendu (mêmes clés):\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Preuves (texte brut):\n{evidence}\n"
    )


def prompt_write_from_facts(section: Dict[str, Any], facts: Dict[str, Any]) -> str:
    base_prompt = (section.get("prompt") or "").strip()
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)

    return (
        f"{base_prompt}\n\n"
        "---\n"
        "FACTS (JSON) = source UNIQUE\n"
        "Tu dois utiliser UNIQUEMENT ces FACTS pour rédiger. Ignore les preuves brutes.\n"
        f"{facts_json}\n"
        "---\n\n"
        "Rappel critique:\n"
        "- N'invente rien.\n"
        "- Si quelque chose est dans unknowns, indique-le explicitement dans 'Points à confirmer'.\n"
        "- Respecte strictement l'intention de la macro-partie.\n"
    )


def prompt_repair(section: Dict[str, Any], facts: Dict[str, Any], draft_text: str, issues: List[str]) -> str:
    mp = section.get("macro_part")
    mp_name = section.get("macro_part_name") or f"Macro {mp}"
    bucket = section.get("bucket_id") or "SECTION"

    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
    issues_txt = "\n- ".join(issues) if issues else "(aucun)"

    return (
        "Tu dois CORRIGER une section de rapport pour supprimer des violations.\n"
        f"Contexte: Macro-partie {mp} - {mp_name} | Bucket {bucket}\n\n"
        "Contraintes:\n"
        "- Ne change pas le fond: reste fidèle aux FACTS.\n"
        "- Ne rajoute aucun fait.\n"
        "- Supprime toute formulation qui viole les règles listées.\n"
        "- Si info manquante: mets-la dans 'Points à confirmer'.\n"
        "- Sortie: UNIQUEMENT le texte corrigé.\n\n"
        f"Issues détectées:\n- {issues_txt}\n\n"
        f"FACTS (source unique):\n{facts_json}\n\n"
        f"TEXTE À CORRIGER:\n{draft_text}\n"
    )


# -----------------------------
# Helpers: parsing + deterministic fallbacks
# -----------------------------
def parse_json_safely(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not raw:
        return None, "empty"
    s = raw.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, None
        return None, "not_a_dict"
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        candidate = s[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, None
            return None, "extracted_not_a_dict"
        except Exception as e:
            return None, f"json_extract_parse_failed:{e}"

    return None, "no_json_object_found"


def default_empty_facts() -> Dict[str, Any]:
    return {
        "observations": [],
        "entities": [],
        "quantities": [],
        "constraints_from_evidence": [],
        "unknowns": [],
        "norm_refs": [],
        "do_not_say": [],
    }


def heuristic_facts_from_evidence(section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic fallback if HF is down: parse your evidence block format.
    It won't be perfect, but it keeps the pipeline alive and avoids stopping on HF 500.
    """
    evidence = (section.get("evidence") or "")
    lines = [ln.strip() for ln in evidence.splitlines() if ln.strip()]

    facts = default_empty_facts()
    for ln in lines:
        # Page titles
        if ln.startswith("## Page:"):
            # treat page title as entity/context
            facts["entities"].append(ln.replace("## Page:", "").strip())
            continue

        # bullet evidence lines in your format: "- [paragraph] ..."
        if ln.startswith("- [") and "]" in ln:
            content = ln.split("]", 1)[1].strip()
            # Question/demand -> unknown or do_not_say
            if "?" in content or content.lower().startswith(("peux-tu", "peux tu", "pouvez-vous", "pouvez vous")):
                facts["unknowns"].append(content)
            else:
                facts["observations"].append(content)
            # crude quantity capture
            for tok in content.split():
                if tok.isdigit():
                    facts["quantities"].append(content)
                    break
            # norm refs capture
            low = content.lower()
            if "iso" in low or "52120" in low or "décret bacs" in low or "decret bacs" in low:
                facts["norm_refs"].append(content)
            continue

        # Top meta lines
        if ln.startswith("- Bucket:") or ln.startswith("- Mots-clés:"):
            facts["constraints_from_evidence"].append(ln)
            continue

    # de-dup while preserving order
    def dedup(seq):
        seen = set()
        out = []
        for x in seq:
            k = x.strip()
            if not k:
                continue
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    for k in list(facts.keys()):
        facts[k] = dedup(facts[k])

    return facts


# -----------------------------
# Diagnostics helpers (per-section)
# -----------------------------
def keyword_hit_ratio(section: Dict[str, Any], text: str) -> float:
    kws = [qs.norm(k) for k in (section.get("keywords") or []) if k]
    t = qs.norm(text or "")
    if not kws:
        return 0.0
    hits = sum(1 for k in kws if k and k in t)
    return hits / len(kws)


def detect_section_issues(section: Dict[str, Any], text: str) -> Tuple[List[str], Dict[str, Any]]:
    mp = section.get("macro_part")
    t = text or ""

    forb = qs.INTENT_FORBIDDEN.get(mp, [])
    forb_hits = qs.count_hits(t, forb)
    ph_hits = qs.count_hits(t, qs.PLACEHOLDERS)
    mp1_presc_hits = qs.count_hits(t, qs.PRESCRIPTIVE_MP1) if mp == 1 else 0
    khr = keyword_hit_ratio(section, t)

    issues = []
    if forb_hits:
        issues.append(f"forbidden_terms_hits={forb_hits}")
    if ph_hits:
        issues.append(f"placeholders_hits={ph_hits}")
    if mp1_presc_hits:
        issues.append(f"mp1_prescriptive_hits={mp1_presc_hits}")
    if (section.get("keywords") or []) and khr < 0.25:
        issues.append(f"low_keyword_alignment={khr:.2f}")

    metrics = {
        "forbidden_hits": forb_hits,
        "placeholder_hits": ph_hits,
        "mp1_prescriptive_hits": mp1_presc_hits,
        "keyword_hit_ratio": round(khr, 3),
    }
    return issues, metrics


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to process/drafts/<case_id>/draft_bundle.json")
    ap.add_argument("--out", default="", help="Output folder (defaults to bundle folder)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--mode", choices=["single", "multistep"], default="single",
                    help="single = one call per section; multistep = facts->write (+repair)")
    ap.add_argument("--facts_max_tokens", type=int, default=500,
                    help="Max tokens for facts extraction (lower reduces HF load)")
    ap.add_argument("--facts_temperature", type=float, default=0.0,
                    help="Temperature for facts extraction")
    ap.add_argument("--repair_max_tokens", type=int, default=900, help="Max tokens for repair call")
    ap.add_argument("--repair_temperature", type=float, default=0.2, help="Temperature for repair call")
    ap.add_argument("--max_repairs", type=int, default=1, help="Repair attempts per section")

    ap.add_argument("--min_quality", type=float, default=0.0, help="If >0, fail when FINAL quality < min_quality")
    ap.add_argument("--quality", default="", help="Path to write quality_report.json (default: out_dir/quality_report.json)")

    # retry controls
    ap.add_argument("--retries", type=int, default=4, help="Retries for HF transient errors")
    ap.add_argument("--retry_sleep", type=float, default=1.2, help="Base sleep for retry backoff")

    args = ap.parse_args()

    bundle_path = Path(args.bundle)
    out_dir = Path(args.out) if args.out else bundle_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_json(bundle_path)
    client = make_client()

    generated = dict(bundle)

    for sec in generated.get("sections", []):
        base_prompt = (sec.get("prompt") or "").strip()
        if not base_prompt:
            sec["generated_text"] = ""
            sec["final_text"] = ""
            sec["llm_raw"] = None
            continue

        # -------------------------
        # SINGLE (legacy)
        # -------------------------
        if args.mode == "single":
            resp, err = safe_chat(
                client, build_messages(base_prompt),
                temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p,
                retries=args.retries, base_sleep=args.retry_sleep
            )
            if resp is None:
                # if HF is down even in single mode, store failure but keep file outputs
                sec["generated_text"] = ""
                sec["final_text"] = ""
                sec["llm_error"] = err
                sec["llm_raw"] = None
            else:
                text = (resp.text or "").strip()
                sec["generated_text"] = text
                sec["final_text"] = text
                sec["llm_raw"] = resp.raw
            continue

        # -------------------------
        # MULTISTEP
        # -------------------------

        # (1) Facts extraction with retry; fallback to deterministic extraction if HF fails
        facts_prompt = prompt_extract_facts(sec)
        facts_resp, facts_err = safe_chat(
            client, build_messages(facts_prompt),
            temperature=args.facts_temperature, max_tokens=args.facts_max_tokens, top_p=1.0,
            retries=args.retries, base_sleep=args.retry_sleep
        )

        if facts_resp is None:
            sec["facts_llm_error"] = facts_err
            facts_obj = heuristic_facts_from_evidence(sec)
            sec["facts_source"] = "heuristic"
            sec["facts_json"] = facts_obj
            sec["llm_raw_facts"] = None
        else:
            facts_obj, parse_err = parse_json_safely((facts_resp.text or ""))
            if facts_obj is None:
                sec["facts_parse_error"] = parse_err
                facts_obj = heuristic_facts_from_evidence(sec)
                sec["facts_source"] = "heuristic_parse_fallback"
            else:
                sec["facts_source"] = "llm"
            sec["facts_json"] = facts_obj
            sec["llm_raw_facts"] = facts_resp.raw

        # (2) Writing from facts; if HF fails, fall back to single-step writing for that section
        write_prompt = prompt_write_from_facts(sec, facts_obj)
        write_resp, write_err = safe_chat(
            client, build_messages(write_prompt),
            temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p,
            retries=args.retries, base_sleep=args.retry_sleep
        )

        if write_resp is None:
            sec["write_llm_error"] = write_err
            # fallback: legacy single prompt for this section only
            sec["fallback_mode"] = "single_due_to_write_error"
            single_resp, single_err = safe_chat(
                client, build_messages(base_prompt),
                temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p,
                retries=args.retries, base_sleep=args.retry_sleep
            )
            if single_resp is None:
                sec["generated_text"] = ""
                sec["final_text"] = ""
                sec["llm_error"] = single_err
                sec["llm_raw"] = None
                continue
            draft_text = (single_resp.text or "").strip()
            sec["draft_text"] = draft_text
            sec["llm_raw_write"] = single_resp.raw
        else:
            draft_text = (write_resp.text or "").strip()
            sec["draft_text"] = draft_text
            sec["llm_raw_write"] = write_resp.raw

        # (3) Optional repair loop; if HF fails on repair, keep best available text
        final_text = draft_text
        repairs_done = 0

        while repairs_done < max(0, args.max_repairs):
            issues, metrics = detect_section_issues(sec, final_text)
            sec["section_metrics"] = metrics
            if not issues:
                break

            rep_prompt = prompt_repair(sec, facts_obj, final_text, issues)
            rep_resp, rep_err = safe_chat(
                client, build_messages(rep_prompt),
                temperature=args.repair_temperature, max_tokens=args.repair_max_tokens, top_p=1.0,
                retries=args.retries, base_sleep=args.retry_sleep
            )
            if rep_resp is None:
                sec.setdefault("repair_errors", []).append({"issues": issues, "error": rep_err})
                break

            repaired = (rep_resp.text or "").strip()
            sec.setdefault("repair_attempts", []).append({
                "issues": issues,
                "repaired_text": repaired,
                "llm_raw_repair": rep_resp.raw,
            })
            final_text = repaired
            repairs_done += 1

        sec["final_text"] = final_text
        sec["generated_text"] = final_text  # backward compat
        sec["llm_raw"] = {
            "facts": sec.get("llm_raw_facts"),
            "write": sec.get("llm_raw_write"),
        }

    # Write generated bundle
    gen_path = out_dir / "generated_bundle.json"
    save_json(gen_path, generated)

    # Assemble report (use final_text)
    assembled = {
        "case_id": generated.get("case_id"),
        "report_type": generated.get("report_type"),
        "macro_parts": []
    }
    by_mp: Dict[int, List[Dict[str, Any]]] = {}
    for sec in generated.get("sections", []):
        mp = sec.get("macro_part")
        by_mp.setdefault(mp, []).append(sec)

    for mp in sorted(by_mp.keys()):
        assembled["macro_parts"].append({
            "macro_part": mp,
            "macro_part_name": by_mp[mp][0].get("macro_part_name"),
            "sections": [
                {"bucket_id": s.get("bucket_id"), "text": (s.get("final_text") or s.get("generated_text") or "").strip()}
                for s in by_mp[mp]
            ]
        })

    assembled_path = out_dir / "assembled_report.json"
    save_json(assembled_path, assembled)

    print(f"Wrote: {gen_path}")
    print(f"Wrote: {assembled_path}")

    # Full-report deterministic scoring (still authoritative gate) [1](https://discuss.huggingface.co/t/500-internal-server-error-with-inference-endpoint/89605)
    quality = evaluate_quality(generated, assembled)
    q_path = Path(args.quality) if args.quality else (out_dir / "quality_report.json")
    save_json(q_path, quality)

    print(f"Wrote: {q_path}")
    print(f"Quality total: {quality.get('total')} / 100")

    if args.min_quality and (quality.get("total", 0) < args.min_quality):
        raise SystemExit(2)


if __name__ == "__main__":
    main()