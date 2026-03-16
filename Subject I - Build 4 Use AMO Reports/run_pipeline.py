#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_pipeline.py

Minor update:
- Pass --onenote <process/onenote/<notebook>> to run_llm_jobs.py so it can resolve image assets
  from processed OneNote pages and populate slides with images.

All previous behavior is preserved.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from section_context import build_section_context, assert_same_section_context

try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

DEFAULT_CASE_ID = "P060011"
DEFAULT_NOTEBOOK_NAME = "test"
DEFAULT_ONENOTE_SECTION = "Clinique - Goussonville"

ROOT = Path(__file__).resolve().parent
REPORT_TYPES_CONFIG = ROOT / "input" / "config" / "report_types.json"
PROMPT_TEMPLATES = ROOT / "input" / "config" / "prompt_templates.json"
PPTX_TEMPLATE = ROOT / "input" / "templates" / "TEMPLATE_AUDIT_BUILD4USE.pptx"
DEFAULT_BACS_RULES = ROOT / "input" / "rules" / "bacs_table6_rules_structured_clean.json"

ONENOTE_EXPORT_ROOT = ROOT / "input" / "onenote-exporter" / "output"
PROCESS_ROOT = ROOT / "process"
ONENOTE_PROCESSED_ROOT = PROCESS_ROOT / "onenote"
PLANS_ROOT = PROCESS_ROOT / "plans"
DRAFTS_ROOT = PROCESS_ROOT / "drafts"
SKELETONS_DIR = PROCESS_ROOT / "learning" / "skeletons"
OUTPUT_ROOT = ROOT / "output"

DEFAULT_MIN_QUALITY_SCORE = 0.0
LLM_TEMPERATURE = "0.2"
LLM_MAX_TOKENS = "1500"
DEFAULT_LLM_MODE = "multistep"


def run(cmd: list[str], *, cwd: Path = ROOT):
    print("\n>", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print(f"\nERROR: Pipeline failed at step: {' '.join(cmd)}")
        sys.exit(res.returncode)


def main():
    ap = argparse.ArgumentParser(description="Run the GTB/BACS generation pipeline end-to-end.")
    ap.add_argument("--case-id", default=DEFAULT_CASE_ID)
    ap.add_argument("--notebook", default=DEFAULT_NOTEBOOK_NAME)
    ap.add_argument("--onenote-section", default=DEFAULT_ONENOTE_SECTION)
    ap.add_argument("--mode", choices=["single", "multistep"], default=DEFAULT_LLM_MODE)
    ap.add_argument("--min-quality", type=float, default=DEFAULT_MIN_QUALITY_SCORE)
    ap.add_argument("--bacs-rules", default=str(DEFAULT_BACS_RULES))
    ap.add_argument("--bacs-building-scope", default="Non résidentiel", choices=["Résidentiel", "Non résidentiel"])
    ap.add_argument("--bacs-targets", default="")
    ap.add_argument("--bacs-part2-slides", action="store_true")
    args = ap.parse_args()

    notebook_name = args.notebook
    onenote_section = (args.onenote_section or "").strip()
    case_id = (args.case_id or "").strip() or DEFAULT_CASE_ID

    llm_mode = args.mode
    min_quality = args.min_quality

    notebook_processed = ONENOTE_PROCESSED_ROOT / notebook_name
    plan_path = PLANS_ROOT / f"{case_id}.json"
    draft_dir = DRAFTS_ROOT / case_id
    output_reports = OUTPUT_ROOT / "reports" / case_id
    output_reports.mkdir(parents=True, exist_ok=True)

    expected_ctx = build_section_context(notebook_name, onenote_section) if onenote_section else None

    # 1) OneNote processing
    run([
        sys.executable,
        "process_onenote.py",
        notebook_name,
        "--input",
        str(ONENOTE_EXPORT_ROOT),
        "--out",
        str(ONENOTE_PROCESSED_ROOT),
        "--transcribe",
    ])

    # 2) Section aggregation
    agg_path = None
    if onenote_section:
        run([
            sys.executable,
            "aggregate_onenote_section.py",
            "--onenote",
            str(notebook_processed),
            "--section",
            onenote_section,
            "--out",
            str((ROOT / "process" / "onenote_aggregates")),
        ])
        section_slug = expected_ctx.get('section_slug') if isinstance(expected_ctx, dict) else None
        if not section_slug:
            section_slug = build_section_context(notebook_name, onenote_section).get('section_slug')
        agg_path = (ROOT / "process" / "onenote_aggregates" / notebook_name / f"{section_slug}.json").resolve()
        agg_obj = json.loads(agg_path.read_text(encoding="utf-8"))
        if expected_ctx and isinstance(agg_obj.get("section_context"), dict):
            assert_same_section_context(expected_ctx, agg_obj["section_context"], "onenote_aggregate")

    # 3) Planning
    run([
        sys.executable,
        "plan_generation.py",
        "--config",
        str(REPORT_TYPES_CONFIG),
        "--skeletons",
        str(SKELETONS_DIR),
        "--onenote",
        str(notebook_processed),
        "--case-id",
        case_id,
        "--out",
        str(PLANS_ROOT),
        "--onenote-section",
        onenote_section,
    ])

    # 4) Draft generation
    cmd = [
        sys.executable,
        "generate_draft.py",
        "--plan",
        str(plan_path),
        "--onenote",
        str(notebook_processed),
        "--templates",
        str(PROMPT_TEMPLATES),
        "--out",
        str(DRAFTS_ROOT),
    ]
    if agg_path:
        cmd += ["--section-aggregate", str(agg_path)]
    run(cmd)

    # 5) LLM + assembly + quality
    llm_cmd = [
        sys.executable,
        "run_llm_jobs.py",
        "--bundle",
        str(draft_dir / "draft_bundle.json"),
        "--mode",
        llm_mode,
        "--temperature",
        LLM_TEMPERATURE,
        "--max_tokens",
        LLM_MAX_TOKENS,
        "--min_quality",
        str(min_quality),
        "--onenote",
        str(notebook_processed),
    ]

    rules_path = Path(args.bacs_rules)
    if rules_path.exists():
        llm_cmd += [
            "--bacs_rules",
            str(rules_path),
            "--bacs_building_scope",
            args.bacs_building_scope,
        ]
    if args.bacs_targets:
        llm_cmd += ["--bacs_targets", args.bacs_targets]
    if args.bacs_part2_slides:
        llm_cmd += ["--bacs_part2_slides"]

    run(llm_cmd)

    # 6) Render PPTX
    out_pptx = output_reports / "Rapport_Audit.pptx"
    run([
        sys.executable,
        "render_report_pptx.py",
        "--template",
        str(PPTX_TEMPLATE),
        "--assembled",
        str(draft_dir / "assembled_report.json"),
        "--out",
        str(out_pptx),
    ])

    print("\nOK: Pipeline completed successfully")
    print(f"Output: {out_pptx}")


if __name__ == "__main__":
    main()
