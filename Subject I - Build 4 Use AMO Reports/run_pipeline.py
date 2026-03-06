#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_pipeline.py

Single entrypoint to run the full GTB / BACS report generation pipeline.

Pipeline
--------
1. process_onenote.py (input -> process/onenote)
2. aggregate_onenote_section.py (process/onenote -> process/onenote_aggregates) [optional]
3. plan_generation.py (process/onenote -> process/plans)
4. generate_draft.py (process/plans + process/onenote -> process/drafts)
5. run_llm_jobs.py (process/drafts -> generated_bundle + assembled_report + quality_report)
6. render_report_pptx.py (process/drafts -> output/reports)

LLM MODE
--------
Default is now: multistep
You can override with: --mode single

HARDENING (small additions)
--------------------------
- Build a canonical section_context when --onenote-section is provided.
- Validate that aggregate/plan/draft/assembled all agree on the same section_context.

No behavior is removed; this is a fail-fast consistency gate.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from section_context import build_section_context, assert_same_section_context

# ============================================================================
# CONFIGURATION (DEFAULTS — can be overridden by CLI)
# ============================================================================
DEFAULT_CASE_ID = "P050011"  # legacy fallback if no OneNote section is provided
DEFAULT_NOTEBOOK_NAME = "test"  # frontmatter notebook: value
DEFAULT_ONENOTE_SECTION = ""  # e.g. "Oseraie - OSNY"

ROOT = Path(__file__).resolve().parent

# Inputs
REPORT_TYPES_CONFIG = ROOT / "input" / "config" / "report_types.json"
PROMPT_TEMPLATES = ROOT / "input" / "config" / "prompt_templates.json"
PPTX_TEMPLATE = ROOT / "input" / "templates" / "TEMPLATE_AUDIT_BUILD4USE.pptx"

# OneNote exporter output (Docker output)
ONENOTE_EXPORT_ROOT = ROOT / "input" / "onenote-exporter" / "output"

# Process folders
PROCESS_ROOT = ROOT / "process"
ONENOTE_PROCESSED_ROOT = PROCESS_ROOT / "onenote"
PLANS_ROOT = PROCESS_ROOT / "plans"
DRAFTS_ROOT = PROCESS_ROOT / "drafts"
SKELETONS_DIR = PROCESS_ROOT / "learning" / "skeletons"

# Outputs
OUTPUT_ROOT = ROOT / "output"

# Quality gate
DEFAULT_MIN_QUALITY_SCORE = 75.0

# LLM parameters
LLM_TEMPERATURE = "0.2"
LLM_MAX_TOKENS = "1200"

# NEW: default LLM mode
DEFAULT_LLM_MODE = "multistep"  # <- default requested


# ============================================================================
# UTIL
# ============================================================================

def run(cmd: list[str], *, cwd: Path = ROOT):
    print("\n▶", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print(f"\n❌ Pipeline failed at step: {' '.join(cmd)}")
        sys.exit(res.returncode)


# ============================================================================
# PIPELINE
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Run the GTB/BACS generation pipeline end-to-end.")
    ap.add_argument("--case-id", default=DEFAULT_CASE_ID, help="Case identifier fallback (used only if --onenote-section is empty)")
    ap.add_argument("--notebook", default=DEFAULT_NOTEBOOK_NAME, help="OneNote notebook name (frontmatter notebook:)")
    ap.add_argument(
        "--onenote-section",
        default=DEFAULT_ONENOTE_SECTION,
        help="OneNote section name that defines the report unit (e.g. 'Oseraie - OSNY'). If provided, case_id becomes its slug.",
    )
    ap.add_argument(
        "--mode",
        choices=["single", "multistep"],
        default=DEFAULT_LLM_MODE,
        help="LLM mode passed to run_llm_jobs.py (default: multistep)",
    )
    ap.add_argument(
        "--min-quality",
        type=float,
        default=DEFAULT_MIN_QUALITY_SCORE,
        help="Minimum quality score gate passed to run_llm_jobs.py",
    )
    args = ap.parse_args()

    def slugify_local(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "section"

    notebook_name = args.notebook
    onenote_section = (args.onenote_section or "").strip()

    case_id = slugify_local(onenote_section) if onenote_section else args.case_id
    llm_mode = args.mode
    min_quality = args.min_quality

    notebook_processed = ONENOTE_PROCESSED_ROOT / notebook_name
    plan_path = PLANS_ROOT / f"{case_id}.json"
    draft_dir = DRAFTS_ROOT / case_id
    output_reports = OUTPUT_ROOT / "reports" / case_id
    output_reports.mkdir(parents=True, exist_ok=True)

    # HARDENING: expected section_context (only in section mode)
    expected_ctx = build_section_context(notebook_name, onenote_section) if onenote_section else None

    # 1) OneNote → structured JSON (process/onenote/<notebook>)
    run(
        [
            sys.executable,
            "process_onenote.py",
            notebook_name,
            "--input",
            str(ONENOTE_EXPORT_ROOT),
            "--out",
            str(ONENOTE_PROCESSED_ROOT),
            "--transcribe",
        ]
    )

    # 2) NEW: OneNote section aggregation (report unit)
    if onenote_section:
        run(
            [
                sys.executable,
                "aggregate_onenote_section.py",
                "--onenote",
                str(notebook_processed),
                "--section",
                onenote_section,
                "--out",
                str((ROOT / "process" / "onenote_aggregates")),
            ]
        )

        # HARDENING: validate aggregate identity
        agg_path = (ROOT / "process" / "onenote_aggregates" / notebook_name / f"{case_id}.json").resolve()
        try:
            agg_obj = json.loads(agg_path.read_text(encoding="utf-8"))
            if expected_ctx and isinstance(agg_obj.get("section_context"), dict):
                assert_same_section_context(expected_ctx, agg_obj["section_context"], "onenote_aggregate")
        except Exception as e:
            print(f"❌ Failed to validate aggregate section_context: {e}")
            raise

    # 3) Planning (process/plans/<case_id>.json)
    run(
        [
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
        ]
    )

    # HARDENING: validate plan identity
    if expected_ctx:
        plan_obj = json.loads(plan_path.read_text(encoding="utf-8"))
        if isinstance(plan_obj.get("section_context"), dict):
            assert_same_section_context(expected_ctx, plan_obj["section_context"], "plan")

    # 4) Draft generation (process/drafts/<case_id>/...)
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

    # If you're using section aggregates, pass its path (derived from section slug)
    if onenote_section:
        agg_path = (ROOT / "process" / "onenote_aggregates" / notebook_name / f"{case_id}.json").resolve()
        cmd += ["--section-aggregate", str(agg_path)]

    run(cmd)

    # HARDENING: validate draft identity
    if expected_ctx:
        draft_obj = json.loads((draft_dir / "draft_bundle.json").read_text(encoding="utf-8"))
        if isinstance(draft_obj.get("section_context"), dict):
            assert_same_section_context(expected_ctx, draft_obj["section_context"], "draft_bundle")

    # 5) LLM execution + assembly + quality gate
    run(
        [
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
        ]
    )

    # HARDENING: validate assembled identity
    if expected_ctx:
        assembled_obj = json.loads((draft_dir / "assembled_report.json").read_text(encoding="utf-8"))
        if isinstance(assembled_obj.get("section_context"), dict):
            assert_same_section_context(expected_ctx, assembled_obj["section_context"], "assembled_report")

    # 6) PPTX rendering → output/reports/<case_id>/Rapport_Audit.pptx
    out_pptx = output_reports / "Rapport_Audit.pptx"
    run(
        [
            sys.executable,
            "render_report_pptx.py",
            "--template",
            str(PPTX_TEMPLATE),
            "--assembled",
            str(draft_dir / "assembled_report.json"),
            "--out",
            str(out_pptx),
        ]
    )

    print("\n✅ Pipeline completed successfully")
    print(f"📄 Output: {out_pptx}")


if __name__ == "__main__":
    main()
