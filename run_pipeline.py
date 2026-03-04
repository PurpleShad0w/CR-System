#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py

Single entrypoint to run the full GTB / BACS report generation pipeline.

NEW FOLDER CONVENTION (project reorg)
-------------------------------------
- Inputs live under: input/
- Pipeline artifacts: process/
- Final reports: output/

Pipeline
--------
1. process_onenote.py        (input -> process/onenote)
2. plan_generation.py        (process/onenote -> process/plans)
3. generate_draft.py         (process/plans + process/onenote -> process/drafts)
4. run_llm_jobs.py           (process/drafts -> generated_bundle + assembled_report + quality_report)
5. render_report_pptx.py     (process/drafts -> output/reports)

LLM MODE
--------
Default is now: multistep
You can override with: --mode single
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION (DEFAULTS — can be overridden by CLI)
# ============================================================================
DEFAULT_CASE_ID = "P050011"
DEFAULT_NOTEBOOK_NAME = "test"  # frontmatter notebook: value

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
    ap.add_argument("--case-id", default=DEFAULT_CASE_ID, help="Case identifier (used for plan + output folder)")
    ap.add_argument("--notebook", default=DEFAULT_NOTEBOOK_NAME, help="OneNote notebook name (frontmatter notebook:)")
    ap.add_argument("--mode", choices=["single", "multistep"], default=DEFAULT_LLM_MODE,
                    help="LLM mode passed to run_llm_jobs.py (default: multistep)")
    ap.add_argument("--min-quality", type=float, default=DEFAULT_MIN_QUALITY_SCORE,
                    help="Minimum quality score gate passed to run_llm_jobs.py")
    args = ap.parse_args()

    case_id = args.case_id
    notebook_name = args.notebook
    llm_mode = args.mode
    min_quality = args.min_quality

    notebook_processed = ONENOTE_PROCESSED_ROOT / notebook_name
    plan_path = PLANS_ROOT / f"{case_id}.json"
    draft_dir = DRAFTS_ROOT / case_id

    output_reports = OUTPUT_ROOT / "reports" / case_id
    output_reports.mkdir(parents=True, exist_ok=True)

    # 1) OneNote → structured JSON (process/onenote/<notebook>)
    run([
        sys.executable,
        "process_onenote.py",
        notebook_name,
        "--input", str(ONENOTE_EXPORT_ROOT),
        "--out", str(ONENOTE_PROCESSED_ROOT),
        "--transcribe",
    ])

    # 2) Planning (process/plans/<case_id>.json)
    run([
        sys.executable,
        "plan_generation.py",
        "--config", str(REPORT_TYPES_CONFIG),
        "--skeletons", str(SKELETONS_DIR),
        "--onenote", str(notebook_processed),
        "--case-id", case_id,
        "--out", str(PLANS_ROOT),
    ])

    # 3) Draft generation (process/drafts/<case_id>/...)
    run([
        sys.executable,
        "generate_draft.py",
        "--plan", str(plan_path),
        "--onenote", str(notebook_processed),
        "--templates", str(PROMPT_TEMPLATES),
        "--out", str(DRAFTS_ROOT),
    ])

    # 4) LLM execution + assembly + quality gate (still under process/drafts/<case_id>)
    # NEW: pass --mode (default multistep, overridable)
    run([
        sys.executable,
        "run_llm_jobs.py",
        "--bundle", str(draft_dir / "draft_bundle.json"),
        "--mode", llm_mode,
        "--temperature", LLM_TEMPERATURE,
        "--max_tokens", LLM_MAX_TOKENS,
        "--min_quality", str(min_quality),
    ])

    # 5) PPTX rendering → output/reports/<case_id>/Rapport_Audit.pptx
    out_pptx = output_reports / "Rapport_Audit.pptx"
    run([
        sys.executable,
        "render_report_pptx.py",
        "--template", str(PPTX_TEMPLATE),
        "--assembled", str(draft_dir / "assembled_report.json"),
        "--out", str(out_pptx),
    ])

    print("\n✅ Pipeline completed successfully")
    print(f"📄 Output: {out_pptx}")

if __name__ == "__main__":
    main()
