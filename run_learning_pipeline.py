#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_learning_pipeline.py

Single entrypoint for the *learning/reference* pipeline.

Folder convention (project reorg):
- input/   : human-made reports
- process/ : intermediate artifacts (corpus, chunks, skeletons)
- output/  : final generated reports (not used here)

This runner performs:
1) process_reports.py  -> normalizes input reports into a corpus
2) build_skeletons.py  -> learns canonical skeletons from that corpus

All parameters are defined below.
"""

import subprocess
import sys
from pathlib import Path

# =============================================================================
# CONFIGURATION (EDIT HERE)
# =============================================================================

ROOT = Path(__file__).resolve().parent

# Input: human-made reports (keep existing tree under this folder)
INPUT_REPORTS_DIR = ROOT / 'input' / 'reports'

# Process outputs
PROCESSED_REPORTS_ROOT = ROOT / 'process' / 'learning' / 'processed_reports'
PROCESSED_COLLECTION = PROCESSED_REPORTS_ROOT / 'Rapports_d_audit'  # created by process_reports.py
SKELETONS_OUT_DIR = ROOT / 'process' / 'learning' / 'skeletons'

# Options
EXTRACT_IMAGES = True   # set False if you don't need PPTX image extraction

# =============================================================================
# UTIL
# =============================================================================

def run(cmd: list[str], *, cwd: Path = ROOT):
    print("\n▶", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        print(f"\n❌ Learning pipeline failed at step: {' '.join(cmd)}")
        sys.exit(res.returncode)

# =============================================================================
# PIPELINE
# =============================================================================

def main():
    # Ensure dirs exist
    INPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    SKELETONS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Ingest/normalize human-made reports -> process/learning/processed_reports/Rapports_d_audit
    cmd1 = [
        sys.executable,
        'process_reports.py',
        '--root', '.',
        '--audits', str(INPUT_REPORTS_DIR),
        '--out', str(PROCESSED_REPORTS_ROOT),
    ]
    if EXTRACT_IMAGES:
        cmd1.append('--extract-images')
    run(cmd1)

    # 2) Learn skeletons -> process/learning/skeletons
    run([
        sys.executable,
        'build_skeletons.py',
        '--processed', str(PROCESSED_COLLECTION),
        '--out', str(SKELETONS_OUT_DIR),
    ])

    print("\n✅ Learning pipeline completed successfully")
    print(f"📦 Corpus: {PROCESSED_COLLECTION}")
    print(f"🦴 Skeletons: {SKELETONS_OUT_DIR}")


if __name__ == '__main__':
    main()
