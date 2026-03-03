# 📘 GTB / BACS Report Generation System

**Purpose of this README**

This file is meant to be **pasted at the start of any future conversation** (human or LLM) so the full context of the project is immediately understood: architecture, data flow, constraints, and current implementation status.


## 0. Repository layout

This project follows a strict **compiler-style** separation of concerns at the filesystem level:

- **input/**: *human-made* or *external* inputs only
  - `input/reports/` → all historical human-made audit reports (keep their internal tree under this folder)
  - `input/onenote-exporter/output/` → OneNote exporter (Docker) output
  - `input/config/` → configuration files (`report_types.json`, `prompt_templates.json`)
  - `input/templates/` → PPTX templates (`TEMPLATE_AUDIT_BUILD4USE.pptx`, etc.)

- **process/**: everything generated *during the pipeline* **except** the final report
  - `process/onenote/<notebook>/...` → processed OneNote page packs
  - `process/plans/<case_id>.json` → per-case plan
  - `process/drafts/<case_id>/...` → prompts + generated_bundle + assembled_report + quality_report
  - `process/learning/...` → learning corpus and skeletons

- **output/**: final generated deliverables
  - `output/reports/<case_id>/Rapport_Audit.pptx`


## 1. What this project does

This project generates **PowerPoint audit reports** from **OneNote field data**, supporting two report families: **BACS / scoring reports (implemented)** and **État GTB reports (defined but not yet implemented)**.


## 2. Hard rules & constraints

- **Nothing under archive/ is used**. No design or implementation must rely on archived files.
- **OneNote extraction is external** and handled via a Docker container (onenote-exporter). Python code consumes its outputs only.
- **Macro‑part 4 exists in configuration but is intentionally never generated** (for all report types).
- The system is designed as a **compiler-style pipeline**, not a conversational generator.


## 3. Report types

Report types are defined in `input/config/report_types.json`. The system currently distinguishes **definition** from **implementation**.

### 3.1 ✅ Implemented report type: BACS_SCORING

**Purpose**
Audit GTB systems under the **Décret BACS / ISO 52120‑1**, with scoring and target projection.

**Macro‑parts**
- **État des lieux GTB** — generated, strictly descriptive
- **Scoring GTB actuel** — generated, norm-based (ISO 52120‑1)
- **Scoring projeté** — generated, inferred target (≥ classe B)
- **Mise en conformité & Bilan** — defined but **never generated**

**Buckets**
- ETAT_DES_LIEUX_GTB → macro‑part 1
- SCORING_ACTUEL → macro‑part 2
- SCORING_PROJETE → macro‑part 3
- CONFORMITE_BACS_BILAN → macro‑part 4 (not generated)

### 3.2 🚧 Defined but NOT implemented: ETAT_GTB_AUDIT

**Purpose**
Pure GTB technical audit **without BACS scoring** (architecture, diagnostics, actions).

**Macro‑parts**
- Généralités
- État du système GTB
- Diagnostics
- Plan d’actions (never generated)

**Buckets (already defined in config)**
- SITE_CONTEXTE → macro‑part 1
- ARCHI_GTB → macro‑part 2
- COMMUNICATIONS → macro‑part 2
- DIAGNOSTICS → macro‑part 3

⚠️ **Status**
The system can detect this report type and plan buckets for it, but **prompting, enforcement, and rendering are not implemented yet**.


## 4. Two distinct pipelines

### 4.0 Convenience runners

- **Generation runner**: `run_pipeline.py`
  - Reads from `input/` and writes intermediates to `process/` then final PPTX to `output/`

- **Learning runner**: `run_learning_pipeline.py`
  - Reads historical reports from `input/reports/` and writes corpus + skeletons into `process/learning/`

### 4.1 Generation pipeline

Used to generate **new audit reports**.

```Shell
OneNote export (Docker)
 ↓
process_onenote.py
 ↓
plan_generation.py
 ↓
generate_draft.py
 ↓
run_llm_jobs.py
 ↓
(enforcement / auto‑regen if enabled)
 ↓
render_report_pptx.py
```

**Output**: `output/reports/<case_id>/Rapport_Audit.pptx`

### 4.2 Learning pipeline

Used to ingest **existing audit reports** and learn structure priors.

```Shell
Existing reports (PDF / DOCX / PPTX)
 ↓
process_reports.py
 ↓
build_skeletons.py
 ↓
skeletons/*.json
```

This pipeline **does not generate reports**.


## 5. OneNote ingestion

### 5.1 onenote-exporter (Docker)
- OneNote data is exported using a Docker container: `onenote-exporter`
- This repository **does not implement OneNote extraction**

### 5.2 Expected exporter output

```Shell
input/onenote-exporter/output/<notebook>/**/**/*.md
```

These Markdown files are consumed by `process_onenote.py`.


## 6. Script reference — generation pipeline

### 6.1 process_onenote.py

**Role**
Convert OneNote Markdown exports into structured JSON page packs.

**Input**
- `input/onenote-exporter/output/<notebook>/**/**/*.md`

**Output**
- `process/onenote/<notebook>/pages/*.json`
- `process/onenote/<notebook>/assets/`
- `process/onenote/<notebook>/manifest.json`, `errors.jsonl`

**Usage**
```Shell
python process_onenote.py <notebook> --transcribe
```

### 6.2 plan_generation.py

**Role**
Build a per-case **generation plan**:
- detect report type
- select macro‑parts (1–3 only)
- select and score buckets
- route OneNote pages as evidence

**Input**
- `input/config/report_types.json`
- `process/onenote/<notebook>/pages/*.json`
- `process/learning/skeletons/*.json` (priors only)

**Output**
- `process/plans/<case_id>.json`

**Usage**
```Shell
python plan_generation.py --config input/config/report_types.json --skeletons process/learning/skeletons --onenote process/onenote/<notebook> --case-id <case_id>
```

### 6.3 generate_draft.py

**Role**
Translate the plan into **LLM writing jobs**.

**Input**
- `process/plans/<case_id>.json`
- `process/onenote/<notebook>/pages/*.json`
- `input/config/prompt_templates.json`

**Output**
- `process/drafts/<case_id>/draft_bundle.json`
- prompt text files for inspection

**Usage**
```Shell
python generate_draft.py --plan process/plans/<case_id>.json --onenote process/onenote/<notebook> --templates input/config/prompt_templates.json --out process/drafts
```

### 6.4 run_llm_jobs.py

**Role**
Execute LLM calls and assemble macro‑parts.

**Input**
- `process/drafts/<case_id>/draft_bundle.json`

**Output**
- `process/drafts/<case_id>/generated_bundle.json`
- `process/drafts/<case_id>/assembled_report.json` (macro‑parts 1–3)
- `process/drafts/<case_id>/quality_report.json`

**Usage**
```Shell
python run_llm_jobs.py --bundle process/drafts/<case_id>/draft_bundle.json
```

### 6.5 llm_client.py

**Role**
HuggingFace chat client wrapper (LLaMA‑3.x).

**Environment variables**
- HF_TOKEN (required)
- HF_MODEL (optional, default provided)

### 6.6 render_report_pptx.py

**Role**
Render a final PowerPoint report from `assembled_report.json` and a PPTX template.

**Input**
- `--template input/templates/TEMPLATE_AUDIT_BUILD4USE.pptx`
- `--assembled process/drafts/<case_id>/assembled_report.json`

**Output**
- `output/reports/<case_id>/Rapport_Audit.pptx`

**Usage**
```Shell
python render_report_pptx.py --template input/templates/TEMPLATE_AUDIT_BUILD4USE.pptx --assembled process/drafts/<case_id>/assembled_report.json --out output/reports/<case_id>/Rapport_Audit.pptx
```


## 7. Script reference — learning tools

### 7.1 process_reports.py

**Role**
Normalize existing audit reports (PDF / DOCX / PPTX) into a structured corpus.

**Input**
- `input/reports/**/{*.pdf,*.docx,*.pptx}`

**Output**
- `process/learning/processed_reports/Rapports_d_audit/`
  - `docs/`
  - `chunks/`
  - `assets/`
  - `manifest.json`

**Usage**
```Shell
python process_reports.py --audits input/reports --out process/learning/processed_reports --extract-images
```

### 7.2 build_skeletons.py

**Role**
Learn canonical report skeletons (structure priors) from historical reports.

**Input**
- `process/learning/processed_reports/Rapports_d_audit/`

**Output**
- `process/learning/skeletons/GLOBAL.json`
- `process/learning/skeletons/BACS.json`
- `process/learning/skeletons/Etat_GTB_PlansActions.json`
- `process/learning/skeletons/Interventions_Avancement.json`

**Usage**
```Shell
python build_skeletons.py --processed process/learning/processed_reports/Rapports_d_audit --out process/learning/skeletons
```

### 7.3 make_template_from_report.py

**Role**
Create a reusable PPTX template from a single report.

**Usage**
```Shell
python make_template_from_report.py --in <report>.pptx --out TEMPLATE_AUDIT_BACS.pptx --keep-parts 1 2 3 --strip-images
```

### 7.4 derive_template_from_reports.py

**Role**
Derive a PPTX template from multiple reports by identifying common slide layouts.


## 8. Implemented vs planned

### Implemented end‑to‑end
- BACS report generation (macro‑parts 1–3)
- OneNote → JSON normalization
- Planning & routing
- LLM draft generation
- PPTX rendering
- Corpus ingestion & skeleton learning

### Not implemented yet
- Full generation + enforcement + rendering for ETAT_GTB_AUDIT


## 9. Mental model recap
- This is a **compiler pipeline**, not a chatbot
- Planning is deterministic
- LLMs are used **only for phrasing**
- Semantics are enforced outside the LLM
- BACS and État‑GTB are distinct report families
- État‑GTB is defined but intentionally not implemented yet


## 10. Credits and Dependencies

- [hkevin01](https://github.com/hkevin01) for their OneNote Exporter.