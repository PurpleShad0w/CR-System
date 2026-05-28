#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / 'src'
# Ensure repo + src are importable
for p in (REPO_ROOT, SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from src.page_cards.runner import run_page_cards
from ui.onenote_cloud import onenote_cloud_ui


def list_local_notebooks(repo_root: Path) -> list[str]:
    base = repo_root / "input" / "onenote-exporter" / "output"
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def load_presets() -> list[dict]:
    for p in (REPO_ROOT / 'input' / 'config' / 'presets.json', REPO_ROOT / 'presets.json'):
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding='utf-8'))
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get('presets'), list):
                return [x for x in obj['presets'] if isinstance(x, dict)]
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
    return []


def main():
    st.set_page_config(page_title='AMO Reports', layout='wide')
    st.title('AMO Reports Generator')
    st.caption('Dual app: Main (Page Cards), OneNote (Microsoft Graph download), Legacy pipeline.')

    tabs = st.tabs(["Page Cards", "OneNote (Graph)", "Legacy"])

    with tabs[0]:
        st.subheader("Page Cards")

        # Presets (si tu veux garder la logique)
        presets = load_presets()
        preset_names = ["—"] + [p.get("name", "") for p in presets if p.get("name")]
        selected = st.selectbox("Preset", options=preset_names, index=0)

        # Valeurs par défaut
        if "pc_notebook" not in st.session_state:
            st.session_state["pc_notebook"] = "test"
        if "pc_case_id" not in st.session_state:
            st.session_state["pc_case_id"] = "Savills"
        if "pc_section" not in st.session_state:
            st.session_state["pc_section"] = 'Visite de site'
        if "pc_max_images" not in st.session_state:
            st.session_state["pc_max_images"] = 6
        if "pc_max_bullets" not in st.session_state:
            st.session_state["pc_max_bullets"] = 20

        # Appliquer preset -> champs (notebook/case/section)
        if selected != "—":
            p = next((x for x in presets if x.get("name") == selected), None)
            if p:
                st.session_state["pc_notebook"] = p.get("notebook", st.session_state["pc_notebook"])
                st.session_state["pc_case_id"] = p.get("case_id", st.session_state["pc_case_id"])
                st.session_state["pc_section"] = p.get("onenote_section", st.session_state["pc_section"])
                st.session_state["pc_max_images"] = int(p.get("max_images", st.session_state["pc_max_images"]))
                st.session_state["pc_max_bullets"] = int(p.get("max_bullets", st.session_state["pc_max_bullets"]))

        # Ligne de saisie: Notebook / Case ID / Section
        c1, c2, c3 = st.columns([1, 1, 2])

        with c1:
            nbs = list_local_notebooks(REPO_ROOT)
            cur = (st.session_state.get("pc_notebook") or "").strip()
            if nbs:
                options = nbs if cur in nbs else ([cur] + nbs if cur else nbs)
                st.selectbox("Notebook", options=options, key="pc_notebook")
            else:
                st.text_input("Notebook", key="pc_notebook")

        with c2:
            st.text_input("Case ID", key="pc_case_id")

        with c3:
            st.text_input("Section OneNote", key="pc_section")

        # Options
        o1, o2 = st.columns([1, 1])
        with o1:
            st.number_input("Max Images", min_value=0, max_value=6, step=1, key="pc_max_images")
        with o2:
            st.number_input("Max Bullets", min_value=10, max_value=30, step=1, key="pc_max_bullets")


        run_btn = st.button("Lancer Page Cards", type="primary")
        out_box = st.empty()

        if run_btn:
            argv = [
                "--case-id", (st.session_state["pc_case_id"] or "").strip(),
                "--section-name", (st.session_state["pc_section"] or "").strip(),
                "--notebook", (st.session_state["pc_notebook"] or "").strip(),
                "--max-images", str(int(st.session_state["pc_max_images"])),
                "--max-bullets", str(int(st.session_state["pc_max_bullets"])),
            ]

            rc, out = run_page_cards(argv, root=REPO_ROOT)
            if rc == 0:
                out_box.success("OK")
            else:
                out_box.error(f"Erreur (code {rc})")
            st.code(out, language="text")

    with tabs[1]:
        onenote_cloud_ui(REPO_ROOT)

    with tabs[2]:
        st.info("Legacy pipeline à brancher ici (si besoin).")


if __name__ == '__main__':
    main()
