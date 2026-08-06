from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from src.dwg_entities import extract_entities_df, layer_summary
from src.layer_classifier import (
    build_review_table,
    decisions_from_review_table,
    load_yaml,
    save_yaml,
)
from src.render_clean_plan import render_clean_plan
from src.render_25d import export_25d_html

st.set_page_config(page_title="Sujet III - DWG cleaner", layout="wide")
st.title("Sujet III - Revue interactive DWG/DXF")

rules_path = st.sidebar.text_input("Règles", "config/default_rules.yaml")
rules = load_yaml(rules_path, default={})
default_decisions_path = (rules.get("layer_decisions_file") or "data/work/layer_review/layer_decisions.yaml")
decisions_path = st.sidebar.text_input("Décisions layers", default_decisions_path)
decisions = load_yaml(decisions_path, default={"keep": [], "drop": [], "undecided": []})

uploaded = st.file_uploader("Fichier DWG/DXF", type=["dxf", "dwg"])
if uploaded is None:
    st.info("Dépose un plan DWG/DXF pour générer entities_df, revoir les layers et produire le rendu propre.")
    st.stop()

with tempfile.TemporaryDirectory() as tmp:
    in_path = Path(tmp) / uploaded.name
    in_path.write_bytes(uploaded.read())
    with st.spinner("Extraction des entités CAD..."):
        df = extract_entities_df(in_path, converter=(rules.get("converter") or {}))

st.subheader("Résumé global")
col1, col2, col3 = st.columns(3)
col1.metric("Entités", len(df))
col2.metric("Layers", df["layer"].nunique() if not df.empty else 0)
col3.metric("Types", df["entity_type"].nunique() if not df.empty else 0)

st.subheader("Revue des layers")
review = build_review_table(df, rules, decisions)
if not review.empty:
    edited = st.data_editor(
        review,
        use_container_width=True,
        hide_index=True,
        column_config={
            "decision": st.column_config.SelectboxColumn("decision", options=["keep", "drop", "undecided"]),
            "suggested_action": st.column_config.TextColumn("suggestion"),
            "wall_score_mean": st.column_config.NumberColumn("wall_score", format="%.2f"),
        },
    )
    if st.button("Enregistrer layer_decisions.yaml", type="primary"):
        save_yaml(decisions_from_review_table(edited), decisions_path)
        st.success(f"Décisions enregistrées dans {decisions_path}")
else:
    st.warning("Aucun layer détecté.")
    st.stop()

st.subheader("Prévisualisation")
preview_dir = Path("output/streamlit_preview")
preview_dir.mkdir(parents=True, exist_ok=True)
preview_png = preview_dir / "rendered_clean_preview.png"
preview_html = preview_dir / "rendered_25d_preview.html"
current_decisions = decisions_from_review_table(edited)

left, right = st.columns([1, 1])
with left:
    if st.button("Rendu 2D propre"):
        render_clean_plan(df, preview_png, rules=rules, decisions=current_decisions, preview=True)
        st.image(str(preview_png), use_container_width=True)
with right:
    if st.button("Vue 2.5D"):
        export_25d_html(df, preview_html, rules=rules, decisions=current_decisions)
        st.components.v1.html(preview_html.read_text(encoding="utf-8"), height=700, scrolling=True)

with st.expander("entities_df"):
    st.dataframe(df.drop(columns=["points"], errors="ignore"), use_container_width=True)
with st.expander("layer_summary"):
    st.dataframe(layer_summary(df), use_container_width=True)
