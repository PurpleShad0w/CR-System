from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from src.dwg_entities import extract_entities_df, layer_summary
from src.feature_classifier import build_review_table, decisions_from_review_table, load_yaml, save_yaml, classify_features
from src.render_clean_plan import render_clean_plan
from src.render_25d import render_aerial_25d

st.set_page_config(page_title="Sujet III - DWG cleaner v2", layout="wide")
st.title("Sujet III - Plan propre + vue aérienne 2.5D")

rules_path = st.sidebar.text_input("Règles", "config/default_rules_aerial_25d.yaml")
rules = load_yaml(rules_path, default={})
default_decisions_path = rules.get("layer_decisions_file") or "data/work/layer_review/layer_decisions.yaml"
decisions_path = st.sidebar.text_input("Décisions layers", default_decisions_path)
decisions = load_yaml(decisions_path, default={"keep": [], "drop": [], "undecided": []})

uploaded = st.file_uploader("Fichier DWG/DXF", type=["dxf", "dwg"])
if uploaded is None:
    st.info("Dépose un plan DWG/DXF. La v2 garde mieux les portes/fenêtres/blocs et génère une vue aérienne, pas une extrusion 3D.")
    st.stop()

with tempfile.TemporaryDirectory() as tmp:
    in_path = Path(tmp) / uploaded.name
    in_path.write_bytes(uploaded.read())
    with st.spinner("Extraction des entités CAD et blocs..."):
        df = extract_entities_df(in_path, converter=(rules.get("converter") or {}), include_blocks=True)

classified = classify_features(df, rules, decisions)
st.subheader("Résumé")
cols = st.columns(6)
cols[0].metric("Entités", len(classified))
cols[1].metric("Layers", classified["layer"].nunique() if not classified.empty else 0)
cols[2].metric("Murs", int((classified["feature_class"] == "wall").sum()))
cols[3].metric("Portes", int((classified["feature_class"] == "door").sum()))
cols[4].metric("Fenêtres", int((classified["feature_class"] == "window").sum()))
cols[5].metric("Escaliers", int((classified["feature_class"] == "stairs").sum()))

st.subheader("Revue des layers")
review = build_review_table(df, rules, decisions)
if review.empty:
    st.warning("Aucun layer détecté.")
    st.stop()

edited = st.data_editor(
    review,
    use_container_width=True,
    hide_index=True,
    column_config={
        "decision": st.column_config.SelectboxColumn("decision", options=["keep", "drop", "undecided"]),
        "wall_score": st.column_config.NumberColumn("wall", format="%.2f"),
        "door_score": st.column_config.NumberColumn("door", format="%.2f"),
        "window_score": st.column_config.NumberColumn("window", format="%.2f"),
        "stairs_score": st.column_config.NumberColumn("stairs", format="%.2f"),
    },
)

if st.button("Enregistrer layer_decisions.yaml", type="primary"):
    save_yaml(decisions_from_review_table(edited), decisions_path)
    st.success(f"Décisions enregistrées dans {decisions_path}")

st.subheader("Prévisualisation")
preview_dir = Path("output/streamlit_preview")
preview_dir.mkdir(parents=True, exist_ok=True)
preview_2d = preview_dir / "clean_plan_v2.png"
preview_aerial = preview_dir / "aerial_25d_v2.png"
current_decisions = decisions_from_review_table(edited)

left, right = st.columns(2)
with left:
    if st.button("Rendu 2D propre v2"):
        render_clean_plan(df, preview_2d, rules=rules, decisions=current_decisions, preview=True)
        st.image(str(preview_2d), use_container_width=True)
with right:
    if st.button("Vue aérienne 2.5D"):
        render_aerial_25d(df, preview_aerial, rules=rules, decisions=current_decisions, preview=True)
        st.image(str(preview_aerial), use_container_width=True)

with st.expander("Entités classifiées"):
    st.dataframe(classified.drop(columns=["points"], errors="ignore"), use_container_width=True)
with st.expander("Résumé layers brut"):
    st.dataframe(layer_summary(df), use_container_width=True)
