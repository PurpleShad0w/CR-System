from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from src.classify_physical import classify_physical, load_yaml, save_yaml, normalize_decisions
from src.dwg_entities import extract_entities_df
from src.noise_filter import clean_physical_entities
from src.render_clean_plan_v3 import render_clean_plan_v3

st.set_page_config(page_title="Sujet III - clean v3", layout="wide")
st.title("Sujet III - DWG cleaner v3")
st.caption("Version bruit réduit : sélection physique, plus gros composant spatial, suppression micro-objets, crop final.")

rules_path = st.sidebar.text_input("Règles", "config/default_rules_clean_v3.yaml")
rules = load_yaml(rules_path, default={})
decisions_path = st.sidebar.text_input("Décisions layers", rules.get("layer_decisions_file", "data/work/layer_review/layer_decisions.yaml"))
decisions = load_yaml(decisions_path, default={"keep": [], "drop": [], "undecided": []})

uploaded = st.file_uploader("Fichier DWG/DXF", type=["dwg", "dxf"])
if uploaded is None:
    st.stop()

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / uploaded.name
    p.write_bytes(uploaded.read())
    df = extract_entities_df(p, converter=(rules.get("converter") or {}), include_blocks=True)

classified = classify_physical(df, rules, decisions)
clean = clean_physical_entities(df, rules, decisions)

cols = st.columns(6)
cols[0].metric("Brut", len(df))
cols[1].metric("Retenu", len(clean))
cols[2].metric("Murs", int((clean["feature_class"] == "wall").sum()) if not clean.empty else 0)
cols[3].metric("Portes", int((clean["feature_class"] == "door").sum()) if not clean.empty else 0)
cols[4].metric("Fenêtres", int((clean["feature_class"] == "window").sum()) if not clean.empty else 0)
cols[5].metric("Escaliers", int((clean["feature_class"] == "stairs").sum()) if not clean.empty else 0)

st.subheader("Diagnostic layers")
summary = classified.groupby("layer").agg(
    n=("entity_id", "count"),
    kept=("decision", lambda s: int((s == "keep").sum())),
    walls=("feature_class", lambda s: int((s == "wall").sum())),
    doors=("feature_class", lambda s: int((s == "door").sum())),
    windows=("feature_class", lambda s: int((s == "window").sum())),
    stairs=("feature_class", lambda s: int((s == "stairs").sum())),
    hard_noise=("hard_noise", "sum"),
    blocks=("source_block", lambda s: ", ".join(sorted({str(x) for x in s if str(x)}))[:250]),
).reset_index().sort_values(["kept", "n"], ascending=False)

edited = st.data_editor(summary, use_container_width=True, hide_index=True)

st.subheader("Rendu")
out_dir = Path("output/streamlit_preview")
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "rendered_clean_v3.png"
if st.button("Générer rendu propre v3", type="primary"):
    render_clean_plan_v3(df, out, rules=rules, decisions=decisions)
    st.image(str(out), use_container_width=True)

with st.expander("Entités physiques retenues"):
    st.dataframe(clean.drop(columns=["points"], errors="ignore"), use_container_width=True)
with st.expander("Entités classifiées brutes"):
    st.dataframe(classified.drop(columns=["points"], errors="ignore"), use_container_width=True)
