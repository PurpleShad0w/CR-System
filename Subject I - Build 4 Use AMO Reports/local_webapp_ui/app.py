import time
from pathlib import Path

import streamlit as st

from pipeline_runner import (
    PipelineConfig,
    OneNoteExportConfig,
    LearningPipelineConfig,
    run_pipeline_streaming,
    run_onenote_export_streaming,
    run_learning_pipeline_streaming,
    compute_expected_outputs,
)
from utils import load_json, safe_read_text

st.set_page_config(page_title="Build4Use – Interface locale", page_icon="🧩", layout="wide")

project_root_default = Path(__file__).resolve().parent.parent

# Sidebar
st.sidebar.title("🧭 Paramètres")
project_root = st.sidebar.text_input("Chemin projet", value=str(project_root_default), help="Dossier contenant run_pipeline.py")
project_root_p = Path(project_root).resolve()

preset_path = Path(__file__).parent / 'presets.json'
presets = load_json(preset_path).get('presets', []) if preset_path.exists() else []
preset_names = [p['name'] for p in presets]
selected_preset = st.sidebar.selectbox("Profil génération", options=(['—'] + preset_names) if preset_names else ['—'])

# Defaults
notebook = "test"
onenote_section = "Clinique - Goussonville"
case_id = "P060011"
mode = "multistep"
min_quality = 0.0
bacs_building_scope = "Non résidentiel"
bacs_part2_slides = False

if presets and selected_preset != '—':
    p = next(x for x in presets if x['name'] == selected_preset)
    notebook = p.get('notebook', notebook)
    onenote_section = p.get('onenote_section', onenote_section)
    case_id = p.get('case_id', case_id)
    mode = p.get('mode', mode)
    min_quality = float(p.get('min_quality', min_quality))
    bacs_building_scope = p.get('bacs_building_scope', bacs_building_scope)
    bacs_part2_slides = bool(p.get('bacs_part2_slides', bacs_part2_slides))

st.title("🧩 Build4Use – Interface locale")
st.caption("Génération rapport + export OneNote + learning pipeline depuis une UI web locale.")

if not (project_root_p / 'run_pipeline.py').exists():
    st.warning(f"run_pipeline.py introuvable dans: {project_root_p} — corrige le chemin dans la sidebar")


tab1, tab2, tab3 = st.tabs(["📄 Générer un rapport", "📤 Export OneNote", "🧠 Learning / Skeletons"])

# ------------- Tab 1: generation -------------
with tab1:
    st.subheader("Génération de rapport")

    colL, colR = st.columns([1, 1])
    with colL:
        notebook_in = st.text_input("Notebook", value=notebook)
        section_in = st.text_input("Section OneNote", value=onenote_section)
        case_in = st.text_input("Case ID", value=case_id)
    with colR:
        mode_in = st.selectbox("Mode LLM", options=["multistep", "single"], index=0 if mode == 'multistep' else 1)
        minq_in = st.slider("Seuil qualité (min)", 0.0, 100.0, float(min_quality), 1.0)
        rules_default = str(project_root_p / 'input' / 'rules' / 'bacs_table6_rules_structured_clean.json')
        bacs_rules = st.text_input("Règles Tableau 6", value=rules_default)
        bscope = st.selectbox("Scope", options=["Non résidentiel", "Résidentiel"], index=0 if bacs_building_scope == 'Non résidentiel' else 1)
        btgt = st.text_input("Targets (optionnel JSON)", value="")
        bslides = st.checkbox("Partie 2 en slides markdown (LLM)", value=bacs_part2_slides)

    run_btn = st.button("🚀 Générer le rapport", type="primary")
    log_box = st.empty()
    status = st.empty()

    if 'logs_gen' not in st.session_state:
        st.session_state.logs_gen = []

    if run_btn:
        st.session_state.logs_gen = []
        cfg = PipelineConfig(
            project_root=project_root_p,
            notebook=notebook_in.strip(),
            onenote_section=section_in.strip(),
            case_id=case_in.strip(),
            mode=mode_in,
            min_quality=float(minq_in),
            bacs_rules=bacs_rules.strip(),
            bacs_building_scope=bscope,
            bacs_targets=btgt.strip(),
            bacs_part2_slides=bool(bslides),
        )
        status.info("Lancement…")
        _, lines = run_pipeline_streaming(cfg)
        for ln in lines:
            st.session_state.logs_gen.append(ln)
            log_box.code("\n".join(st.session_state.logs_gen[-400:]), language='text')
            time.sleep(0.01)

        outs = compute_expected_outputs(project_root_p, cfg.case_id)
        pptx_path = outs['pptx']
        if pptx_path.exists():
            status.success(f"Rapport généré : {pptx_path}")
            with open(pptx_path, 'rb') as f:
                st.download_button("⬇️ Télécharger le PPTX", data=f.read(), file_name=pptx_path.name,
                                   mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**assembled_report.json**")
                st.code(safe_read_text(outs['assembled']), language='json')
            with col2:
                st.markdown("**quality_report.json**")
                st.code(safe_read_text(outs['quality_report']), language='json')
        else:
            status.warning("PPTX non trouvé — consulte les logs.")

# ------------- Tab 2: OneNote export (with picker) -------------
with tab2:
    st.subheader("Export OneNote (native, Microsoft Graph)")
    st.caption("Le bouton 'Lister' récupère les notebooks puis tu peux en sélectionner un pour remplir automatiquement les champs.")

    def parse_notebooks(lines: list[str]) -> list[dict]:
        rows = []
        for ln in lines:
            ln = (ln or '').strip()
            if not ln or '\t' not in ln:
                continue
            name, nid = ln.split('\t', 1)
            name = name.strip()
            nid = nid.strip()
            if name and nid:
                rows.append({'Nom': name, 'ID': nid})
        # Dedup
        seen = set(); out = []
        for r in rows:
            if r['ID'] in seen:
                continue
            seen.add(r['ID'])
            out.append(r)
        return out

    if 'onenote_rows' not in st.session_state:
        st.session_state.onenote_rows = []
    if 'onenote_pick_name' not in st.session_state:
        st.session_state.onenote_pick_name = ''
    if 'onenote_pick_id' not in st.session_state:
        st.session_state.onenote_pick_id = ''

    colA, colB = st.columns([1, 1])
    with colA:
        env_path = st.text_input("Fichier .env", value=str(project_root_p / '.env'))
        out_dir = st.text_input("Output dir", value=str(project_root_p / 'input' / 'onenote-exporter' / 'output'))
        token_cache = st.text_input("Token cache", value=str(project_root_p / 'input' / 'onenote-exporter' / 'cache' / 'token_cache.json'))
        formats = st.text_input("Formats", value='md')
        merge = st.checkbox("Générer merged.md", value=False)
    with colB:
        notebook_name = st.text_input("Notebook (nom, match partiel)", value=st.session_state.onenote_pick_name)
        notebook_id = st.text_input("Notebook ID (exact)", value=st.session_state.onenote_pick_id)

    colX, colY = st.columns([1, 1])
    list_btn = colX.button("📋 Lister les notebooks")
    export_btn = colY.button("📤 Exporter")

    log_box2 = st.empty(); status2 = st.empty()
    if 'logs_export' not in st.session_state:
        st.session_state.logs_export = []

    def run_export(list_only: bool):
        st.session_state.logs_export = []
        cfg = OneNoteExportConfig(
            project_root=project_root_p,
            env_path=env_path.strip(),
            list_only=list_only,
            notebook_name=notebook_name.strip(),
            notebook_id=notebook_id.strip(),
            merge=bool(merge),
            formats=formats.strip(),
            output_dir=out_dir.strip(),
            token_cache=token_cache.strip(),
        )
        status2.info("Lancement…")
        _, lines = run_onenote_export_streaming(cfg)
        collected = []
        for ln in lines:
            collected.append(ln)
            st.session_state.logs_export.append(ln)
            log_box2.code("\n".join(st.session_state.logs_export[-400:]), language='text')
            time.sleep(0.01)
        return collected

    if list_btn:
        lines = run_export(True)
        rows = parse_notebooks(lines)
        st.session_state.onenote_rows = rows
        if rows:
            status2.success(f"Notebooks détectés : {len(rows)}")
        else:
            status2.warning("Aucun notebook détecté. Vérifie .env / permissions.")

    if st.session_state.onenote_rows:
        st.markdown("---")
        st.markdown("### Sélection rapide")
        filt = st.text_input("Filtrer", value="", placeholder="ex: Goussonville")
        rows = st.session_state.onenote_rows
        if filt.strip():
            f = filt.strip().lower()
            rows = [r for r in rows if f in r['Nom'].lower()]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        names = [r['Nom'] for r in rows]
        if names:
            chosen = st.selectbox("Choisir un notebook", options=names, index=0)
            chosen_row = next((r for r in rows if r['Nom'] == chosen), None)
            if chosen_row:
                st.session_state.onenote_pick_name = chosen_row['Nom']
                st.session_state.onenote_pick_id = chosen_row['ID']
                st.success("Sélection appliquée (champs remplis ci-dessus)")
                st.code(chosen_row['ID'], language='text')

    if export_btn:
        run_export(False)
        status2.success("Export terminé (voir logs).")

    with st.expander("ℹ️ Notes"):
        st.markdown("""
- Cette action exécute `python -m onenote_exporter.cli` du repo.
- Si `CLIENT_ID` manque dans `.env`, l'export s'arrête avec une erreur.
- Le device-flow peut demander un code à saisir dans le navigateur.
""")

# ------------- Tab 3: learning -------------
with tab3:
    st.subheader("Learning pipeline (corpus + skeletons)")
    st.caption("Exécute run_learning_pipeline.py : process_reports.py puis build_skeletons.py")

    run_learn = st.button("🧠 Lancer learning pipeline")
    log_box3 = st.empty(); status3 = st.empty()

    if 'logs_learn' not in st.session_state:
        st.session_state.logs_learn = []

    if run_learn:
        st.session_state.logs_learn = []
        cfg = LearningPipelineConfig(project_root=project_root_p)
        status3.info("Lancement…")
        _, lines = run_learning_pipeline_streaming(cfg)
        for ln in lines:
            st.session_state.logs_learn.append(ln)
            log_box3.code("\n".join(st.session_state.logs_learn[-500:]), language='text')
            time.sleep(0.01)
        status3.success("Learning terminé (voir logs).")
