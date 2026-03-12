
import time
from pathlib import Path

import streamlit as st

from pipeline_runner import (
    PipelineConfig,
    OneNoteExportConfig,
    OneNoteProcessConfig,
    LearningPipelineConfig,
    run_pipeline_streaming,
    run_onenote_export_streaming,
    run_onenote_process_streaming,
    run_learning_pipeline_streaming,
    compute_expected_outputs,
)
from utils import load_json, safe_read_text

st.set_page_config(page_title="Build4Use – Interface locale", page_icon="🧩", layout="wide")

# UI polish
st.markdown("""<style>
:root {
  --b4u-blue: #0A3D62;
  --b4u-accent: #1B9CFC;
  --b4u-soft: #f5f8ff;
  --b4u-border: #e6ecff;
}
.block-container { padding-top: 1.0rem; }
.small { font-size: 0.9rem; color: #5b6b7f; }
.kpi { background: var(--b4u-soft); border: 1px solid var(--b4u-border); padding: 12px 14px; border-radius: 12px; }
.kpi h4 { margin: 0 0 4px 0; }
hr { border: none; border-top: 1px solid #e8eef6; margin: 1rem 0; }
</style>""", unsafe_allow_html=True)

project_root_default = Path(__file__).resolve().parent.parent

# Sidebar
st.sidebar.title("🧭 Paramètres")
project_root = st.sidebar.text_input("Chemin projet", value=str(project_root_default), help="Dossier contenant run_pipeline.py")
project_root_p = Path(project_root).resolve()

preset_path = Path(__file__).parent / 'presets.json'
presets = load_json(preset_path).get('presets', []) if preset_path.exists() else []
preset_names = [p['name'] for p in presets]
selected_preset = st.sidebar.selectbox("Profil génération", options=(['—'] + preset_names) if preset_names else ['—'])

# ---- session defaults (cross-tab sync) ----
def _init(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# Generation defaults
_init('gen_notebook', 'test')
_init('gen_onenote_section', 'Clinique - Goussonville')
_init('gen_case_id', 'P060011')
_init('gen_mode', 'multistep')
_init('gen_min_quality', 0.0)
_init('gen_bacs_scope', 'Non résidentiel')
_init('gen_bacs_part2_slides', False)
_init('gen_bacs_targets', '')
_init('gen_bacs_rules', str(project_root_p / 'input' / 'rules' / 'bacs_table6_rules_structured_clean.json'))

# Export defaults
_init('exp_env_path', str(project_root_p / '.env'))
_init('exp_out_dir', str(project_root_p / 'input' / 'onenote-exporter' / 'output'))
_init('exp_token_cache', str(project_root_p / 'input' / 'onenote-exporter' / 'cache' / 'token_cache.json'))
_init('exp_formats', 'md')
_init('exp_merge', False)
_init('exp_notebook_name', '')
_init('exp_notebook_id', '')
_init('exp_rows', [])
_init('exp_last_click_id', '')
_init('exp_last_click_ts', 0.0)
_init('exp_autorun_doubleclick', True)
_init('exp_autorun_pending', False)

# Processing defaults
_init('proc_input_root', str(project_root_p / 'input' / 'onenote-exporter' / 'output'))
_init('proc_out_root', str(project_root_p / 'process' / 'onenote'))
_init('proc_transcribe', True)
_init('proc_copy_assets', False)

# Apply preset
if presets and selected_preset != '—':
    p = next(x for x in presets if x['name'] == selected_preset)
    st.session_state['gen_notebook'] = p.get('notebook', st.session_state['gen_notebook'])
    st.session_state['gen_onenote_section'] = p.get('onenote_section', st.session_state['gen_onenote_section'])
    st.session_state['gen_case_id'] = p.get('case_id', st.session_state['gen_case_id'])
    st.session_state['gen_mode'] = p.get('mode', st.session_state['gen_mode'])
    st.session_state['gen_min_quality'] = float(p.get('min_quality', st.session_state['gen_min_quality']))
    st.session_state['gen_bacs_scope'] = p.get('bacs_building_scope', st.session_state['gen_bacs_scope'])
    st.session_state['gen_bacs_part2_slides'] = bool(p.get('bacs_part2_slides', st.session_state['gen_bacs_part2_slides']))

st.title("🧩 Build4Use – Interface locale")
st.markdown("<div class='small'>Workflow recommandé : <b>1) Export OneNote</b> → <b>2) Traiter l'export</b> → <b>3) Générer le rapport</b> → <b>4) Learning</b></div>", unsafe_allow_html=True)

# Top cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='kpi'><h4>📤 Export</h4><div class='small'>Récupérer les notebooks & exporter</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='kpi'><h4>⚙️ Traitement</h4><div class='small'>process_onenote.py → page packs</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='kpi'><h4>📄 Rapport</h4><div class='small'>run_pipeline.py → PPTX</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='kpi'><h4>🧠 Learning</h4><div class='small'>process_reports + skeletons</div></div>", unsafe_allow_html=True)

if not (project_root_p / 'run_pipeline.py').exists():
    st.warning(f"run_pipeline.py introuvable dans: {project_root_p} — corrige le chemin dans la sidebar")


tab1, tab2, tab3 = st.tabs(["📄 Générer un rapport", "📤 Export & Traitement OneNote", "🧠 Learning / Skeletons"])

# ---------------- Tab 1 ----------------
with tab1:
    st.subheader("Génération de rapport")

    colL, colR = st.columns([1, 1])
    with colL:
        st.text_input("Notebook", key='gen_notebook')
        st.text_input("Section OneNote", key='gen_onenote_section')
        st.text_input("Case ID", key='gen_case_id')
    with colR:
        st.selectbox("Mode LLM", options=["multistep", "single"], key='gen_mode')
        st.slider("Seuil qualité (min)", 0.0, 100.0, step=1.0, key='gen_min_quality')
        with st.expander("⚙️ Options BACS", expanded=True):
            st.text_input("Règles Tableau 6", key='gen_bacs_rules')
            st.selectbox("Scope", options=["Non résidentiel", "Résidentiel"], key='gen_bacs_scope')
            st.text_input("Targets (optionnel JSON)", key='gen_bacs_targets')
            st.checkbox("Partie 2 en slides markdown (LLM)", key='gen_bacs_part2_slides')

    run_btn = st.button("🚀 Générer le rapport", type="primary")
    log_box = st.empty(); status = st.empty()

    if 'logs_gen' not in st.session_state:
        st.session_state.logs_gen = []

    if run_btn:
        st.session_state.logs_gen = []
        cfg = PipelineConfig(
            project_root=project_root_p,
            notebook=st.session_state['gen_notebook'].strip(),
            onenote_section=st.session_state['gen_onenote_section'].strip(),
            case_id=st.session_state['gen_case_id'].strip(),
            mode=st.session_state['gen_mode'],
            min_quality=float(st.session_state['gen_min_quality']),
            bacs_rules=st.session_state['gen_bacs_rules'].strip(),
            bacs_building_scope=st.session_state['gen_bacs_scope'],
            bacs_targets=st.session_state['gen_bacs_targets'].strip(),
            bacs_part2_slides=bool(st.session_state['gen_bacs_part2_slides']),
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

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Export & Traitement OneNote")
    st.caption("Clique sur une ligne = sélection. Double-clic (rapide) = export immédiat (optionnel). Ensuite, bouton de traitement (process_onenote.py).")

    def parse_notebooks(lines: list[str]) -> list[dict]:
        rows = []
        for ln in lines:
            ln = (ln or '').strip()
            if not ln or '\t' not in ln:
                continue
            name, nid = ln.split('\t', 1)
            name = name.strip(); nid = nid.strip()
            if name and nid:
                rows.append({'Nom': name, 'ID': nid})
        seen = set(); out = []
        for r in rows:
            if r['ID'] in seen:
                continue
            seen.add(r['ID']); out.append(r)
        return out

    colA, colB = st.columns([1, 1])
    with colA:
        st.text_input("Fichier .env", key='exp_env_path')
        st.text_input("Output dir (export)", key='exp_out_dir')
        st.text_input("Token cache", key='exp_token_cache')
        st.text_input("Formats", key='exp_formats')
        st.checkbox("Générer merged.md", key='exp_merge')
        st.checkbox("Double-clic = Export immédiat", key='exp_autorun_doubleclick')
    with colB:
        st.text_input("Notebook (nom, match partiel)", key='exp_notebook_name')
        st.text_input("Notebook ID (exact)", key='exp_notebook_id')

        if st.session_state.get('exp_notebook_name') or st.session_state.get('exp_notebook_id'):
            if st.button("✅ Utiliser ce notebook pour la génération"):
                chosen_name = st.session_state.get('exp_notebook_name') or st.session_state.get('exp_notebook_id')
                st.session_state['gen_notebook'] = chosen_name
                st.success("Notebook copié dans l'onglet Génération (champ Notebook).")

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
            env_path=st.session_state['exp_env_path'].strip(),
            list_only=list_only,
            notebook_name=st.session_state['exp_notebook_name'].strip(),
            notebook_id=st.session_state['exp_notebook_id'].strip(),
            merge=bool(st.session_state['exp_merge']),
            formats=st.session_state['exp_formats'].strip(),
            output_dir=st.session_state['exp_out_dir'].strip(),
            token_cache=st.session_state['exp_token_cache'].strip(),
        )
        status2.info("Lancement export…")
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
        st.session_state['exp_rows'] = parse_notebooks(lines)
        if st.session_state['exp_rows']:
            status2.success(f"Notebooks détectés : {len(st.session_state['exp_rows'])}")
        else:
            status2.warning("Aucun notebook détecté. Vérifie .env / permissions.")

    if st.session_state.get('exp_rows'):
        st.markdown("---")
        st.markdown("### Notebooks")
        filt = st.text_input("Filtrer", value="", placeholder="ex: Goussonville")
        rows = st.session_state['exp_rows']
        if filt.strip():
            f = filt.strip().lower()
            rows = [r for r in rows if f in r['Nom'].lower()]

        event = st.dataframe(rows, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
        sel = getattr(event, 'selection', None)
        if sel and sel.get('rows'):
            idx = sel['rows'][0]
            if 0 <= idx < len(rows):
                chosen = rows[idx]
                st.session_state['exp_notebook_name'] = chosen['Nom']
                st.session_state['exp_notebook_id'] = chosen['ID']

                now = time.time()
                last_id = st.session_state.get('exp_last_click_id', '')
                last_ts = float(st.session_state.get('exp_last_click_ts', 0.0))
                st.session_state['exp_last_click_id'] = chosen['ID']
                st.session_state['exp_last_click_ts'] = now

                if (
                    bool(st.session_state.get('exp_autorun_doubleclick'))
                    and chosen['ID'] == last_id
                    and (now - last_ts) < 0.9
                ):
                    st.session_state['exp_autorun_pending'] = True

                st.success("Sélection appliquée (champs remplis ci-dessus)")
                st.code(chosen['ID'], language='text')

        if st.session_state.get('exp_autorun_pending'):
            st.session_state['exp_autorun_pending'] = False
            status2.warning("Double-clic détecté → export en cours…")
            run_export(False)
            status2.success("Export terminé (voir logs).")

    if export_btn:
        run_export(False)
        status2.success("Export terminé (voir logs).")

    st.markdown("---")
    st.subheader("⚙️ Traiter l’export (process_onenote.py)")
    st.caption("Convertit les Markdown exportés en JSON page packs sous process/onenote/<notebook>/pages/*.json")

    colP1, colP2 = st.columns([1, 1])
    with colP1:
        proc_notebook_default = st.session_state.get('exp_notebook_name') or st.session_state.get('gen_notebook')
        proc_notebook = st.text_input("Notebook à traiter", value=proc_notebook_default)
        st.text_input("Input root", key='proc_input_root')
        st.text_input("Output root", key='proc_out_root')
    with colP2:
        st.checkbox("Transcrire audio (--transcribe)", key='proc_transcribe')
        st.checkbox("Copier assets (--copy-assets)", key='proc_copy_assets')
        process_btn = st.button("➡️ Lancer le traitement", type="secondary")

    log_boxP = st.empty(); statusP = st.empty()
    if 'logs_process' not in st.session_state:
        st.session_state.logs_process = []

    if process_btn:
        st.session_state.logs_process = []
        cfgp = OneNoteProcessConfig(
            project_root=project_root_p,
            notebook=proc_notebook.strip(),
            input_root=st.session_state['proc_input_root'].strip(),
            out_root=st.session_state['proc_out_root'].strip(),
            transcribe=bool(st.session_state['proc_transcribe']),
            copy_assets=bool(st.session_state['proc_copy_assets']),
        )
        statusP.info("Traitement en cours…")
        _, lines = run_onenote_process_streaming(cfgp)
        for ln in lines:
            st.session_state.logs_process.append(ln)
            log_boxP.code("\n".join(st.session_state.logs_process[-500:]), language='text')
            time.sleep(0.01)
        statusP.success("Traitement terminé (voir logs).")

    with st.expander("ℹ️ Notes"):
        st.markdown("""
- Export: `python -m onenote_exporter.cli ...`
- Traitement: `python process_onenote.py <notebook> --input ... --out ...` (+ options)
""")

# ---------------- Tab 3 ----------------
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
