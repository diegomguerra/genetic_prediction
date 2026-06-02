"""
DSII Genetic Predictor — Streamlit Web App
Roda a predição V11 completa via browser.
"""
import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import pickle, csv, re, time, io
from pathlib import Path
from scipy.stats import rankdata

# ============================================================
# CONFIG
# ============================================================
BASE = Path(__file__).parent
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")
DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
V11_DIR = BASE / 'dsii_v11_results'

st.set_page_config(
    page_title="DSII Genetic Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# IMPORT ALL CONSTANTS FROM PRODUCTION SCRIPT
# ============================================================
from dsii_production_predict import (
    VALIDATED_RECALL, GROUP_TRAITS, CATEGORIES, CORE_TRAITS,
    GENETIC_SD, HERITABILITY, GENETIC_CORRELATIONS,
    BULL_COL, CDCB_COL, STUD_MAP,
    sf, _normalize_bull_row, normalize_naab, get_bull_ptas, lookup_bull,
    compute_pa, build_features_v11, predict_batch_v11, classify_groups,
)

# ============================================================
# CACHED LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_databases():
    """Carrega bases de touros e modelos V11 uma unica vez."""
    bulls = {}
    bulls_by_name = {}
    with open(BULLS_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            row = _normalize_bull_row(row)
            naab = row.get('NAAB','').strip()
            if naab:
                bulls[naab] = row
                reg_name = row.get('Registration Name','').strip().upper()
                if reg_name: bulls_by_name[reg_name] = (naab, row)
                name = row.get('Name','').strip().upper()
                if name: bulls_by_name[name] = (naab, row)

    cdcb_bulls = {}
    cdcb_to_ss = {}
    cdcb_path = DOWNLOADS / 'Bull_Report (1).csv'
    if cdcb_path.exists():
        with open(cdcb_path, 'r', encoding='latin-1') as f:
            for row in csv.DictReader(f):
                naab = row.get('NAAB_CODE','').strip()
                if naab:
                    cdcb_bulls[naab] = row
                    name = row.get('NAME','').strip().upper()
                    if name and name in bulls_by_name:
                        cdcb_to_ss[naab] = bulls_by_name[name][0]

    bulls_by_num = {}
    for naab, row in bulls.items():
        m = re.match(r'^(\d+)(HO|BS)(\d+)$', naab)
        if m: bulls_by_num.setdefault(m.group(3), []).append((naab, row))

    cdcb_by_num = {}
    for naab, row in cdcb_bulls.items():
        m = re.match(r'^(\d+)(HO|BS)(\d+)$', naab)
        if m: cdcb_by_num.setdefault(m.group(3), []).append((naab, row))

    with open(V11_DIR / 'v11_models.pkl', 'rb') as f: saved_models = pickle.load(f)
    with open(V11_DIR / 'v11_sire_profiles.pkl', 'rb') as f: sire_profiles = pickle.load(f)
    with open(V11_DIR / 'v11_mgs_profiles.pkl', 'rb') as f: mgs_profiles = pickle.load(f)

    return {
        'bulls': bulls, 'cdcb_bulls': cdcb_bulls, 'cdcb_to_ss': cdcb_to_ss,
        'bulls_by_num': bulls_by_num, 'cdcb_by_num': cdcb_by_num,
        'saved_models': saved_models, 'sire_profiles': sire_profiles,
        'mgs_profiles': mgs_profiles,
    }

# ============================================================
# PREDICTION PIPELINE
# ============================================================
def run_prediction(df, header_row, db, progress_cb=None):
    """Executa a predição completa e retorna resultados."""
    bulls = db['bulls']
    cdcb_bulls = db['cdcb_bulls']
    cdcb_to_ss = db['cdcb_to_ss']
    bulls_by_num = db['bulls_by_num']
    cdcb_by_num = db['cdcb_by_num']
    saved_models = db['saved_models']
    sire_profiles = db['sire_profiles']
    mgs_profiles = db['mgs_profiles']

    ALL_TRAITS = list(saved_models.keys())

    cols = list(df.columns)
    id_col, pai_col, avo_col = cols[0], cols[1], cols[2]
    bis_col = cols[3] if len(cols) > 3 else None

    # Resolve pedigrees
    if progress_cb: progress_cb(0.05, "Resolvendo pedigrees...")
    excluded = []
    valid_animals = []

    for _, row in df.iterrows():
        animal_id = row[id_col]
        pai_naab = str(row[pai_col]).strip()
        avo_naab = str(row[avo_col]).strip() if avo_col else 'nan'
        bis_naab = str(row[bis_col]).strip() if bis_col else 'nan'

        pai_row, pai_src, pai_resolved, pai_ptas = lookup_bull(
            pai_naab, bulls, cdcb_bulls, cdcb_to_ss, bulls_by_num, cdcb_by_num)

        if not pai_row or not pai_ptas:
            excluded.append({'ID': animal_id, 'Pai': pai_naab, 'Motivo': f"Pai não encontrado ({pai_naab})"})
            continue

        avo_row, avo_src, avo_resolved, avo_ptas = (None, None, None, {})
        if avo_naab and avo_naab not in ('nan', 'None', ''):
            avo_row, avo_src, avo_resolved, avo_ptas = lookup_bull(
                avo_naab, bulls, cdcb_bulls, cdcb_to_ss, bulls_by_num, cdcb_by_num)

        bis_row, bis_src, bis_resolved, bis_ptas = (None, None, None, {})
        if bis_naab and bis_naab not in ('nan', 'None', ''):
            bis_row, bis_src, bis_resolved, bis_ptas = lookup_bull(
                bis_naab, bulls, cdcb_bulls, cdcb_to_ss, bulls_by_num, cdcb_by_num)

        meta = {
            'ID': animal_id,
            'Pai_NAAB': pai_naab,
            'Pai_Resolved': pai_resolved or pai_naab,
            'Pai_Src': pai_src or 'N/A',
            'N_Traits_Pai': len(pai_ptas),
        }
        valid_animals.append((meta, pai_ptas, avo_ptas, pai_resolved, avo_resolved, bis_ptas))

    if not valid_animals:
        return None, None, None, excluded, "Nenhum animal válido para predição."

    # Predict
    animals_input = [(a[1], a[2], a[3], a[4], a[5]) for a in valid_animals]

    if progress_cb: progress_cb(0.15, "Predizendo PL (base)...")
    pl_preds = predict_batch_v11('PL', animals_input, saved_models, sire_profiles, mgs_profiles)

    results = [{**a[0]} for a in valid_animals]
    for ti, trait in enumerate(ALL_TRAITS):
        pct = 0.15 + 0.65 * (ti / len(ALL_TRAITS))
        if progress_cb: progress_cb(pct, f"Predizendo {trait}...")
        preds = predict_batch_v11(trait, animals_input, saved_models, sire_profiles, mgs_profiles, pl_preds)
        for i, p in enumerate(preds):
            if p is not None:
                results[i][trait] = p

    # Group classification
    if progress_cb: progress_cb(0.85, "Classificando grupos...")
    n_animals = len(results)
    group_data = {}
    for trait in GROUP_TRAITS:
        if trait not in ALL_TRAITS: continue
        preds_list = [(i, results[i].get(trait)) for i in range(len(results))]
        groups = classify_groups(preds_list, trait, n_animals)
        group_data[trait] = groups

    # Build DataFrames
    if progress_cb: progress_cb(0.95, "Montando resultados...")

    # Predictions
    meta_cols = ['ID', 'Pai_NAAB', 'Pai_Resolved', 'Pai_Src', 'N_Traits_Pai']
    ordered_traits = []
    for cat_traits in CATEGORIES.values():
        for t in cat_traits:
            if t in ALL_TRAITS: ordered_traits.append(t)
    for t in ALL_TRAITS:
        if t not in ordered_traits: ordered_traits.append(t)

    final_cols = [c for c in meta_cols if c in results[0]] + [t for t in ordered_traits if t in results[0]]
    res_df = pd.DataFrame(results)[final_cols]

    # Groups
    group_rows = []
    for i in range(len(results)):
        row = {'ID': results[i]['ID']}
        for trait in GROUP_TRAITS:
            if trait not in group_data: continue
            gi = group_data[trait].get(i)
            if gi:
                row[f'{trait}_Grupo'] = gi['label']
                row[f'{trait}_Pctl'] = gi['percentile']
                row[f'{trait}_Recall'] = f"{gi['recall']}%" if gi['recall'] != '-' else '-'
        group_rows.append(row)
    group_df = pd.DataFrame(group_rows)

    # Confidence
    conf_rows = []
    for trait in GROUP_TRAITS:
        recall_data = VALIDATED_RECALL.get(trait, {})
        conf_rows.append({
            'Trait': trait,
            'N_Animais': n_animals,
            'Top_5%_Animais': max(1, int(n_animals * 0.05)),
            'Top_10%_Animais': max(1, int(n_animals * 0.10)),
            'Recall@5%': f"{recall_data.get(0.05, '?')}%",
            'Recall@10%': f"{recall_data.get(0.10, '?')}%",
            'Recall@15%': f"{recall_data.get(0.15, '?')}%",
            'Recall@20%': f"{recall_data.get(0.20, '?')}%",
        })
    conf_df = pd.DataFrame(conf_rows)

    return res_df, group_df, conf_df, excluded, ordered_traits, group_data, ALL_TRAITS

# ============================================================
# EXCEL EXPORT
# ============================================================
def to_excel_bytes(res_df, group_df, conf_df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        res_df.to_excel(writer, sheet_name='Predicoes', index=False)
        group_df.to_excel(writer, sheet_name='Grupos', index=False)
        conf_df.to_excel(writer, sheet_name='Confianca', index=False)
    return buf.getvalue()

# ============================================================
# CUSTOM CSS
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp { background-color: #0f1117; }
    .main .block-container { max-width: 1400px; padding-top: 1rem; }

    /* Header */
    .app-header {
        display: flex; align-items: center; gap: 1rem;
        padding: 1rem 0; border-bottom: 1px solid #2d3242; margin-bottom: 1.5rem;
    }
    .app-header h1 { margin: 0; font-size: 1.4rem; font-weight: 700; }
    .app-header .version {
        background: linear-gradient(135deg, #4f8cff, #6c5ce7);
        color: #fff; font-size: .7rem; padding: 3px 10px; border-radius: 12px; font-weight: 600;
    }
    .app-header .status { margin-left: auto; font-size: .8rem; color: #8b92a5; }
    .app-header .status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #00b894; margin-right: 6px; }

    /* KPI */
    .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
    .kpi-card {
        background: #21252f; border: 1px solid #2d3242; border-radius: 12px; padding: 1rem;
    }
    .kpi-card.highlight { border-color: #4f8cff; background: linear-gradient(135deg, rgba(79,140,255,.08), rgba(108,92,231,.08)); }
    .kpi-card .label { font-size: .7rem; color: #8b92a5; text-transform: uppercase; letter-spacing: .5px; }
    .kpi-card .value { font-size: 1.6rem; font-weight: 700; margin: .2rem 0; }
    .kpi-card .sub { font-size: .7rem; color: #8b92a5; }

    /* Group cards */
    .group-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: .8rem; margin-bottom: 1.5rem; }
    .group-card {
        background: #21252f; border: 1px solid #2d3242; border-radius: 10px;
        padding: .8rem; text-align: center;
    }
    .group-card .icon { font-size: 1.3rem; }
    .group-card .name { font-size: .72rem; font-weight: 600; }
    .group-card .count { font-size: 1.4rem; font-weight: 700; }
    .group-card .pct { font-size: .65rem; color: #8b92a5; }

    /* Recall box */
    .recall-box {
        background: #1a1d27; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 3px solid #4f8cff; margin-bottom: 1rem; font-size: .82rem;
    }
    .recall-box .title { color: #4f8cff; font-weight: 600; margin-bottom: .3rem; }
    .recall-box .text { color: #8b92a5; line-height: 1.5; }

    /* Pills */
    .pill { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: .7rem; font-weight: 600; }
    .pill-elite { background: rgba(255,215,0,.15); color: #ffd700; }
    .pill-top10 { background: rgba(0,184,148,.15); color: #00b894; }
    .pill-sup20 { background: rgba(79,140,255,.15); color: #4f8cff; }
    .pill-above { background: rgba(162,155,254,.12); color: #a29bfe; }
    .pill-media { background: rgba(99,110,114,.15); color: #636e72; }
    .pill-below { background: rgba(225,112,85,.15); color: #e17055; }

    /* Engine box */
    .engine-box {
        background: #1a1d27; border: 1px solid #2d3242; border-radius: 10px;
        padding: .8rem 1rem; margin-top: .5rem; font-size: .8rem; color: #8b92a5;
    }

    /* Hide streamlit defaults */
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# GROUP LABEL TO PILL
# ============================================================
def pill_html(label):
    cls_map = {
        'Elite 5%': 'pill-elite', 'Top 10%': 'pill-top10',
        'Superior 20%': 'pill-sup20', 'Acima Media': 'pill-above',
        'Media': 'pill-media', 'Abaixo Media': 'pill-below',
    }
    cls = cls_map.get(label, 'pill-media')
    return f'<span class="pill {cls}">{label}</span>'

GROUP_COLORS = {
    'Elite 5%': '#ffd700', 'Top 10%': '#00b894', 'Superior 20%': '#4f8cff',
    'Acima Media': '#a29bfe', 'Media': '#636e72', 'Abaixo Media': '#e17055',
}
GROUP_ICONS = {
    'Elite 5%': '⭐', 'Top 10%': '🏆', 'Superior 20%': '🔵',
    'Acima Media': '🟣', 'Media': '⚪', 'Abaixo Media': '🔻',
}

# ============================================================
# APP
# ============================================================
def main():
    inject_css()

    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🧬 DSII Genetic Predictor</h1>
        <span class="version">V11 PRODUCTION</span>
        <div class="status"><span class="dot"></span>Modelos carregados (47 traits)</div>
    </div>
    """, unsafe_allow_html=True)

    # Load databases (cached)
    with st.spinner("Carregando bases de touros e modelos V11..."):
        db = load_databases()

    n_bulls = len(db['bulls'])
    n_cdcb = len(db['cdcb_bulls'])
    n_traits = len(db['saved_models'])

    # Upload section
    col_up, col_cfg = st.columns([2, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Upload da planilha de animais (.xlsx)",
            type=['xlsx', 'xls'],
            help="Colunas esperadas: ANIMAL, SIRE (NAAB), MGS (NAAB), MMGS (NAAB)"
        )

    with col_cfg:
        st.markdown("**Configurações**")
        client_name = st.text_input("Nome do rebanho / cliente", placeholder="Ex: Fazenda São José")
        header_row = 0

    if not uploaded:
        # Limpa resultados anteriores quando arquivo é removido
        for k in ['res_df','group_df','conf_df','excluded','ordered_traits','group_data',
                   'ALL_TRAITS','elapsed','client_name','n_animals','_last_file']:
            st.session_state.pop(k, None)
        st.stop()

    # Detecta troca de arquivo e limpa resultados antigos
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get('_last_file') != file_id:
        for k in ['res_df','group_df','conf_df','excluded','ordered_traits','group_data',
                   'ALL_TRAITS','elapsed','client_name','n_animals']:
            st.session_state.pop(k, None)
        st.session_state['_last_file'] = file_id

    # Read uploaded file
    try:
        df = pd.read_excel(uploaded, engine='openpyxl', header=header_row)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        st.stop()

    st.success(f"Arquivo carregado: **{uploaded.name}** — {len(df)} animais detectados")

    # Preview
    with st.expander("Preview das primeiras linhas", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    # Run prediction
    if st.button("▶ Executar Predição", type="primary", use_container_width=True):
        t0 = time.time()
        progress = st.progress(0, text="Iniciando...")

        def update_progress(pct, msg):
            progress.progress(pct, text=msg)

        result = run_prediction(df, header_row, db, progress_cb=update_progress)

        if result[0] is None:
            st.error(f"Erro: {result[4]}")
            if result[3]:
                st.dataframe(pd.DataFrame(result[3]))
            st.stop()

        res_df, group_df, conf_df, excluded, ordered_traits, group_data, ALL_TRAITS = result
        elapsed = time.time() - t0

        progress.progress(1.0, text=f"Concluído em {elapsed:.0f}s!")

        # Store results in session state
        st.session_state['res_df'] = res_df
        st.session_state['group_df'] = group_df
        st.session_state['conf_df'] = conf_df
        st.session_state['excluded'] = excluded
        st.session_state['ordered_traits'] = ordered_traits
        st.session_state['group_data'] = group_data
        st.session_state['ALL_TRAITS'] = ALL_TRAITS
        st.session_state['elapsed'] = elapsed
        st.session_state['client_name'] = client_name or uploaded.name.replace('.xlsx','')
        st.session_state['n_animals'] = len(res_df)

    # Show results if available
    if 'res_df' not in st.session_state:
        st.stop()

    res_df = st.session_state['res_df']
    group_df = st.session_state['group_df']
    conf_df = st.session_state['conf_df']
    excluded = st.session_state['excluded']
    group_data = st.session_state['group_data']
    elapsed = st.session_state['elapsed']
    n_animals = st.session_state['n_animals']
    ordered_traits = st.session_state['ordered_traits']

    st.divider()

    # KPIs
    n_traits_pred = len([c for c in res_df.columns if c not in ['ID','Pai_NAAB','Pai_Resolved','Pai_Src','N_Traits_Pai']])
    n_sire_found = (res_df['Pai_Src'] != 'N/A').sum() if 'Pai_Src' in res_df.columns else n_animals
    pct_found = n_sire_found / n_animals * 100 if n_animals > 0 else 0

    # Elite count for TPI
    tpi_elite = 0
    if 'TPI' in group_data:
        tpi_elite = sum(1 for g in group_data['TPI'].values() if g['label'] == 'Elite 5%')

    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card highlight">
            <div class="label">Animais Processados</div>
            <div class="value" style="color:#4f8cff;">{n_animals}</div>
            <div class="sub">{n_animals} válidos / {len(excluded)} excluídos</div>
        </div>
        <div class="kpi-card">
            <div class="label">Traits Preditos</div>
            <div class="value">{n_traits_pred}</div>
            <div class="sub">{len([t for t in ordered_traits if t in ['MILK','FAT','FAT%','PRO','PRO%','PL','SCS','DPR','CCR','LIV','MAST']])} produção + tipo</div>
        </div>
        <div class="kpi-card">
            <div class="label">Touros Encontrados</div>
            <div class="value" style="color:#00b894;">{pct_found:.0f}%</div>
            <div class="sub">{n_sire_found} de {n_animals}</div>
        </div>
        <div class="kpi-card">
            <div class="label">Grupo Elite 5% (TPI)</div>
            <div class="value" style="color:#ffd700;">{tpi_elite}</div>
            <div class="sub">animais no topo</div>
        </div>
        <div class="kpi-card">
            <div class="label">Tempo</div>
            <div class="value">{elapsed:.0f}s</div>
            <div class="sub">{n_traits_pred} traits × {n_animals} animais</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # Group distribution for selected trait
    trait_for_groups = st.selectbox("Trait para distribuição de grupos:", GROUP_TRAITS, index=0)

    if trait_for_groups in group_data:
        labels_list = [g['label'] for g in group_data[trait_for_groups].values()]
        group_order = ['Elite 5%', 'Top 10%', 'Superior 20%', 'Acima Media', 'Media', 'Abaixo Media']

        cards_html = '<div class="group-grid">'
        for grp in group_order:
            cnt = labels_list.count(grp)
            pct = cnt / len(labels_list) * 100 if labels_list else 0
            color = GROUP_COLORS.get(grp, '#636e72')
            icon = GROUP_ICONS.get(grp, '⚪')
            highlight = ' style="border-color:' + color + ';"' if grp in ('Elite 5%', 'Top 10%', 'Superior 20%') else ''
            cards_html += f"""
            <div class="group-card"{highlight}>
                <div class="icon">{icon}</div>
                <div class="name" style="color:{color};">{grp}</div>
                <div class="count" style="color:{color};">{cnt}</div>
                <div class="pct">{pct:.1f}%</div>
            </div>
            """
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

    # Recall info
    recall_data = VALIDATED_RECALL.get(trait_for_groups, {})
    r10 = recall_data.get(0.10, '?')
    r15 = recall_data.get(0.15, '?')
    st.markdown(f"""
    <div class="recall-box">
        <div class="title">📊 Confiança da Classificação — {trait_for_groups}</div>
        <div class="text">
            Ao selecionar o <strong style="color:#e6e9ef;">Top 10% predito</strong>, capturamos <strong style="color:#e6e9ef;">{r10}%</strong> dos animais realmente top 5%.
            Com <strong style="color:#e6e9ef;">Top 15%</strong>, a captura sobe para <strong style="color:#e6e9ef;">{r15}%</strong>.
            Validado em 4.156 animais com genômica real.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Predições por Trait", "🏷️ Classificação de Grupos", "🔒 Confiança (Recall)", "⚠️ Excluídos"])

    with tab1:
        # Show predictions table
        display_cols = ['ID']
        show_traits = [t for t in ['TPI', 'NM$', 'CM$', 'MILK', 'FAT', 'FAT%', 'PRO', 'PRO%',
                                    'PL', 'DPR', 'SCS', 'PTAT', 'UDC', 'FLC'] if t in res_df.columns]
        display_cols += show_traits

        st.dataframe(
            res_df[display_cols].style.format({t: '{:.1f}' for t in show_traits if t in res_df.columns}),
            use_container_width=True,
            height=500,
        )

        with st.expander("Ver todos os traits"):
            st.dataframe(res_df, use_container_width=True, height=500)

    with tab2:
        # Group classification table with colored pills
        if not group_df.empty:
            # Show a simplified view
            group_display = group_df.copy()
            st.dataframe(group_display, use_container_width=True, height=500)

    with tab3:
        st.dataframe(conf_df, use_container_width=True)

    with tab4:
        if excluded:
            st.warning(f"{len(excluded)} animais excluídos (pai não encontrado)")
            st.dataframe(pd.DataFrame(excluded), use_container_width=True)
        else:
            st.success("Nenhum animal excluído — todos os pais foram encontrados!")

    # Download
    st.divider()
    col_dl1, col_dl2 = st.columns([3, 1])
    with col_dl1:
        st.markdown(f"""
        **📥 Exportar Resultados** — Excel com 3 abas: Predições (valores), Grupos (classificação), Confiança (recall por trait)
        """)
    with col_dl2:
        client_label = st.session_state.get('client_name', 'rebanho')
        excel_data = to_excel_bytes(res_df, group_df, conf_df)
        st.download_button(
            label="⬇ Baixar Excel",
            data=excel_data,
            file_name=f"{client_label}_DSII_V11.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True,
        )

    # ============================================================
    # VALIDATION SECTION — só aparece APÓS a predição
    # ============================================================
    st.divider()
    st.markdown("""
    <div class="recall-box" style="border-left-color:#ffd700;">
        <div class="title" style="color:#ffd700;">🔬 Validação (opcional)</div>
        <div class="text">
            Suba a planilha com os valores reais de <strong style="color:#e6e9ef;">PA</strong> e/ou
            <strong style="color:#e6e9ef;">Genômica</strong> para comparar com as predições do DSII.
            Esta seção só fica disponível após rodar o modelo para não viesar a análise.
        </div>
    </div>
    """, unsafe_allow_html=True)

    val_col1, val_col2 = st.columns(2)
    with val_col1:
        pa_file = st.file_uploader(
            "Upload PA real (.xlsx)",
            type=['xlsx', 'xls'],
            help="Planilha com colunas: ID + traits com valores de Parent Average",
            key="pa_upload"
        )
    with val_col2:
        gen_file = st.file_uploader(
            "Upload Genômica real (.xlsx)",
            type=['xlsx', 'xls'],
            help="Planilha com colunas: ID + traits com valores genômicos reais",
            key="gen_upload"
        )

    if pa_file or gen_file:
        _show_validation(res_df, pa_file, gen_file, ordered_traits)

    # Footer
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;color:#8b92a5;font-size:.72rem;border-top:1px solid #2d3242;margin-top:2rem;">
        DSII Genetic Predictor v11.0 — Select Sires do Brasil &nbsp;|&nbsp; Modelo treinado em 7.739 touros (Mai/2026) &nbsp;|&nbsp; Dados CDCB + banco proprietário
    </div>
    """, unsafe_allow_html=True)


def _show_validation(res_df, pa_file, gen_file, ordered_traits):
    """Compara DSII vs PA vs Genômica quando os dados reais são fornecidos."""
    from scipy.stats import spearmanr, pearsonr

    st.subheader("Comparativo: DSII vs PA vs Genômica")

    # Read validation files
    pa_df = None
    gen_df = None
    if pa_file:
        try:
            pa_df = pd.read_excel(pa_file, engine='openpyxl')
            st.success(f"PA carregado: {len(pa_df)} animais")
        except Exception as e:
            st.error(f"Erro ao ler PA: {e}")
    if gen_file:
        try:
            gen_df = pd.read_excel(gen_file, engine='openpyxl')
            st.success(f"Genômica carregado: {len(gen_df)} animais")
        except Exception as e:
            st.error(f"Erro ao ler Genômica: {e}")

    if gen_df is None and pa_df is None:
        return

    # Identify ID column (first column)
    dsii_id_col = res_df.columns[0]  # 'ID'

    # Try to match on ID
    comparison_rows = []
    trait_cols = [c for c in res_df.columns if c not in ['ID', 'Pai_NAAB', 'Pai_Resolved', 'Pai_Src', 'N_Traits_Pai']]

    # Find common traits between DSII and genomic/PA
    gen_id_col = gen_df.columns[0] if gen_df is not None else None
    pa_id_col = pa_df.columns[0] if pa_df is not None else None

    # Detect traits in genomic file
    gen_traits = []
    pa_traits = []
    if gen_df is not None:
        gen_traits = [c for c in gen_df.columns[1:] if c in trait_cols or c.upper() in [t.upper() for t in trait_cols]]
    if pa_df is not None:
        pa_traits = [c for c in pa_df.columns[1:] if c in trait_cols or c.upper() in [t.upper() for t in trait_cols]]

    # Determine which traits to compare
    compare_traits = list(set(gen_traits) | set(pa_traits))
    if not compare_traits:
        # Try case-insensitive matching
        dsii_upper = {t.upper(): t for t in trait_cols}
        if gen_df is not None:
            for c in gen_df.columns[1:]:
                if c.upper() in dsii_upper:
                    gen_traits.append(c)
        if pa_df is not None:
            for c in pa_df.columns[1:]:
                if c.upper() in dsii_upper:
                    pa_traits.append(c)
        compare_traits = list(set(gen_traits) | set(pa_traits))

    if not compare_traits:
        st.warning("Nenhum trait em comum encontrado entre DSII e os arquivos de validação. Verifique se os nomes das colunas coincidem.")
        return

    st.info(f"Traits em comum para comparação: **{', '.join(sorted(compare_traits))}**")

    # Merge data
    merged = res_df.copy()
    merged[dsii_id_col] = merged[dsii_id_col].astype(str).str.strip()

    if gen_df is not None:
        gen_df[gen_id_col] = gen_df[gen_id_col].astype(str).str.strip()
        gen_renamed = gen_df.rename(columns={gen_id_col: dsii_id_col})
        for c in gen_traits:
            gen_renamed = gen_renamed.rename(columns={c: f'GEN_{c}'})
        merged = merged.merge(gen_renamed[[dsii_id_col] + [f'GEN_{c}' for c in gen_traits]], on=dsii_id_col, how='left')

    if pa_df is not None:
        pa_df[pa_id_col] = pa_df[pa_id_col].astype(str).str.strip()
        pa_renamed = pa_df.rename(columns={pa_id_col: dsii_id_col})
        for c in pa_traits:
            pa_renamed = pa_renamed.rename(columns={c: f'PA_{c}'})
        merged = merged.merge(pa_renamed[[dsii_id_col] + [f'PA_{c}' for c in pa_traits]], on=dsii_id_col, how='left')

    # Compute comparison metrics
    metrics_rows = []
    for trait in sorted(compare_traits):
        row = {'Trait': trait}

        # DSII vs Genomic
        if f'GEN_{trait}' in merged.columns and trait in merged.columns:
            mask = merged[[trait, f'GEN_{trait}']].notna().all(axis=1)
            if mask.sum() >= 5:
                dsii_vals = merged.loc[mask, trait].values.astype(float)
                gen_vals = merged.loc[mask, f'GEN_{trait}'].values.astype(float)
                rho, _ = spearmanr(dsii_vals, gen_vals)
                r, _ = pearsonr(dsii_vals, gen_vals)
                mae = np.mean(np.abs(dsii_vals - gen_vals))
                bias = np.mean(dsii_vals - gen_vals)
                row['N_Gen'] = int(mask.sum())
                row['DSII_vs_Gen_Spearman'] = round(rho, 3)
                row['DSII_vs_Gen_Pearson'] = round(r, 3)
                row['DSII_vs_Gen_MAE'] = round(mae, 2)
                row['DSII_vs_Gen_Bias'] = round(bias, 2)

        # PA vs Genomic
        if f'PA_{trait}' in merged.columns and f'GEN_{trait}' in merged.columns:
            mask = merged[[f'PA_{trait}', f'GEN_{trait}']].notna().all(axis=1)
            if mask.sum() >= 5:
                pa_vals = merged.loc[mask, f'PA_{trait}'].values.astype(float)
                gen_vals = merged.loc[mask, f'GEN_{trait}'].values.astype(float)
                rho, _ = spearmanr(pa_vals, gen_vals)
                mae = np.mean(np.abs(pa_vals - gen_vals))
                row['PA_vs_Gen_Spearman'] = round(rho, 3)
                row['PA_vs_Gen_MAE'] = round(mae, 2)

        # DSII vs PA
        if f'PA_{trait}' in merged.columns and trait in merged.columns:
            mask = merged[[trait, f'PA_{trait}']].notna().all(axis=1)
            if mask.sum() >= 5:
                dsii_vals = merged.loc[mask, trait].values.astype(float)
                pa_vals = merged.loc[mask, f'PA_{trait}'].values.astype(float)
                rho, _ = spearmanr(dsii_vals, pa_vals)
                row['DSII_vs_PA_Spearman'] = round(rho, 3)

        if len(row) > 1:
            metrics_rows.append(row)

    if not metrics_rows:
        st.warning("Não foi possível calcular métricas — verifique se os IDs coincidem entre os arquivos.")
        return

    metrics_df = pd.DataFrame(metrics_rows)

    # Summary KPIs
    if 'DSII_vs_Gen_Spearman' in metrics_df.columns and 'PA_vs_Gen_Spearman' in metrics_df.columns:
        dsii_mean = metrics_df['DSII_vs_Gen_Spearman'].mean()
        pa_mean = metrics_df['PA_vs_Gen_Spearman'].mean()
        dsii_wins = (metrics_df['DSII_vs_Gen_Spearman'] > metrics_df['PA_vs_Gen_Spearman']).sum()
        total = len(metrics_df)

        kpi_val = f"""
        <div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);">
            <div class="kpi-card highlight">
                <div class="label">DSII vs Genômica</div>
                <div class="value" style="color:#00b894;">{dsii_mean:.3f}</div>
                <div class="sub">Spearman médio</div>
            </div>
            <div class="kpi-card">
                <div class="label">PA vs Genômica</div>
                <div class="value" style="color:#ffa94d;">{pa_mean:.3f}</div>
                <div class="sub">Spearman médio</div>
            </div>
            <div class="kpi-card">
                <div class="label">DSII Vence</div>
                <div class="value" style="color:#ffd700;">{dsii_wins}/{total}</div>
                <div class="sub">traits com Spearman maior</div>
            </div>
            <div class="kpi-card">
                <div class="label">Vantagem DSII</div>
                <div class="value" style="color:#00b894;">+{(dsii_mean - pa_mean):.3f}</div>
                <div class="sub">vs PA (Spearman)</div>
            </div>
        </div>
        """
        st.markdown(kpi_val, unsafe_allow_html=True)

    # Detailed table
    st.markdown("**Métricas detalhadas por trait**")

    # Style the dataframe to highlight winners
    def highlight_winner(row):
        styles = [''] * len(row)
        if 'DSII_vs_Gen_Spearman' in row.index and 'PA_vs_Gen_Spearman' in row.index:
            dsii = row.get('DSII_vs_Gen_Spearman', 0)
            pa = row.get('PA_vs_Gen_Spearman', 0)
            if pd.notna(dsii) and pd.notna(pa):
                dsii_idx = row.index.get_loc('DSII_vs_Gen_Spearman')
                pa_idx = row.index.get_loc('PA_vs_Gen_Spearman')
                if dsii > pa:
                    styles[dsii_idx] = 'background-color: rgba(0,184,148,0.2); font-weight: bold'
                elif pa > dsii:
                    styles[pa_idx] = 'background-color: rgba(255,169,77,0.2); font-weight: bold'
        return styles

    st.dataframe(
        metrics_df.style.apply(highlight_winner, axis=1),
        use_container_width=True,
        height=400,
    )

    # Top group comparison (Recall real)
    if gen_df is not None and f'GEN_{compare_traits[0]}' in merged.columns:
        st.markdown("**Recall real — Top 5% genômico capturado pelo DSII**")
        recall_rows = []
        for trait in sorted(compare_traits):
            if f'GEN_{trait}' not in merged.columns or trait not in merged.columns:
                continue
            mask = merged[[trait, f'GEN_{trait}']].notna().all(axis=1)
            if mask.sum() < 20:
                continue
            dsii_vals = merged.loc[mask, trait].values.astype(float)
            gen_vals = merged.loc[mask, f'GEN_{trait}'].values.astype(float)
            n = len(gen_vals)
            n_top5 = max(1, int(n * 0.05))

            real_top = set(np.argsort(gen_vals)[-n_top5:])

            for sel_pct, label in [(0.05, 'Top5%'), (0.10, 'Top10%'), (0.15, 'Top15%'), (0.20, 'Top20%')]:
                n_sel = max(1, int(n * sel_pct))
                pred_top = set(np.argsort(dsii_vals)[-n_sel:])
                recall = len(real_top & pred_top) / len(real_top) * 100
                recall_rows.append({'Trait': trait, 'Estratégia': label, 'Recall': f'{recall:.0f}%', 'N': n})

        if recall_rows:
            recall_df = pd.DataFrame(recall_rows)
            recall_pivot = recall_df.pivot(index='Trait', columns='Estratégia', values='Recall')
            recall_pivot = recall_pivot[['Top5%', 'Top10%', 'Top15%', 'Top20%']]
            st.dataframe(recall_pivot, use_container_width=True)


if __name__ == '__main__':
    main()
