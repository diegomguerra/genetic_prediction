"""
DSII Genetic Predictor — Streamlit Web App
Roda a predição V11 completa via browser.
Touros servidos via Supabase (SelectSires Platform).
"""
import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import pickle, csv, re, time, io, json, urllib.request, urllib.error, urllib.parse
from pathlib import Path
from scipy.stats import rankdata

# ============================================================
# CONFIG
# ============================================================
BASE = Path(__file__).parent
V11_DIR = BASE / 'dsii_v11_results'

# Supabase SelectSires Platform
SUPABASE_URL = "https://odactdxpecpiyiyaqfgi.supabase.co/rest/v1"
import os as _os
SUPABASE_KEY = _os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kYWN0ZHhwZWNwaXlpeWFxZmdpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzMzMjQ4MTYsImV4cCI6MjA4ODkwMDgxNn0.1Ybzv5-oqg1yHu2W1vBRDHN23tj0YzN_AbwBJyKcrpY")

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
# SUPABASE BULL FETCHER
# ============================================================
# DB col -> original CSV col mapping (SS)
_DB_TO_SS = {
    'naab':'NAAB','name':'Name','registration_name':'Registration Name',
    'tpi':'TPI','nm_dollar':'NM$','cm_dollar':'CM$','ptam':'PTAM','cfp':'CFP',
    'ptaf':'PTAF','ptaf_pct':'PTAF%','ptap':'PTAP','ptap_pct':'PTAP%',
    'pl':'PL','dpr':'DPR','liv':'LIV','scs':'SCS','mast':'MAST','ptat':'PTAT',
    'udc':'UDC','flc':'FLC','sce':'SCE','dce':'DCE','ssb':'SSB','dsb':'DSB',
    'sta':'STA','str':'STR','dfm':'DFM','rua':'RUA','rls':'RLS','rtp':'RTP',
    'ftl':'FTL','ccr':'CCR','hcr':'HCR','bwc':'BWC','bod':'BOD','rw':'RW',
    'rlr':'RLR','fta':'FTA','fls':'FLS','fua':'FUA','ruh':'RUH','ruw':'RUW',
    'ucl':'UCL','udp':'UDP','ftp':'FTP','met':'MET','rp':'RP','da':'DA',
    'ket':'KET','mf':'MF','gfi':'GFI','rfi':'RFI','efc':'EFC',
    'h_liv':'H LIV','fi':'FI','gl':'GL','f_sav':'F SAV',
}
_DB_TO_CDCB = {
    'naab_code':'NAAB_CODE','name':'NAME',
    'nm_dollar_pta':'NM$_PTA','cm_dollar_pta':'CM$_PTA',
    'mlk_pta':'MLK_PTA','fat_pta':'FAT_PTA','fat_pct':'FAT%',
    'pro_pta':'PRO_PTA','pro_pct':'PRO%','pl_pta':'PL_PTA',
    'scs_pta':'SCS_PTA','dpr_pta':'DPR_PTA','hcr_pta':'HCR_PTA',
    'ccr_pta':'CCR_PTA','liv_pta':'LIV_PTA','gl_pta':'GL_PTA',
    'rfi_pta':'RFI_PTA','mfv_pta':'MFV_PTA','dab_pta':'DAB_PTA',
    'ket_pta':'KET_PTA','mas_pta':'MAS_PTA','met_pta':'MET_PTA',
    'rpl_pta':'RPL_PTA','efc_pta':'EFC_PTA','hlv_pta':'HLV_PTA',
    'fs_pta':'FS_PTA','typ_pta':'TYP_PTA','udc_pta':'UDC_PTA','flc_pta':'FLC_PTA',
}

def _supabase_get(table, params):
    """Faz GET no Supabase REST API."""
    url = f"{SUPABASE_URL}/{table}?{params}"
    req = urllib.request.Request(url)
    req.add_header('apikey', SUPABASE_KEY)
    req.add_header('Authorization', f'Bearer {SUPABASE_KEY}')
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except Exception as e:
        return []

def _row_to_ss(db_row):
    """Converte row do Supabase para formato esperado pelo get_bull_ptas."""
    return {csv_col: (str(db_row[db_col]) if db_row.get(db_col) is not None else '')
            for db_col, csv_col in _DB_TO_SS.items()}

def _row_to_cdcb(db_row):
    return {csv_col: (str(db_row[db_col]) if db_row.get(db_col) is not None else '')
            for db_col, csv_col in _DB_TO_CDCB.items()}

def fetch_bulls_for_naabs(raw_naabs):
    """Busca touros do Supabase apenas para os NAABs necessarios."""
    # Collect all NAAB variations and numeric suffixes
    all_naabs = set()
    all_nums = set()
    for raw in raw_naabs:
        candidates = normalize_naab(raw)
        all_naabs.update(candidates)
        m = re.match(r'^(\d+)(HO|BS|H|B)0*(\d+)$', str(raw).strip())
        if m:
            all_nums.add(m.group(3))

    # Query SS bulls by exact NAAB match (batch in chunks of 200)
    naab_list = list(all_naabs)
    ss_rows = []
    for i in range(0, len(naab_list), 200):
        chunk = naab_list[i:i+200]
        filter_val = ','.join(chunk)
        rows = _supabase_get('bulls_ss', f'naab=in.({urllib.parse.quote(filter_val)})&limit=10000')
        ss_rows.extend(rows)

    # Numeric suffix fallback: batch with OR filter (max 20 per request)
    num_list = list(all_nums)
    for i in range(0, len(num_list), 20):
        chunk = num_list[i:i+20]
        or_parts = ','.join(f'naab.like.*{num}' for num in chunk)
        rows = _supabase_get('bulls_ss', f'or=({urllib.parse.quote(or_parts)})&limit=5000')
        ss_rows.extend(rows)

    # Build bulls dict
    bulls = {}
    bulls_by_name = {}
    seen_naabs = set()
    for r in ss_rows:
        naab = r.get('naab','')
        if not naab or naab in seen_naabs: continue
        seen_naabs.add(naab)
        row = _row_to_ss(r)
        bulls[naab] = row
        name = (r.get('name') or '').strip().upper()
        if name: bulls_by_name[name] = (naab, row)
        regname = (r.get('registration_name') or '').strip().upper()
        if regname: bulls_by_name[regname] = (naab, row)

    # Query CDCB bulls (same approach)
    cdcb_rows = []
    for i in range(0, len(naab_list), 200):
        chunk = naab_list[i:i+200]
        filter_val = ','.join(chunk)
        rows = _supabase_get('bulls_cdcb', f'naab_code=in.({urllib.parse.quote(filter_val)})&limit=10000')
        cdcb_rows.extend(rows)
    for i in range(0, len(num_list), 20):
        chunk = num_list[i:i+20]
        or_parts = ','.join(f'naab_code.like.*{num}' for num in chunk)
        rows = _supabase_get('bulls_cdcb', f'or=({urllib.parse.quote(or_parts)})&limit=5000')
        cdcb_rows.extend(rows)

    cdcb_bulls = {}
    cdcb_to_ss = {}
    seen_cdcb = set()
    for r in cdcb_rows:
        naab = r.get('naab_code','')
        if not naab or naab in seen_cdcb: continue
        seen_cdcb.add(naab)
        row = _row_to_cdcb(r)
        cdcb_bulls[naab] = row
        name = (r.get('name') or '').strip().upper()
        if name and name in bulls_by_name:
            cdcb_to_ss[naab] = bulls_by_name[name][0]

    # Build numeric indexes
    bulls_by_num = {}
    for naab, row in bulls.items():
        m = re.match(r'^(\d+)(HO|BS)(\d+)$', naab)
        if m: bulls_by_num.setdefault(m.group(3), []).append((naab, row))

    cdcb_by_num = {}
    for naab, row in cdcb_bulls.items():
        m = re.match(r'^(\d+)(HO|BS)(\d+)$', naab)
        if m: cdcb_by_num.setdefault(m.group(3), []).append((naab, row))

    return bulls, cdcb_bulls, cdcb_to_ss, bulls_by_num, cdcb_by_num

# ============================================================
# CACHED MODEL LOADING — from Supabase Storage or local cache
# ============================================================
STORAGE_URL = "https://odactdxpecpiyiyaqfgi.supabase.co/storage/v1/object/public/dsii-models"
MODEL_CACHE_DIR = BASE / '_model_cache'

def _download_file(url, dest):
    """Baixa arquivo da URL para dest."""
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=120)
    with open(dest, 'wb') as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk: break
            f.write(chunk)

def _ensure_models():
    """Garante que os modelos estao em cache local, baixando do Supabase Storage se necessario."""
    import gzip
    MODEL_CACHE_DIR.mkdir(exist_ok=True)

    models_pkl = MODEL_CACHE_DIR / 'v11_models.pkl'
    sire_pkl = MODEL_CACHE_DIR / 'v11_sire_profiles.pkl'
    mgs_pkl = MODEL_CACHE_DIR / 'v11_mgs_profiles.pkl'

    # Try local V11_DIR first (dev environment)
    if (V11_DIR / 'v11_models.pkl').exists():
        return V11_DIR / 'v11_models.pkl', V11_DIR / 'v11_sire_profiles.pkl', V11_DIR / 'v11_mgs_profiles.pkl'

    # Download from Supabase Storage if not cached
    if not sire_pkl.exists():
        _download_file(f"{STORAGE_URL}/v11_sire_profiles.pkl", sire_pkl)

    if not mgs_pkl.exists():
        _download_file(f"{STORAGE_URL}/v11_mgs_profiles.pkl", mgs_pkl)

    if not models_pkl.exists():
        # Download compressed parts and reassemble
        part0 = MODEL_CACHE_DIR / 'part0.gz'
        part1 = MODEL_CACHE_DIR / 'part1.gz'
        if not part0.exists():
            _download_file(f"{STORAGE_URL}/v11_models.pkl.gz.part0", part0)
        if not part1.exists():
            _download_file(f"{STORAGE_URL}/v11_models.pkl.gz.part1", part1)

        # Reassemble and decompress
        compressed = part0.read_bytes() + part1.read_bytes()
        decompressed = gzip.decompress(compressed)
        models_pkl.write_bytes(decompressed)

        # Clean up parts
        part0.unlink(missing_ok=True)
        part1.unlink(missing_ok=True)

    return models_pkl, sire_pkl, mgs_pkl

@st.cache_resource(show_spinner=False)
def load_models():
    """Carrega modelos V11 uma unica vez."""
    models_path, sire_path, mgs_path = _ensure_models()
    with open(models_path, 'rb') as f: saved_models = pickle.load(f)
    with open(sire_path, 'rb') as f: sire_profiles = pickle.load(f)
    with open(mgs_path, 'rb') as f: mgs_profiles = pickle.load(f)
    return saved_models, sire_profiles, mgs_profiles

# ============================================================
# PREDICTION PIPELINE
# ============================================================
def run_prediction(df, header_row, saved_models, sire_profiles, mgs_profiles, progress_cb=None):
    """Executa a predição completa e retorna resultados."""

    ALL_TRAITS = list(saved_models.keys())

    # Smart column detection by name patterns
    cols = list(df.columns)
    col_lower = {c: c.lower().replace('_', '').replace('-', '').replace(' ', '') for c in cols}

    id_col = None
    pai_col = None
    avo_col = None
    bis_col = None

    for c, cl in col_lower.items():
        if cl in ('id', 'idfazenda', 'animal', 'nome', 'brinco', 'registro') and id_col is None:
            id_col = c
        elif cl in ('naabpai', 'pai', 'sire', 'naabsire', 'painaab', 'sirenaab') and pai_col is None:
            pai_col = c
        elif cl in ('naabavomaterno', 'avomaterno', 'mgs', 'naabmgs', 'avo', 'mgsnaab') and avo_col is None:
            avo_col = c
        elif cl in ('naabbisamaterno', 'bisaMaterno', 'bisamaterno', 'mmgs', 'naabmmgs', 'bisavo', 'mmgsnaab') and bis_col is None:
            bis_col = c

    # Fallback: use positional if no match found
    if pai_col is None:
        # Find columns that look like NAAB codes (contain HO or BS patterns)
        naab_cols = []
        for c in cols:
            sample_vals = df[c].dropna().astype(str).head(10)
            if sample_vals.str.contains(r'\d+HO\d+|\d+BS\d+', regex=True).any():
                naab_cols.append(c)
        if len(naab_cols) >= 1: pai_col = naab_cols[0]
        if len(naab_cols) >= 2: avo_col = naab_cols[1]
        if len(naab_cols) >= 3: bis_col = naab_cols[2]

    if id_col is None:
        # Use first column that is NOT a NAAB column
        for c in cols:
            if c not in (pai_col, avo_col, bis_col):
                id_col = c
                break

    if pai_col is None:
        return None, None, None, [], "Coluna do pai (NAAB) não encontrada. Verifique o arquivo."

    # Collect all NAAB codes from the file and fetch from Supabase
    if progress_cb: progress_cb(0.02, "Buscando touros no banco de dados...")
    raw_naabs = set()
    for _, row in df.iterrows():
        raw_naabs.add(str(row[pai_col]).strip())
        if avo_col: raw_naabs.add(str(row[avo_col]).strip())
        if bis_col: raw_naabs.add(str(row[bis_col]).strip())
    raw_naabs.discard('nan')
    raw_naabs.discard('')
    raw_naabs.discard('None')

    bulls, cdcb_bulls, cdcb_to_ss, bulls_by_num, cdcb_by_num = fetch_bulls_for_naabs(raw_naabs)

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
# CUSTOM CSS — Select Sires Branding
# ============================================================
SS_RED = '#c41230'
SS_RED_DARK = '#9e0f28'
SS_RED_LIGHT = 'rgba(196,18,48,.08)'

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #111318; }
    .main .block-container { max-width: 1400px; padding-top: 0; }

    /* Navbar */
    .ss-navbar {
        background: linear-gradient(135deg, #c41230 0%, #9e0f28 100%);
        margin: -1rem -1rem 1.5rem -1rem; padding: 0.8rem 2rem;
        display: flex; align-items: center; gap: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,.3);
    }
    .ss-navbar .logo { font-size: 1.3rem; font-weight: 800; color: #fff; letter-spacing: -.5px; }
    .ss-navbar .divider { width: 1px; height: 24px; background: rgba(255,255,255,.3); }
    .ss-navbar .subtitle { color: rgba(255,255,255,.85); font-size: .82rem; font-weight: 500; }
    .ss-navbar .badge {
        margin-left: auto; background: rgba(255,255,255,.15); color: #fff;
        font-size: .65rem; padding: 3px 10px; border-radius: 20px; font-weight: 600;
        backdrop-filter: blur(4px);
    }

    /* Upload area */
    .upload-section {
        background: #1a1d25; border: 1px solid #2a2d38; border-radius: 12px;
        padding: 1.5rem; margin-bottom: 1.5rem;
    }
    .upload-section h3 { margin: 0 0 .5rem; font-size: 1rem; color: #e8e9ec; font-weight: 600; }
    .upload-section .hint { font-size: .75rem; color: #6b7280; }

    /* KPI */
    .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: .8rem; margin-bottom: 1.5rem; }
    .kpi-card {
        background: #1a1d25; border: 1px solid #2a2d38; border-radius: 10px; padding: .9rem 1rem;
    }
    .kpi-card.highlight { border-color: #c41230; background: linear-gradient(135deg, rgba(196,18,48,.06), rgba(196,18,48,.02)); }
    .kpi-card .label { font-size: .65rem; color: #6b7280; text-transform: uppercase; letter-spacing: .8px; font-weight: 600; }
    .kpi-card .value { font-size: 1.5rem; font-weight: 800; margin: .15rem 0; color: #e8e9ec; }
    .kpi-card .sub { font-size: .68rem; color: #6b7280; }

    /* Group cards */
    .group-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: .6rem; margin-bottom: 1.2rem; }
    .group-card {
        background: #1a1d25; border: 1px solid #2a2d38; border-radius: 8px;
        padding: .7rem .5rem; text-align: center;
    }
    .group-card .icon { font-size: 1.1rem; margin-bottom: .15rem; }
    .group-card .name { font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .3px; }
    .group-card .count { font-size: 1.3rem; font-weight: 800; }
    .group-card .pct { font-size: .62rem; color: #6b7280; }

    /* Info box */
    .info-box {
        background: #1a1d25; border-radius: 8px; padding: .9rem 1.1rem;
        border-left: 3px solid #c41230; margin-bottom: 1rem; font-size: .8rem;
    }
    .info-box .title { color: #c41230; font-weight: 700; margin-bottom: .25rem; font-size: .82rem; }
    .info-box .text { color: #9ca3af; line-height: 1.6; }
    .info-box.gold { border-left-color: #d4a017; }
    .info-box.gold .title { color: #d4a017; }

    /* Button override */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #c41230, #9e0f28) !important;
        border: none !important; font-weight: 700 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #d41838, #b01030) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #2a2d38; }
    .stTabs [data-baseweb="tab"] { color: #6b7280; font-weight: 600; font-size: .82rem; }
    .stTabs [aria-selected="true"] { color: #c41230 !important; border-bottom-color: #c41230 !important; }

    /* Footer */
    .ss-footer {
        text-align: center; padding: 1.5rem 0 1rem; color: #4b5563; font-size: .68rem;
        border-top: 1px solid #2a2d38; margin-top: 2rem; letter-spacing: .2px;
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
# GROUP CONFIG
# ============================================================
GROUP_COLORS = {
    'Elite 5%': '#d4a017', 'Top 10%': '#22c55e', 'Superior 20%': '#3b82f6',
    'Acima Media': '#8b5cf6', 'Media': '#6b7280', 'Abaixo Media': '#ef4444',
}
GROUP_ICONS = {
    'Elite 5%': '★', 'Top 10%': '▲', 'Superior 20%': '●',
    'Acima Media': '◆', 'Media': '○', 'Abaixo Media': '▼',
}

# ============================================================
# APP
# ============================================================
def main():
    inject_css()

    # Navbar
    st.markdown("""
    <div class="ss-navbar">
        <div class="logo">SELECT SIRES</div>
        <div class="divider"></div>
        <div class="subtitle">DSII Genetic Predictor</div>
        <div class="badge">V11 PRODUCTION</div>
    </div>
    """, unsafe_allow_html=True)

    # Load ML models (cached, downloads from Supabase Storage on first run)
    with st.status("Carregando modelos ML...", expanded=False) as status:
        saved_models, sire_profiles, mgs_profiles = load_models()
        status.update(label=f"Modelos carregados — {len(saved_models)} traits", state="complete", expanded=False)

    # Upload section
    col_up, col_cfg = st.columns([3, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Upload da planilha de animais (.xlsx, .csv)",
            type=['xlsx', 'xls', 'csv'],
            help="Colunas: ID, PAI (NAAB), AVO MATERNO (NAAB), BISAVO (NAAB)",
            label_visibility="collapsed",
        )

    with col_cfg:
        client_name = st.text_input("Rebanho / Cliente", placeholder="Ex: Fazenda Canada")
        header_row = 0

    if not uploaded:
        for k in ['res_df','group_df','conf_df','excluded','ordered_traits','group_data',
                   'ALL_TRAITS','elapsed','client_name','n_animals','_last_file']:
            st.session_state.pop(k, None)
        # Landing message
        st.markdown("""
        <div class="info-box">
            <div class="title">Como usar</div>
            <div class="text">
                1. Suba uma planilha com colunas de NAAB do <strong style="color:#e8e9ec;">Pai</strong>,
                <strong style="color:#e8e9ec;">Avo Materno</strong> e opcionalmente <strong style="color:#e8e9ec;">Bisavo</strong><br>
                2. Clique em <strong style="color:#e8e9ec;">Executar Predicao</strong><br>
                3. Visualize os resultados e baixe o Excel com predicoes e classificacao de grupos<br><br>
                Formatos aceitos: <strong style="color:#e8e9ec;">.xlsx</strong>, <strong style="color:#e8e9ec;">.csv</strong> (separador <code>;</code> ou <code>,</code>)
                &nbsp;&bull;&nbsp; 47 traits &nbsp;&bull;&nbsp; 258k touros no banco
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Detecta troca de arquivo
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get('_last_file') != file_id:
        for k in ['res_df','group_df','conf_df','excluded','ordered_traits','group_data',
                   'ALL_TRAITS','elapsed','client_name','n_animals']:
            st.session_state.pop(k, None)
        st.session_state['_last_file'] = file_id

    # Read uploaded file
    try:
        if uploaded.name.endswith('.csv'):
            uploaded.seek(0)
            sample = uploaded.read(2048).decode('utf-8', errors='replace')
            uploaded.seek(0)
            sep = ';' if sample.count(';') > sample.count(',') else ','
            df = pd.read_csv(uploaded, sep=sep, header=header_row, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded, engine='openpyxl', header=header_row)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        st.stop()

    st.markdown(f"""
    <div class="info-box" style="border-left-color:#22c55e;">
        <div class="title" style="color:#22c55e;">Arquivo carregado</div>
        <div class="text"><strong style="color:#e8e9ec;">{uploaded.name}</strong> &mdash; {len(df)} animais &nbsp;&bull;&nbsp; {len(df.columns)} colunas</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Preview dos dados", expanded=False):
        st.dataframe(df.head(8), use_container_width=True)

    # Run prediction
    if st.button("Executar Predicao", type="primary", use_container_width=True):
        t0 = time.time()
        progress = st.progress(0, text="Iniciando...")

        def update_progress(pct, msg):
            progress.progress(pct, text=msg)

        result = run_prediction(df, header_row, saved_models, sire_profiles, mgs_profiles, progress_cb=update_progress)

        if result[0] is None:
            st.error(f"Erro: {result[4]}")
            if result[3]:
                st.dataframe(pd.DataFrame(result[3]))
            st.stop()

        res_df, group_df, conf_df, excluded, ordered_traits, group_data, ALL_TRAITS = result
        elapsed = time.time() - t0

        progress.progress(1.0, text=f"Concluido em {elapsed:.0f}s!")

        st.session_state['res_df'] = res_df
        st.session_state['group_df'] = group_df
        st.session_state['conf_df'] = conf_df
        st.session_state['excluded'] = excluded
        st.session_state['ordered_traits'] = ordered_traits
        st.session_state['group_data'] = group_data
        st.session_state['ALL_TRAITS'] = ALL_TRAITS
        st.session_state['elapsed'] = elapsed
        st.session_state['client_name'] = client_name or uploaded.name.replace('.xlsx','').replace('.csv','')
        st.session_state['n_animals'] = len(res_df)

    # Show results
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

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # KPIs
    n_traits_pred = len([c for c in res_df.columns if c not in ['ID','Pai_NAAB','Pai_Resolved','Pai_Src','N_Traits_Pai']])
    n_sire_found = (res_df['Pai_Src'] != 'N/A').sum() if 'Pai_Src' in res_df.columns else n_animals
    pct_found = n_sire_found / n_animals * 100 if n_animals > 0 else 0
    tpi_elite = sum(1 for g in group_data.get('TPI', {}).values() if g['label'] == 'Elite 5%')

    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card highlight">
            <div class="label">Animais</div>
            <div class="value" style="color:#c41230;">{n_animals}</div>
            <div class="sub">{n_animals} validos / {len(excluded)} excluidos</div>
        </div>
        <div class="kpi-card">
            <div class="label">Traits</div>
            <div class="value">{n_traits_pred}</div>
            <div class="sub">producao + tipo + saude</div>
        </div>
        <div class="kpi-card">
            <div class="label">Touros no Banco</div>
            <div class="value" style="color:#22c55e;">{pct_found:.0f}%</div>
            <div class="sub">{n_sire_found} de {n_animals} pais</div>
        </div>
        <div class="kpi-card">
            <div class="label">Elite 5% TPI</div>
            <div class="value" style="color:#d4a017;">{tpi_elite}</div>
            <div class="sub">animais no topo</div>
        </div>
        <div class="kpi-card">
            <div class="label">Tempo</div>
            <div class="value">{elapsed:.0f}s</div>
            <div class="sub">{n_traits_pred} x {n_animals}</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # Group distribution
    trait_for_groups = st.selectbox("Distribuicao de grupos por trait:", GROUP_TRAITS, index=0, label_visibility="collapsed")

    if trait_for_groups in group_data:
        labels_list = [g['label'] for g in group_data[trait_for_groups].values()]
        group_order = ['Elite 5%', 'Top 10%', 'Superior 20%', 'Acima Media', 'Media', 'Abaixo Media']

        cards_html = '<div class="group-grid">'
        for grp in group_order:
            cnt = labels_list.count(grp)
            pct = cnt / len(labels_list) * 100 if labels_list else 0
            color = GROUP_COLORS.get(grp, '#6b7280')
            icon = GROUP_ICONS.get(grp, '○')
            border = f' style="border-color:{color};"' if grp in ('Elite 5%', 'Top 10%') else ''
            cards_html += f"""
            <div class="group-card"{border}>
                <div class="icon" style="color:{color};">{icon}</div>
                <div class="name" style="color:{color};">{grp}</div>
                <div class="count" style="color:{color};">{cnt}</div>
                <div class="pct">{pct:.1f}%</div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

    # Recall info
    recall_data = VALIDATED_RECALL.get(trait_for_groups, {})
    r10 = recall_data.get(0.10, '?')
    r15 = recall_data.get(0.15, '?')
    st.markdown(f"""
    <div class="info-box">
        <div class="title">Confianca — {trait_for_groups}</div>
        <div class="text">
            Top 10% predito captura <strong style="color:#e8e9ec;">{r10}%</strong> dos realmente top 5%.
            Top 15% captura <strong style="color:#e8e9ec;">{r15}%</strong>.
            Validado em 4.156 animais com genomica real.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Predicoes", "Grupos", "Confianca", "Excluidos"])

    with tab1:
        display_cols = ['ID']
        show_traits = [t for t in ['TPI', 'NM$', 'CM$', 'MILK', 'FAT', 'FAT%', 'PRO', 'PRO%',
                                    'PL', 'DPR', 'SCS', 'PTAT', 'UDC', 'FLC'] if t in res_df.columns]
        display_cols += show_traits
        st.dataframe(
            res_df[display_cols].style.format({t: '{:.1f}' for t in show_traits if t in res_df.columns}),
            use_container_width=True, height=500,
        )
        with st.expander("Todos os traits"):
            st.dataframe(res_df, use_container_width=True, height=500)

    with tab2:
        if not group_df.empty:
            st.dataframe(group_df, use_container_width=True, height=500)

    with tab3:
        st.dataframe(conf_df, use_container_width=True)

    with tab4:
        if excluded:
            st.warning(f"{len(excluded)} animais excluidos (pai nao encontrado)")
            st.dataframe(pd.DataFrame(excluded), use_container_width=True)
        else:
            st.success("Nenhum animal excluido!")

    # Download
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    col_dl1, col_dl2 = st.columns([3, 1])
    with col_dl1:
        st.markdown("**Exportar Resultados** — Excel com 3 abas: Predicoes, Grupos, Confianca")
    with col_dl2:
        client_label = st.session_state.get('client_name', 'rebanho')
        excel_data = to_excel_bytes(res_df, group_df, conf_df)
        st.download_button(
            label="Baixar Excel",
            data=excel_data,
            file_name=f"{client_label}_DSII_V11.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary", use_container_width=True,
        )

    # Validation section
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box gold">
        <div class="title">Validacao (opcional)</div>
        <div class="text">
            Suba a planilha com valores reais de <strong style="color:#e8e9ec;">PA</strong> e/ou
            <strong style="color:#e8e9ec;">Genomica</strong> para comparar com as predicoes do DSII.
        </div>
    </div>
    """, unsafe_allow_html=True)

    val_col1, val_col2 = st.columns(2)
    with val_col1:
        pa_file = st.file_uploader("Upload PA real (.xlsx)", type=['xlsx','xls'], key="pa_upload")
    with val_col2:
        gen_file = st.file_uploader("Upload Genomica real (.xlsx)", type=['xlsx','xls'], key="gen_upload")

    if pa_file or gen_file:
        _show_validation(res_df, pa_file, gen_file, ordered_traits)

    # Footer
    st.markdown("""
    <div class="ss-footer">
        DSII Genetic Predictor v11.0 &mdash; Select Sires do Brasil &nbsp;&bull;&nbsp; 47 traits &nbsp;&bull;&nbsp; 258k touros &nbsp;&bull;&nbsp; Validado em 4.156 animais
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
