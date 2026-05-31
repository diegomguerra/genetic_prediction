"""
DSII PRODUCTION — Predicao Final V11 + Classificacao por Grupo
Arquitetura: PA + V11 Deviation + Recall@K Group Classification

CAMADA 1: Valores individuais por trait (PA + V11 deviation)
CAMADA 2: Classificacao por grupo (Elite 5%, Top 10%, Superior 20%, etc.)
           com indicador de confianca baseado no recall validado (Ethan, N=4156)

Uso:
  python dsii_production_predict.py <arquivo_cliente.xlsx> [--output <saida.xlsx>]

O arquivo de entrada deve ter colunas: ID, Pai (NAAB), Avo (NAAB), Bisavo (NAAB)
As primeiras 4 colunas sao usadas como ID, Pai, Avo, Bisavo.
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, pickle, csv, re, numpy as np, sys, time, argparse
from pathlib import Path
from collections import Counter
from scipy.stats import rankdata

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# CONFIGURACAO
# ============================================================
BASE = Path(r"C:\Users\DiegoGuerra\Documents\Projetos\genetic_prediction")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")
DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
V11_DIR = BASE / 'dsii_v11_results'

# Recall validado no Ethan (4156 animais USA com genomica real)
# Para cada trait: recall do top 5% real ao selecionar top K% predito
VALIDATED_RECALL = {
    # trait: {select_pct: recall_pct}
    'TPI':  {0.05: 43, 0.10: 74, 0.15: 85, 0.20: 88, 0.25: 94},
    'MILK': {0.05: 42, 0.10: 54, 0.15: 67, 0.20: 75, 0.25: 84},
    'FAT':  {0.05: 58, 0.10: 85, 0.15: 96, 0.20: 97, 0.25: 99},
    'FAT%': {0.05: 42, 0.10: 67, 0.15: 79, 0.20: 84, 0.25: 90},
    'PRO':  {0.05: 44, 0.10: 63, 0.15: 78, 0.20: 87, 0.25: 94},
    'PRO%': {0.05: 35, 0.10: 55, 0.15: 70, 0.20: 81, 0.25: 86},
    'PL':   {0.05: 44, 0.10: 66, 0.15: 80, 0.20: 83, 0.25: 87},
    'DPR':  {0.05: 59, 0.10: 79, 0.15: 85, 0.20: 89, 0.25: 93},
    'CCR':  {0.05: 52, 0.10: 73, 0.15: 79, 0.20: 84, 0.25: 88},
    'LIV':  {0.05: 35, 0.10: 54, 0.15: 66, 0.20: 77, 0.25: 84},
    'SCS':  {0.05: 44, 0.10: 71, 0.15: 82, 0.20: 87, 0.25: 93},
    'MAST': {0.05: 31, 0.10: 49, 0.15: 59, 0.20: 71, 0.25: 78},
    'UDC':  {0.05: 43, 0.10: 70, 0.15: 78, 0.20: 85, 0.25: 89},
    'FLC':  {0.05: 57, 0.10: 79, 0.15: 89, 0.20: 92, 0.25: 97},
    'HCR':  {0.05: 40, 0.10: 60, 0.15: 72, 0.20: 80, 0.25: 86},
    'NM$':  {0.05: 50, 0.10: 75, 0.15: 87, 0.20: 92, 0.25: 96},
    'PTAT': {0.05: 45, 0.10: 68, 0.15: 80, 0.20: 86, 0.25: 91},
}

# Traits to classify in groups (production-relevant)
GROUP_TRAITS = ['TPI', 'NM$', 'MILK', 'FAT', 'FAT%', 'PRO', 'PRO%',
                'PL', 'LIV', 'SCS', 'MAST', 'DPR', 'HCR', 'CCR',
                'PTAT', 'UDC', 'FLC']

# Category ordering for output
CATEGORIES = {
    'Indices': ['TPI','NM$','CM$'],
    'Producao': ['MILK','FAT','FAT%','PRO','PRO%','CFP'],
    'Longevidade': ['PL','LIV','H_LIV'],
    'Saude': ['SCS','MAST','MET','RP','DA','KET','MF'],
    'Reproducao': ['DPR','HCR','CCR','SCE','DCE','SSB','DSB','GL','EFC'],
    'Eficiencia': ['FI','F_SAV','RFI','GFI'],
    'Tipo Composto': ['PTAT','UDC','FLC'],
    'Tipo Linear': ['STA','STR','DFM','BOD','RUA','RW','RLS','RLR','FTA','FLS',
                    'FUA','RUH','RUW','UCL','UDP','FTP','RTP'],
}

CORE_TRAITS = ['TPI','MILK','FAT','FAT%','PRO','PRO%','PL','SCS','DPR','LIV','FI','CCR',
               'PTAT','UDC','FLC','MAST','NM$','HCR','SCE','F_SAV']

GENETIC_SD = {
    'TPI':250,'MILK':675,'FAT':29,'FAT%':0.05,'PRO':19,'PRO%':0.02,'CFP':40,
    'PL':1.85,'LIV':1.2,'H_LIV':1.0,
    'SCS':0.14,'MAST':1.0,'MET':1.0,'RP':0.5,'DA':0.5,'KET':0.8,'MF':0.3,
    'DPR':1.3,'HCR':1.5,'CCR':1.65,'SCE':0.8,'DCE':0.8,'SSB':1.0,'DSB':1.0,'GL':0.8,'EFC':1.0,
    'FI':1.0,'F_SAV':50,'RFI':30,'GFI':3.0,
    'PTAT':0.70,'UDC':0.75,'FLC':0.65,
    'STA':0.80,'STR':0.70,'DFM':0.80,'BOD':0.60,'RUA':0.60,'RW':0.70,
    'RLS':0.50,'RLR':0.50,'FTA':0.60,'FLS':0.50,'FTL':0.50,
    'FUA':0.80,'RUH':0.80,'RUW':0.80,'UCL':0.60,'UDP':0.80,'FTP':0.70,'RTP':0.60,'TL':0.50,
    'BWC':0.50,'NM$':200,'CM$':200,
}
HERITABILITY = {
    'TPI':0.30,'MILK':0.25,'FAT':0.25,'FAT%':0.50,'PRO':0.25,'PRO%':0.50,'CFP':0.25,
    'PL':0.08,'LIV':0.05,'H_LIV':0.05,
    'SCS':0.12,'MAST':0.04,'MET':0.03,'RP':0.02,'DA':0.04,'KET':0.04,'MF':0.02,
    'DPR':0.04,'HCR':0.04,'CCR':0.04,'SCE':0.08,'DCE':0.06,'SSB':0.05,'DSB':0.04,'GL':0.40,'EFC':0.06,
    'FI':0.06,'F_SAV':0.15,'RFI':0.15,'GFI':0.15,
    'PTAT':0.30,'UDC':0.25,'FLC':0.15,
    'STA':0.42,'STR':0.27,'DFM':0.25,'BOD':0.30,'RUA':0.30,'RW':0.25,
    'RLS':0.15,'RLR':0.10,'FTA':0.10,'FLS':0.10,'FTL':0.10,
    'FUA':0.25,'RUH':0.20,'RUW':0.20,'UCL':0.15,'UDP':0.30,'FTP':0.25,'RTP':0.20,'TL':0.25,
    'BWC':0.20,'NM$':0.30,'CM$':0.30,
}
GENETIC_CORRELATIONS = {
    ('MILK','DPR'):-0.35,('MILK','CCR'):-0.30,('MILK','SCS'):0.10,('MILK','PL'):-0.15,
    ('FAT','PRO'):0.65,('FAT','DPR'):-0.25,('PRO','DPR'):-0.30,('DPR','CCR'):0.55,
    ('DPR','PL'):0.45,('SCS','PL'):-0.30,('SCS','MAST'):0.70,('PTAT','UDC'):0.40,
    ('PTAT','FLC'):0.30,('MILK','FAT'):0.55,('MILK','PRO'):0.85,('PL','LIV'):0.85,('DPR','FI'):0.70,
    ('HCR','CCR'):0.60,('HCR','DPR'):0.50,('SCE','DCE'):0.60,('SSB','DSB'):0.55,
    ('MAST','MET'):0.30,('MAST','DA'):0.20,('KET','DA'):0.35,('MET','RP'):0.30,
    ('STA','STR'):0.25,('STA','BOD'):0.40,('PTAT','STA'):0.30,
    ('UDC','FUA'):0.70,('UDC','RUH'):0.65,('UDC','UDP'):0.60,('UDC','UCL'):0.40,
    ('FLC','FTA'):0.50,('FLC','RLS'):0.35,('FLC','RLR'):0.40,('FLC','FLS'):0.60,
    ('PL','H_LIV'):0.65,('LIV','H_LIV'):0.70,('PL','MAST'):-0.30,
    ('F_SAV','RFI'):-0.80,('FI','F_SAV'):0.70,('FI','RFI'):-0.60,
    ('NM$','TPI'):0.90,('NM$','CM$'):0.95,('NM$','PL'):0.50,('NM$','FAT'):0.40,
    ('GL','SCE'):-0.30,('GL','SSB'):-0.25,
}
BULL_COL = {
    'TPI':'TPI','MILK':'PTAM','FAT':'PTAF','FAT%':'PTAF%','PRO':'PTAP','PRO%':'PTAP%','CFP':'CFP',
    'PL':'PL','LIV':'LIV','H_LIV':'H LIV',
    'SCS':'SCS','MAST':'MAST','MET':'MET','RP':'RP','DA':'DA','KET':'KET','MF':'MF',
    'DPR':'DPR','HCR':'HCR','CCR':'CCR','SCE':'SCE','DCE':'DCE','SSB':'SSB','DSB':'DSB','GL':'GL','EFC':'EFC',
    'FI':'FI','F_SAV':'F SAV','RFI':'RFI','GFI':'GFI',
    'PTAT':'PTAT','UDC':'UDC','FLC':'FLC',
    'STA':'STA','STR':'STR','DFM':'DFM','BOD':'BOD','RUA':'RUA','RW':'RW',
    'RLS':'RLS','RLR':'RLR','FTA':'FTA','FLS':'FLS','FTL':'FTL',
    'FUA':'FUA','RUH':'RUH','RUW':'RUW','UCL':'UCL','UDP':'UDP','FTP':'FTP','RTP':'RTP','TL':None,
    'BWC':'BWC','NM$':'NM$','CM$':'CM$',
}
CDCB_COL = {
    'TPI':None,'MILK':'MLK_PTA','FAT':'FAT_PTA','FAT%':'FAT%','PRO':'PRO_PTA','PRO%':'PRO%','CFP':None,
    'PL':'PL_PTA','LIV':'LIV_PTA','H_LIV':'HLV_PTA',
    'SCS':'SCS_PTA','MAST':'MAS_PTA','MET':'MET_PTA','RP':'RPL_PTA','DA':'DAB_PTA','KET':'KET_PTA','MF':'MFV_PTA',
    'DPR':'DPR_PTA','HCR':'HCR_PTA','CCR':'CCR_PTA','SCE':None,'DCE':None,'SSB':None,'DSB':None,'GL':'GL_PTA','EFC':'EFC_PTA',
    'FI':'FS_PTA','F_SAV':None,'RFI':'RFI_PTA','GFI':None,
    'PTAT':'TYP_PTA','UDC':'UDC_PTA','FLC':'FLC_PTA',
    'STA':None,'STR':None,'DFM':None,'BOD':None,'RUA':None,'RW':None,
    'RLS':None,'RLR':None,'FTA':None,'FLS':None,'FTL':None,
    'FUA':None,'RUH':None,'RUW':None,'UCL':None,'UDP':None,'FTP':None,'RTP':None,'TL':None,
    'BWC':None,'NM$':'NM$_PTA','CM$':'CM$_PTA',
}
STUD_MAP = {
    '507':'7','509':'9','550':'50','551':'51','559':'59','518':'18','521':'21',
    '523':'23','581':'81','585':'85','501':'1','601':'1','604':'4','614':'14',
    '629':'29','751':'151','752':'152','814':'14','250':'200',
    '001':'1','007':'7','009':'9','011':'11','029':'29','097':'97',
    '100':'100','200':'200','777':'777','796':'796','745':'745',
}

# ============================================================
# BULL DATABASE FUNCTIONS
# ============================================================
def sf(v):
    if v is None: return None
    try:
        x = float(v)
        return x if not np.isnan(x) else None
    except: return None

def _normalize_bull_row(row):
    out = {}
    for k, v in row.items():
        if k is None: continue
        if '$' in k:
            idx = k.index('$')
            k = k[:idx+1] + ''.join(c for c in k[idx+1:] if c.isalnum() or c in ' _').rstrip()
        out[k] = v
    return out

def normalize_naab(naab):
    if not naab or str(naab).strip() in ('','nan','None'): return []
    naab = str(naab).strip()
    m = re.match(r'^(\d+)(HO|BS|H|B)0*(\d+)$', naab)
    if not m: return [naab]
    org, breed_raw, num = m.group(1), m.group(2), m.group(3)
    if breed_raw in ('H', 'HO'): breeds = ['HO', 'H']
    elif breed_raw in ('B', 'BS'): breeds = ['BS', 'B']
    else: breeds = [breed_raw]
    orgs = set([org, org.lstrip('0') or org])
    if org in STUD_MAP: orgs.add(STUD_MAP[org])
    if org.startswith('5'): orgs.add(org[1:])
    nums = [num]
    num_padded = num.zfill(5)
    if num_padded != num: nums.append(num_padded)
    return list(dict.fromkeys(f'{o}{b}{n}' for o in orgs for b in breeds for n in nums))

def get_bull_ptas(bull_row, source='SS'):
    ptas = {}
    col_map = BULL_COL if source == 'SS' else CDCB_COL
    for trait, col in col_map.items():
        if col is None: continue
        v = sf(bull_row.get(col))
        if v is not None: ptas[trait] = v
    return ptas

def lookup_bull(naab_raw, bulls, cdcb_bulls, cdcb_to_ss, bulls_by_num, cdcb_by_num):
    m_raw = re.match(r'^(\d+)(HO|BS|H|B)0*(\d+)$', str(naab_raw).strip())
    for c in normalize_naab(naab_raw):
        if c in bulls:
            return bulls[c], 'SS', c, get_bull_ptas(bulls[c], 'SS')
        if c in cdcb_bulls:
            if c in cdcb_to_ss:
                ss_naab = cdcb_to_ss[c]
                return bulls[ss_naab], 'SS', ss_naab, get_bull_ptas(bulls[ss_naab], 'SS')
            return cdcb_bulls[c], 'CDCB', c, get_bull_ptas(cdcb_bulls[c], 'CDCB')
    if not m_raw:
        return None, None, None, {}
    num = m_raw.group(3)
    orig_org = m_raw.group(1).lstrip('0') or m_raw.group(1)
    if num in bulls_by_num:
        matches = bulls_by_num[num]
        if len(matches) == 1:
            naab, row = matches[0]
            return row, 'SS', naab, get_bull_ptas(row, 'SS')
        for naab, row in matches:
            m2 = re.match(r'^(\d+)(HO|BS)(\d+)$', naab)
            if m2 and m2.group(1).lstrip('0') == orig_org:
                return row, 'SS', naab, get_bull_ptas(row, 'SS')
        naab, row = matches[0]
        return row, 'SS', naab, get_bull_ptas(row, 'SS')
    if num in cdcb_by_num:
        matches = cdcb_by_num[num]
        for naab, row in matches:
            if naab in cdcb_to_ss:
                ss_naab = cdcb_to_ss[naab]
                return bulls[ss_naab], 'SS', ss_naab, get_bull_ptas(bulls[ss_naab], 'SS')
        for naab, row in matches:
            m2 = re.match(r'^(\d+)(HO|BS)(\d+)$', naab)
            if m2 and m2.group(1).lstrip('0') == orig_org:
                return row, 'CDCB', naab, get_bull_ptas(row, 'CDCB')
        naab, row = matches[0]
        return row, 'CDCB', naab, get_bull_ptas(row, 'CDCB')
    return None, None, None, {}

# ============================================================
# V11 PREDICTION ENGINE
# ============================================================
def compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait):
    s = sire_ptas.get(trait)
    if s is None: return None
    pa = s / 2
    mg = mgs_ptas.get(trait) if mgs_ptas else None
    if mg is not None:
        pa += mg / 4
        mmg = mmgs_ptas.get(trait) if mmgs_ptas else None
        if mmg is not None:
            pa += mmg / 8
    return pa

def build_features_v11(sire_ptas, mgs_ptas, trait, sire_naab, mgs_naab, mmgs_ptas,
                       sire_profiles, mgs_profiles):
    s = sire_ptas.get(trait)
    if s is None: return None

    f = {}
    sd = GENETIC_SD.get(trait, 1)
    h2 = HERITABILITY.get(trait, 0.15)

    pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)
    f['pa'] = pa
    f['pa_z'] = pa / sd if sd else 0

    f['sire'] = s
    f['sire_z'] = s / sd if sd else 0
    f['sire_sq'] = s * s
    f['sire_h2'] = s * h2
    f['sire_dev_from_pa'] = s - pa * 2

    mg = mgs_ptas.get(trait) if mgs_ptas else None
    f['mgs_val'] = mg if mg is not None else 0
    f['mgs_z'] = (mg / sd if sd else 0) if mg is not None else 0
    f['mgs_sq'] = mg * mg if mg is not None else 0
    f['has_mgs'] = 1 if mg is not None else 0

    if mg is not None:
        f['sire_x_mgs'] = s * mg
        f['sire_mgs_diff'] = s - mg
        f['sire_mgs_ratio'] = s / mg if mg != 0 else 0
        f['sire_mgs_mean'] = (s + mg) / 2
        f['mendelian_range'] = abs(s - mg)
        f['mendelian_range_z'] = abs(s - mg) / sd if sd else 0
    else:
        f['sire_x_mgs'] = 0; f['sire_mgs_diff'] = 0; f['sire_mgs_ratio'] = 0
        f['sire_mgs_mean'] = s / 2; f['mendelian_range'] = 0; f['mendelian_range_z'] = 0

    mmg = mmgs_ptas.get(trait) if mmgs_ptas else None
    f['mmgs_val'] = mmg if mmg is not None else 0
    f['has_mmgs'] = 1 if mmg is not None else 0
    f['sire_x_mmgs'] = s * mmg if mmg is not None else 0

    for ot in CORE_TRAITS:
        if ot == trait: continue
        sv = sire_ptas.get(ot)
        if sv is not None: f[f'sire_{ot}'] = sv
        mv = mgs_ptas.get(ot) if mgs_ptas else None
        if mv is not None: f[f'mgs_{ot}'] = mv
        ot_pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, ot)
        if ot_pa is not None: f[f'pa_{ot}'] = ot_pa

    for (t1, t2), corr in GENETIC_CORRELATIONS.items():
        if t1 == trait or t2 == trait:
            other = t2 if t1 == trait else t1
            ot_pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, other)
            if ot_pa is not None: f[f'gc_pa_{other}'] = ot_pa * corr
            sv = sire_ptas.get(other)
            if sv is not None: f[f'gc_sire_{other}'] = sv * corr
            mv = mgs_ptas.get(other) if mgs_ptas else None
            if mv is not None: f[f'gc_mgs_{other}'] = mv * corr

    if sire_naab and sire_naab in sire_profiles:
        sp = sire_profiles[sire_naab]
        tp = sp.get(trait)
        if tp and tp['n'] >= 2:
            f['sire_prof_n'] = tp['n']; f['sire_prof_mean'] = tp['mean']
            f['sire_prof_std'] = tp['std']; f['sire_prof_ratio'] = tp['trans_ratio']
            f['sire_prof_resid'] = tp['residual']
            f['sire_prof_dev_pa'] = tp['mean'] - pa
        else:
            f['sire_prof_n'] = 0; f['sire_prof_mean'] = 0; f['sire_prof_std'] = 0
            f['sire_prof_ratio'] = 1.0; f['sire_prof_resid'] = 0; f['sire_prof_dev_pa'] = 0
        for ot in CORE_TRAITS:
            if ot == trait: continue
            otp = sp.get(ot)
            if otp and otp['n'] >= 2: f[f'sire_prof_{ot}'] = otp['residual']
    else:
        f['sire_prof_n'] = 0; f['sire_prof_mean'] = 0; f['sire_prof_std'] = 0
        f['sire_prof_ratio'] = 1.0; f['sire_prof_resid'] = 0; f['sire_prof_dev_pa'] = 0

    if mgs_naab and mgs_naab in mgs_profiles:
        mp = mgs_profiles[mgs_naab]
        tp = mp.get(trait)
        if tp and tp['n'] >= 2:
            f['mgs_prof_n'] = tp['n']; f['mgs_prof_mean'] = tp['mean']; f['mgs_prof_resid'] = tp['residual']
        else:
            f['mgs_prof_n'] = 0; f['mgs_prof_mean'] = 0; f['mgs_prof_resid'] = 0
    else:
        f['mgs_prof_n'] = 0; f['mgs_prof_mean'] = 0; f['mgs_prof_resid'] = 0

    return f

def predict_batch_v11(trait, animals, saved_models, sire_profiles, mgs_profiles, pl_preds=None):
    info = saved_models.get(trait)
    if not info: return [None] * len(animals)

    feat_rows = []
    pa_vals = []
    valid_idx = []
    for i, (sire_ptas, mgs_ptas, sire_naab, mgs_naab, mmgs_ptas) in enumerate(animals):
        feat = build_features_v11(sire_ptas, mgs_ptas, trait, sire_naab, mgs_naab, mmgs_ptas,
                                  sire_profiles, mgs_profiles)
        if feat is None: continue

        pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)
        if pa is None: pa = 0

        if info.get('v10b') == 'pl_anchored':
            pl_val = pl_preds[i] if pl_preds and pl_preds[i] is not None else sire_ptas.get('PL', 0) / 2
            feat['pl_pred'] = pl_val
            feat['pl_pred_sq'] = pl_val ** 2

        feat_rows.append(feat)
        pa_vals.append(pa)
        valid_idx.append(i)

    if not feat_rows: return [None] * len(animals)

    feat_df = pd.DataFrame(feat_rows)
    for c in info['feature_cols']:
        if c not in feat_df.columns: feat_df[c] = 0
    feat_df = feat_df[info['feature_cols']].fillna(0)
    X = feat_df.values.astype(np.float64)
    pa_arr = np.array(pa_vals, dtype=np.float64)

    if info['type'] == 'stack':
        base_preds = np.column_stack([info['base_models'][mn].predict(X) for mn in info['base_names']])
        dev_preds = info['meta_model'].predict(base_preds)
    else:
        dev_preds = info['model'].predict(X)

    preds_v11 = pa_arr + dev_preds

    # Hybrid 3-way
    if info.get('v10b') == 'hybrid_3way':
        sign_clf = info.get('sign_clf')
        if sign_clf is not None:
            sign_probs = sign_clf.predict_proba(X)[:, 1]
        else:
            sign_probs = np.ones(len(X))
        mags = np.maximum(info['mag_model'].predict(X), 0)

        final_preds = np.zeros(len(X))
        for j, vi in enumerate(valid_idx):
            sire_ptas, mgs_ptas, sire_naab, mgs_naab, mmgs_ptas = animals[vi]
            pred_sign = 1.0 if sign_probs[j] >= 0.5 else -1.0
            sign_conf = abs(sign_probs[j] - 0.5) * 2
            n_d = sire_profiles.get(sire_naab, {}).get(trait, {}).get('n', 0)
            prof_conf = min(n_d / 30, 1.0) if n_d > 0 else 0.2
            total_conf = 0.6 * sign_conf + 0.4 * prof_conf
            pred_2stage = pred_sign * mags[j] * total_conf

            sp = sire_profiles.get(sire_naab, {})
            tp = sp.get(trait)
            mgs_val = mgs_ptas.get(trait, 0) if mgs_ptas else 0
            if tp and tp['n'] >= 5:
                pred_anc = tp['mean'] * 0.85 + mgs_val * 0.15
            elif tp and tp['n'] >= 2:
                pred_anc = tp['mean'] * 0.6 + (sire_ptas.get(trait, 0) / 2) * 0.25 + mgs_val * 0.15
            else:
                pred_anc = pa_arr[j]

            blend = np.array([[preds_v11[j], pred_2stage, pred_anc, pa_arr[j]]])
            final_preds[j] = info['blend_ridge'].predict(blend)[0]
        preds_v11 = final_preds

    out = [None] * len(animals)
    for j, vi in enumerate(valid_idx):
        out[vi] = round(float(preds_v11[j]), 2)
    return out

# ============================================================
# GROUP CLASSIFICATION (CAMADA 2)
# ============================================================
def classify_groups(predictions, trait, n_animals):
    """
    Classifica animais em grupos percentuais e atribui confianca.

    Returns dict per animal index:
      group_label: 'Elite 5%', 'Top 10%', 'Superior 20%', 'Acima Media', 'Media', 'Abaixo Media'
      group_rank: percentile rank (0-100, 100=melhor)
      recall: confianca validada de que animal esta corretamente classificado
    """
    valid = [(i, v) for i, v in enumerate(predictions) if v is not None]
    if len(valid) < 10:
        return {}

    indices = [v[0] for v in valid]
    values = np.array([v[1] for v in valid])
    n = len(values)

    # Percentile rank (100 = melhor)
    ranks = rankdata(values) / n * 100

    # Recall lookup
    recall_data = VALIDATED_RECALL.get(trait, {})

    result = {}
    for j, idx in enumerate(indices):
        pct = ranks[j]

        # Group label and recall
        if pct >= 95:
            label = 'Elite 5%'
            recall = recall_data.get(0.05, '?')
        elif pct >= 90:
            label = 'Top 10%'
            recall = recall_data.get(0.10, '?')
        elif pct >= 80:
            label = 'Superior 20%'
            recall = recall_data.get(0.20, '?')
        elif pct >= 50:
            label = 'Acima Media'
            recall = '-'
        elif pct >= 20:
            label = 'Media'
            recall = '-'
        else:
            label = 'Abaixo Media'
            recall = '-'

        result[idx] = {
            'label': label,
            'percentile': round(pct, 1),
            'recall': recall,
        }

    return result

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='DSII Production Predict V11')
    parser.add_argument('input_file', nargs='?', default=None, help='Arquivo Excel do cliente')
    parser.add_argument('--output', '-o', default=None, help='Arquivo de saida')
    parser.add_argument('--header-row', type=int, default=0, help='Linha do cabecalho (0-based)')
    parser.add_argument('--col-map', default=None, help='Mapeamento de colunas (JSON)')
    args = parser.parse_args()

    if args.input_file is None:
        # Default: Jose
        input_file = DOWNLOADS / 'Cliente Jose Goncalvez.xlsx'
        if not input_file.exists():
            input_file = DOWNLOADS / 'Cliente Jose Gonçalvez.xlsx'
        if not input_file.exists():
            # Try accent variations
            for f in DOWNLOADS.glob('Cliente Jos*'):
                input_file = f; break
    else:
        input_file = Path(args.input_file)

    if not input_file.exists():
        print(f"ERRO: Arquivo nao encontrado: {input_file}")
        sys.exit(1)

    output_file = Path(args.output) if args.output else input_file.parent / f"{input_file.stem}_DSII_V11.xlsx"
    excluded_file = input_file.parent / f"{input_file.stem}_excluidos.xlsx"

    t0 = time.time()

    # === LOAD DATABASES ===
    print("=" * 120)
    print("  DSII PRODUCTION V11 — Predicao Genetica com Classificacao por Grupo")
    print("=" * 120)

    print("\n  Carregando bases de touros...", flush=True)
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
    print(f"    bulls.csv: {len(bulls)} touros", flush=True)

    cdcb_bulls = {}
    cdcb_to_ss = {}
    try:
        with open(DOWNLOADS / 'Bull_Report (1).csv', 'r', encoding='latin-1') as f:
            for row in csv.DictReader(f):
                naab = row.get('NAAB_CODE','').strip()
                if naab:
                    cdcb_bulls[naab] = row
                    name = row.get('NAME','').strip().upper()
                    if name and name in bulls_by_name:
                        cdcb_to_ss[naab] = bulls_by_name[name][0]
        print(f"    CDCB: {len(cdcb_bulls)} touros", flush=True)
    except:
        print("    CDCB: nao disponivel", flush=True)

    bulls_by_num = {}
    for naab, row in bulls.items():
        m = re.match(r'^(\d+)(HO|BS)(\d+)$', naab)
        if m: bulls_by_num.setdefault(m.group(3), []).append((naab, row))

    cdcb_by_num = {}
    for naab, row in cdcb_bulls.items():
        m = re.match(r'^(\d+)(HO|BS)(\d+)$', naab)
        if m: cdcb_by_num.setdefault(m.group(3), []).append((naab, row))

    # Load V11 models
    print("  Carregando modelos V11...", flush=True)
    with open(V11_DIR / 'v11_models.pkl', 'rb') as f: saved_models = pickle.load(f)
    with open(V11_DIR / 'v11_sire_profiles.pkl', 'rb') as f: sire_profiles = pickle.load(f)
    with open(V11_DIR / 'v11_mgs_profiles.pkl', 'rb') as f: mgs_profiles = pickle.load(f)
    print(f"    {len(saved_models)} traits disponiveis", flush=True)

    ALL_TRAITS = list(saved_models.keys())

    # === LOAD CLIENT FILE ===
    print(f"\n  Arquivo: {input_file.name}", flush=True)
    df = pd.read_excel(input_file, engine='openpyxl', header=args.header_row)
    cols = list(df.columns)
    id_col, pai_col, avo_col = cols[0], cols[1], cols[2]
    bis_col = cols[3] if len(cols) > 3 else None
    print(f"  Total animais: {len(df)}", flush=True)

    # === RESOLVE PEDIGREES ===
    print("\n  Resolvendo pedigrees...", flush=True)
    excluded = []
    valid_animals = []

    for _, row in df.iterrows():
        animal_id = row[id_col]
        pai_naab = str(row[pai_col]).strip()
        avo_naab = str(row[avo_col]).strip() if avo_col else 'nan'
        bis_naab = str(row[bis_col]).strip() if bis_col else 'nan'

        pai_row, pai_src, pai_resolved, pai_ptas = lookup_bull(pai_naab, bulls, cdcb_bulls, cdcb_to_ss, bulls_by_num, cdcb_by_num)

        if not pai_row or not pai_ptas:
            excluded.append({'ID': animal_id, 'Pai': pai_naab, 'motivo': f"Pai nao encontrado ({pai_naab})"})
            continue

        avo_row, avo_src, avo_resolved, avo_ptas = (None, None, None, {})
        if avo_naab and avo_naab not in ('nan', 'None', ''):
            avo_row, avo_src, avo_resolved, avo_ptas = lookup_bull(avo_naab, bulls, cdcb_bulls, cdcb_to_ss, bulls_by_num, cdcb_by_num)

        bis_row, bis_src, bis_resolved, bis_ptas = (None, None, None, {})
        if bis_naab and bis_naab not in ('nan', 'None', ''):
            bis_row, bis_src, bis_resolved, bis_ptas = lookup_bull(bis_naab, bulls, cdcb_bulls, cdcb_to_ss, bulls_by_num, cdcb_by_num)

        meta = {
            'ID': animal_id,
            'Pai_NAAB': pai_naab, 'Pai_Resolved': pai_resolved or pai_naab,
            'Pai_Src': pai_src or 'N/A',
            'N_Traits_Pai': len(pai_ptas),
        }
        valid_animals.append((meta, pai_ptas, avo_ptas, pai_resolved, avo_resolved, bis_ptas))

    print(f"    Validos: {len(valid_animals)}/{len(df)}", flush=True)
    print(f"    Excluidos: {len(excluded)}", flush=True)

    if not valid_animals:
        print("\n  ERRO: Nenhum animal valido para predicao!")
        sys.exit(1)

    # === PREDICT ===
    animals_input = [(a[1], a[2], a[3], a[4], a[5]) for a in valid_animals]

    print(f"\n  CAMADA 1: Predicao de valores individuais ({len(ALL_TRAITS)} traits)", flush=True)
    print(f"  {'='*80}", flush=True)

    # PL first (for PL-anchored traits)
    pl_preds = predict_batch_v11('PL', animals_input, saved_models, sire_profiles, mgs_profiles)

    results = [{**a[0]} for a in valid_animals]
    for trait in ALL_TRAITS:
        t1 = time.time()
        preds = predict_batch_v11(trait, animals_input, saved_models, sire_profiles, mgs_profiles, pl_preds)
        for i, p in enumerate(preds):
            if p is not None:
                results[i][trait] = p
        n_valid = sum(1 for p in preds if p is not None)
        elapsed = time.time() - t1
        print(f"    {trait:>6}: {n_valid:>4} predicoes ({elapsed:.1f}s)", flush=True)

    # === GROUP CLASSIFICATION ===
    n_animals = len(results)
    print(f"\n  CAMADA 2: Classificacao por grupo (N={n_animals})", flush=True)
    print(f"  {'='*80}", flush=True)

    group_data = {}  # group_data[trait] = {idx: {label, percentile, recall}}

    for trait in GROUP_TRAITS:
        if trait not in ALL_TRAITS: continue
        preds_list = [(i, results[i].get(trait)) for i in range(len(results))]
        groups = classify_groups(preds_list, trait, n_animals)
        group_data[trait] = groups

        # Stats
        labels = [g['label'] for g in groups.values()]
        elite_n = labels.count('Elite 5%')
        top_n = labels.count('Top 10%')
        recall_info = VALIDATED_RECALL.get(trait, {})
        r5 = recall_info.get(0.05, '?')
        r10 = recall_info.get(0.10, '?')
        print(f"    {trait:>6}: Elite={elite_n} Top10={top_n} | Recall@5%={r5}% Recall@10%={r10}%", flush=True)

    # === BUILD OUTPUT ===
    print(f"\n  Gerando arquivo de saida...", flush=True)

    # Sheet 1: Predicoes individuais (valores por trait)
    meta_cols = ['ID', 'Pai_NAAB', 'Pai_Resolved', 'Pai_Src', 'N_Traits_Pai']
    ordered_traits = []
    for cat_traits in CATEGORIES.values():
        for t in cat_traits:
            if t in ALL_TRAITS: ordered_traits.append(t)
    for t in ALL_TRAITS:
        if t not in ordered_traits: ordered_traits.append(t)

    final_cols = [c for c in meta_cols if c in results[0]] + [t for t in ordered_traits if t in results[0]]
    res_df = pd.DataFrame(results)[final_cols]

    # Sheet 2: Classificacao por grupo (com percentil e recall)
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

    # Sheet 3: Resumo de confianca por trait
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
            'Interpretacao': (
                f"Dos {max(1,int(n_animals*0.05))} animais realmente Elite 5%, "
                f"capturamos ~{recall_data.get(0.05,'?')}% ao selecionar top 5% predito "
                f"e ~{recall_data.get(0.10,'?')}% ao selecionar top 10% predito."
            )
        })
    conf_df = pd.DataFrame(conf_rows)

    # Write Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        res_df.to_excel(writer, sheet_name='Predicoes', index=False)
        group_df.to_excel(writer, sheet_name='Grupos', index=False)
        conf_df.to_excel(writer, sheet_name='Confianca', index=False)

    print(f"\n  Arquivo salvo: {output_file}", flush=True)
    print(f"  - Aba 'Predicoes': valores individuais por trait ({len(ordered_traits)} traits)")
    print(f"  - Aba 'Grupos': classificacao por grupo + percentil + recall")
    print(f"  - Aba 'Confianca': tabela de confianca por trait")

    if excluded:
        exc_df = pd.DataFrame(excluded)
        exc_df.to_excel(excluded_file, index=False)
        print(f"  Excluidos: {excluded_file} ({len(excluded)} animais)")

    # === PRINT SUMMARY ===
    elapsed = time.time() - t0
    print(f"\n{'='*120}")
    print(f"  RESUMO DA PREDICAO")
    print(f"{'='*120}")
    print(f"  Animais preditos: {len(results)}")
    print(f"  Traits preditas:  {len(ordered_traits)}")
    print(f"  Tempo total:      {elapsed:.0f}s")

    print(f"\n  DISTRIBUICAO POR GRUPO (traits principais):")
    print(f"  {'Trait':>6} | {'Elite5%':>8} | {'Top10%':>8} | {'Sup20%':>8} | {'AcMedia':>8} | {'Media':>8} | {'AbMedia':>8} | {'Recall@10%':>10}")
    print(f"  {'-'*90}")
    for trait in GROUP_TRAITS:
        if trait not in group_data: continue
        groups = group_data[trait]
        labels = [g['label'] for g in groups.values()]
        recall_data = VALIDATED_RECALL.get(trait, {})
        print(f"  {trait:>6} | {labels.count('Elite 5%'):>8} | {labels.count('Top 10%'):>8} | "
              f"{labels.count('Superior 20%'):>8} | {labels.count('Acima Media'):>8} | "
              f"{labels.count('Media'):>8} | {labels.count('Abaixo Media'):>8} | "
              f"{recall_data.get(0.10, '?'):>9}%")

    print(f"\n  Como interpretar os grupos:")
    print(f"    Elite 5%  = Animal predito entre os 5% melhores do rebanho")
    print(f"    Top 10%   = Animal predito entre os 10% melhores do rebanho")
    print(f"    Recall@10% = Se voce selecionar os 10% melhores preditos,")
    print(f"                 essa e a % dos 5% realmente melhores que voce captura")
    print(f"    Exemplo FAT: selecionar top 10% predito captura 85% dos top 5% reais")
    print(f"\n  Predicao concluida!")

if __name__ == '__main__':
    main()
