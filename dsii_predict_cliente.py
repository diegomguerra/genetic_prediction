"""
DSII V10-B — Predicao Cliente Jose Goncalvez (52 traits, prova completa)
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, pickle, csv, re, numpy as np, sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BULLS_CSV = Path('C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv')
DOWNLOADS = Path('C:/Users/DiegoGuerra/Downloads')
RESULTS_DIR = Path('C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v10_results')

INPUT_FILE = DOWNLOADS / 'Cliente José Gonçalvez.xlsx'
OUTPUT_FILE = RESULTS_DIR / 'jose_goncalvez_predictions_v10b.xlsx'
EXCLUDED_FILE = RESULTS_DIR / 'jose_goncalvez_excluidos.xlsx'

# Load V10-B models
with open(RESULTS_DIR / 'v10_models.pkl', 'rb') as f: saved_models = pickle.load(f)
with open(RESULTS_DIR / 'v10_sire_profiles.pkl', 'rb') as f: sire_profiles = pickle.load(f)
with open(RESULTS_DIR / 'v10_mgs_profiles.pkl', 'rb') as f: mgs_profiles = pickle.load(f)

ALL_TRAITS = list(saved_models.keys())
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
            k = k[:idx+1] + ''.join(c for c in k[idx+1:] if c.isalnum() or c in ' _')
            k = k.rstrip()
        out[k] = v
    return out

STUD_MAP = {
    '507':'7','509':'9','550':'50','551':'51','559':'59','518':'18','521':'21',
    '523':'23','581':'81','585':'85','501':'1','601':'1','604':'4','614':'14',
    '629':'29','751':'151','752':'152','814':'14','250':'200',
    '001':'1','007':'7','009':'9','011':'11','029':'29','097':'97',
    '100':'100','200':'200','777':'777','796':'796','745':'745',
}

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

# ============================================================
# LOAD BULL DATABASES
# ============================================================
print("Carregando bases de touros...")

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
print(f"  bulls.csv: {len(bulls)} touros, {len(bulls_by_name)} nomes")

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
    print(f"  CDCB: {len(cdcb_bulls)} touros, {len(cdcb_to_ss)} cross-ref")
except:
    print("  CDCB: nao disponivel")

bulls_by_num = {}
for naab, row in bulls.items():
    m = re.match(r'^(\d+)(HO|BS)(\d+)$', naab)
    if m: bulls_by_num.setdefault(m.group(3), []).append((naab, row))

cdcb_by_num = {}
for naab, row in cdcb_bulls.items():
    m = re.match(r'^(\d+)(HO|BS)(\d+)$', naab)
    if m: cdcb_by_num.setdefault(m.group(3), []).append((naab, row))

def get_bull_ptas(bull_row, source='SS'):
    ptas = {}
    col_map = BULL_COL if source == 'SS' else CDCB_COL
    for trait, col in col_map.items():
        if col is None: continue
        v = sf(bull_row.get(col))
        if v is not None: ptas[trait] = v
    return ptas

def lookup_bull(naab_raw):
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

def build_features_v10(sire_ptas, mgs_ptas, trait, sire_naab=None, mgs_naab=None, mmgs_ptas=None):
    s = sire_ptas.get(trait)
    if s is None: return None
    f = {}
    sd = GENETIC_SD.get(trait, 1); h2 = HERITABILITY.get(trait, 0.15)
    f['sire'] = s; f['sire_z'] = s/sd if sd else 0; f['sire_sq'] = s*s; f['sire_h2'] = s*h2
    mg = mgs_ptas.get(trait) if mgs_ptas else None
    f['mgs_val'] = mg if mg is not None else 0
    f['mgs_z'] = (mg/sd if sd else 0) if mg is not None else 0
    f['mgs_sq'] = mg*mg if mg is not None else 0
    f['has_mgs'] = 1 if mg is not None else 0
    if mg is not None:
        f['sire_x_mgs'] = s*mg; f['sire_mgs_diff'] = s-mg; f['sire_mgs_ratio'] = s/mg if mg != 0 else 0
    else:
        f['sire_x_mgs'] = 0; f['sire_mgs_diff'] = 0; f['sire_mgs_ratio'] = 0
    mmg = mmgs_ptas.get(trait) if mmgs_ptas else None
    f['mmgs_val'] = mmg if mmg is not None else 0; f['has_mmgs'] = 1 if mmg is not None else 0
    f['sire_x_mmgs'] = s*mmg if mmg is not None else 0
    for ot in CORE_TRAITS:
        if ot == trait: continue
        sv = sire_ptas.get(ot)
        if sv is not None: f[f'sire_{ot}'] = sv
        mv = mgs_ptas.get(ot) if mgs_ptas else None
        if mv is not None: f[f'mgs_{ot}'] = mv
    for (t1, t2), corr in GENETIC_CORRELATIONS.items():
        if t1 == trait or t2 == trait:
            other = t2 if t1 == trait else t1
            sv = sire_ptas.get(other)
            if sv is not None: f[f'gc_sire_{other}'] = sv * corr
            mv = mgs_ptas.get(other) if mgs_ptas else None
            if mv is not None: f[f'gc_mgs_{other}'] = mv * corr
    if sire_naab and sire_naab in sire_profiles:
        sp = sire_profiles[sire_naab]; tp = sp.get(trait)
        if tp and tp['n'] >= 2:
            f['sire_prof_n']=tp['n'];f['sire_prof_mean']=tp['mean'];f['sire_prof_std']=tp['std']
            f['sire_prof_ratio']=tp['trans_ratio'];f['sire_prof_resid']=tp['residual']
        else:
            f['sire_prof_n']=0;f['sire_prof_mean']=0;f['sire_prof_std']=0;f['sire_prof_ratio']=1.0;f['sire_prof_resid']=0
        for ot in CORE_TRAITS:
            if ot == trait: continue
            otp = sp.get(ot)
            if otp and otp['n'] >= 2: f[f'sire_prof_{ot}'] = otp['residual']
    else:
        f['sire_prof_n']=0;f['sire_prof_mean']=0;f['sire_prof_std']=0;f['sire_prof_ratio']=1.0;f['sire_prof_resid']=0
    if mgs_naab and mgs_naab in mgs_profiles:
        mp2 = mgs_profiles[mgs_naab]; tp2 = mp2.get(trait)
        if tp2 and tp2['n'] >= 2:
            f['mgs_prof_n']=tp2['n'];f['mgs_prof_mean']=tp2['mean'];f['mgs_prof_resid']=tp2['residual']
        else:
            f['mgs_prof_n']=0;f['mgs_prof_mean']=0;f['mgs_prof_resid']=0
    else:
        f['mgs_prof_n']=0;f['mgs_prof_mean']=0;f['mgs_prof_resid']=0
    return f

def predict_batch(trait, animals, pl_preds=None):
    """Predict a trait for all animals at once (batch mode for speed)."""
    info = saved_models.get(trait)
    if not info: return [None] * len(animals)

    feat_rows = []
    valid_idx = []
    for i, (sire_ptas, mgs_ptas, sire_naab, mgs_naab, mmgs_ptas) in enumerate(animals):
        feat = build_features_v10(sire_ptas, mgs_ptas, trait, sire_naab, mgs_naab, mmgs_ptas)
        if feat is None:
            continue
        if info.get('v10b') == 'pl_anchored':
            pl_val = pl_preds[i] if pl_preds and pl_preds[i] is not None else sire_ptas.get('PL', 0) / 2
            feat['pl_pred'] = pl_val
            feat['pl_pred_sq'] = pl_val ** 2
        feat_rows.append(feat)
        valid_idx.append(i)

    if not feat_rows:
        return [None] * len(animals)

    feat_df = pd.DataFrame(feat_rows)
    for c in info['feature_cols']:
        if c not in feat_df.columns: feat_df[c] = 0
    feat_df = feat_df[info['feature_cols']]
    X = feat_df.values.astype(np.float64)

    if info['type'] == 'stack':
        base_preds = np.column_stack([info['base_models'][mn].predict(X) for mn in info['base_names']])
        preds_v10 = info['meta_model'].predict(base_preds)
    else:
        preds_v10 = info['model'].predict(X)

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
                pred_anc = sire_ptas.get(trait, 0) * 0.35 + mgs_val * 0.15
            blend = np.array([[preds_v10[j], pred_2stage, pred_anc]])
            final_preds[j] = info['blend_ridge'].predict(blend)[0]
        preds_v10 = final_preds

    out = [None] * len(animals)
    for j, vi in enumerate(valid_idx):
        out[vi] = round(float(preds_v10[j]), 2)
    return out

# ============================================================
# LOAD CLIENT FILE AND PREDICT
# ============================================================
print(f"\nCarregando {INPUT_FILE.name}...")
df = pd.read_excel(INPUT_FILE, engine='openpyxl')
cols = list(df.columns)
id_col, pai_col, avo_col, bis_col = cols[0], cols[1], cols[2], cols[3]
print(f"  Total animais: {len(df)}")

print(f"\n{'='*120}")
print(f"  DSII V10-B — PREDICAO CLIENTE JOSE GONCALVEZ (52 traits)")
print(f"{'='*120}")

# === FASE 1: VALIDACAO ===
print(f"\n  FASE 1: Validacao de linhagem")
print(f"  {'-'*80}")

# First pass: check all unique NAABs
all_naabs = set()
for _, row in df.iterrows():
    for c in [pai_col, avo_col, bis_col]:
        n = str(row[c]).strip()
        if n and n != 'nan': all_naabs.add(n)

print(f"  NAABs unicos no arquivo: {len(all_naabs)}")

found_ss, found_cdcb, not_found = 0, 0, 0
not_found_list = []
for naab in sorted(all_naabs):
    _, src, resolved, ptas = lookup_bull(naab)
    if src == 'SS': found_ss += 1
    elif src == 'CDCB': found_cdcb += 1
    else:
        not_found += 1
        not_found_list.append(naab)

print(f"  Touros encontrados no bulls.csv (SS): {found_ss}")
print(f"  Touros encontrados no CDCB: {found_cdcb}")
print(f"  Touros NAO encontrados: {not_found}")
if not_found_list:
    print(f"  NAABs nao encontrados: {', '.join(not_found_list[:20])}")
    if len(not_found_list) > 20:
        print(f"    ... e mais {len(not_found_list)-20}")

# === FASE 2: PREDICAO ===
print(f"\n  FASE 2: Predicao")
print(f"  {'-'*80}")

excluded = []
valid_animals = []  # list of (animal_meta, sire_ptas, mgs_ptas, sire_naab, mgs_naab, mmgs_ptas)

for _, row in df.iterrows():
    animal_id = row[id_col]
    pai_naab = str(row[pai_col]).strip()
    avo_naab = str(row[avo_col]).strip()
    bis_naab = str(row[bis_col]).strip()

    pai_row, pai_src, pai_resolved, pai_ptas = lookup_bull(pai_naab)
    avo_row, avo_src, avo_resolved, avo_ptas = lookup_bull(avo_naab)
    bis_row, bis_src, bis_resolved, bis_ptas = lookup_bull(bis_naab)

    missing = []
    if not pai_row: missing.append(f"Pai({pai_naab})")
    if not avo_row: missing.append(f"Avo({avo_naab})")
    if not bis_row: missing.append(f"Bis({bis_naab})")

    if missing:
        excluded.append({'ID': animal_id, 'Pai': pai_naab, 'Avo': avo_naab, 'Bis': bis_naab,
                         'motivo': ', '.join(missing)})
        continue

    if not pai_ptas:
        excluded.append({'ID': animal_id, 'Pai': pai_naab, 'Avo': avo_naab, 'Bis': bis_naab,
                         'motivo': 'Pai sem PTAs'})
        continue

    meta = {
        'ID': animal_id,
        'Pai_NAAB': pai_naab, 'Pai_Resolved': pai_resolved, 'Pai_Src': pai_src,
        'Avo_NAAB': avo_naab, 'Avo_Resolved': avo_resolved, 'Avo_Src': avo_src,
        'Bis_NAAB': bis_naab, 'Bis_Resolved': bis_resolved, 'Bis_Src': bis_src,
        'Pai_Traits': len(pai_ptas), 'Avo_Traits': len(avo_ptas), 'Bis_Traits': len(bis_ptas),
    }
    valid_animals.append((meta, pai_ptas, avo_ptas, pai_resolved, avo_resolved, bis_ptas))

# Batch prediction: one trait at a time across all animals
animals_input = [(a[1], a[2], a[3], a[4], a[5]) for a in valid_animals]

# PL first (needed for LIV anchoring)
import time
t0 = time.time()
pl_preds = predict_batch('PL', animals_input)
print(f"    PL predito em {time.time()-t0:.1f}s")

# All other traits in batch
results = [{**a[0]} for a in valid_animals]
for trait in ALL_TRAITS:
    t0 = time.time()
    preds = predict_batch(trait, animals_input, pl_preds)
    for i, p in enumerate(preds):
        if p is not None:
            results[i][trait] = p
    elapsed = time.time() - t0
    n_valid = sum(1 for p in preds if p is not None)
    print(f"    {trait:>6}: {n_valid:>4} predicoes em {elapsed:.2f}s")
    sys.stdout.flush()

print(f"  Animais VALIDOS: {len(results)}/{len(df)} ({100*len(results)/len(df):.1f}%)")
print(f"  Animais EXCLUIDOS: {len(excluded)}/{len(df)}")

if results:
    ss_pai = sum(1 for r in results if r['Pai_Src'] == 'SS')
    cdcb_pai = sum(1 for r in results if r['Pai_Src'] == 'CDCB')
    print(f"\n  Fontes dos Pais:")
    print(f"    bulls.csv (provas atualizadas): {ss_pai}")
    print(f"    CDCB: {cdcb_pai}")

if excluded:
    motivos = Counter(e['motivo'].split(',')[0].strip() for e in excluded)
    print(f"\n  Motivos de exclusao:")
    for m, c in motivos.most_common():
        print(f"    {m}: {c}")

# === RESULTADOS ===
if results:
    res_df = pd.DataFrame(results)
    trait_cols = [t for t in ALL_TRAITS if t in res_df.columns]

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

    print(f"\n  FASE 3: Estatisticas das predicoes (N={len(results)})")
    for cat_name, cat_traits in CATEGORIES.items():
        available = [t for t in cat_traits if t in trait_cols]
        if not available: continue
        print(f"\n  --- {cat_name} ---")
        print(f"  {'Trait':>6} | {'N':>4} | {'Media':>8} | {'Min':>8} | {'Max':>8} | {'SD':>8}")
        print(f"  {'-'*60}")
        for t in available:
            vals = res_df[t].dropna()
            if len(vals) > 0:
                print(f"  {t:>6} | {len(vals):>4} | {vals.mean():>8.2f} | {vals.min():>8.2f} | {vals.max():>8.2f} | {vals.std():>8.2f}")

    # Save — order columns by category
    meta_cols = ['ID','Pai_NAAB','Pai_Resolved','Pai_Src','Avo_NAAB','Avo_Resolved','Avo_Src',
                 'Bis_NAAB','Bis_Resolved','Bis_Src','Pai_Traits','Avo_Traits','Bis_Traits']
    ordered_traits = []
    for cat_traits in CATEGORIES.values():
        for t in cat_traits:
            if t in res_df.columns: ordered_traits.append(t)
    for t in trait_cols:
        if t not in ordered_traits: ordered_traits.append(t)

    final_cols = [c for c in meta_cols if c in res_df.columns] + ordered_traits
    res_df = res_df[final_cols]
    res_df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n  Arquivo salvo: {OUTPUT_FILE}")
    print(f"  Total traits: {len(ordered_traits)}")

    if excluded:
        exc_df = pd.DataFrame(excluded)
        exc_df.to_excel(EXCLUDED_FILE, index=False)
        print(f"  Excluidos: {EXCLUDED_FILE}")

print(f"\n  Predicao Jose Goncalvez concluida!")
