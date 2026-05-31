"""
DSII V12b — BTB-Only Training + Daughter Calibration
Variant of V12 for comparison:
  - Phase 1: Train ML ONLY on 198k Bull-to-Bull records (Sire Stack)
  - Phase 2: Enhanced Sire Transmission Profiles from 250k bulls
  - Phase 3: Calibrate on 7.7k daughter records (independent validation)
  - Phase 4: Adaptive shrinkage per trait
  - Incremental save after each trait
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, csv, pickle, sys, re, glob, time
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingClassifier, RandomForestClassifier,
                               HistGradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor

DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")
OUTPUT_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v12b_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def sf(v):
    if v is None: return None
    try:
        x = float(v)
        return x if not np.isnan(x) else None
    except: return None

def flush_print(*args, **kwargs):
    print(*args, **kwargs); sys.stdout.flush()

# ============================================================
# NAAB NORMALIZATION
# ============================================================
STUD_MAP = {'507':'7','509':'9','550':'50','551':'51','559':'59','518':'18','521':'21',
            '523':'23','581':'81','585':'85','501':'1','601':'1','604':'4','614':'14',
            '629':'29','751':'151','752':'152','814':'14','250':'200',
            '001':'1','007':'7','009':'9','011':'11','029':'29','097':'97',
            '100':'100','200':'200','777':'777','796':'796','745':'745'}

def normalize_naab(naab):
    if not naab or str(naab).strip() in ('','nan','None'): return []
    naab = str(naab).strip()
    m = re.match(r'^(\d+)(HO|BS|H|B)0*(\d+)$', naab)
    if not m: return [naab]
    org, breed_raw, num = m.group(1), m.group(2), m.group(3)
    if breed_raw in ('H', 'HO'): breeds = ['HO', 'H']
    elif breed_raw in ('B', 'BS'): breeds = ['BS', 'B']
    else: breeds = [breed_raw]
    orgs = [org]
    if org.startswith('5'): orgs.append(org[1:])
    if org in STUD_MAP: orgs.append(STUD_MAP[org])
    orgs.append(org.lstrip('0') or org)
    nums = [num]
    num_padded = num.zfill(5)
    if num_padded != num: nums.append(num_padded)
    return list(dict.fromkeys(f'{o}{b}{n}' for o in orgs for b in breeds for n in nums))

# ============================================================
# DOMAIN KNOWLEDGE
# ============================================================
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
BANCO_COL = {
    'TPI':'TPI','MILK':'MILK','FAT':'FAT','FAT%':'FAT %','PRO':'PROT','PRO%':'PROT%','CFP':None,
    'PL':'PL','LIV':'LIV','H_LIV':None,
    'SCS':'SCS','MAST':'CDCB_MAST','MET':None,'RP':None,'DA':None,'KET':None,'MF':None,
    'DPR':'DPR','HCR':None,'CCR':'CCR','SCE':None,'DCE':None,'SSB':None,'DSB':None,'GL':None,'EFC':None,
    'FI':'FI','F_SAV':None,'RFI':None,'GFI':None,
    'PTAT':'TYPE FS','UDC':'UDC','FLC':'FLC',
    'STA':'ST','STR':'SG','DFM':None,'BOD':None,'RUA':None,'RW':None,
    'RLS':None,'RLR':None,'FTA':None,'FLS':None,'FTL':None,
    'FUA':None,'RUH':None,'RUW':None,'UCL':'UC','UDP':'UD','FTP':None,'RTP':None,'TL':'TL',
    'BWC':None,'NM$':None,'CM$':None,
}
MAY_COL = {
    'TPI':'GTPI','MILK':'MILK','FAT':'FAT','FAT%':'%F','PRO':'PRO','PRO%':'%P','CFP':'CFP',
    'PL':'PL','LIV':'LIV','H_LIV':'HLIV',
    'SCS':'SCS','MAST':'MAST','MET':None,'RP':None,'DA':None,'KET':None,'MF':None,
    'DPR':'DPR','HCR':'HCR','CCR':'CCR','SCE':'SCE','DCE':'DCE','SSB':'SSB','DSB':'DSB','GL':'GL','EFC':'EFC',
    'FI':'FI','F_SAV':'FSAV','RFI':None,'GFI':'GFI',
    'PTAT':'PTAT','UDC':'UDC','FLC':'FLC',
    'STA':None,'STR':None,'DFM':None,'BOD':None,'RUA':None,'RW':None,
    'RLS':None,'RLR':None,'FTA':None,'FLS':None,'FTL':None,
    'FUA':None,'RUH':None,'RUW':None,'UCL':None,'UDP':None,'FTP':None,'RTP':None,'TL':None,
    'BWC':None,'NM$':'NM$','CM$':'CM$',
}
DAM_COL = {
    'TPI':'TPI','MILK':'PTA Milk','FAT':'PTA Fat','FAT%':'% Fat','PRO':'PTA Pro','PRO%':'% Pro','CFP':'CFP',
    'PL':'PL','LIV':'PTA LIV','H_LIV':'Heifer Livability',
    'SCS':'SCS','MAST':'Mastitis','MET':'Metritis','RP':'Retained Placenta',
    'DA':'Displaced Abomasum','KET':'Ketosis','MF':'Milk Fever',
    'DPR':'PTA DPR','HCR':'HCR','CCR':'CCR','SCE':'SCE','DCE':'DCE','SSB':'SSB','DSB':'DSB',
    'GL':'PTA GL','EFC':'Early First Calving',
    'FI':None,'F_SAV':'Feed Saved','RFI':'RFI','GFI':None,
    'PTAT':'PTA Type','UDC':'UDC','FLC':'FLC',
    'STA':'STA','STR':'STR','DFM':'DF','BOD':'BD','RUA':'RA','RW':'TW',
    'RLS':'RLS','RLR':'RLR','FTA':'FA','FLS':'FLS','FTL':None,
    'FUA':'FUA','RUH':'RUH','RUW':'RUW','UCL':'UC','UDP':'UD','FTP':'FTP','RTP':'RTP','TL':'TL',
    'BWC':None,'NM$':'Net Merit','CM$':'CM$',
}

ALL_TRAITS = list(BULL_COL.keys())
CORE_TRAITS = ['TPI','MILK','FAT','FAT%','PRO','PRO%','PL','SCS','DPR','LIV','FI','CCR',
               'PTAT','UDC','FLC','MAST','NM$','HCR','SCE','F_SAV']

REL_COL = {
    'TPI': 'Rel', 'NM$': 'NM$ Rel', 'CM$': 'NM$ Rel',
    'PL': 'PL Rel', 'DPR': 'DPR Rel', 'CCR': 'CCR Rel', 'HCR': 'HCR Rel',
    'PTAT': 'PTAT Rel', 'SCS': 'SCS Rel', 'SCE': 'SCE Rel',
    'LIV': 'LIV Rel', 'MAST': 'MAST Rel',
}

# ============================================================
# LOAD BULL DATABASE
# ============================================================
flush_print("=" * 95)
flush_print("  DSII V12b — BTB-Only Training + Daughter Calibration")
flush_print("=" * 95)

flush_print("\n  Loading bull database...")

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

bulls = {}
name_to_naab = {}
with open(BULLS_CSV, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        row = _normalize_bull_row(row)
        naab = row.get('NAAB','').strip()
        if naab:
            bulls[naab] = row
            name = row.get('Registration Name','').strip().upper()
            if name: name_to_naab[name] = naab
            name2 = row.get('Name','').strip().upper()
            if name2: name_to_naab[name2] = naab
flush_print(f"    bulls.csv: {len(bulls)} bulls, {len(name_to_naab)} name mappings")

cdcb_bulls = {}
try:
    with open(DOWNLOADS / "Bull_Report (1).csv", 'r', encoding='latin-1') as f:
        for row in csv.DictReader(f):
            naab = row.get('NAAB_CODE','').strip()
            if naab: cdcb_bulls[naab] = row
    flush_print(f"    CDCB: {len(cdcb_bulls)} bulls")
except:
    flush_print(f"    CDCB: not available")

def lookup_bull(naab):
    for c in normalize_naab(naab):
        if c in bulls: return bulls[c], 'SS'
        if c in cdcb_bulls: return cdcb_bulls[c], 'CDCB'
    return None, None

def lookup_bull_by_name(name):
    if not name or str(name).strip() in ('','nan','None'): return None, None
    key = str(name).strip().upper()
    if key in name_to_naab:
        naab = name_to_naab[key]
        return bulls.get(naab), 'SS'
    return None, None

def get_bull_ptas(bull_row, source='SS'):
    ptas = {}
    col_map = BULL_COL if source == 'SS' else CDCB_COL
    for trait, col in col_map.items():
        if col is None: continue
        v = sf(bull_row.get(col))
        if v is not None: ptas[trait] = v
    return ptas

def get_bull_dob_year(bull_row):
    dob = bull_row.get('DOB_RAW', '')
    if not dob: return None
    try:
        dob = str(dob).strip()
        if len(dob) >= 4:
            yr = int(dob[:4])
            if 1960 <= yr <= 2025: return yr
    except: pass
    return None

# ============================================================
# PHASE 1A: BTB TRAINING DATA (198k records) — TRAIN SET
# ============================================================
flush_print("\n  PHASE 1A: Building BTB training data (TRAIN set)...")

btb_records = []
btb_sire_obs = defaultdict(list)
btb_mgs_obs = defaultdict(list)

for naab, row in bulls.items():
    sire_stack = row.get('Sire Stack', '').strip()
    if not sire_stack or sire_stack in ('nan', 'None', ''): continue
    parts = sire_stack.split(' x ')
    if len(parts) < 2: continue

    sire_name = parts[0].strip().upper()
    mgs_name = parts[1].strip().upper()

    sire_row_l, sire_src_l = lookup_bull_by_name(sire_name)
    if not sire_row_l: continue
    sire_ptas_l = get_bull_ptas(sire_row_l, sire_src_l)
    if not sire_ptas_l: continue
    sire_naab_l = sire_row_l.get('NAAB', '')

    mgs_row_l, mgs_src_l = lookup_bull_by_name(mgs_name)
    if not mgs_row_l: continue
    mgs_ptas_l = get_bull_ptas(mgs_row_l, mgs_src_l)
    if not mgs_ptas_l: continue
    mgs_naab_l = mgs_row_l.get('NAAB', '')

    son_ptas = get_bull_ptas(row, 'SS')
    if not son_ptas: continue

    son_rel = sf(row.get('Rel', ''))
    if son_rel is not None and son_rel <= 75: continue

    # Filter by birth year > 2005
    dob_year = get_bull_dob_year(row)
    if dob_year is not None and dob_year <= 2005: continue

    mmgs_ptas_l = {}
    if len(parts) >= 3:
        mmgs_name = parts[2].strip().upper()
        mmgs_row_l, _ = lookup_bull_by_name(mmgs_name)
        if mmgs_row_l: mmgs_ptas_l = get_bull_ptas(mmgs_row_l, 'SS')

    for t, v in son_ptas.items():
        btb_sire_obs[sire_naab_l].append((t, v))
        btb_mgs_obs[mgs_naab_l].append((t, v))

    yr = dob_year
    w = 0.5 + 0.5 * min((yr - 1990) / 30, 1.0) if yr and yr >= 1990 else 0.5
    rel = son_rel if son_rel else 70
    w *= (rel / 100.0)

    btb_records.append({
        'source': 'btb', 'sire_naab': sire_naab_l,
        'mgs_naab': mgs_naab_l, 'sire_ptas': sire_ptas_l,
        'mgs_ptas': mgs_ptas_l, 'mmgs_ptas': mmgs_ptas_l,
        'son_ptas': son_ptas, 'weight': w,
    })

flush_print(f"    BTB records (Rel>75%, DOB>2005): {len(btb_records)}")

# ============================================================
# PHASE 1B: DAUGHTER DATA — CALIBRATION SET (held out)
# ============================================================
flush_print("\n  PHASE 1B: Loading daughter data (CALIBRATION set)...")

daughter_records = []
dau_sire_obs = defaultdict(list)
dau_mgs_obs = defaultdict(list)

# banco21052026
raw = pd.read_excel(DOWNLOADS / "banco21052026.xlsx", header=None, engine='openpyxl')
hdr = [str(c).strip() if c else f'col_{i}' for i, c in enumerate(raw.iloc[10])]
banco = raw.iloc[11:].copy(); banco.columns = hdr; banco = banco.reset_index(drop=True)
flush_print(f"    banco21052026: {len(banco)} daughters")

banco_ok = 0
for _, row in banco.iterrows():
    sire_naab = str(row.get('Sire of Record NAAB', '')).strip()
    mgs_naab = str(row.get('Maternal Grandsire NAAB', '')).strip()
    if sire_naab in ('', 'nan', 'None'): continue
    sire_row, sire_src = lookup_bull(sire_naab)
    if not sire_row: continue
    sire_ptas = get_bull_ptas(sire_row, sire_src)
    if not sire_ptas: continue
    mgs_ptas = {}
    if mgs_naab not in ('', 'nan', 'None'):
        mgs_row, mgs_src = lookup_bull(mgs_naab)
        if mgs_row: mgs_ptas = get_bull_ptas(mgs_row, mgs_src)
    daughter_ptas = {}
    for trait, bcol in BANCO_COL.items():
        if bcol and bcol in row.index:
            v = sf(row[bcol])
            if v is not None: daughter_ptas[trait] = v
    if not daughter_ptas: continue
    banco_ok += 1
    for t, v in daughter_ptas.items():
        dau_sire_obs[sire_naab].append((t, v))
        if mgs_ptas: dau_mgs_obs[mgs_naab].append((t, v))
    daughter_records.append({
        'source': 'banco', 'sire_naab': sire_naab,
        'mgs_naab': mgs_naab if mgs_ptas else None,
        'sire_ptas': sire_ptas, 'mgs_ptas': mgs_ptas,
        'daughter_ptas': daughter_ptas,
    })
flush_print(f"    banco: {banco_ok} usable")

# May 2026
may_df = pd.read_excel(DOWNLOADS / "May 2026.xlsx", engine='openpyxl')
may_ok = 0
for _, row in may_df.iterrows():
    sire_name = str(row.get('SIRENAME', '')).strip()
    sire_row, sire_src = lookup_bull_by_name(sire_name)
    if not sire_row:
        sire_reg = str(row.get('SIREREGNUM2', '')).strip()
        sire_row, sire_src = lookup_bull(sire_reg)
    if not sire_row: continue
    sire_ptas = get_bull_ptas(sire_row, sire_src)
    if not sire_ptas: continue
    sire_naab = sire_row.get('NAAB', '') if sire_src == 'SS' else ''
    mgs_name = str(row.get('MGSNAME2', '')).strip()
    mgs_ptas = {}; mgs_naab = ''
    if mgs_name not in ('', 'nan', 'None'):
        mgs_row, mgs_src = lookup_bull_by_name(mgs_name)
        if mgs_row:
            mgs_ptas = get_bull_ptas(mgs_row, mgs_src)
            mgs_naab = mgs_row.get('NAAB', '') if mgs_src == 'SS' else ''
    daughter_ptas = {}
    for trait, mcol in MAY_COL.items():
        if mcol and mcol in row.index:
            v = sf(row[mcol])
            if v is not None: daughter_ptas[trait] = v
    if not daughter_ptas: continue
    may_ok += 1
    if sire_naab:
        for t, v in daughter_ptas.items(): dau_sire_obs[sire_naab].append((t, v))
    if mgs_naab and mgs_ptas:
        for t, v in daughter_ptas.items(): dau_mgs_obs[mgs_naab].append((t, v))
    daughter_records.append({
        'source': 'may2026', 'sire_naab': sire_naab,
        'mgs_naab': mgs_naab if mgs_ptas else None,
        'sire_ptas': sire_ptas, 'mgs_ptas': mgs_ptas,
        'daughter_ptas': daughter_ptas,
    })
flush_print(f"    May 2026: {may_ok} usable")

# DAM1-20
dam_frames = []
for fp in sorted(glob.glob(str(DOWNLOADS / "DAM*.xlsx"))):
    dam_frames.append(pd.read_excel(fp, engine='openpyxl'))
dam_df = pd.concat(dam_frames, ignore_index=True) if dam_frames else pd.DataFrame()
dam_ok = 0
for _, row in dam_df.iterrows():
    sire_name = str(row.get('Sire', '')).strip()
    mgs_name = str(row.get('MGS', '')).strip()
    mggs_name = str(row.get('MGGS', '')).strip()
    sire_row_d, sire_src_d = lookup_bull_by_name(sire_name)
    if not sire_row_d: continue
    sire_ptas_d = get_bull_ptas(sire_row_d, sire_src_d)
    if not sire_ptas_d: continue
    sire_naab_d = sire_row_d.get('NAAB', '') if sire_src_d == 'SS' else ''
    mgs_ptas_d = {}; mgs_naab_d = ''
    if mgs_name not in ('', 'nan', 'None'):
        mgs_row_d, mgs_src_d = lookup_bull_by_name(mgs_name)
        if mgs_row_d:
            mgs_ptas_d = get_bull_ptas(mgs_row_d, mgs_src_d)
            mgs_naab_d = mgs_row_d.get('NAAB', '') if mgs_src_d == 'SS' else ''
    mmgs_ptas_d = {}
    if mggs_name not in ('', 'nan', 'None'):
        mmgs_row_d, _ = lookup_bull_by_name(mggs_name)
        if mmgs_row_d: mmgs_ptas_d = get_bull_ptas(mmgs_row_d, 'SS')
    daughter_ptas_d = {}
    for trait, dcol in DAM_COL.items():
        if dcol and dcol in row.index:
            v = sf(row[dcol])
            if v is not None: daughter_ptas_d[trait] = v
    if not daughter_ptas_d: continue
    dam_ok += 1
    if sire_naab_d:
        for t, v in daughter_ptas_d.items(): dau_sire_obs[sire_naab_d].append((t, v))
    if mgs_naab_d and mgs_ptas_d:
        for t, v in daughter_ptas_d.items(): dau_mgs_obs[mgs_naab_d].append((t, v))
    daughter_records.append({
        'source': 'dam', 'sire_naab': sire_naab_d,
        'mgs_naab': mgs_naab_d if mgs_ptas_d else None,
        'sire_ptas': sire_ptas_d, 'mgs_ptas': mgs_ptas_d,
        'mmgs_ptas': mmgs_ptas_d, 'daughter_ptas': daughter_ptas_d,
    })
flush_print(f"    DAM1-20: {dam_ok} usable")
flush_print(f"    TOTAL daughter records (calibration): {len(daughter_records)}")
flush_print(f"    TOTAL BTB records (training): {len(btb_records)}")

# ============================================================
# PHASE 2: TRANSMISSION PROFILES (BTB + Daughters combined)
# ============================================================
flush_print("\n  PHASE 2: Computing transmission profiles...")

combined_sire_obs = defaultdict(list)
combined_mgs_obs = defaultdict(list)
for k, v in btb_sire_obs.items(): combined_sire_obs[k].extend(v)
for k, v in dau_sire_obs.items(): combined_sire_obs[k].extend(v)
for k, v in btb_mgs_obs.items(): combined_mgs_obs[k].extend(v)
for k, v in dau_mgs_obs.items(): combined_mgs_obs[k].extend(v)

sire_profiles = {}
for sire_naab, obs in combined_sire_obs.items():
    sire_row, src = lookup_bull(sire_naab)
    if not sire_row: continue
    sire_ptas = get_bull_ptas(sire_row, src)
    profile = {}
    trait_vals = defaultdict(list)
    for t, v in obs: trait_vals[t].append(v)
    for t, vals in trait_vals.items():
        if len(vals) >= 2 and t in sire_ptas:
            sp = sire_ptas[t]
            mean_d = np.mean(vals)
            profile[t] = {
                'n': len(vals), 'mean': mean_d, 'std': np.std(vals),
                'trans_ratio': mean_d / (sp / 2) if sp != 0 else 1.0,
                'residual': mean_d - sp / 2,
            }
    if profile: sire_profiles[sire_naab] = profile

mgs_profiles = {}
for mgs_naab, obs in combined_mgs_obs.items():
    mgs_row, src = lookup_bull(mgs_naab)
    if not mgs_row: continue
    mgs_ptas_l = get_bull_ptas(mgs_row, src)
    profile = {}
    trait_vals = defaultdict(list)
    for t, v in obs: trait_vals[t].append(v)
    for t, vals in trait_vals.items():
        if len(vals) >= 2 and t in mgs_ptas_l:
            mean_d = np.mean(vals)
            profile[t] = {
                'n': len(vals), 'mean': mean_d, 'std': np.std(vals),
                'residual': mean_d - mgs_ptas_l[t] / 4,
            }
    if profile: mgs_profiles[mgs_naab] = profile

flush_print(f"    Sire profiles: {len(sire_profiles)}")
flush_print(f"    MGS profiles: {len(mgs_profiles)}")

# ============================================================
# FEATURES
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

def build_features(sire_ptas, mgs_ptas, trait, sire_naab=None, mgs_naab=None, mmgs_ptas=None):
    s = sire_ptas.get(trait)
    if s is None: return None
    f = {}
    sd = GENETIC_SD.get(trait, 1)
    h2 = HERITABILITY.get(trait, 0.15)
    pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)
    f['pa'] = pa; f['pa_z'] = pa / sd if sd else 0
    f['sire'] = s; f['sire_z'] = s / sd if sd else 0
    f['sire_sq'] = s * s; f['sire_h2'] = s * h2
    f['sire_dev_from_pa'] = s - pa * 2
    mg = mgs_ptas.get(trait) if mgs_ptas else None
    f['mgs_val'] = mg if mg is not None else 0
    f['mgs_z'] = (mg / sd if sd else 0) if mg is not None else 0
    f['mgs_sq'] = mg * mg if mg is not None else 0
    f['has_mgs'] = 1 if mg is not None else 0
    if mg is not None:
        f['sire_x_mgs'] = s * mg; f['sire_mgs_diff'] = s - mg
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
        sp = sire_profiles[sire_naab]; tp = sp.get(trait)
        if tp and tp['n'] >= 2:
            f['sire_prof_n'] = tp['n']; f['sire_prof_mean'] = tp['mean']
            f['sire_prof_std'] = tp['std']; f['sire_prof_ratio'] = tp['trans_ratio']
            f['sire_prof_resid'] = tp['residual']; f['sire_prof_dev_pa'] = tp['mean'] - pa
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
        mp = mgs_profiles[mgs_naab]; tp = mp.get(trait)
        if tp and tp['n'] >= 2:
            f['mgs_prof_n'] = tp['n']; f['mgs_prof_mean'] = tp['mean']; f['mgs_prof_resid'] = tp['residual']
        else:
            f['mgs_prof_n'] = 0; f['mgs_prof_mean'] = 0; f['mgs_prof_resid'] = 0
    else:
        f['mgs_prof_n'] = 0; f['mgs_prof_mean'] = 0; f['mgs_prof_resid'] = 0
    return f

# ============================================================
# MODELS
# ============================================================
def get_models(h2=0.25):
    reg_factor = max(0.5, 1.0 - h2)
    n_est = 200
    return {
        'LGBM': LGBMRegressor(n_estimators=n_est, max_depth=5, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.6,
                               min_child_samples=max(5, int(10 * reg_factor)),
                               reg_alpha=0.5 * reg_factor, reg_lambda=2.0 * reg_factor,
                               random_state=42, verbose=-1, n_jobs=-1),
        'XGB': XGBRegressor(n_estimators=n_est, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.6,
                             reg_alpha=0.5 * reg_factor, reg_lambda=2.0 * reg_factor,
                             min_child_weight=max(5, int(10 * reg_factor)),
                             random_state=42, verbosity=0, n_jobs=-1),
    }

# ============================================================
# PHASE 3: TRAIN ON BTB, CALIBRATE ON DAUGHTERS
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  PHASE 3: TRAIN on BTB ({len(btb_records)}), CALIBRATE on Daughters ({len(daughter_records)})")
flush_print(f"{'='*95}")

flush_print(f"\n  {'Trait':>6} | {'N_BTB':>6} | {'N_Cal':>5} | {'Best':>5} | {'R2_btb':>7} | "
            f"{'R2_cal':>7} | {'PA_cal':>6} | {'b0':>7} | {'b1':>5} | {'Lam':>5} | {'Final':>7} | Win")
flush_print(f"  {'-'*105}")

# === RESUME: load previously saved models if available ===
saved_models = {}
all_results = []
calibration_params = {}
_resumed_traits = set()

if (OUTPUT_DIR / "v12b_models.pkl").exists():
    try:
        with open(OUTPUT_DIR / "v12b_models.pkl", 'rb') as f:
            saved_models = pickle.load(f)
        _resumed_traits = set(saved_models.keys())
        flush_print(f"\n  RESUME: {len(_resumed_traits)} traits already trained: {list(_resumed_traits)}")
    except: pass
if (OUTPUT_DIR / "v12b_calibration.pkl").exists():
    try:
        with open(OUTPUT_DIR / "v12b_calibration.pkl", 'rb') as f:
            calibration_params = pickle.load(f)
    except: pass
if (OUTPUT_DIR / "v12b_training_results.csv").exists():
    try:
        _prev = pd.read_csv(OUTPUT_DIR / "v12b_training_results.csv")
        all_results = _prev.to_dict('records')
    except: pass

for trait in ALL_TRAITS:
    if trait == 'FI':
        flush_print(f"  {trait:>6} | SKIP (calculated from formula: 0.4*DPR + 0.4*CCR + 0.1*HCR + 0.1*EFC)")
        continue
    if trait in _resumed_traits:
        flush_print(f"  {trait:>6} | SKIP (already trained)")
        continue
    t0 = time.time()
    h2 = HERITABILITY.get(trait, 0.15)

    # === BTB TRAINING DATA ===
    btb_feat, btb_y, btb_pa, btb_grp, btb_w = [], [], [], [], []
    for rec in btb_records:
        dv = rec['son_ptas'].get(trait)
        if dv is None: continue
        feat = build_features(rec['sire_ptas'], rec['mgs_ptas'], trait,
                               rec['sire_naab'], rec.get('mgs_naab'), rec.get('mmgs_ptas'))
        if feat is None: continue
        pa = compute_pa(rec['sire_ptas'], rec['mgs_ptas'], rec.get('mmgs_ptas'), trait)
        btb_feat.append(feat); btb_y.append(dv)
        btb_pa.append(pa if pa is not None else 0)
        btb_grp.append(rec['sire_naab']); btb_w.append(rec['weight'])

    # === DAUGHTER CALIBRATION DATA ===
    cal_feat, cal_y, cal_pa, cal_grp = [], [], [], []
    for rec in daughter_records:
        dv = rec['daughter_ptas'].get(trait)
        if dv is None: continue
        feat = build_features(rec['sire_ptas'], rec['mgs_ptas'], trait,
                               rec['sire_naab'], rec.get('mgs_naab'), rec.get('mmgs_ptas'))
        if feat is None: continue
        pa = compute_pa(rec['sire_ptas'], rec['mgs_ptas'], rec.get('mmgs_ptas'), trait)
        cal_feat.append(feat); cal_y.append(dv)
        cal_pa.append(pa if pa is not None else 0); cal_grp.append(rec['sire_naab'])

    n_btb = len(btb_feat)
    n_cal = len(cal_feat)

    if n_btb < 50:
        flush_print(f"  {trait:>6} | SKIP (N_BTB={n_btb})")
        continue

    # === BUILD TRAINING MATRICES (BTB only) ===
    feat_df = pd.DataFrame(btb_feat)
    feat_df = feat_df.dropna(axis=1, thresh=int(len(feat_df) * 0.3))
    feature_cols = list(feat_df.columns)
    for c in feature_cols:
        if feat_df[c].isna().any(): feat_df[c] = feat_df[c].fillna(feat_df[c].median())
    X_train = feat_df.values.astype(np.float64)
    y_train = np.array(btb_y, dtype=np.float64)
    pa_train = np.array(btb_pa, dtype=np.float64)
    w_train = np.array(btb_w, dtype=np.float64)
    grp_train = np.array(btb_grp)
    y_dev_train = y_train - pa_train

    # GroupKFold on BTB for model selection
    unique_sires = np.unique(grp_train)
    if len(unique_sires) >= 3:
        gkf = GroupKFold(n_splits=min(3, len(unique_sires)))
        cv_splits = list(gkf.split(X_train, y_dev_train, grp_train))
    else:
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_splits = list(kf.split(X_train))

    # === TRAIN MODELS (BTB only) ===
    models = get_models(h2)
    model_scores = {}
    model_preds_oof = {}

    for mn, model in models.items():
        try:
            preds = np.zeros(len(y_dev_train))
            r2s = []
            for tr, te in cv_splits:
                m = get_models(h2)[mn]
                if mn in ('LGBM', 'XGB'):
                    m.fit(X_train[tr], y_dev_train[tr], sample_weight=w_train[tr])
                else:
                    m.fit(X_train[tr], y_dev_train[tr])
                pred = m.predict(X_train[te])
                preds[te] = pred
                r2s.append(r2_score(y_dev_train[te], pred))
            model_scores[mn] = np.mean(r2s)
            model_preds_oof[mn] = preds
        except: pass

    if not model_scores: continue

    best_name = max(model_scores, key=lambda k: model_scores[k])
    best_r2_btb = model_scores[best_name]

    # Stack top 3
    sorted_m = sorted(model_scores.items(), key=lambda x: -x[1])
    top3 = [m[0] for m in sorted_m[:3]]
    sX = np.column_stack([model_preds_oof[m] for m in top3])
    stack_preds = np.zeros(len(y_dev_train))
    for tr, te in cv_splits:
        meta = Ridge(alpha=1.0); meta.fit(sX[tr], y_dev_train[tr])
        stack_preds[te] = meta.predict(sX[te])
    stack_r2 = r2_score(y_dev_train, stack_preds)

    if stack_r2 > best_r2_btb:
        winner = 'STACK'; r2_btb_final = stack_r2
    else:
        winner = best_name; r2_btb_final = best_r2_btb

    # === TRAIN FINAL MODELS ON ALL BTB DATA ===
    if winner == 'STACK':
        base_models = {}
        for mn in top3:
            m = get_models(h2)[mn]
            if mn in ('LGBM', 'XGB'):
                m.fit(X_train, y_dev_train, sample_weight=w_train)
            else:
                m.fit(X_train, y_dev_train)
            base_models[mn] = m
        base_preds_all = np.column_stack([model_preds_oof[m] for m in top3])
        meta_model = Ridge(alpha=1.0); meta_model.fit(base_preds_all, y_dev_train)
    else:
        final_model = get_models(h2)[winner]
        if winner in ('LGBM', 'XGB', 'HGBR'):
            final_model.fit(X_train, y_dev_train, sample_weight=w_train)
        else:
            final_model.fit(X_train, y_dev_train)

    # === PHASE 3B: CALIBRATE ON DAUGHTERS (independent) ===
    b0, b1, best_lambda = 0.0, 1.0, 1.0
    r2_cal = -999; pa_r2_cal = -999

    if n_cal >= 20:
        # Build calibration features using SAME feature_cols
        cal_df = pd.DataFrame(cal_feat)
        for c in feature_cols:
            if c not in cal_df.columns: cal_df[c] = 0
        cal_df = cal_df[feature_cols].fillna(0)
        X_cal = cal_df.values.astype(np.float64)
        y_cal = np.array(cal_y, dtype=np.float64)
        pa_cal = np.array(cal_pa, dtype=np.float64)

        # Predict on daughters using BTB-trained model
        if winner == 'STACK':
            bp = np.column_stack([base_models[mn].predict(X_cal) for mn in top3])
            raw_dev_cal = meta_model.predict(bp)
        else:
            raw_dev_cal = final_model.predict(X_cal)

        # Linear calibration: actual_dev = b0 + b1 * predicted_dev
        y_dev_cal = y_cal - pa_cal
        mask = np.isfinite(raw_dev_cal) & np.isfinite(y_dev_cal)
        if mask.sum() > 10:
            cal_reg = LinearRegression()
            cal_reg.fit(raw_dev_cal[mask].reshape(-1, 1), y_dev_cal[mask])
            b0 = cal_reg.intercept_
            b1 = cal_reg.coef_[0]

        # Apply calibration
        if abs(b1) > 0.01:
            cal_dev = (raw_dev_cal - b0) / b1
        else:
            cal_dev = raw_dev_cal

        # Optimal shrinkage on daughters
        best_lambda = 1.0; best_r2_lam = -999
        for lam in np.arange(0.0, 1.51, 0.05):
            pred = pa_cal + lam * cal_dev
            r2t = r2_score(y_cal, pred)
            if r2t > best_r2_lam:
                best_r2_lam = r2t; best_lambda = lam
        # Fine-tune
        for lam in np.arange(max(0, best_lambda - 0.1), min(1.5, best_lambda + 0.1), 0.01):
            pred = pa_cal + lam * cal_dev
            r2t = r2_score(y_cal, pred)
            if r2t > best_r2_lam:
                best_r2_lam = r2t; best_lambda = round(lam, 2)

        r2_cal = best_r2_lam
        pa_r2_cal = r2_score(y_cal, pa_cal)
    else:
        r2_cal = -999; pa_r2_cal = -999

    flush_print(f"  {trait:>6} | {n_btb:>6} | {n_cal:>5} | {best_name:>5} | {r2_btb_final:>7.4f} | "
                f"{r2_cal:>7.4f} | {pa_r2_cal:>6.4f} | {b0:>+7.3f} | {b1:>5.3f} | {best_lambda:>5.2f} | "
                f"{r2_cal:>7.4f} | {winner}")

    # === SAVE ===
    calibration_params[trait] = {
        'b0': round(b0, 6), 'b1': round(b1, 6), 'lambda': best_lambda,
        'pa_r2_cal': round(pa_r2_cal, 4), 'r2_cal': round(r2_cal, 4),
    }

    model_info = {
        'feature_cols': feature_cols,
        'r2_btb': round(r2_btb_final, 4),
        'r2_cal': round(r2_cal, 4),
        'pa_r2_cal': round(pa_r2_cal, 4),
        'target': 'deviation',
        'calibration': {'b0': b0, 'b1': b1},
        'shrinkage_lambda': best_lambda,
    }

    if winner == 'STACK':
        model_info.update({
            'type': 'stack', 'base_models': base_models, 'base_names': top3,
            'meta_model': meta_model,
        })
    else:
        model_info.update({
            'type': 'single', 'model': final_model, 'model_name': winner,
        })

    saved_models[trait] = model_info

    all_results.append({
        'Trait': trait, 'N_BTB': n_btb, 'N_Cal': n_cal,
        'R2_BTB': round(r2_btb_final, 4),
        'R2_Cal': round(r2_cal, 4), 'PA_R2_Cal': round(pa_r2_cal, 4),
        'b0': round(b0, 4), 'b1': round(b1, 4), 'Lambda': best_lambda,
        'Winner': winner, 'Time_s': round(time.time() - t0, 1),
    })

    # === INCREMENTAL SAVE ===
    with open(OUTPUT_DIR / "v12b_models.pkl", 'wb') as f: pickle.dump(saved_models, f)
    with open(OUTPUT_DIR / "v12b_sire_profiles.pkl", 'wb') as f: pickle.dump(sire_profiles, f)
    with open(OUTPUT_DIR / "v12b_mgs_profiles.pkl", 'wb') as f: pickle.dump(mgs_profiles, f)
    with open(OUTPUT_DIR / "v12b_calibration.pkl", 'wb') as f: pickle.dump(calibration_params, f)
    pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "v12b_training_results.csv", index=False)

# ============================================================
# SUMMARY
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  V12b ENGINE — TRAINING SUMMARY")
flush_print(f"{'='*95}")
flush_print(f"  BTB training records: {len(btb_records)}")
flush_print(f"  Daughter calibration records: {len(daughter_records)}")
flush_print(f"  Sire profiles: {len(sire_profiles)}")
flush_print(f"  MGS profiles: {len(mgs_profiles)}")
flush_print(f"  Traits trained: {len(saved_models)}")

if all_results:
    flush_print(f"\n  {'Trait':>6} | {'R2_BTB':>7} | {'R2_Cal':>7} | {'PA_Cal':>6} | {'Gain':>6} | {'Lam':>5} | {'b1':>5} | Win")
    flush_print(f"  {'-'*75}")
    for r in all_results:
        gain = r['R2_Cal'] - r['PA_R2_Cal']
        flush_print(f"  {r['Trait']:>6} | {r['R2_BTB']:>7.4f} | {r['R2_Cal']:>7.4f} | {r['PA_R2_Cal']:>6.4f} | "
                    f"{'+' if gain > 0 else ''}{gain:>5.4f} | {r['Lambda']:>5.2f} | {r['b1']:>5.3f} | {r['Winner']}")

# Final save
with open(OUTPUT_DIR / "v12b_models.pkl", 'wb') as f: pickle.dump(saved_models, f)
with open(OUTPUT_DIR / "v12b_sire_profiles.pkl", 'wb') as f: pickle.dump(sire_profiles, f)
with open(OUTPUT_DIR / "v12b_mgs_profiles.pkl", 'wb') as f: pickle.dump(mgs_profiles, f)
with open(OUTPUT_DIR / "v12b_calibration.pkl", 'wb') as f: pickle.dump(calibration_params, f)
pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "v12b_training_results.csv", index=False)

flush_print(f"\n  All models saved to {OUTPUT_DIR}")
flush_print(f"  V12b engine complete!")
