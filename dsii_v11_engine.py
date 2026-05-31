"""
DSII V11 — Production Engine (PA-Deviation Architecture)
Key improvements over V10-B:
  1. Target = deviation from Parent Average (PA), not absolute value
  2. GroupKFold by sire — no data leakage
  3. Explicit PA features + Mendelian sampling range
  4. Hybrid 3-way expanded to all weak traits
  5. HistGradientBoosting as 5th model
  6. Stronger regularization for low-h2 traits
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, csv, pickle, sys, re, glob, time
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               GradientBoostingClassifier, RandomForestClassifier,
                               HistGradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor

DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")
OUTPUT_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v11_results")
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
    'STA':'ST','STR':'SG','DFM':None,'BOD':None,'RUA':None,'RW':'RW',
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

# Traits that benefit from hybrid 3-way (low h2, sign matters, V10 struggled)
HYBRID_TRAITS = ['DPR', 'CCR', 'MAST', 'MILK', 'FAT', 'PL', 'SCS', 'HCR',
                 'PRO', 'LIV', 'H_LIV', 'MET', 'RP', 'DA', 'KET', 'MF',
                 'DCE', 'SSB', 'DSB', 'GL', 'EFC', 'FI', 'RFI', 'GFI']

# ============================================================
# LOAD BULL DATABASES
# ============================================================
flush_print("=" * 95)
flush_print("  DSII V11 — PA-Deviation Engine (GroupKFold + Hybrid Expanded)")
flush_print("=" * 95)

flush_print("\n  Loading bull databases...")

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
    for k, naab in name_to_naab.items():
        if key in k or k in key:
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

# ============================================================
# LOAD ALL TRAINING DATA
# ============================================================
flush_print("\n  Loading training data...")

# Source 1: banco21052026
raw = pd.read_excel(DOWNLOADS / "banco21052026.xlsx", header=None, engine='openpyxl')
hdr = [str(c).strip() if c else f'col_{i}' for i, c in enumerate(raw.iloc[10])]
banco = raw.iloc[11:].copy()
banco.columns = hdr
banco = banco.reset_index(drop=True)
flush_print(f"    banco21052026: {len(banco)} daughters")

# Source 2: May 2026
may_df = pd.read_excel(DOWNLOADS / "May 2026.xlsx", engine='openpyxl')
flush_print(f"    May 2026: {len(may_df)} daughters")

# Source 3: DAM1-20
dam_frames = []
for fp in sorted(glob.glob(str(DOWNLOADS / "DAM*.xlsx"))):
    dam_frames.append(pd.read_excel(fp, engine='openpyxl'))
dam_df = pd.concat(dam_frames, ignore_index=True) if dam_frames else pd.DataFrame()
flush_print(f"    DAM1-20: {len(dam_df)} animals ({dam_df['MGGS'].notna().sum() if len(dam_df) else 0} with MGGS)")

# ============================================================
# ASSEMBLE TRAINING RECORDS
# ============================================================
flush_print("\n  Assembling training records...")

all_records = []
sire_daughters = defaultdict(list)
mgs_daughters = defaultdict(list)

# From banco21052026
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
        sire_daughters[sire_naab].append((t, v))
        if mgs_ptas: mgs_daughters[mgs_naab].append((t, v))
    all_records.append({
        'source': 'banco', 'sire_naab': sire_naab,
        'mgs_naab': mgs_naab if mgs_ptas else None,
        'sire_ptas': sire_ptas, 'mgs_ptas': mgs_ptas,
        'daughter_ptas': daughter_ptas,
    })
flush_print(f"    banco21052026: {banco_ok} usable records")

# From May 2026
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
        for t, v in daughter_ptas.items(): sire_daughters[sire_naab].append((t, v))
    if mgs_naab and mgs_ptas:
        for t, v in daughter_ptas.items(): mgs_daughters[mgs_naab].append((t, v))
    all_records.append({
        'source': 'may2026', 'sire_naab': sire_naab,
        'mgs_naab': mgs_naab if mgs_ptas else None,
        'sire_ptas': sire_ptas, 'mgs_ptas': mgs_ptas,
        'daughter_ptas': daughter_ptas,
    })
flush_print(f"    May 2026: {may_ok} usable records")

# From DAM1-20
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
        mmgs_row_d, mmgs_src_d = lookup_bull_by_name(mggs_name)
        if mmgs_row_d: mmgs_ptas_d = get_bull_ptas(mmgs_row_d, mmgs_src_d)
    daughter_ptas_d = {}
    for trait, dcol in DAM_COL.items():
        if dcol and dcol in row.index:
            v = sf(row[dcol])
            if v is not None: daughter_ptas_d[trait] = v
    if not daughter_ptas_d: continue
    dam_ok += 1
    if sire_naab_d:
        for t, v in daughter_ptas_d.items(): sire_daughters[sire_naab_d].append((t, v))
    if mgs_naab_d and mgs_ptas_d:
        for t, v in daughter_ptas_d.items(): mgs_daughters[mgs_naab_d].append((t, v))
    all_records.append({
        'source': 'dam', 'sire_naab': sire_naab_d,
        'mgs_naab': mgs_naab_d if mgs_ptas_d else None,
        'sire_ptas': sire_ptas_d, 'mgs_ptas': mgs_ptas_d,
        'mmgs_ptas': mmgs_ptas_d, 'daughter_ptas': daughter_ptas_d,
    })
flush_print(f"    DAM1-20: {dam_ok} usable records")
flush_print(f"    TOTAL training records: {len(all_records)}")

# ============================================================
# COMPUTE TRANSMISSION PROFILES
# ============================================================
flush_print("\n  Computing transmission profiles...")

sire_profiles = {}
for sire_naab, daughter_vals in sire_daughters.items():
    sire_row, src = lookup_bull(sire_naab)
    if not sire_row: continue
    sire_ptas = get_bull_ptas(sire_row, src)
    profile = {}
    trait_vals = defaultdict(list)
    for t, v in daughter_vals: trait_vals[t].append(v)
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
for mgs_naab, daughter_vals in mgs_daughters.items():
    mgs_row, src = lookup_bull(mgs_naab)
    if not mgs_row: continue
    mgs_ptas_l = get_bull_ptas(mgs_row, src)
    profile = {}
    trait_vals = defaultdict(list)
    for t, v in daughter_vals: trait_vals[t].append(v)
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
# BUILD FEATURES — V11 (PA-Deviation Architecture)
# ============================================================
def compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait):
    """Compute Parent Average: Sire/2 + MGS/4 + MMGS/8"""
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

def build_features_v11(sire_ptas, mgs_ptas, trait, sire_naab=None, mgs_naab=None, mmgs_ptas=None):
    s = sire_ptas.get(trait)
    if s is None: return None

    f = {}
    sd = GENETIC_SD.get(trait, 1)
    h2 = HERITABILITY.get(trait, 0.15)

    # === PARENT AVERAGE (key anchor) ===
    pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)
    f['pa'] = pa
    f['pa_z'] = pa / sd if sd else 0

    # === SIRE FEATURES ===
    f['sire'] = s
    f['sire_z'] = s / sd if sd else 0
    f['sire_sq'] = s * s
    f['sire_h2'] = s * h2
    f['sire_dev_from_pa'] = s - pa * 2  # how far sire is from 2*PA

    # === MGS FEATURES ===
    mg = mgs_ptas.get(trait) if mgs_ptas else None
    f['mgs_val'] = mg if mg is not None else 0
    f['mgs_z'] = (mg / sd if sd else 0) if mg is not None else 0
    f['mgs_sq'] = mg * mg if mg is not None else 0
    f['has_mgs'] = 1 if mg is not None else 0

    # === SIRE x MGS INTERACTIONS ===
    if mg is not None:
        f['sire_x_mgs'] = s * mg
        f['sire_mgs_diff'] = s - mg
        f['sire_mgs_ratio'] = s / mg if mg != 0 else 0
        f['sire_mgs_mean'] = (s + mg) / 2
        f['mendelian_range'] = abs(s - mg)  # expected sampling variance
        f['mendelian_range_z'] = abs(s - mg) / sd if sd else 0
    else:
        f['sire_x_mgs'] = 0; f['sire_mgs_diff'] = 0; f['sire_mgs_ratio'] = 0
        f['sire_mgs_mean'] = s / 2; f['mendelian_range'] = 0; f['mendelian_range_z'] = 0

    # === MMGS FEATURES ===
    mmg = mmgs_ptas.get(trait) if mmgs_ptas else None
    f['mmgs_val'] = mmg if mmg is not None else 0
    f['has_mmgs'] = 1 if mmg is not None else 0
    f['sire_x_mmgs'] = s * mmg if mmg is not None else 0

    # === CROSS-TRAIT PA (Parent Averages for correlated traits) ===
    for ot in CORE_TRAITS:
        if ot == trait: continue
        sv = sire_ptas.get(ot)
        if sv is not None: f[f'sire_{ot}'] = sv
        mv = mgs_ptas.get(ot) if mgs_ptas else None
        if mv is not None: f[f'mgs_{ot}'] = mv
        # Cross-trait PA
        ot_pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, ot)
        if ot_pa is not None: f[f'pa_{ot}'] = ot_pa

    # === GENETIC CORRELATIONS ===
    for (t1, t2), corr in GENETIC_CORRELATIONS.items():
        if t1 == trait or t2 == trait:
            other = t2 if t1 == trait else t1
            # Use PA of correlated trait (more stable than raw sire)
            ot_pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, other)
            if ot_pa is not None:
                f[f'gc_pa_{other}'] = ot_pa * corr
            sv = sire_ptas.get(other)
            if sv is not None: f[f'gc_sire_{other}'] = sv * corr
            mv = mgs_ptas.get(other) if mgs_ptas else None
            if mv is not None: f[f'gc_mgs_{other}'] = mv * corr

    # === SIRE TRANSMISSION PROFILE ===
    if sire_naab and sire_naab in sire_profiles:
        sp = sire_profiles[sire_naab]
        tp = sp.get(trait)
        if tp and tp['n'] >= 2:
            f['sire_prof_n'] = tp['n']
            f['sire_prof_mean'] = tp['mean']
            f['sire_prof_std'] = tp['std']
            f['sire_prof_ratio'] = tp['trans_ratio']
            f['sire_prof_resid'] = tp['residual']
            f['sire_prof_dev_pa'] = tp['mean'] - pa  # profile deviation from PA
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

    # === MGS TRANSMISSION PROFILE ===
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

# ============================================================
# TRAIN V11 MODELS
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  TRAINING V11 MODELS — PA-Deviation + GroupKFold + 5 Architectures")
flush_print(f"{'='*95}")

def get_models(h2=0.25):
    """Get model set with regularization adapted to heritability."""
    # Lower h2 = more noise = need more regularization
    reg_factor = max(0.5, 1.0 - h2)  # 0.5 for h2=0.5, 1.0 for h2=0
    n_est = 300
    return {
        'GBR': GradientBoostingRegressor(n_estimators=n_est, max_depth=4, learning_rate=0.05,
                                          min_samples_leaf=max(5, int(10 * reg_factor)),
                                          subsample=0.8, random_state=42),
        'LGBM': LGBMRegressor(n_estimators=n_est, max_depth=5, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.6, min_child_samples=max(5, int(10 * reg_factor)),
                               reg_alpha=0.5 * reg_factor, reg_lambda=2.0 * reg_factor,
                               random_state=42, verbose=-1),
        'XGB': XGBRegressor(n_estimators=n_est, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.6,
                             reg_alpha=0.5 * reg_factor, reg_lambda=2.0 * reg_factor,
                             min_child_weight=max(5, int(10 * reg_factor)),
                             random_state=42, verbosity=0),
        'RF': RandomForestRegressor(n_estimators=n_est, max_depth=12, min_samples_leaf=max(4, int(8 * reg_factor)),
                                     random_state=42, n_jobs=-1),
        'HGBR': HistGradientBoostingRegressor(max_iter=n_est, max_depth=5, learning_rate=0.05,
                                               min_samples_leaf=max(5, int(15 * reg_factor)),
                                               l2_regularization=2.0 * reg_factor,
                                               random_state=42),
    }

def get_clf_models():
    return {
        'GBC': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                           min_samples_leaf=10, subsample=0.8, random_state=42),
        'RFC': RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                       random_state=42, n_jobs=-1),
        'LGBC': LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
                                random_state=42, verbose=-1),
    }

flush_print(f"\n  {'Trait':>6} | {'N':>5} | {'Feats':>5} | {'Best':>5} | {'Best R2':>7} | {'Stack R2':>8} | "
            f"{'PA-Dev':>6} | {'PA R2':>7} | Winner")
flush_print(f"  {'-'*100}")

saved_models = {}
all_results = []

for trait in ALL_TRAITS:
    t0 = time.time()
    h2 = HERITABILITY.get(trait, 0.15)
    feat_rows, y_vals, pa_vals, sire_groups = [], [], [], []

    for rec in all_records:
        dv = rec['daughter_ptas'].get(trait)
        if dv is None: continue
        feat = build_features_v11(rec['sire_ptas'], rec['mgs_ptas'], trait,
                                   rec['sire_naab'], rec.get('mgs_naab'),
                                   rec.get('mmgs_ptas'))
        if feat is None: continue
        pa = compute_pa(rec['sire_ptas'], rec['mgs_ptas'], rec.get('mmgs_ptas'), trait)
        feat_rows.append(feat)
        y_vals.append(dv)
        pa_vals.append(pa if pa is not None else 0)
        sire_groups.append(rec['sire_naab'])

    if len(feat_rows) < 50:
        flush_print(f"  {trait:>6} | SKIP (N={len(feat_rows)})")
        continue

    feat_df = pd.DataFrame(feat_rows)
    feat_df = feat_df.dropna(axis=1, thresh=int(len(feat_df) * 0.3))
    feature_cols = list(feat_df.columns)
    for c in feature_cols:
        if feat_df[c].isna().any(): feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    X = feat_df.values.astype(np.float64)
    y = np.array(y_vals, dtype=np.float64)
    pa_arr = np.array(pa_vals, dtype=np.float64)
    groups = np.array(sire_groups)

    # Target: deviation from PA
    y_dev = y - pa_arr

    # GroupKFold by sire (honest CV — no sire in both train and test)
    unique_sires = np.unique(groups)
    n_groups = len(unique_sires)

    if n_groups >= 3:
        gkf = GroupKFold(n_splits=min(3, n_groups))
        cv_splits = list(gkf.split(X, y_dev, groups))
    else:
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_splits = list(kf.split(X))

    # === TRAIN ON DEVIATION TARGET ===
    models = get_models(h2)
    model_scores = {}
    model_preds = {}

    for model_name, model in models.items():
        try:
            preds = np.zeros(len(y_dev))
            r2s = []
            for tr, te in cv_splits:
                m = get_models(h2)[model_name]
                m.fit(X[tr], y_dev[tr])
                pred = m.predict(X[te])
                preds[te] = pred
                r2s.append(r2_score(y_dev[te], pred))
            model_scores[model_name] = {'r2': np.mean(r2s)}
            model_preds[model_name] = preds
        except: pass

    if not model_scores: continue

    best_name = max(model_scores, key=lambda k: model_scores[k]['r2'])
    best_r2_dev = model_scores[best_name]['r2']

    # Stacking: top 3
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['r2'], reverse=True)
    top3_names = [m[0] for m in sorted_models[:3]]
    stack_X = np.column_stack([model_preds[m] for m in top3_names if m in model_preds])
    stack_preds = np.zeros(len(y_dev))
    stack_r2s = []
    for tr, te in cv_splits:
        meta = Ridge(alpha=1.0)
        meta.fit(stack_X[tr], y_dev[tr])
        pred = meta.predict(stack_X[te])
        stack_preds[te] = pred
        stack_r2s.append(r2_score(y_dev[te], pred))
    stack_r2_dev = np.mean(stack_r2s)

    # Convert back to absolute for comparison: pred_absolute = PA + pred_deviation
    best_preds_abs = pa_arr + model_preds[best_name]
    stack_preds_abs = pa_arr + stack_preds
    pa_r2 = r2_score(y, pa_arr)  # PA-only baseline

    best_r2_abs = r2_score(y, best_preds_abs)
    stack_r2_abs = r2_score(y, stack_preds_abs)

    if stack_r2_abs > best_r2_abs:
        winner = 'STACK'
        final_r2_abs = stack_r2_abs
        final_r2_dev = stack_r2_dev
    else:
        winner = best_name
        final_r2_abs = best_r2_abs
        final_r2_dev = best_r2_dev

    flush_print(f"  {trait:>6} | {len(y):>5} | {len(feature_cols):>5} | {best_name:>5} | {best_r2_dev:>7.4f} | "
                f"{stack_r2_dev:>8.4f} | {'Yes':>6} | {pa_r2:>7.4f} | {winner} (abs={final_r2_abs:.4f})")

    # Train final model on all data
    if winner == 'STACK':
        base_models = {}
        for mn in top3_names:
            m = get_models(h2)[mn]
            m.fit(X, y_dev)
            base_models[mn] = m
        base_preds_train = np.column_stack([model_preds[m] for m in top3_names])
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(base_preds_train, y_dev)
        saved_models[trait] = {
            'type': 'stack', 'base_models': base_models, 'base_names': top3_names,
            'meta_model': meta_model, 'feature_cols': feature_cols,
            'r2_cv': round(final_r2_abs, 4), 'r2_dev': round(final_r2_dev, 4),
            'pa_r2': round(pa_r2, 4), 'target': 'deviation',
        }
    else:
        final_model = get_models(h2)[winner]
        final_model.fit(X, y_dev)
        saved_models[trait] = {
            'type': 'single', 'model': final_model, 'model_name': winner,
            'feature_cols': feature_cols,
            'r2_cv': round(final_r2_abs, 4), 'r2_dev': round(final_r2_dev, 4),
            'pa_r2': round(pa_r2, 4), 'target': 'deviation',
        }

    all_results.append({
        'Trait': trait, 'N': len(y), 'Features': len(feature_cols),
        'Best_Single': best_name, 'Best_R2_dev': round(best_r2_dev, 4),
        'Stack_R2_dev': round(stack_r2_dev, 4),
        'PA_R2': round(pa_r2, 4), 'Final_R2_abs': round(final_r2_abs, 4),
        'Winner': winner,
        'All_Scores': {k: round(v['r2'], 4) for k, v in model_scores.items()},
        'Time_s': round(time.time() - t0, 1),
    })

    # === INCREMENTAL SAVE after each trait ===
    with open(OUTPUT_DIR / "v11_models.pkl", 'wb') as f:
        pickle.dump(saved_models, f)
    with open(OUTPUT_DIR / "v11_sire_profiles.pkl", 'wb') as f:
        pickle.dump(sire_profiles, f)
    with open(OUTPUT_DIR / "v11_mgs_profiles.pkl", 'wb') as f:
        pickle.dump(mgs_profiles, f)
    pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "v11_training_results.csv", index=False)

# ============================================================
# V11 ENHANCEMENTS — Hybrid 3-way for weak traits
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  V11 ENHANCEMENTS — Hybrid 3-way (expanded)")
flush_print(f"{'='*95}")

def prepare_data_v11(trait):
    feat_rows, y_vals, pa_vals, recs_out, groups = [], [], [], [], []
    for rec in all_records:
        dv = rec['daughter_ptas'].get(trait)
        if dv is None: continue
        feat = build_features_v11(rec['sire_ptas'], rec['mgs_ptas'], trait,
                                   rec['sire_naab'], rec.get('mgs_naab'), rec.get('mmgs_ptas'))
        if feat is None: continue
        pa = compute_pa(rec['sire_ptas'], rec['mgs_ptas'], rec.get('mmgs_ptas'), trait)
        feat_rows.append(feat)
        y_vals.append(dv)
        pa_vals.append(pa if pa is not None else 0)
        recs_out.append(rec)
        groups.append(rec['sire_naab'])
    feat_df = pd.DataFrame(feat_rows)
    feat_df = feat_df.dropna(axis=1, thresh=int(len(feat_df)*0.3))
    feature_cols = list(feat_df.columns)
    for c in feature_cols:
        if feat_df[c].isna().any(): feat_df[c] = feat_df[c].fillna(feat_df[c].median())
    X = feat_df.values.astype(np.float64)
    y = np.array(y_vals, dtype=np.float64)
    pa_arr = np.array(pa_vals, dtype=np.float64)
    return X, y, pa_arr, recs_out, feature_cols, np.array(groups)

def oof_tournament_v11(X, y_dev, groups, h2=0.25):
    unique_sires = np.unique(groups)
    if len(unique_sires) >= 3:
        gkf = GroupKFold(n_splits=min(3, len(unique_sires)))
        splits = list(gkf.split(X, y_dev, groups))
    else:
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        splits = list(kf.split(X))

    models = get_models(h2)
    model_preds = {}
    for mn, model in models.items():
        try:
            preds = np.zeros(len(y_dev))
            for tr, te in splits:
                m = get_models(h2)[mn]
                m.fit(X[tr], y_dev[tr])
                preds[te] = m.predict(X[te])
            model_preds[mn] = preds
        except: pass
    if not model_preds:
        return np.zeros(len(y_dev)), 'NONE', model_preds, [], splits
    best_name = max(model_preds, key=lambda k: r2_score(y_dev, model_preds[k]))
    sorted_m = sorted(model_preds.items(), key=lambda x: -r2_score(y_dev, x[1]))
    top3 = [m[0] for m in sorted_m[:3]]
    sX = np.column_stack([model_preds[m] for m in top3])
    stack_preds = np.zeros(len(y_dev))
    for tr, te in splits:
        meta = Ridge(alpha=1.0)
        meta.fit(sX[tr], y_dev[tr])
        stack_preds[te] = meta.predict(sX[te])
    if r2_score(y_dev, stack_preds) > r2_score(y_dev, model_preds[best_name]):
        return stack_preds, f'STACK({",".join(top3)})', model_preds, top3, splits
    return model_preds[best_name], best_name, model_preds, [best_name], splits

# --- LIV: PL-anchored ---
flush_print(f"\n  LIV: Retraining with PL-anchored features...")
if 'LIV' in saved_models and 'PL' in saved_models:
    X_pl, y_pl, pa_pl, recs_pl, cols_pl, groups_pl = prepare_data_v11('PL')
    y_dev_pl = y_pl - pa_pl
    pl_oof, _, _, _, _ = oof_tournament_v11(X_pl, y_dev_pl, groups_pl, HERITABILITY.get('PL', 0.08))

    pl_by_key = {}
    for i, rec in enumerate(recs_pl):
        key = (rec['sire_naab'], rec.get('mgs_naab', ''))
        pl_by_key.setdefault(key, []).append(pa_pl[i] + pl_oof[i])  # absolute PL pred

    X_liv, y_liv, pa_liv, recs_liv, cols_liv, groups_liv = prepare_data_v11('LIV')
    y_dev_liv = y_liv - pa_liv
    pl_aug = np.zeros(len(y_liv))
    for i, rec in enumerate(recs_liv):
        key = (rec['sire_naab'], rec.get('mgs_naab', ''))
        if key in pl_by_key:
            pl_aug[i] = np.mean(pl_by_key[key])
        else:
            pl_aug[i] = rec['sire_ptas'].get('PL', 0) / 2

    X_liv_aug = np.column_stack([X_liv, pl_aug, pl_aug**2])
    cols_liv_aug = cols_liv + ['pl_pred', 'pl_pred_sq']

    liv_oof_aug, liv_winner, _, _, _ = oof_tournament_v11(X_liv_aug, y_dev_liv, groups_liv, HERITABILITY.get('LIV', 0.05))
    r2_aug_abs = r2_score(y_liv, pa_liv + liv_oof_aug)
    r2_orig = saved_models['LIV']['r2_cv']

    flush_print(f"    LIV original R2={r2_orig:.4f} -> PL-anchored R2={r2_aug_abs:.4f}")

    if r2_aug_abs > r2_orig:
        final_model = get_models(HERITABILITY.get('LIV', 0.05))[liv_winner.split('(')[0] if 'STACK' not in liv_winner else 'GBR']
        if 'STACK' in liv_winner:
            # Reuse best single for simplicity
            best_single = max(get_models(0.05).keys(), key=lambda k: True)
            final_model = get_models(0.05)['LGBM']
        final_model.fit(X_liv_aug, y_dev_liv)
        saved_models['LIV'] = {
            'type': 'single', 'model': final_model, 'model_name': 'LGBM',
            'feature_cols': cols_liv_aug,
            'r2_cv': round(r2_aug_abs, 4), 'target': 'deviation', 'v10b': 'pl_anchored',
        }
        flush_print(f"    LIV model UPDATED with PL-anchored features")
    else:
        flush_print(f"    LIV: no improvement, keeping original")

# --- Hybrid 3-way for weak traits ---
flush_print(f"\n  Hybrid 3-way for {len(HYBRID_TRAITS)} traits...")

for trait in HYBRID_TRAITS:
    if trait not in saved_models: continue
    h2 = HERITABILITY.get(trait, 0.15)
    X, y, pa_arr, recs_t, cols_t, groups_t = prepare_data_v11(trait)
    N = len(y)
    if N < 100: continue
    y_dev = y - pa_arr

    # V11 OOF predictions (deviation)
    v11_oof, v11_winner, v11_all_preds, v11_top3, splits = oof_tournament_v11(X, y_dev, groups_t, h2)

    # Sign classification (on absolute y)
    y_binary = (np.sign(y) >= 0).astype(int)
    n_classes = len(np.unique(y_binary))

    sign_preds_oof = np.zeros(N)
    best_clf_name = 'SKIP'
    if n_classes >= 2:
        best_clf_acc = 0
        for cname, clf_model in get_clf_models().items():
            proba = np.zeros(N)
            for tr, te in splits:
                clf = get_clf_models()[cname]
                clf.fit(X[tr], y_binary[tr])
                proba[te] = clf.predict_proba(X[te])[:, 1]
            acc = np.mean((proba >= 0.5) == y_binary)
            if acc > best_clf_acc:
                best_clf_acc = acc
                best_clf_name = cname
                sign_preds_oof = proba
    else:
        sign_preds_oof[:] = 1.0

    # Magnitude regression on |y|
    y_abs = np.abs(y)
    mag_oof, mag_winner, _, _, _ = oof_tournament_v11(X, y_abs, groups_t, h2)
    mag_oof = np.maximum(mag_oof, 0)

    # 2-stage combination
    pred_sign_hard = np.where(sign_preds_oof >= 0.5, 1.0, -1.0)
    sign_confidence = np.abs(sign_preds_oof - 0.5) * 2

    preds_2stage = np.zeros(N)
    for i, rec in enumerate(recs_t):
        sn = rec['sire_naab']
        n_d = sire_profiles.get(sn, {}).get(trait, {}).get('n', 0)
        prof_conf = min(n_d / 30, 1.0) if n_d > 0 else 0.2
        total_conf = 0.6 * sign_confidence[i] + 0.4 * prof_conf
        preds_2stage[i] = pred_sign_hard[i] * mag_oof[i] * total_conf

    # Profile-anchored
    preds_anchored = np.zeros(N)
    for i, rec in enumerate(recs_t):
        sn = rec['sire_naab']
        sp = sire_profiles.get(sn, {})
        tp = sp.get(trait)
        mgs_val = rec['mgs_ptas'].get(trait, 0) if rec['mgs_ptas'] else 0
        if tp and tp['n'] >= 5:
            preds_anchored[i] = tp['mean'] * 0.85 + mgs_val * 0.15
        elif tp and tp['n'] >= 2:
            preds_anchored[i] = tp['mean'] * 0.6 + (rec['sire_ptas'].get(trait, 0) / 2) * 0.25 + mgs_val * 0.15
        else:
            preds_anchored[i] = pa_arr[i]  # Use PA as fallback

    # PA-only as 4th component
    # Blend: V11 (PA+dev), 2-stage, anchored, PA-only
    v11_abs = pa_arr + v11_oof
    blend_X = np.column_stack([v11_abs, preds_2stage, preds_anchored, pa_arr])
    preds_hybrid = np.zeros(N)
    for tr, te in splits:
        ridge = Ridge(alpha=1.0)
        ridge.fit(blend_X[tr], y[tr])
        preds_hybrid[te] = ridge.predict(blend_X[te])

    r2_v11 = r2_score(y, v11_abs)
    r2_hyb = r2_score(y, preds_hybrid)
    r2_pa = r2_score(y, pa_arr)

    flush_print(f"    {trait:>6}: PA={r2_pa:.4f} V11={r2_v11:.4f} Hybrid={r2_hyb:.4f} "
                f"{'IMPROVED' if r2_hyb > r2_v11 else 'no change'}")

    if r2_hyb > r2_v11:
        # Train final models
        final_clf = None
        if n_classes >= 2 and best_clf_name != 'SKIP':
            final_clf = get_clf_models()[best_clf_name]
            final_clf.fit(X, y_binary)

        final_mag = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                               min_samples_leaf=8, subsample=0.8, random_state=42)
        final_mag.fit(X, y_abs)

        blend_ridge = Ridge(alpha=1.0)
        blend_ridge.fit(blend_X, y)

        saved_models[trait]['v10b'] = 'hybrid_3way'
        saved_models[trait]['sign_clf'] = final_clf
        saved_models[trait]['sign_clf_name'] = best_clf_name
        saved_models[trait]['mag_model'] = final_mag
        saved_models[trait]['blend_ridge'] = blend_ridge
        saved_models[trait]['r2_cv'] = round(r2_hyb, 4)

    # === INCREMENTAL SAVE after each hybrid trait ===
    with open(OUTPUT_DIR / "v11_models.pkl", 'wb') as f:
        pickle.dump(saved_models, f)

# ============================================================
# PREDICTION FUNCTION — V11
# ============================================================
def predict_animal(sire_ptas, mgs_ptas, sire_naab=None, mgs_naab=None, mmgs_ptas=None):
    results = {}

    # PL first (for LIV anchoring)
    pl_pred = None
    pl_info = saved_models.get('PL')
    if pl_info:
        feat = build_features_v11(sire_ptas, mgs_ptas, 'PL', sire_naab, mgs_naab, mmgs_ptas)
        if feat:
            pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, 'PL')
            feat_df = pd.DataFrame([feat])
            for c in pl_info['feature_cols']:
                if c not in feat_df.columns: feat_df[c] = 0
            feat_df = feat_df[pl_info['feature_cols']]
            X = feat_df.values.astype(np.float64)
            if pl_info['type'] == 'stack':
                base_preds = np.column_stack([pl_info['base_models'][mn].predict(X) for mn in pl_info['base_names']])
                dev_pred = pl_info['meta_model'].predict(base_preds)[0]
            else:
                dev_pred = pl_info['model'].predict(X)[0]
            pl_pred = (pa if pa else 0) + dev_pred
            results['PL'] = round(pl_pred, 2)

    for trait in ALL_TRAITS:
        if trait == 'PL': continue
        info = saved_models.get(trait)
        if not info: continue

        feat = build_features_v11(sire_ptas, mgs_ptas, trait, sire_naab, mgs_naab, mmgs_ptas)
        if feat is None: continue

        pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)
        if pa is None: pa = 0

        feat_df = pd.DataFrame([feat])

        if info.get('v10b') == 'pl_anchored':
            pl_val = pl_pred if pl_pred is not None else sire_ptas.get('PL', 0) / 2
            feat_df['pl_pred'] = pl_val
            feat_df['pl_pred_sq'] = pl_val ** 2

        for c in info['feature_cols']:
            if c not in feat_df.columns: feat_df[c] = 0
        feat_df = feat_df[info['feature_cols']]
        X = feat_df.values.astype(np.float64)

        # Base prediction (deviation from PA)
        if info['type'] == 'stack':
            base_preds = np.column_stack([info['base_models'][mn].predict(X) for mn in info['base_names']])
            dev_pred = info['meta_model'].predict(base_preds)[0]
        else:
            dev_pred = info['model'].predict(X)[0]

        pred_v11 = pa + dev_pred

        # Hybrid 3-way
        if info.get('v10b') == 'hybrid_3way':
            sign_clf = info.get('sign_clf')
            if sign_clf is not None:
                sign_prob = sign_clf.predict_proba(X)[0, 1]
            else:
                sign_prob = 1.0
            mag = max(info['mag_model'].predict(X)[0], 0)
            pred_sign = 1.0 if sign_prob >= 0.5 else -1.0
            sign_conf = abs(sign_prob - 0.5) * 2

            n_d = sire_profiles.get(sire_naab, {}).get(trait, {}).get('n', 0)
            prof_conf = min(n_d / 30, 1.0) if n_d > 0 else 0.2
            total_conf = 0.6 * sign_conf + 0.4 * prof_conf
            pred_2stage = pred_sign * mag * total_conf

            sp = sire_profiles.get(sire_naab, {})
            tp = sp.get(trait)
            mgs_val = mgs_ptas.get(trait, 0) if mgs_ptas else 0
            if tp and tp['n'] >= 5:
                pred_anc = tp['mean'] * 0.85 + mgs_val * 0.15
            elif tp and tp['n'] >= 2:
                pred_anc = tp['mean'] * 0.6 + (sire_ptas.get(trait, 0) / 2) * 0.25 + mgs_val * 0.15
            else:
                pred_anc = pa

            blend = np.array([[pred_v11, pred_2stage, pred_anc, pa]])
            pred_final = info['blend_ridge'].predict(blend)[0]
        else:
            pred_final = pred_v11

        results[trait] = round(pred_final, 4)

    return results

# ============================================================
# SUMMARY
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  V11 ENGINE — TRAINING SUMMARY")
flush_print(f"{'='*95}")
flush_print(f"  Total training records: {len(all_records)}")
flush_print(f"  Sire profiles: {len(sire_profiles)}")
flush_print(f"  MGS profiles: {len(mgs_profiles)}")
flush_print(f"  Traits trained: {len(saved_models)}")

if all_results:
    flush_print(f"\n  {'Trait':>6} | {'R2_abs':>7} | {'R2_dev':>7} | {'PA_R2':>6} | {'Winner':>30} | {'V11':>15}")
    flush_print(f"  {'-'*85}")
    for r in all_results:
        trait = r['Trait']
        info = saved_models.get(trait, {})
        v11_label = info.get('v10b', 'standard')
        final_r2 = info.get('r2_cv', r['Final_R2_abs'])
        winner = r['Winner']
        pa_r2 = r['PA_R2']
        r2_dev = r.get('Stack_R2_dev', r['Best_R2_dev'])
        if v11_label == 'pl_anchored':
            winner = f"{winner} +PL"
        elif v11_label == 'hybrid_3way':
            clf_name = info.get('sign_clf_name', '')
            winner = f"HYBRID({winner}+{clf_name})"
        flush_print(f"  {trait:>6} | {final_r2:>7.4f} | {r2_dev:>7.4f} | {pa_r2:>6.4f} | {winner:>30} | {v11_label:>15}")

# ============================================================
# SAVE
# ============================================================
with open(OUTPUT_DIR / "v11_models.pkl", 'wb') as f:
    pickle.dump(saved_models, f)
with open(OUTPUT_DIR / "v11_sire_profiles.pkl", 'wb') as f:
    pickle.dump(sire_profiles, f)
with open(OUTPUT_DIR / "v11_mgs_profiles.pkl", 'wb') as f:
    pickle.dump(mgs_profiles, f)
pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "v11_training_results.csv", index=False)

flush_print(f"\n  Models saved: {OUTPUT_DIR / 'v11_models.pkl'}")
flush_print(f"  V11 PA-Deviation Engine ready!")
