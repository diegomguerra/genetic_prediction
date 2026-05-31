"""
DSII V12 — Lineage Learning + Calibration + Shrinkage Engine
Key improvements over V11:
  1. Bull-to-Bull training with 186k+ records from bulls.csv (Sire Stack)
  2. Enhanced Sire Transmission Profiles from 250k bulls
  3. Post-hoc linear calibration (BIF-standard: b0=0, b1=1.0)
  4. Adaptive shrinkage: pred = PA + lambda * ML_deviation_calibrated
  5. Reliability-weighted training (Rel >= 70% filter)
  6. Generation weighting (recent bulls = more weight)
  7. Incremental save after each trait
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
from sklearn.linear_model import Ridge, LinearRegression
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor

DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")
OUTPUT_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v12_results")
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
    ('FAT','PRO'):0.65,('FAT','DPR'):-0.25,('PRO','DPR'):-0.30,('DPR','CCR'):0.75,
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
    ('CCR','PL'):0.35,('HCR','PL'):0.30,
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

# Reliability column mapping for bulls.csv
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
flush_print("  DSII V12 — Lineage Learning + Calibration + Shrinkage Engine")
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
    # Skip slow fuzzy search for BTB — exact match only for performance
    return None, None

def get_bull_ptas(bull_row, source='SS'):
    ptas = {}
    col_map = BULL_COL if source == 'SS' else CDCB_COL
    for trait, col in col_map.items():
        if col is None: continue
        v = sf(bull_row.get(col))
        if v is not None: ptas[trait] = v
    return ptas

def get_bull_rel(bull_row, trait):
    """Get trait-specific reliability from bulls.csv"""
    col = REL_COL.get(trait, 'Rel')
    v = sf(bull_row.get(col))
    return v if v is not None else None

def get_bull_dob_year(bull_row):
    """Extract birth year from DOB_RAW"""
    dob = bull_row.get('DOB_RAW', '')
    if not dob: return None
    try:
        dob = str(dob).strip()
        # Format could be YYYYMMDD or other
        if len(dob) >= 4:
            yr = int(dob[:4])
            if 1960 <= yr <= 2025: return yr
    except: pass
    return None

# ============================================================
# PHASE 1: BULL-TO-BULL TRAINING DATA (186k+ records)
# ============================================================
flush_print("\n  PHASE 1: Building Bull-to-Bull training data from Sire Stack...")

btb_records = []  # bull-to-bull records
btb_sire_daughters = defaultdict(list)
btb_mgs_daughters = defaultdict(list)

btb_stats = {'total': 0, 'parsed': 0, 'sire_found': 0, 'mgs_found': 0,
             'both_found': 0, 'rel_ok': 0, 'skipped_low_rel': 0}

for naab, row in bulls.items():
    btb_stats['total'] += 1
    sire_stack = row.get('Sire Stack', '').strip()
    if not sire_stack or sire_stack in ('nan', 'None', ''): continue

    # Parse "Sire x MGS" format
    parts = sire_stack.split(' x ')
    if len(parts) < 2: continue
    btb_stats['parsed'] += 1

    sire_name = parts[0].strip().upper()
    mgs_name = parts[1].strip().upper()

    # Look up sire
    sire_row_l, sire_src_l = lookup_bull_by_name(sire_name)
    if not sire_row_l: continue
    btb_stats['sire_found'] += 1
    sire_ptas_l = get_bull_ptas(sire_row_l, sire_src_l)
    if not sire_ptas_l: continue
    sire_naab_l = sire_row_l.get('NAAB', '')

    # Look up MGS
    mgs_row_l, mgs_src_l = lookup_bull_by_name(mgs_name)
    mgs_ptas_l = {}
    mgs_naab_l = ''
    if mgs_row_l:
        btb_stats['mgs_found'] += 1
        mgs_ptas_l = get_bull_ptas(mgs_row_l, mgs_src_l)
        mgs_naab_l = mgs_row_l.get('NAAB', '')

    if not mgs_ptas_l: continue
    btb_stats['both_found'] += 1

    # Get son's PTAs (this is the target)
    son_ptas = get_bull_ptas(row, 'SS')
    if not son_ptas: continue

    # Get son's reliability — filter low Rel
    son_rel = sf(row.get('Rel', ''))
    if son_rel is not None and son_rel <= 75:
        btb_stats['skipped_low_rel'] += 1
        continue

    # Filter by birth year > 2005
    dob_year = get_bull_dob_year(row)
    if dob_year is not None and dob_year <= 2005:
        btb_stats['skipped_low_rel'] += 1
        continue
    btb_stats['rel_ok'] += 1

    # Get MMGS if available (3rd part of Sire Stack)
    mmgs_ptas_l = {}
    if len(parts) >= 3:
        mmgs_name = parts[2].strip().upper()
        mmgs_row_l, mmgs_src_l = lookup_bull_by_name(mmgs_name)
        if mmgs_row_l:
            mmgs_ptas_l = get_bull_ptas(mmgs_row_l, mmgs_src_l)

    # Get birth year for generation weighting
    dob_year = get_bull_dob_year(row)

    # Build record
    for t, v in son_ptas.items():
        btb_sire_daughters[sire_naab_l].append((t, v))
        if mgs_ptas_l: btb_mgs_daughters[mgs_naab_l].append((t, v))

    btb_records.append({
        'source': 'btb', 'sire_naab': sire_naab_l,
        'mgs_naab': mgs_naab_l if mgs_ptas_l else None,
        'sire_ptas': sire_ptas_l, 'mgs_ptas': mgs_ptas_l,
        'mmgs_ptas': mmgs_ptas_l,
        'son_ptas': son_ptas,  # target = son's PTA (proven bull)
        'son_rel': son_rel if son_rel else 99,
        'son_naab': naab,
        'dob_year': dob_year,
    })

flush_print(f"    Total bulls scanned: {btb_stats['total']}")
flush_print(f"    Sire Stack parsed: {btb_stats['parsed']}")
flush_print(f"    Both Sire+MGS found: {btb_stats['both_found']}")
flush_print(f"    Skipped low Rel (<70): {btb_stats['skipped_low_rel']}")
flush_print(f"    Bull-to-Bull records (Rel>=70): {btb_stats['rel_ok']}")
flush_print(f"    Final BTB records: {len(btb_records)}")

# ============================================================
# PHASE 1B: LOAD DAUGHTER TRAINING DATA (same as V11)
# ============================================================
flush_print("\n  PHASE 1B: Loading daughter training data...")

daughter_records = []
daughter_sire_daughters = defaultdict(list)
daughter_mgs_daughters = defaultdict(list)

# Source 1: banco21052026
raw = pd.read_excel(DOWNLOADS / "banco21052026.xlsx", header=None, engine='openpyxl')
hdr = [str(c).strip() if c else f'col_{i}' for i, c in enumerate(raw.iloc[10])]
banco = raw.iloc[11:].copy()
banco.columns = hdr
banco = banco.reset_index(drop=True)
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
        daughter_sire_daughters[sire_naab].append((t, v))
        if mgs_ptas: daughter_mgs_daughters[mgs_naab].append((t, v))
    daughter_records.append({
        'source': 'banco', 'sire_naab': sire_naab,
        'mgs_naab': mgs_naab if mgs_ptas else None,
        'sire_ptas': sire_ptas, 'mgs_ptas': mgs_ptas,
        'daughter_ptas': daughter_ptas,
    })
flush_print(f"    banco21052026: {banco_ok} usable records")

# Source 2: May 2026
may_df = pd.read_excel(DOWNLOADS / "May 2026.xlsx", engine='openpyxl')
flush_print(f"    May 2026: {len(may_df)} daughters")

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
        for t, v in daughter_ptas.items(): daughter_sire_daughters[sire_naab].append((t, v))
    if mgs_naab and mgs_ptas:
        for t, v in daughter_ptas.items(): daughter_mgs_daughters[mgs_naab].append((t, v))
    daughter_records.append({
        'source': 'may2026', 'sire_naab': sire_naab,
        'mgs_naab': mgs_naab if mgs_ptas else None,
        'sire_ptas': sire_ptas, 'mgs_ptas': mgs_ptas,
        'daughter_ptas': daughter_ptas,
    })
flush_print(f"    May 2026: {may_ok} usable records")

# Source 3: DAM1-20
dam_frames = []
for fp in sorted(glob.glob(str(DOWNLOADS / "DAM*.xlsx"))):
    dam_frames.append(pd.read_excel(fp, engine='openpyxl'))
dam_df = pd.concat(dam_frames, ignore_index=True) if dam_frames else pd.DataFrame()
flush_print(f"    DAM1-20: {len(dam_df)} animals")

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
        for t, v in daughter_ptas_d.items(): daughter_sire_daughters[sire_naab_d].append((t, v))
    if mgs_naab_d and mgs_ptas_d:
        for t, v in daughter_ptas_d.items(): daughter_mgs_daughters[mgs_naab_d].append((t, v))
    daughter_records.append({
        'source': 'dam', 'sire_naab': sire_naab_d,
        'mgs_naab': mgs_naab_d if mgs_ptas_d else None,
        'sire_ptas': sire_ptas_d, 'mgs_ptas': mgs_ptas_d,
        'mmgs_ptas': mmgs_ptas_d, 'daughter_ptas': daughter_ptas_d,
    })
flush_print(f"    DAM1-20: {dam_ok} usable records")
flush_print(f"    TOTAL daughter records: {len(daughter_records)}")
flush_print(f"    TOTAL BTB records: {len(btb_records)}")
flush_print(f"    COMBINED: {len(daughter_records) + len(btb_records)}")

# ============================================================
# PHASE 2: ENHANCED TRANSMISSION PROFILES (BTB + Daughters)
# ============================================================
flush_print("\n  PHASE 2: Computing enhanced transmission profiles...")

# Merge sire daughter observations from both sources
combined_sire_daughters = defaultdict(list)
combined_mgs_daughters = defaultdict(list)

for k, v in daughter_sire_daughters.items():
    combined_sire_daughters[k].extend(v)
for k, v in btb_sire_daughters.items():
    combined_sire_daughters[k].extend(v)
for k, v in daughter_mgs_daughters.items():
    combined_mgs_daughters[k].extend(v)
for k, v in btb_mgs_daughters.items():
    combined_mgs_daughters[k].extend(v)

sire_profiles = {}
for sire_naab, obs_vals in combined_sire_daughters.items():
    sire_row, src = lookup_bull(sire_naab)
    if not sire_row: continue
    sire_ptas = get_bull_ptas(sire_row, src)
    profile = {}
    trait_vals = defaultdict(list)
    for t, v in obs_vals: trait_vals[t].append(v)
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
for mgs_naab, obs_vals in combined_mgs_daughters.items():
    mgs_row, src = lookup_bull(mgs_naab)
    if not mgs_row: continue
    mgs_ptas_l = get_bull_ptas(mgs_row, src)
    profile = {}
    trait_vals = defaultdict(list)
    for t, v in obs_vals: trait_vals[t].append(v)
    for t, vals in trait_vals.items():
        if len(vals) >= 2 and t in mgs_ptas_l:
            mean_d = np.mean(vals)
            profile[t] = {
                'n': len(vals), 'mean': mean_d, 'std': np.std(vals),
                'residual': mean_d - mgs_ptas_l[t] / 4,
            }
    if profile: mgs_profiles[mgs_naab] = profile

flush_print(f"    Sire profiles: {len(sire_profiles)} (enhanced with BTB)")
flush_print(f"    MGS profiles: {len(mgs_profiles)} (enhanced with BTB)")

# ============================================================
# BUILD FEATURES — V12
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

def build_features_v12(sire_ptas, mgs_ptas, trait, sire_naab=None, mgs_naab=None, mmgs_ptas=None):
    s = sire_ptas.get(trait)
    if s is None: return None

    f = {}
    sd = GENETIC_SD.get(trait, 1)
    h2 = HERITABILITY.get(trait, 0.15)

    # === PARENT AVERAGE ===
    pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)
    f['pa'] = pa
    f['pa_z'] = pa / sd if sd else 0

    # === SIRE FEATURES ===
    f['sire'] = s
    f['sire_z'] = s / sd if sd else 0
    f['sire_sq'] = s * s
    f['sire_h2'] = s * h2
    f['sire_dev_from_pa'] = s - pa * 2

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
        f['mendelian_range'] = abs(s - mg)
        f['mendelian_range_z'] = abs(s - mg) / sd if sd else 0
    else:
        f['sire_x_mgs'] = 0; f['sire_mgs_diff'] = 0; f['sire_mgs_ratio'] = 0
        f['sire_mgs_mean'] = s / 2; f['mendelian_range'] = 0; f['mendelian_range_z'] = 0

    # === MMGS FEATURES ===
    mmg = mmgs_ptas.get(trait) if mmgs_ptas else None
    f['mmgs_val'] = mmg if mmg is not None else 0
    f['has_mmgs'] = 1 if mmg is not None else 0
    f['sire_x_mmgs'] = s * mmg if mmg is not None else 0

    # === CROSS-TRAIT PA (raw + z-scored) ===
    for ot in CORE_TRAITS:
        if ot == trait: continue
        ot_sd = GENETIC_SD.get(ot, 1)
        sv = sire_ptas.get(ot)
        if sv is not None:
            f[f'sire_{ot}'] = sv
            f[f'sire_{ot}_z'] = sv / ot_sd if ot_sd else 0
        mv = mgs_ptas.get(ot) if mgs_ptas else None
        if mv is not None:
            f[f'mgs_{ot}'] = mv
            f[f'mgs_{ot}_z'] = mv / ot_sd if ot_sd else 0
        ot_pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, ot)
        if ot_pa is not None:
            f[f'pa_{ot}'] = ot_pa
            f[f'pa_{ot}_z'] = ot_pa / ot_sd if ot_sd else 0

    # === GENETIC CORRELATIONS (raw + z-scored borrowing) ===
    for (t1, t2), corr in GENETIC_CORRELATIONS.items():
        if t1 == trait or t2 == trait:
            other = t2 if t1 == trait else t1
            ot_sd = GENETIC_SD.get(other, 1)
            ot_pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, other)
            if ot_pa is not None:
                f[f'gc_pa_{other}'] = ot_pa * corr
                f[f'gc_pa_{other}_z'] = (ot_pa / ot_sd) * corr if ot_sd else 0
            sv = sire_ptas.get(other)
            if sv is not None:
                f[f'gc_sire_{other}'] = sv * corr
                f[f'gc_sire_{other}_z'] = (sv / ot_sd) * corr if ot_sd else 0
            mv = mgs_ptas.get(other) if mgs_ptas else None
            if mv is not None:
                f[f'gc_mgs_{other}'] = mv * corr
                f[f'gc_mgs_{other}_z'] = (mv / ot_sd) * corr if ot_sd else 0

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
# MODEL DEFINITIONS
# ============================================================
def get_models(h2=0.25):
    reg_factor = max(0.5, 1.0 - h2)
    n_est = 200
    return {
        'LGBM': LGBMRegressor(n_estimators=n_est, max_depth=5, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.6, min_child_samples=max(5, int(10 * reg_factor)),
                               reg_alpha=0.5 * reg_factor, reg_lambda=2.0 * reg_factor,
                               random_state=42, verbose=-1, n_jobs=-1),
        'XGB': XGBRegressor(n_estimators=n_est, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.6,
                             reg_alpha=0.5 * reg_factor, reg_lambda=2.0 * reg_factor,
                             min_child_weight=max(5, int(10 * reg_factor)),
                             random_state=42, verbosity=0, n_jobs=-1),
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

# ============================================================
# PHASE 3: TRAIN V12 MODELS — Bull-to-Bull + Daughters Combined
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  PHASE 3: TRAINING V12 MODELS — BTB({len(btb_records)}) + Daughters({len(daughter_records)})")
flush_print(f"{'='*95}")

flush_print(f"\n  {'Trait':>6} | {'N_BTB':>6} | {'N_Dau':>6} | {'N_Tot':>6} | {'Best':>5} | {'R2_dev':>7} | "
            f"{'R2_abs':>7} | {'PA_R2':>6} | {'Daughter Cal':>22} | {'N_dau':>5} | Winner")
flush_print(f"  {'-'*110}")

# === RESUME: load previously saved models if available ===
saved_models = {}
all_results = []
calibration_params = {}
_resumed_traits = set()

if (OUTPUT_DIR / "v12_models.pkl").exists():
    try:
        with open(OUTPUT_DIR / "v12_models.pkl", 'rb') as f:
            saved_models = pickle.load(f)
        _resumed_traits = set(saved_models.keys())
        flush_print(f"\n  RESUME: {len(_resumed_traits)} traits already trained: {list(_resumed_traits)}")
    except: pass
if (OUTPUT_DIR / "v12_calibration.pkl").exists():
    try:
        with open(OUTPUT_DIR / "v12_calibration.pkl", 'rb') as f:
            calibration_params = pickle.load(f)
    except: pass
if (OUTPUT_DIR / "v12_training_results.csv").exists():
    try:
        _prev = pd.read_csv(OUTPUT_DIR / "v12_training_results.csv")
        all_results = _prev.to_dict('records')
    except: pass

FI_FORMULA_TRAITS = ['DPR', 'CCR', 'HCR', 'EFC']

for trait in ALL_TRAITS:
    if trait == 'FI':
        flush_print(f"  {trait:>6} | SKIP (calculated from formula: 0.4*DPR + 0.4*CCR + 0.1*HCR + 0.1*EFC)")
        continue
    if trait in _resumed_traits:
        flush_print(f"  {trait:>6} | SKIP (already trained)")
        continue
    t0 = time.time()
    h2 = HERITABILITY.get(trait, 0.15)

    # === COLLECT BTB TRAINING DATA ===
    btb_feat_rows, btb_y_vals, btb_pa_vals, btb_groups, btb_weights = [], [], [], [], []

    for rec in btb_records:
        dv = rec['son_ptas'].get(trait)
        if dv is None: continue
        feat = build_features_v12(rec['sire_ptas'], rec['mgs_ptas'], trait,
                                   rec['sire_naab'], rec.get('mgs_naab'),
                                   rec.get('mmgs_ptas'))
        if feat is None: continue
        pa = compute_pa(rec['sire_ptas'], rec['mgs_ptas'], rec.get('mmgs_ptas'), trait)
        btb_feat_rows.append(feat)
        btb_y_vals.append(dv)
        btb_pa_vals.append(pa if pa is not None else 0)
        btb_groups.append(rec['sire_naab'])
        # Generation weight: more recent = more weight
        yr = rec.get('dob_year')
        if yr and yr >= 1990:
            w = 0.5 + 0.5 * min((yr - 1990) / 30, 1.0)  # 0.5-1.0
        else:
            w = 0.5
        # Reliability weight
        rel = rec.get('son_rel', 70)
        w *= (rel / 100.0)
        btb_weights.append(w)

    # === COLLECT DAUGHTER TRAINING DATA ===
    dau_feat_rows, dau_y_vals, dau_pa_vals, dau_groups = [], [], [], []

    for rec in daughter_records:
        dv = rec['daughter_ptas'].get(trait)
        if dv is None: continue
        feat = build_features_v12(rec['sire_ptas'], rec['mgs_ptas'], trait,
                                   rec['sire_naab'], rec.get('mgs_naab'),
                                   rec.get('mmgs_ptas'))
        if feat is None: continue
        pa = compute_pa(rec['sire_ptas'], rec['mgs_ptas'], rec.get('mmgs_ptas'), trait)
        dau_feat_rows.append(feat)
        dau_y_vals.append(dv)
        dau_pa_vals.append(pa if pa is not None else 0)
        dau_groups.append(rec['sire_naab'])

    n_btb = len(btb_feat_rows)
    n_dau = len(dau_feat_rows)
    n_total = n_btb + n_dau

    if n_total < 50:
        flush_print(f"  {trait:>6} | SKIP (N={n_total})")
        continue

    # === COMBINE BTB + DAUGHTER DATA ===
    all_feat_rows = btb_feat_rows + dau_feat_rows
    all_y = np.array(btb_y_vals + dau_y_vals, dtype=np.float64)
    all_pa = np.array(btb_pa_vals + dau_pa_vals, dtype=np.float64)
    all_groups = np.array(btb_groups + dau_groups)

    # Weights: BTB gets generation+rel weight, daughters get weight=1.0
    all_weights = np.array(btb_weights + [1.0] * n_dau, dtype=np.float64)

    feat_df = pd.DataFrame(all_feat_rows)
    feat_df = feat_df.dropna(axis=1, thresh=int(len(feat_df) * 0.3))
    feature_cols = list(feat_df.columns)
    for c in feature_cols:
        if feat_df[c].isna().any(): feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    X = feat_df.values.astype(np.float64)
    y_dev = all_y - all_pa  # Target: deviation from PA

    # GroupKFold by sire
    unique_sires = np.unique(all_groups)
    n_groups = len(unique_sires)

    if n_groups >= 3:
        gkf = GroupKFold(n_splits=min(3, n_groups))
        cv_splits = list(gkf.split(X, y_dev, all_groups))
    else:
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_splits = list(kf.split(X))

    # === TRAIN WITH SAMPLE WEIGHTS ===
    models = get_models(h2)
    model_scores = {}
    model_preds = {}

    for model_name, model in models.items():
        try:
            preds = np.zeros(len(y_dev))
            r2s = []
            for tr, te in cv_splits:
                m = get_models(h2)[model_name]
                # Use sample_weight for models that support it
                if model_name in ('LGBM', 'XGB'):
                    m.fit(X[tr], y_dev[tr], sample_weight=all_weights[tr])
                else:
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

    # === PHASE 3A: 2-FACTOR CALIBRATION (PA dampening) ===
    # Fit PTA = c0 + c1*PA + c2*raw_dev  (data decides PA weight)
    if stack_r2_dev > best_r2_dev:
        raw_oof_dev = stack_preds
        winner = 'STACK'
    else:
        raw_oof_dev = model_preds[best_name]
        winner = best_name

    # 2-factor regression: PTA_real = c0 + c1*PA + c2*ML_deviation
    mask_valid = np.isfinite(raw_oof_dev) & np.isfinite(all_y) & np.isfinite(all_pa)
    if mask_valid.sum() > 10:
        cal_X = np.column_stack([all_pa[mask_valid], raw_oof_dev[mask_valid]])
        cal_reg = LinearRegression()
        cal_reg.fit(cal_X, all_y[mask_valid])
        c0 = cal_reg.intercept_
        c1 = cal_reg.coef_[0]  # PA weight (< 1 means reduced PA dependency)
        c2 = cal_reg.coef_[1]  # ML deviation weight
    else:
        c0, c1, c2 = 0.0, 1.0, 1.0

    # Clamp c1 to [0.3, 1.0] to keep PA meaningful but not dominant
    c1 = max(0.3, min(1.0, c1))

    # Final predictions with 2-factor calibration
    pred_2f = c0 + c1 * all_pa + c2 * raw_oof_dev

    # Final absolute R2 comparisons
    pa_r2 = r2_score(all_y, all_pa)
    raw_abs = all_pa + raw_oof_dev
    raw_r2_abs = r2_score(all_y, raw_abs)
    final_r2_abs = r2_score(all_y, pred_2f)

    # === DAUGHTER-SPECIFIC CALIBRATION (for prediction population) ===
    dau_mask = np.zeros(n_total, dtype=bool)
    dau_mask[n_btb:n_btb + n_dau] = True
    dau_valid = dau_mask & mask_valid
    n_dau_valid = int(dau_valid.sum())

    if n_dau_valid >= 50:
        cal_dau_X = np.column_stack([all_pa[dau_valid], raw_oof_dev[dau_valid]])
        cal_dau_reg = LinearRegression()
        cal_dau_reg.fit(cal_dau_X, all_y[dau_valid])
        d0 = cal_dau_reg.intercept_
        d1 = max(0.3, min(1.5, cal_dau_reg.coef_[0]))
        d2 = cal_dau_reg.coef_[1]

        # Variance expansion: deviation from PA (pred vs target)
        pred_dau_cal = d0 + d1 * all_pa[dau_valid] + d2 * raw_oof_dev[dau_valid]
        dev_pred_dau = pred_dau_cal - all_pa[dau_valid]
        dev_target_dau = all_y[dau_valid] - all_pa[dau_valid]
        pred_dev_sd = np.std(dev_pred_dau)
        target_dev_sd = np.std(dev_target_dau)
        var_expansion = target_dev_sd / pred_dev_sd if pred_dev_sd > 0.001 else 1.0
        var_expansion = max(1.0, min(2.5, var_expansion))

        # R2 on daughter data with expansion
        pred_dau_exp = all_pa[dau_valid] + (pred_dau_cal - all_pa[dau_valid]) * var_expansion
        r2_dau = r2_score(all_y[dau_valid], pred_dau_exp)
    else:
        d0, d1, d2 = c0, c1, c2
        var_expansion = 1.0
        r2_dau = -999

    flush_print(f"  {trait:>6} | {n_btb:>6} | {n_dau:>6} | {n_total:>6} | {best_name:>5} | "
                f"{best_r2_dev:>7.4f} | {final_r2_abs:>7.4f} | {pa_r2:>6.4f} | "
                f"d1={d1:.3f},d2={d2:.3f},vx={var_expansion:.2f} | {n_dau_valid:>5} | {winner}")

    # Store calibration params
    calibration_params[trait] = {
        'c0': round(c0, 6), 'c1': round(c1, 6), 'c2': round(c2, 6),
        'd0': round(d0, 6), 'd1': round(d1, 6), 'd2': round(d2, 6),
        'var_expansion': round(var_expansion, 4),
        'pa_r2': round(pa_r2, 4),
        'raw_r2': round(raw_r2_abs, 4),
        'calibrated_r2': round(final_r2_abs, 4),
        'r2_daughter': round(r2_dau, 4),
    }

    # === TRAIN FINAL MODELS ON ALL DATA ===
    if winner == 'STACK':
        base_models = {}
        for mn in top3_names:
            m = get_models(h2)[mn]
            if mn in ('LGBM', 'XGB'):
                m.fit(X, y_dev, sample_weight=all_weights)
            else:
                m.fit(X, y_dev)
            base_models[mn] = m
        base_preds_train = np.column_stack([model_preds[m] for m in top3_names])
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(base_preds_train, y_dev)
        saved_models[trait] = {
            'type': 'stack', 'base_models': base_models, 'base_names': top3_names,
            'meta_model': meta_model, 'feature_cols': feature_cols,
            'r2_cv': round(final_r2_abs, 4), 'r2_dev': round(best_r2_dev, 4),
            'pa_r2': round(pa_r2, 4), 'target': 'deviation',
            'calibration_2f': {'c0': c0, 'c1': c1, 'c2': c2},
            'calibration_daughter': {'d0': d0, 'd1': d1, 'd2': d2, 'var_expansion': var_expansion},
        }
    else:
        final_model = get_models(h2)[winner]
        if winner in ('GBR', 'LGBM', 'XGB', 'HGBR'):
            final_model.fit(X, y_dev, sample_weight=all_weights)
        else:
            final_model.fit(X, y_dev)
        saved_models[trait] = {
            'type': 'single', 'model': final_model, 'model_name': winner,
            'feature_cols': feature_cols,
            'r2_cv': round(final_r2_abs, 4), 'r2_dev': round(best_r2_dev, 4),
            'pa_r2': round(pa_r2, 4), 'target': 'deviation',
            'calibration_2f': {'c0': c0, 'c1': c1, 'c2': c2},
            'calibration_daughter': {'d0': d0, 'd1': d1, 'd2': d2, 'var_expansion': var_expansion},
        }

    all_results.append({
        'Trait': trait, 'N_BTB': n_btb, 'N_Daughters': n_dau, 'N_Total': n_total,
        'Features': len(feature_cols), 'Best_Single': best_name,
        'Best_R2_dev': round(best_r2_dev, 4), 'Stack_R2_dev': round(stack_r2_dev, 4),
        'PA_R2': round(pa_r2, 4), 'Raw_R2_abs': round(raw_r2_abs, 4),
        'Calibrated_R2_abs': round(final_r2_abs, 4),
        'c0': round(c0, 4), 'c1': round(c1, 4), 'c2': round(c2, 4),
        'd0': round(d0, 4), 'd1': round(d1, 4), 'd2': round(d2, 4),
        'var_expansion': round(var_expansion, 4), 'R2_daughter': round(r2_dau, 4),
        'Winner': winner, 'Time_s': round(time.time() - t0, 1),
    })

    # === INCREMENTAL SAVE ===
    with open(OUTPUT_DIR / "v12_models.pkl", 'wb') as f:
        pickle.dump(saved_models, f)
    with open(OUTPUT_DIR / "v12_sire_profiles.pkl", 'wb') as f:
        pickle.dump(sire_profiles, f)
    with open(OUTPUT_DIR / "v12_mgs_profiles.pkl", 'wb') as f:
        pickle.dump(mgs_profiles, f)
    with open(OUTPUT_DIR / "v12_calibration.pkl", 'wb') as f:
        pickle.dump(calibration_params, f)
    pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "v12_training_results.csv", index=False)

# ============================================================
# PHASE 4: HYBRID 3-WAY FOR WEAK TRAITS (with calibration)
# ============================================================
# Only traits with R2 < 0.60 benefit from hybrid 3-way
HYBRID_TRAITS = ['MAST', 'SCS', 'HCR', 'LIV']

flush_print(f"\n{'='*95}")
flush_print(f"  PHASE 4: Hybrid 3-way for {len(HYBRID_TRAITS)} traits (with calibration)")
flush_print(f"{'='*95}")

def prepare_combined_data(trait):
    feat_rows, y_vals, pa_vals, groups, weights = [], [], [], [], []
    # BTB
    for rec in btb_records:
        dv = rec['son_ptas'].get(trait)
        if dv is None: continue
        feat = build_features_v12(rec['sire_ptas'], rec['mgs_ptas'], trait,
                                   rec['sire_naab'], rec.get('mgs_naab'), rec.get('mmgs_ptas'))
        if feat is None: continue
        pa = compute_pa(rec['sire_ptas'], rec['mgs_ptas'], rec.get('mmgs_ptas'), trait)
        feat_rows.append(feat)
        y_vals.append(dv)
        pa_vals.append(pa if pa is not None else 0)
        groups.append(rec['sire_naab'])
        yr = rec.get('dob_year')
        w = 0.5 + 0.5 * min((yr - 1990) / 30, 1.0) if yr and yr >= 1990 else 0.5
        w *= (rec.get('son_rel', 70) / 100.0)
        weights.append(w)
    # Daughters
    for rec in daughter_records:
        dv = rec['daughter_ptas'].get(trait)
        if dv is None: continue
        feat = build_features_v12(rec['sire_ptas'], rec['mgs_ptas'], trait,
                                   rec['sire_naab'], rec.get('mgs_naab'), rec.get('mmgs_ptas'))
        if feat is None: continue
        pa = compute_pa(rec['sire_ptas'], rec['mgs_ptas'], rec.get('mmgs_ptas'), trait)
        feat_rows.append(feat)
        y_vals.append(dv)
        pa_vals.append(pa if pa is not None else 0)
        groups.append(rec['sire_naab'])
        weights.append(1.0)

    feat_df = pd.DataFrame(feat_rows)
    feat_df = feat_df.dropna(axis=1, thresh=int(len(feat_df)*0.3))
    feature_cols = list(feat_df.columns)
    for c in feature_cols:
        if feat_df[c].isna().any(): feat_df[c] = feat_df[c].fillna(feat_df[c].median())
    X = feat_df.values.astype(np.float64)
    y = np.array(y_vals, dtype=np.float64)
    pa_arr = np.array(pa_vals, dtype=np.float64)
    return X, y, pa_arr, feature_cols, np.array(groups), np.array(weights)

def oof_tournament_v12(X, y_dev, groups, weights, h2=0.25):
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
                if mn in ('LGBM', 'XGB'):
                    m.fit(X[tr], y_dev[tr], sample_weight=weights[tr])
                else:
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

for trait in HYBRID_TRAITS:
    if trait not in saved_models: continue
    # Skip if hybrid already applied (resume)
    if saved_models[trait].get('hybrid') == 'hybrid_3way':
        flush_print(f"    {trait:>6}: SKIP (hybrid already applied)")
        continue
    h2 = HERITABILITY.get(trait, 0.15)
    X, y, pa_arr, cols_t, groups_t, weights_t = prepare_combined_data(trait)
    N = len(y)
    if N < 100: continue
    y_dev = y - pa_arr

    # V12 OOF predictions (deviation)
    v12_oof, v12_winner, v12_all_preds, v12_top3, splits = oof_tournament_v12(X, y_dev, groups_t, weights_t, h2)

    # Sign classification
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

    # Magnitude regression
    y_abs = np.abs(y)
    mag_oof, mag_winner, _, _, _ = oof_tournament_v12(X, y_abs, groups_t, weights_t, h2)
    mag_oof = np.maximum(mag_oof, 0)

    pred_sign_hard = np.where(sign_preds_oof >= 0.5, 1.0, -1.0)
    sign_confidence = np.abs(sign_preds_oof - 0.5) * 2

    preds_2stage = np.zeros(N)
    for i in range(N):
        sn = groups_t[i]
        n_d = sire_profiles.get(sn, {}).get(trait, {}).get('n', 0)
        prof_conf = min(n_d / 30, 1.0) if n_d > 0 else 0.2
        total_conf = 0.6 * sign_confidence[i] + 0.4 * prof_conf
        preds_2stage[i] = pred_sign_hard[i] * mag_oof[i] * total_conf

    # Profile-anchored
    preds_anchored = np.zeros(N)
    for i in range(N):
        sn = groups_t[i]
        sp = sire_profiles.get(sn, {})
        tp = sp.get(trait)
        # Get MGS val from the combined records
        if i < len(btb_records):
            rec = btb_records[i] if i < len(btb_records) else None
        else:
            rec = daughter_records[i - len(btb_records)] if (i - len(btb_records)) < len(daughter_records) else None
        mgs_val = 0
        if tp and tp['n'] >= 5:
            preds_anchored[i] = tp['mean'] * 0.85 + mgs_val * 0.15
        elif tp and tp['n'] >= 2:
            preds_anchored[i] = tp['mean'] * 0.6 + pa_arr[i] * 0.4
        else:
            preds_anchored[i] = pa_arr[i]

    # Blend: V12 (PA+calibrated_dev), 2-stage, anchored, PA-only
    # Apply daughter calibration to v12_oof before blending
    cal_dau = saved_models[trait].get('calibration_daughter')
    cal2f = saved_models[trait].get('calibration_2f')
    if cal_dau:
        d0, d1, d2 = cal_dau['d0'], cal_dau['d1'], cal_dau['d2']
        var_exp = cal_dau.get('var_expansion', 1.0)
        v12_raw = d0 + d1 * pa_arr + d2 * v12_oof
        v12_abs = pa_arr + (v12_raw - pa_arr) * var_exp
    elif cal2f:
        v12_abs = cal2f['c0'] + cal2f['c1'] * pa_arr + cal2f['c2'] * v12_oof
    else:
        v12_abs = pa_arr + v12_oof

    blend_X = np.column_stack([v12_abs, preds_2stage, preds_anchored, pa_arr])
    preds_hybrid = np.zeros(N)
    for tr, te in splits:
        ridge = Ridge(alpha=1.0)
        ridge.fit(blend_X[tr], y[tr])
        preds_hybrid[te] = ridge.predict(blend_X[te])

    r2_v12 = r2_score(y, v12_abs)
    r2_hyb = r2_score(y, preds_hybrid)
    r2_pa = r2_score(y, pa_arr)

    flush_print(f"    {trait:>6}: PA={r2_pa:.4f} V12={r2_v12:.4f} Hybrid={r2_hyb:.4f} "
                f"{'IMPROVED' if r2_hyb > r2_v12 else 'no change'}")

    if r2_hyb > r2_v12:
        final_clf = None
        if n_classes >= 2 and best_clf_name != 'SKIP':
            final_clf = get_clf_models()[best_clf_name]
            final_clf.fit(X, y_binary)

        final_mag = HistGradientBoostingRegressor(max_iter=200, max_depth=5, learning_rate=0.05,
                                                   min_samples_leaf=8, random_state=42)
        final_mag.fit(X, y_abs)

        blend_ridge = Ridge(alpha=1.0)
        blend_ridge.fit(blend_X, y)

        saved_models[trait]['hybrid'] = 'hybrid_3way'
        saved_models[trait]['sign_clf'] = final_clf
        saved_models[trait]['sign_clf_name'] = best_clf_name
        saved_models[trait]['mag_model'] = final_mag
        saved_models[trait]['blend_ridge'] = blend_ridge
        saved_models[trait]['r2_cv'] = round(r2_hyb, 4)

    # === INCREMENTAL SAVE ===
    with open(OUTPUT_DIR / "v12_models.pkl", 'wb') as f:
        pickle.dump(saved_models, f)

# ============================================================
# PREDICTION FUNCTION — V12
# ============================================================
def predict_animal_v12(sire_ptas, mgs_ptas, sire_naab=None, mgs_naab=None, mmgs_ptas=None):
    results = {}

    for trait in ALL_TRAITS:
        info = saved_models.get(trait)
        if not info: continue

        feat = build_features_v12(sire_ptas, mgs_ptas, trait, sire_naab, mgs_naab, mmgs_ptas)
        if feat is None: continue

        pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)
        if pa is None: pa = 0

        feat_df = pd.DataFrame([feat])
        for c in info['feature_cols']:
            if c not in feat_df.columns: feat_df[c] = 0
        feat_df = feat_df[info['feature_cols']]
        feat_df = feat_df.fillna(0)
        X = feat_df.values.astype(np.float64)

        # Base prediction (deviation from PA)
        if info['type'] == 'stack':
            base_preds = np.column_stack([info['base_models'][mn].predict(X) for mn in info['base_names']])
            dev_pred = info['meta_model'].predict(base_preds)[0]
        else:
            dev_pred = info['model'].predict(X)[0]

        # Apply daughter-specific calibration (preferred for new predictions)
        cal_dau = info.get('calibration_daughter')
        cal2f = info.get('calibration_2f')
        if cal_dau:
            d0, d1, d2 = cal_dau['d0'], cal_dau['d1'], cal_dau['d2']
            var_exp = cal_dau.get('var_expansion', 1.0)
            pred_raw = d0 + d1 * pa + d2 * dev_pred
            # Expand deviation from PA to match expected genetic variance
            pred_v12 = pa + (pred_raw - pa) * var_exp
        elif cal2f:
            c0, c1, c2 = cal2f['c0'], cal2f['c1'], cal2f['c2']
            pred_v12 = c0 + c1 * pa + c2 * dev_pred
        else:
            pred_v12 = pa + dev_pred

        # Hybrid 3-way
        if info.get('hybrid') == 'hybrid_3way':
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

            blend = np.array([[pred_v12, pred_2stage, pred_anc, pa]])
            pred_final = info['blend_ridge'].predict(blend)[0]
        else:
            pred_final = pred_v12

        results[trait] = round(pred_final, 4)

    # FI calculated from formula (CDCB 2024): FI = 0.4*DPR + 0.4*CCR + 0.1*HCR + 0.1*EFC
    dpr_val = results.get('DPR')
    ccr_val = results.get('CCR')
    hcr_val = results.get('HCR')
    efc_val = results.get('EFC')
    if dpr_val is not None and ccr_val is not None and hcr_val is not None:
        efc = efc_val if efc_val is not None else 0
        results['FI'] = round(0.4 * dpr_val + 0.4 * ccr_val + 0.1 * hcr_val + 0.1 * efc, 4)

    return results

# ============================================================
# SUMMARY
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  V12 ENGINE — TRAINING SUMMARY")
flush_print(f"{'='*95}")
flush_print(f"  Bull-to-Bull records: {len(btb_records)}")
flush_print(f"  Daughter records: {len(daughter_records)}")
flush_print(f"  Total training records: {len(btb_records) + len(daughter_records)}")
flush_print(f"  Sire profiles: {len(sire_profiles)} (enhanced)")
flush_print(f"  MGS profiles: {len(mgs_profiles)} (enhanced)")
flush_print(f"  Traits trained: {len(saved_models)}")

if all_results:
    flush_print(f"\n  {'Trait':>6} | {'N_Tot':>6} | {'R2_abs':>7} | {'PA_R2':>6} | {'Gain':>6} | {'d1(PA)':>6} | {'d2(ML)':>6} | {'VarExp':>6} | {'R2_dau':>7} | Winner")
    flush_print(f"  {'-'*100}")
    for r in all_results:
        trait = r['Trait']
        info = saved_models.get(trait, {})
        final_r2 = info.get('r2_cv', r['Calibrated_R2_abs'])
        pa_r2 = r['PA_R2']
        gain = final_r2 - pa_r2
        d1_val = r.get('d1', r.get('c1', 1.0))
        d2_val = r.get('d2', r.get('c2', 1.0))
        vx = r.get('var_expansion', 1.0)
        r2_dau = r.get('R2_daughter', -999)
        r2_dau_str = f"{r2_dau:>7.4f}" if r2_dau > -900 else "    N/A"
        winner = r['Winner']
        hyb = info.get('hybrid', '')
        if hyb: winner = f"{winner}+HYB"
        flush_print(f"  {trait:>6} | {r['N_Total']:>6} | {final_r2:>7.4f} | {pa_r2:>6.4f} | "
                    f"{'+' if gain > 0 else ''}{gain:>5.4f} | {d1_val:>5.3f} | {d2_val:>5.3f} | {vx:>6.2f} | {r2_dau_str} | {winner}")

# Final save
with open(OUTPUT_DIR / "v12_models.pkl", 'wb') as f:
    pickle.dump(saved_models, f)
with open(OUTPUT_DIR / "v12_sire_profiles.pkl", 'wb') as f:
    pickle.dump(sire_profiles, f)
with open(OUTPUT_DIR / "v12_mgs_profiles.pkl", 'wb') as f:
    pickle.dump(mgs_profiles, f)
with open(OUTPUT_DIR / "v12_calibration.pkl", 'wb') as f:
    pickle.dump(calibration_params, f)
pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "v12_training_results.csv", index=False)

flush_print(f"\n  All models saved to {OUTPUT_DIR}")
flush_print(f"  V12 engine complete!")
