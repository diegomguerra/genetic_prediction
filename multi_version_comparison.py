"""
DSII Multi-Version Location Accuracy Comparison
Compara PA, V10, V11, V12-OLD, V12-2F, V12-DAU contra genomica real (Jose, 92 animais)
Analisa localizacao em grupos de 5%, 10%, 20%
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pickle, csv, re, sys
from pathlib import Path
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r"C:\Users\DiegoGuerra\Documents\Projetos\genetic_prediction")
DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")

TRAITS = ['TPI', 'NM$', 'MILK', 'FAT', 'PRO', 'FAT%', 'PRO%', 'PL', 'DPR', 'HCR']

# ============================================================
# SHARED BULL INFRASTRUCTURE
# ============================================================
STUD_MAP = {
    '507':'7','509':'9','550':'50','551':'51','559':'59','518':'18','521':'21',
    '523':'23','581':'81','585':'85','501':'1','601':'1','604':'4','614':'14',
    '629':'29','751':'151','752':'152','814':'14','250':'200',
    '001':'1','007':'7','009':'9','011':'11','029':'29','097':'97',
    '100':'100','200':'200','777':'777','796':'796','745':'745',
}
GENETIC_SD = {
    'TPI':250,'MILK':675,'FAT':29,'FAT%':0.05,'PRO':19,'PRO%':0.02,'CFP':40,
    'PL':1.85,'LIV':1.2,'SCS':0.14,'MAST':1.0,'MET':1.0,'RP':0.5,'DA':0.5,'KET':0.8,'MF':0.3,
    'DPR':1.3,'HCR':1.5,'CCR':1.65,'SCE':0.8,'DCE':0.8,'SSB':1.0,'DSB':1.0,
    'FI':1.0,'GFI':3.0,'PTAT':0.70,'UDC':0.75,'FLC':0.65,
    'STA':0.80,'STR':0.70,'DFM':0.80,'BOD':0.60,'RUA':0.60,'RW':0.70,
    'RLS':0.50,'RLR':0.50,'FTA':0.60,'FLS':0.50,'FTL':0.50,
    'FUA':0.80,'RUH':0.80,'RUW':0.80,'UCL':0.60,'UDP':0.80,'FTP':0.70,'RTP':0.60,
    'BWC':0.50,'NM$':200,'CM$':200,'H_LIV':1.0,'F_SAV':50,'RFI':30,'GL':0.8,'EFC':1.0,
}
HERITABILITY = {
    'TPI':0.30,'MILK':0.25,'FAT':0.25,'FAT%':0.50,'PRO':0.25,'PRO%':0.50,'CFP':0.25,
    'PL':0.08,'LIV':0.05,'SCS':0.12,'MAST':0.04,'MET':0.03,'RP':0.02,'DA':0.04,'KET':0.04,'MF':0.02,
    'DPR':0.04,'HCR':0.04,'CCR':0.04,'SCE':0.08,'DCE':0.06,'SSB':0.05,'DSB':0.04,
    'FI':0.06,'GFI':0.15,'PTAT':0.30,'UDC':0.25,'FLC':0.15,
    'STA':0.42,'STR':0.27,'DFM':0.25,'BOD':0.30,'RUA':0.30,'RW':0.25,
    'RLS':0.15,'RLR':0.10,'FTA':0.10,'FLS':0.10,'FTL':0.10,
    'FUA':0.25,'RUH':0.20,'RUW':0.20,'UCL':0.15,'UDP':0.30,'FTP':0.25,'RTP':0.20,
    'BWC':0.20,'NM$':0.30,'CM$':0.30,'H_LIV':0.05,'F_SAV':0.15,'RFI':0.15,'GL':0.40,'EFC':0.06,
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
    ('GL','SCE'):-0.30,('GL','SSB'):-0.25,('CCR','PL'):0.35,('HCR','PL'):0.30,
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
CORE_TRAITS = ['TPI','MILK','FAT','FAT%','PRO','PRO%','PL','SCS','DPR','LIV','FI','CCR',
               'PTAT','UDC','FLC','MAST','NM$','HCR','SCE','F_SAV']
V10_ALL_TRAITS = list(BULL_COL.keys())

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
print(f"  bulls.csv: {len(bulls)} touros")

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
    print(f"  CDCB: {len(cdcb_bulls)} touros")
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
        naab, row = matches[0]
        return row, 'CDCB', naab, get_bull_ptas(row, 'CDCB')
    return None, None, None, {}

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

# ============================================================
# LOAD ALL MODEL VERSIONS
# ============================================================
print("\nCarregando modelos...")

# V10
with open(BASE / 'dsii_v10_results/v10_models.pkl', 'rb') as f: v10_models = pickle.load(f)
with open(BASE / 'dsii_v10_results/v10_sire_profiles.pkl', 'rb') as f: v10_sire_profiles = pickle.load(f)
with open(BASE / 'dsii_v10_results/v10_mgs_profiles.pkl', 'rb') as f: v10_mgs_profiles = pickle.load(f)
print(f"  V10: {len(v10_models)} traits")

# V11
with open(BASE / 'dsii_v11_results/v11_models.pkl', 'rb') as f: v11_models = pickle.load(f)
with open(BASE / 'dsii_v11_results/v11_sire_profiles.pkl', 'rb') as f: v11_sire_profiles = pickle.load(f)
with open(BASE / 'dsii_v11_results/v11_mgs_profiles.pkl', 'rb') as f: v11_mgs_profiles = pickle.load(f)
print(f"  V11: {len(v11_models)} traits")

# V12 (current pkl has both cal_2f and cal_dau)
with open(BASE / 'dsii_v12_results/v12_models.pkl', 'rb') as f: v12_models = pickle.load(f)
with open(BASE / 'dsii_v12_results/v12_sire_profiles.pkl', 'rb') as f: v12_sire_profiles = pickle.load(f)
with open(BASE / 'dsii_v12_results/v12_mgs_profiles.pkl', 'rb') as f: v12_mgs_profiles = pickle.load(f)
print(f"  V12: {len(v12_models)} traits")

# ============================================================
# FEATURE BUILDERS
# ============================================================
def build_features_v10(sire_ptas, mgs_ptas, trait, sire_naab=None, mgs_naab=None, mmgs_ptas=None):
    """V10: absolute PTA prediction, no PA features"""
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
        f['sire_x_mgs'] = s*mg; f['sire_mgs_diff'] = s-mg
        f['sire_mgs_ratio'] = s/mg if mg != 0 else 0
    else:
        f['sire_x_mgs'] = 0; f['sire_mgs_diff'] = 0; f['sire_mgs_ratio'] = 0
    mmg = mmgs_ptas.get(trait) if mmgs_ptas else None
    f['mmgs_val'] = mmg if mmg is not None else 0
    f['has_mmgs'] = 1 if mmg is not None else 0
    f['sire_x_mmgs'] = s*mmg if mmg is not None else 0
    # V10 uses V10_ALL_TRAITS for cross-traits (broader than CORE)
    v10_traits = list({k for k in BULL_COL.keys() if k in v10_models})
    for ot in v10_traits:
        if ot == trait: continue
        sv = sire_ptas.get(ot)
        if sv is not None: f[f'sire_{ot}'] = sv
        mv = mgs_ptas.get(ot) if mgs_ptas else None
        if mv is not None: f[f'mgs_{ot}'] = mv
    for (t1,t2),corr in GENETIC_CORRELATIONS.items():
        if t1==trait or t2==trait:
            other = t2 if t1==trait else t1
            sv = sire_ptas.get(other)
            if sv is not None: f[f'gc_sire_{other}'] = sv*corr
            mv = mgs_ptas.get(other) if mgs_ptas else None
            if mv is not None: f[f'gc_mgs_{other}'] = mv*corr
    if sire_naab and sire_naab in v10_sire_profiles:
        sp = v10_sire_profiles[sire_naab]; tp = sp.get(trait)
        if tp and tp['n'] >= 2:
            f['sire_prof_n']=tp['n']; f['sire_prof_mean']=tp['mean']
            f['sire_prof_std']=tp['std']; f['sire_prof_ratio']=tp['trans_ratio']
            f['sire_prof_resid']=tp['residual']
        else:
            f['sire_prof_n']=0; f['sire_prof_mean']=0; f['sire_prof_std']=0
            f['sire_prof_ratio']=1.0; f['sire_prof_resid']=0
        for ot in v10_traits:
            if ot==trait: continue
            otp = sp.get(ot)
            if otp and otp['n']>=2: f[f'sire_prof_{ot}'] = otp['residual']
    else:
        f['sire_prof_n']=0; f['sire_prof_mean']=0; f['sire_prof_std']=0
        f['sire_prof_ratio']=1.0; f['sire_prof_resid']=0
    if mgs_naab and mgs_naab in v10_mgs_profiles:
        mp = v10_mgs_profiles[mgs_naab]; tp = mp.get(trait)
        if tp and tp['n']>=2:
            f['mgs_prof_n']=tp['n']; f['mgs_prof_mean']=tp['mean']; f['mgs_prof_resid']=tp['residual']
        else:
            f['mgs_prof_n']=0; f['mgs_prof_mean']=0; f['mgs_prof_resid']=0
    else:
        f['mgs_prof_n']=0; f['mgs_prof_mean']=0; f['mgs_prof_resid']=0
    return f

def build_features_v11(sire_ptas, mgs_ptas, trait, sire_naab=None, mgs_naab=None, mmgs_ptas=None):
    """V11: PA-deviation, no z-scored cross-traits"""
    s = sire_ptas.get(trait)
    if s is None: return None
    f = {}
    sd = GENETIC_SD.get(trait, 1); h2 = HERITABILITY.get(trait, 0.15)
    pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)
    f['pa'] = pa; f['pa_z'] = pa / sd if sd else 0
    f['sire'] = s; f['sire_z'] = s / sd if sd else 0; f['sire_sq'] = s * s; f['sire_h2'] = s * h2
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
        f['mendelian_range'] = abs(s - mg); f['mendelian_range_z'] = abs(s - mg) / sd if sd else 0
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
    if sire_naab and sire_naab in v11_sire_profiles:
        sp = v11_sire_profiles[sire_naab]; tp = sp.get(trait)
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
    if mgs_naab and mgs_naab in v11_mgs_profiles:
        mp = v11_mgs_profiles[mgs_naab]; tp = mp.get(trait)
        if tp and tp['n'] >= 2:
            f['mgs_prof_n'] = tp['n']; f['mgs_prof_mean'] = tp['mean']; f['mgs_prof_resid'] = tp['residual']
        else:
            f['mgs_prof_n'] = 0; f['mgs_prof_mean'] = 0; f['mgs_prof_resid'] = 0
    else:
        f['mgs_prof_n'] = 0; f['mgs_prof_mean'] = 0; f['mgs_prof_resid'] = 0
    return f

def build_features_v12(sire_ptas, mgs_ptas, trait, sire_naab=None, mgs_naab=None, mmgs_ptas=None):
    """V12: PA-deviation, z-scored cross-traits"""
    s = sire_ptas.get(trait)
    if s is None: return None
    f = {}
    sd = GENETIC_SD.get(trait, 1); h2 = HERITABILITY.get(trait, 0.15)
    pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)
    f['pa'] = pa; f['pa_z'] = pa / sd if sd else 0
    f['sire'] = s; f['sire_z'] = s / sd if sd else 0; f['sire_sq'] = s * s; f['sire_h2'] = s * h2
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
        f['mendelian_range'] = abs(s - mg); f['mendelian_range_z'] = abs(s - mg) / sd if sd else 0
    else:
        f['sire_x_mgs'] = 0; f['sire_mgs_diff'] = 0; f['sire_mgs_ratio'] = 0
        f['sire_mgs_mean'] = s / 2; f['mendelian_range'] = 0; f['mendelian_range_z'] = 0
    mmg = mmgs_ptas.get(trait) if mmgs_ptas else None
    f['mmgs_val'] = mmg if mmg is not None else 0
    f['has_mmgs'] = 1 if mmg is not None else 0
    f['sire_x_mmgs'] = s * mmg if mmg is not None else 0
    for ot in CORE_TRAITS:
        if ot == trait: continue
        ot_sd = GENETIC_SD.get(ot, 1)
        sv = sire_ptas.get(ot)
        if sv is not None:
            f[f'sire_{ot}'] = sv; f[f'sire_{ot}_z'] = sv / ot_sd if ot_sd else 0
        mv = mgs_ptas.get(ot) if mgs_ptas else None
        if mv is not None:
            f[f'mgs_{ot}'] = mv; f[f'mgs_{ot}_z'] = mv / ot_sd if ot_sd else 0
        ot_pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, ot)
        if ot_pa is not None:
            f[f'pa_{ot}'] = ot_pa; f[f'pa_{ot}_z'] = ot_pa / ot_sd if ot_sd else 0
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
    if sire_naab and sire_naab in v12_sire_profiles:
        sp = v12_sire_profiles[sire_naab]; tp = sp.get(trait)
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
    if mgs_naab and mgs_naab in v12_mgs_profiles:
        mp = v12_mgs_profiles[mgs_naab]; tp = mp.get(trait)
        if tp and tp['n'] >= 2:
            f['mgs_prof_n'] = tp['n']; f['mgs_prof_mean'] = tp['mean']; f['mgs_prof_resid'] = tp['residual']
        else:
            f['mgs_prof_n'] = 0; f['mgs_prof_mean'] = 0; f['mgs_prof_resid'] = 0
    else:
        f['mgs_prof_n'] = 0; f['mgs_prof_mean'] = 0; f['mgs_prof_resid'] = 0
    return f

# ============================================================
# PREDICT FUNCTIONS
# ============================================================
def predict_v10_single(feat, trait):
    if trait not in v10_models: return None
    mi = v10_models[trait]
    X = np.array([[feat.get(c, 0) for c in mi['feature_cols']]])
    if mi['type'] == 'single':
        return mi['model'].predict(X)[0]
    elif mi['type'] == 'stack':
        base_preds = [mi['base_models'][mn].predict(X)[0] for mn in mi['base_names']]
        return mi['meta_model'].predict(np.array([base_preds]))[0]
    return None

def predict_v11_single(feat, trait, pa_val):
    if trait not in v11_models: return None
    info = v11_models[trait]
    X = np.array([[feat.get(c, 0) for c in info['feature_cols']]])
    if info['type'] == 'stack':
        base_preds = np.column_stack([info['base_models'][mn].predict(X) for mn in info['base_names']])
        dev = info['meta_model'].predict(base_preds)[0]
    elif info['type'] == 'single':
        dev = info['model'].predict(X)[0]
    else:
        return None
    return pa_val + dev

def predict_v12_variants(feat, trait, pa_val):
    """Returns dict with 3 V12 variants: OLD (pa+dev), 2F (calibrated), DAU (daughter cal)"""
    if trait not in v12_models: return {}
    info = v12_models[trait]
    X = np.array([[feat.get(c, 0) for c in info['feature_cols']]])
    # Stacking prediction
    if info['type'] == 'stack':
        base_preds = np.column_stack([info['base_models'][mn].predict(X) for mn in info['base_names']])
        dev = info['meta_model'].predict(base_preds)[0]
    elif info['type'] == 'single':
        dev = info['model'].predict(X)[0]
    else:
        return {}

    results = {}
    # V12-OLD: just PA + dev
    results['V12_OLD'] = pa_val + dev
    # V12-2F: 2-factor calibration
    cal2f = info.get('calibration_2f')
    if cal2f:
        c0, c1, c2 = cal2f['c0'], cal2f['c1'], cal2f['c2']
        results['V12_2F'] = c0 + c1 * pa_val + c2 * dev
    # V12-DAU: daughter calibration + variance expansion
    cal_dau = info.get('calibration_daughter')
    if cal_dau:
        d0, d1, d2 = cal_dau['d0'], cal_dau['d1'], cal_dau['d2']
        var_exp = cal_dau.get('var_expansion', 1.0)
        pred_raw = d0 + d1 * pa_val + d2 * dev
        results['V12_DAU'] = pa_val + (pred_raw - pa_val) * var_exp
    return results

# ============================================================
# LOAD JOSE INPUT + GENOMIC
# ============================================================
print("\nCarregando dados de Jose...")
df = pd.read_excel(DOWNLOADS / 'Cliente José Gonçalvez.xlsx', engine='openpyxl')
cols = list(df.columns)
id_col, pai_col, avo_col, bis_col = cols[0], cols[1], cols[2], cols[3]
print(f"  Input: {len(df)} animais")

genomic = pd.read_csv(BASE / 'jose_genomic_results.csv')
gen_ids = set(genomic['ID'].values)
print(f"  Genomic: {len(genomic)} animais")

# ============================================================
# GENERATE ALL PREDICTIONS
# ============================================================
print("\nGerando predicoes multi-versao...")

all_predictions = []  # list of dicts: {ID, trait, PA, V10, V11, V12_OLD, V12_2F, V12_DAU, GENOMIC}

for _, row in df.iterrows():
    animal_id = int(row[id_col]) if not pd.isna(row[id_col]) else None
    if animal_id is None or animal_id not in gen_ids:
        continue

    sire_naab = str(row[pai_col]).strip() if not pd.isna(row[pai_col]) else ''
    avo_naab = str(row[avo_col]).strip() if not pd.isna(row[avo_col]) else ''
    bis_naab = str(row[bis_col]).strip() if not pd.isna(row[bis_col]) else ''

    _, _, sire_resolved, sire_ptas = lookup_bull(sire_naab)
    _, _, avo_resolved, mgs_ptas = lookup_bull(avo_naab)
    _, _, bis_resolved, mmgs_ptas = lookup_bull(bis_naab)

    if not sire_ptas:
        continue

    gen_row = genomic[genomic['ID'] == animal_id].iloc[0]

    for trait in TRAITS:
        gen_val = gen_row.get(trait)
        if pd.isna(gen_val): continue

        pa_val = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)

        rec = {'ID': animal_id, 'Trait': trait, 'GENOMIC': float(gen_val), 'PA': pa_val}

        # V10
        feat_v10 = build_features_v10(sire_ptas, mgs_ptas, trait, sire_resolved, avo_resolved, mmgs_ptas)
        if feat_v10:
            rec['V10'] = predict_v10_single(feat_v10, trait)

        # V11
        feat_v11 = build_features_v11(sire_ptas, mgs_ptas, trait, sire_resolved, avo_resolved, mmgs_ptas)
        if feat_v11 and pa_val is not None:
            rec['V11'] = predict_v11_single(feat_v11, trait, pa_val)

        # V12 variants
        feat_v12 = build_features_v12(sire_ptas, mgs_ptas, trait, sire_resolved, avo_resolved, mmgs_ptas)
        if feat_v12 and pa_val is not None:
            v12_res = predict_v12_variants(feat_v12, trait, pa_val)
            rec.update(v12_res)

        all_predictions.append(rec)

pred_df = pd.DataFrame(all_predictions)
pred_df.to_pickle(BASE / 'dsii_v12_results' / 'multi_version_predictions.pkl')
print(f"  Total predicoes: {len(pred_df)} (saved to pkl)")
print(f"  Animais: {pred_df['ID'].nunique()}")
print(f"  Traits com predicao por versao:")
VERSIONS = ['PA', 'V10', 'V11', 'V12_OLD', 'V12_2F', 'V12_DAU']
for v in VERSIONS:
    if v in pred_df.columns:
        n = pred_df[v].notna().sum()
        print(f"    {v}: {n}")

# ============================================================
# LOCATION ACCURACY ANALYSIS
# ============================================================

def percentile_group(series, n_groups):
    """Assign items to percentile groups (1=worst, n_groups=best)"""
    return pd.qcut(series.rank(method='first'), q=n_groups, labels=False) + 1

def location_accuracy(gen_vals, pred_vals, n_groups):
    """Compute location accuracy metrics for given number of groups"""
    n = len(gen_vals)
    if n < n_groups * 2:
        return None

    gen_g = percentile_group(pd.Series(gen_vals), n_groups).values
    pred_g = percentile_group(pd.Series(pred_vals), n_groups).values

    exact = (gen_g == pred_g).mean() * 100
    adj1 = (np.abs(gen_g - pred_g) <= 1).mean() * 100

    # Top group accuracy
    gen_top = set(np.where(gen_g == n_groups)[0])
    pred_top = set(np.where(pred_g == n_groups)[0])
    top_acc = len(gen_top & pred_top) / len(gen_top) * 100 if gen_top else 0

    # Bottom group accuracy
    gen_bot = set(np.where(gen_g == 1)[0])
    pred_bot = set(np.where(pred_g == 1)[0])
    bot_acc = len(gen_bot & pred_bot) / len(gen_bot) * 100 if gen_bot else 0

    random_exact = 100.0 / n_groups

    return {
        'exact': exact, 'adj1': adj1,
        'top': top_acc, 'bottom': bot_acc,
        'random': random_exact,
        'gain': exact - random_exact,
    }

# Granularities to analyze
GRANULARITIES = {
    'Quintil (20%)': 5,
    'Decil (10%)': 10,
    'Vigintil (5%)': 20,
}

print("\n" + "=" * 120)
print("  ANALISE MULTI-VERSAO DE ACURACIA DE LOCALIZACAO")
print("  Jose - 92 animais com genomica real")
print("=" * 120)

# Per-trait, per-version, per-granularity analysis
all_results = []

for trait in TRAITS:
    trait_df = pred_df[pred_df['Trait'] == trait].copy()
    gen_vals = trait_df['GENOMIC'].values

    for gran_name, n_groups in GRANULARITIES.items():
        for version in VERSIONS:
            if version not in trait_df.columns:
                continue
            valid = trait_df[version].notna()
            if valid.sum() < n_groups * 2:
                continue

            pred_vals = trait_df.loc[valid, version].values.astype(float)
            gen_v = trait_df.loc[valid, 'GENOMIC'].values.astype(float)

            acc = location_accuracy(gen_v, pred_vals, n_groups)
            if acc is None: continue

            rho, _ = spearmanr(pred_vals, gen_v)

            all_results.append({
                'Trait': trait, 'Granularity': gran_name, 'N_Groups': n_groups,
                'Version': version, 'N': len(pred_vals),
                'Exact%': acc['exact'], 'Adj1%': acc['adj1'],
                'Top%': acc['top'], 'Bottom%': acc['bottom'],
                'Random%': acc['random'], 'Gain_pp': acc['gain'],
                'Spearman': rho,
            })

results_df = pd.DataFrame(all_results)

# ============================================================
# PRINT: RESUMO POR GRANULARIDADE
# ============================================================
for gran_name, n_groups in GRANULARITIES.items():
    gdf = results_df[results_df['N_Groups'] == n_groups]
    if gdf.empty: continue

    print(f"\n{'='*120}")
    print(f"  {gran_name.upper()} ({100//n_groups}% por grupo, {n_groups} grupos)")
    print(f"{'='*120}")

    # Table: trait x version
    print(f"\n  ACERTO EXATO (aleatorio = {100/n_groups:.0f}%)")
    print(f"  {'Trait':>6}", end='')
    for v in VERSIONS:
        if v in gdf['Version'].values:
            print(f" | {v:>8}", end='')
    print(f" | {'MELHOR':>8}")
    print("  " + "-" * (10 + 11 * len([v for v in VERSIONS if v in gdf['Version'].values]) + 11))

    trait_bests = {}
    for trait in TRAITS:
        tdf = gdf[gdf['Trait'] == trait]
        if tdf.empty: continue
        print(f"  {trait:>6}", end='')
        best_v = ''; best_val = -999
        for v in VERSIONS:
            row = tdf[tdf['Version'] == v]
            if not row.empty:
                val = row.iloc[0]['Exact%']
                print(f" | {val:>7.1f}%", end='')
                if val > best_val:
                    best_val = val; best_v = v
            else:
                print(f" | {'---':>8}", end='')
        print(f" | {best_v:>8}")
        trait_bests[trait] = best_v

    # Averages
    print("  " + "-" * (10 + 11 * len([v for v in VERSIONS if v in gdf['Version'].values]) + 11))
    print(f"  {'MEDIA':>6}", end='')
    version_avgs = {}
    for v in VERSIONS:
        vdf = gdf[gdf['Version'] == v]
        if not vdf.empty:
            avg = vdf['Exact%'].mean()
            version_avgs[v] = avg
            print(f" | {avg:>7.1f}%", end='')
        else:
            print(f" | {'---':>8}", end='')
    best_avg_v = max(version_avgs, key=version_avgs.get) if version_avgs else ''
    print(f" | {best_avg_v:>8}")

    # Top group accuracy
    print(f"\n  ACERTO TOP {100//n_groups}%")
    print(f"  {'Trait':>6}", end='')
    for v in VERSIONS:
        if v in gdf['Version'].values:
            print(f" | {v:>8}", end='')
    print(f" | {'MELHOR':>8}")
    print("  " + "-" * (10 + 11 * len([v for v in VERSIONS if v in gdf['Version'].values]) + 11))
    for trait in TRAITS:
        tdf = gdf[gdf['Trait'] == trait]
        if tdf.empty: continue
        print(f"  {trait:>6}", end='')
        best_v = ''; best_val = -999
        for v in VERSIONS:
            row = tdf[tdf['Version'] == v]
            if not row.empty:
                val = row.iloc[0]['Top%']
                print(f" | {val:>7.1f}%", end='')
                if val > best_val:
                    best_val = val; best_v = v
            else:
                print(f" | {'---':>8}", end='')
        print(f" | {best_v:>8}")
    print("  " + "-" * (10 + 11 * len([v for v in VERSIONS if v in gdf['Version'].values]) + 11))
    print(f"  {'MEDIA':>6}", end='')
    for v in VERSIONS:
        vdf = gdf[gdf['Version'] == v]
        if not vdf.empty:
            print(f" | {vdf['Top%'].mean():>7.1f}%", end='')
        else:
            print(f" | {'---':>8}", end='')
    print()

# ============================================================
# WINNER SUMMARY
# ============================================================
print(f"\n{'='*120}")
print(f"  RESUMO: MELHOR VERSAO POR TRAIT E GRANULARIDADE (Acerto Exato)")
print(f"{'='*120}")
print(f"  {'Trait':>6}", end='')
for gran_name in GRANULARITIES:
    print(f" | {gran_name:>16}", end='')
print(f" | {'Spearman_Best':>14}")
print("  " + "-" * 70)

for trait in TRAITS:
    print(f"  {trait:>6}", end='')
    for gran_name, n_groups in GRANULARITIES.items():
        gdf = results_df[(results_df['Trait'] == trait) & (results_df['N_Groups'] == n_groups)]
        if gdf.empty:
            print(f" | {'---':>16}", end='')
            continue
        best_row = gdf.loc[gdf['Exact%'].idxmax()]
        print(f" | {best_row['Version']:>9}({best_row['Exact%']:4.0f}%)", end='')
    # Best Spearman overall
    tdf = results_df[(results_df['Trait'] == trait) & (results_df['N_Groups'] == 5)]
    if not tdf.empty:
        best_sp = tdf.loc[tdf['Spearman'].idxmax()]
        print(f" | {best_sp['Version']:>7}({best_sp['Spearman']:.3f})", end='')
    print()

# ============================================================
# OVERALL VERSION RANKING
# ============================================================
print(f"\n{'='*120}")
print(f"  RANKING GERAL DAS VERSOES (media de acerto exato em todas as traits)")
print(f"{'='*120}")
for gran_name, n_groups in GRANULARITIES.items():
    gdf = results_df[results_df['N_Groups'] == n_groups]
    print(f"\n  {gran_name}:")
    version_stats = []
    for v in VERSIONS:
        vdf = gdf[gdf['Version'] == v]
        if vdf.empty: continue
        avg_exact = vdf['Exact%'].mean()
        avg_top = vdf['Top%'].mean()
        avg_spear = vdf['Spearman'].mean()
        n_wins = 0
        for trait in TRAITS:
            tdf = gdf[gdf['Trait'] == trait]
            if tdf.empty: continue
            best = tdf.loc[tdf['Exact%'].idxmax()]
            if best['Version'] == v: n_wins += 1
        version_stats.append((v, avg_exact, avg_top, avg_spear, n_wins))
    version_stats.sort(key=lambda x: -x[1])
    print(f"    {'Versao':>10} | {'Exact%':>7} | {'Top%':>7} | {'Spearman':>8} | {'Wins':>5}")
    print(f"    " + "-" * 50)
    for v, ex, tp, sp, w in version_stats:
        medal = ' <-- MELHOR' if v == version_stats[0][0] else ''
        print(f"    {v:>10} | {ex:>6.1f}% | {tp:>6.1f}% | {sp:>8.3f} | {w:>5}{medal}")

# ============================================================
# EVOLUTION: como cada versao evoluiu
# ============================================================
print(f"\n{'='*120}")
print(f"  EVOLUCAO V10 -> V11 -> V12 (Quintil - acerto exato medio)")
print(f"{'='*120}")
evolution_versions = ['PA', 'V10', 'V11', 'V12_OLD', 'V12_2F', 'V12_DAU']
gdf = results_df[results_df['N_Groups'] == 5]
for v in evolution_versions:
    vdf = gdf[gdf['Version'] == v]
    if vdf.empty: continue
    avg = vdf['Exact%'].mean()
    bar = '#' * int(avg / 2)
    print(f"  {v:>10}: {avg:>5.1f}% {bar}")

print(f"\n  PA = baseline (Parent Average puro)")
print(f"  Ganho total (melhor V12 vs PA):", end='')
pa_avg = gdf[gdf['Version'] == 'PA']['Exact%'].mean() if 'PA' in gdf['Version'].values else 0
best_v12 = max(
    gdf[gdf['Version'].str.startswith('V12')].groupby('Version')['Exact%'].mean().items(),
    key=lambda x: x[1], default=(None, 0)
)
if best_v12[0]:
    print(f" {best_v12[1]-pa_avg:+.1f}pp ({best_v12[0]})")
else:
    print(" N/A")

print()
