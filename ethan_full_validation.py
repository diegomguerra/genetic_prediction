"""
VALIDACAO COMPLETA NO ETHAN (5624 animais USA com genomica real)
PA + todos os metodos ML por trait - Granularidade ate Vigintil 5%
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pickle, csv, re, sys
from pathlib import Path
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r"C:\Users\DiegoGuerra\Documents\Projetos\genetic_prediction")
DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")

# Traits to evaluate (map Ethan column names to standard)
ETHAN_COL_MAP = {
    'TPI': 'TPI', 'MILK': 'MILK', 'FAT': 'FAT', 'FAT %': 'FAT%',
    'PROT': 'PRO', 'PROT%': 'PRO%', 'PL': 'PL', 'DPR': 'DPR',
    'CCR': 'CCR', 'LIV': 'LIV', 'SCS': 'SCS', 'CDCB_MAST': 'MAST',
    'UDC': 'UDC', 'FLC': 'FLC', 'FI': 'FI',
}
TRAITS = ['TPI', 'MILK', 'FAT', 'FAT%', 'PRO', 'PRO%', 'PL', 'DPR',
          'CCR', 'LIV', 'SCS', 'MAST', 'UDC', 'FLC']

# ============================================================
# SHARED BULL INFRASTRUCTURE (same as multi_version)
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
        if c in bulls: return bulls[c], 'SS', c, get_bull_ptas(bulls[c], 'SS')
        if c in cdcb_bulls:
            if c in cdcb_to_ss:
                ss_naab = cdcb_to_ss[c]
                return bulls[ss_naab], 'SS', ss_naab, get_bull_ptas(bulls[ss_naab], 'SS')
            return cdcb_bulls[c], 'CDCB', c, get_bull_ptas(cdcb_bulls[c], 'CDCB')
    if not m_raw: return None, None, None, {}
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
        if mmg is not None: pa += mmg / 8
    return pa

# ============================================================
# LOAD ALL MODELS
# ============================================================
print("\nCarregando modelos...")
with open(BASE / 'dsii_v10_results/v10_models.pkl', 'rb') as f: v10_models = pickle.load(f)
with open(BASE / 'dsii_v10_results/v10_sire_profiles.pkl', 'rb') as f: v10_sp = pickle.load(f)
with open(BASE / 'dsii_v10_results/v10_mgs_profiles.pkl', 'rb') as f: v10_mp = pickle.load(f)
print(f"  V10: {len(v10_models)} traits")

with open(BASE / 'dsii_v11_results/v11_models.pkl', 'rb') as f: v11_models = pickle.load(f)
with open(BASE / 'dsii_v11_results/v11_sire_profiles.pkl', 'rb') as f: v11_sp = pickle.load(f)
with open(BASE / 'dsii_v11_results/v11_mgs_profiles.pkl', 'rb') as f: v11_mp = pickle.load(f)
print(f"  V11: {len(v11_models)} traits")

with open(BASE / 'dsii_v12_results/v12_models.pkl', 'rb') as f: v12_models = pickle.load(f)
with open(BASE / 'dsii_v12_results/v12_sire_profiles.pkl', 'rb') as f: v12_sp = pickle.load(f)
with open(BASE / 'dsii_v12_results/v12_mgs_profiles.pkl', 'rb') as f: v12_mp = pickle.load(f)
print(f"  V12: {len(v12_models)} traits")

# ============================================================
# FEATURE BUILDERS (compact versions)
# ============================================================
def _base_features(sire_ptas, mgs_ptas, mmgs_ptas, trait):
    s = sire_ptas.get(trait)
    if s is None: return None
    f = {}
    sd = GENETIC_SD.get(trait, 1); h2 = HERITABILITY.get(trait, 0.15)
    pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)
    f['pa'] = pa; f['pa_z'] = pa / sd if sd else 0
    f['sire'] = s; f['sire_z'] = s / sd if sd else 0; f['sire_sq'] = s*s; f['sire_h2'] = s*h2
    f['sire_dev_from_pa'] = s - pa * 2
    mg = mgs_ptas.get(trait) if mgs_ptas else None
    f['mgs_val'] = mg if mg is not None else 0
    f['mgs_z'] = (mg/sd if sd else 0) if mg is not None else 0
    f['mgs_sq'] = mg*mg if mg is not None else 0
    f['has_mgs'] = 1 if mg is not None else 0
    if mg is not None:
        f['sire_x_mgs'] = s*mg; f['sire_mgs_diff'] = s-mg
        f['sire_mgs_ratio'] = s/mg if mg != 0 else 0
        f['sire_mgs_mean'] = (s+mg)/2
        f['mendelian_range'] = abs(s-mg); f['mendelian_range_z'] = abs(s-mg)/sd if sd else 0
    else:
        f['sire_x_mgs']=0; f['sire_mgs_diff']=0; f['sire_mgs_ratio']=0
        f['sire_mgs_mean']=s/2; f['mendelian_range']=0; f['mendelian_range_z']=0
    mmg = mmgs_ptas.get(trait) if mmgs_ptas else None
    f['mmgs_val'] = mmg if mmg is not None else 0
    f['has_mmgs'] = 1 if mmg is not None else 0
    f['sire_x_mmgs'] = s*mmg if mmg is not None else 0
    return f

def _add_cross_traits(f, sire_ptas, mgs_ptas, mmgs_ptas, trait, cross_list, add_z=False):
    for ot in cross_list:
        if ot == trait: continue
        ot_sd = GENETIC_SD.get(ot, 1)
        sv = sire_ptas.get(ot)
        if sv is not None:
            f[f'sire_{ot}'] = sv
            if add_z: f[f'sire_{ot}_z'] = sv/ot_sd if ot_sd else 0
        mv = mgs_ptas.get(ot) if mgs_ptas else None
        if mv is not None:
            f[f'mgs_{ot}'] = mv
            if add_z: f[f'mgs_{ot}_z'] = mv/ot_sd if ot_sd else 0
        ot_pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, ot)
        if ot_pa is not None:
            f[f'pa_{ot}'] = ot_pa
            if add_z: f[f'pa_{ot}_z'] = ot_pa/ot_sd if ot_sd else 0

def _add_gc(f, sire_ptas, mgs_ptas, mmgs_ptas, trait, add_z=False):
    for (t1,t2),corr in GENETIC_CORRELATIONS.items():
        if t1==trait or t2==trait:
            other = t2 if t1==trait else t1
            ot_sd = GENETIC_SD.get(other, 1)
            ot_pa = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, other)
            if ot_pa is not None:
                f[f'gc_pa_{other}'] = ot_pa * corr
                if add_z: f[f'gc_pa_{other}_z'] = (ot_pa/ot_sd)*corr if ot_sd else 0
            sv = sire_ptas.get(other)
            if sv is not None:
                f[f'gc_sire_{other}'] = sv * corr
                if add_z: f[f'gc_sire_{other}_z'] = (sv/ot_sd)*corr if ot_sd else 0
            mv = mgs_ptas.get(other) if mgs_ptas else None
            if mv is not None:
                f[f'gc_mgs_{other}'] = mv * corr
                if add_z: f[f'gc_mgs_{other}_z'] = (mv/ot_sd)*corr if ot_sd else 0

def _add_profiles(f, trait, sire_naab, mgs_naab, pa, sire_profiles, mgs_profiles, cross_list=None):
    if sire_naab and sire_naab in sire_profiles:
        sp = sire_profiles[sire_naab]; tp = sp.get(trait)
        if tp and tp['n']>=2:
            f['sire_prof_n']=tp['n']; f['sire_prof_mean']=tp['mean']
            f['sire_prof_std']=tp['std']; f['sire_prof_ratio']=tp['trans_ratio']
            f['sire_prof_resid']=tp['residual']
            if pa is not None: f['sire_prof_dev_pa'] = tp['mean'] - pa
        else:
            f['sire_prof_n']=0; f['sire_prof_mean']=0; f['sire_prof_std']=0
            f['sire_prof_ratio']=1.0; f['sire_prof_resid']=0
            if pa is not None: f['sire_prof_dev_pa'] = 0
        if cross_list:
            for ot in cross_list:
                if ot==trait: continue
                otp = sp.get(ot)
                if otp and otp['n']>=2: f[f'sire_prof_{ot}'] = otp['residual']
    else:
        f['sire_prof_n']=0; f['sire_prof_mean']=0; f['sire_prof_std']=0
        f['sire_prof_ratio']=1.0; f['sire_prof_resid']=0
        if pa is not None: f['sire_prof_dev_pa'] = 0
    if mgs_naab and mgs_naab in mgs_profiles:
        mp = mgs_profiles[mgs_naab]; tp = mp.get(trait)
        if tp and tp['n']>=2:
            f['mgs_prof_n']=tp['n']; f['mgs_prof_mean']=tp['mean']; f['mgs_prof_resid']=tp['residual']
        else:
            f['mgs_prof_n']=0; f['mgs_prof_mean']=0; f['mgs_prof_resid']=0
    else:
        f['mgs_prof_n']=0; f['mgs_prof_mean']=0; f['mgs_prof_resid']=0

def build_v10(sire_ptas, mgs_ptas, trait, sire_naab, mgs_naab, mmgs_ptas):
    s = sire_ptas.get(trait)
    if s is None: return None
    sd = GENETIC_SD.get(trait,1); h2 = HERITABILITY.get(trait,0.15)
    f = {'sire':s, 'sire_z':s/sd if sd else 0, 'sire_sq':s*s, 'sire_h2':s*h2}
    mg = mgs_ptas.get(trait) if mgs_ptas else None
    f['mgs_val'] = mg if mg is not None else 0
    f['mgs_z'] = (mg/sd if sd else 0) if mg is not None else 0
    f['mgs_sq'] = mg*mg if mg is not None else 0
    f['has_mgs'] = 1 if mg is not None else 0
    if mg is not None:
        f['sire_x_mgs']=s*mg; f['sire_mgs_diff']=s-mg; f['sire_mgs_ratio']=s/mg if mg!=0 else 0
    else:
        f['sire_x_mgs']=0; f['sire_mgs_diff']=0; f['sire_mgs_ratio']=0
    mmg = mmgs_ptas.get(trait) if mmgs_ptas else None
    f['mmgs_val'] = mmg if mmg is not None else 0
    f['has_mmgs'] = 1 if mmg is not None else 0
    f['sire_x_mmgs'] = s*mmg if mmg is not None else 0
    v10t = [k for k in BULL_COL if k in v10_models]
    for ot in v10t:
        if ot==trait: continue
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
    _add_profiles(f, trait, sire_naab, mgs_naab, None, v10_sp, v10_mp, v10t)
    return f

def build_v11(sire_ptas, mgs_ptas, trait, sire_naab, mgs_naab, mmgs_ptas):
    f = _base_features(sire_ptas, mgs_ptas, mmgs_ptas, trait)
    if f is None: return None
    _add_cross_traits(f, sire_ptas, mgs_ptas, mmgs_ptas, trait, CORE_TRAITS, add_z=False)
    _add_gc(f, sire_ptas, mgs_ptas, mmgs_ptas, trait, add_z=False)
    _add_profiles(f, trait, sire_naab, mgs_naab, f['pa'], v11_sp, v11_mp, CORE_TRAITS)
    return f

def build_v12(sire_ptas, mgs_ptas, trait, sire_naab, mgs_naab, mmgs_ptas):
    f = _base_features(sire_ptas, mgs_ptas, mmgs_ptas, trait)
    if f is None: return None
    _add_cross_traits(f, sire_ptas, mgs_ptas, mmgs_ptas, trait, CORE_TRAITS, add_z=True)
    _add_gc(f, sire_ptas, mgs_ptas, mmgs_ptas, trait, add_z=True)
    _add_profiles(f, trait, sire_naab, mgs_naab, f['pa'], v12_sp, v12_mp, CORE_TRAITS)
    return f

# ============================================================
# PREDICT FUNCTIONS
# ============================================================
def predict_single(feat, models, trait):
    if trait not in models: return None
    info = models[trait]
    X = np.array([[feat.get(c, 0) for c in info['feature_cols']]])
    if info['type'] == 'stack':
        base_preds = np.column_stack([info['base_models'][mn].predict(X) for mn in info['base_names']])
        return info['meta_model'].predict(base_preds)[0]
    elif info['type'] == 'single':
        return info['model'].predict(X)[0]
    return None

# ============================================================
# LOAD ETHAN DATA
# ============================================================
print("\nCarregando Ethan...")
ethan = pd.read_excel(DOWNLOADS / 'Cliente Ethan.xlsx', engine='openpyxl', header=10)
print(f"  {len(ethan)} animais totais")

# Rename columns to standard
ethan_renamed = {}
for ecol, std in ETHAN_COL_MAP.items():
    if ecol in ethan.columns:
        ethan_renamed[ecol] = std
ethan = ethan.rename(columns=ethan_renamed)

# Filter: must have Sire NAAB
ethan = ethan[ethan['Sire of Record NAAB'].notna()].copy()
print(f"  {len(ethan)} com Sire")

# ============================================================
# GENERATE PREDICTIONS FOR ALL ETHAN ANIMALS
# ============================================================
print("\nGerando predicoes para Ethan (pode demorar ~2-3 min)...")
sys.stdout.flush()

VERSIONS = ['PA', 'V10', 'V11', 'V12_OLD', 'V12_2F', 'V12_DAU']
all_preds = []
n_resolved = 0
n_skipped = 0

for idx, row in ethan.iterrows():
    animal_id = row.get('Animal ID')
    sire_naab = str(row['Sire of Record NAAB']).strip()
    mgs_naab = str(row.get('Maternal Grandsire NAAB', '')).strip()
    if mgs_naab in ('nan', 'None', ''): mgs_naab = None

    _, _, sire_resolved, sire_ptas = lookup_bull(sire_naab)
    if not sire_ptas:
        n_skipped += 1
        continue

    mgs_ptas = {}
    mgs_resolved = None
    if mgs_naab:
        _, _, mgs_resolved, mgs_ptas = lookup_bull(mgs_naab)

    mmgs_ptas = {}  # Ethan doesn't have MMGS
    n_resolved += 1

    for trait in TRAITS:
        gen_val = row.get(trait)
        if pd.isna(gen_val): continue
        gen_val = float(gen_val)

        pa_val = compute_pa(sire_ptas, mgs_ptas, mmgs_ptas, trait)
        rec = {'ID': animal_id, 'Trait': trait, 'GENOMIC': gen_val, 'PA': pa_val}

        # V10 (absolute PTA)
        feat_v10 = build_v10(sire_ptas, mgs_ptas, trait, sire_resolved, mgs_resolved, mmgs_ptas)
        if feat_v10: rec['V10'] = predict_single(feat_v10, v10_models, trait)

        # V11 (PA + dev)
        feat_v11 = build_v11(sire_ptas, mgs_ptas, trait, sire_resolved, mgs_resolved, mmgs_ptas)
        if feat_v11 and pa_val is not None:
            dev = predict_single(feat_v11, v11_models, trait)
            if dev is not None: rec['V11'] = pa_val + dev

        # V12 variants
        feat_v12 = build_v12(sire_ptas, mgs_ptas, trait, sire_resolved, mgs_resolved, mmgs_ptas)
        if feat_v12 and pa_val is not None:
            dev = predict_single(feat_v12, v12_models, trait)
            if dev is not None:
                rec['V12_OLD'] = pa_val + dev
                cal2f = v12_models[trait].get('calibration_2f')
                if cal2f:
                    rec['V12_2F'] = cal2f['c0'] + cal2f['c1']*pa_val + cal2f['c2']*dev
                cal_dau = v12_models[trait].get('calibration_daughter')
                if cal_dau:
                    d0,d1,d2 = cal_dau['d0'],cal_dau['d1'],cal_dau['d2']
                    ve = cal_dau.get('var_expansion',1.0)
                    raw = d0 + d1*pa_val + d2*dev
                    rec['V12_DAU'] = pa_val + (raw - pa_val) * ve

        all_preds.append(rec)

    if n_resolved % 500 == 0:
        print(f"  ... {n_resolved} animais processados", flush=True)

print(f"\n  Resolvidos: {n_resolved}, Pulados: {n_skipped}")
pred_df = pd.DataFrame(all_preds)
print(f"  Total predicoes: {len(pred_df)}")
for v in VERSIONS:
    if v in pred_df.columns:
        print(f"    {v}: {pred_df[v].notna().sum()}")

# Save for reuse
pred_df.to_pickle(BASE / 'dsii_v12_results' / 'ethan_multi_version_predictions.pkl')

# ============================================================
# LOCATION ACCURACY ANALYSIS
# ============================================================
def percentile_group(vals, n_groups):
    return pd.qcut(pd.Series(vals).rank(method='first'), q=n_groups, labels=False).values + 1

def analyze(gen_vals, pred_vals, n_groups):
    n = len(gen_vals)
    if n < n_groups * 2: return None
    gen_g = percentile_group(gen_vals, n_groups)
    pred_g = percentile_group(pred_vals, n_groups)
    exact = (gen_g == pred_g).mean() * 100
    adj1 = (np.abs(gen_g - pred_g) <= 1).mean() * 100
    gen_top = set(np.where(gen_g == n_groups)[0])
    pred_top = set(np.where(pred_g == n_groups)[0])
    top_acc = len(gen_top & pred_top) / len(gen_top) * 100 if gen_top else 0
    gen_bot = set(np.where(gen_g == 1)[0])
    pred_bot = set(np.where(pred_g == 1)[0])
    bot_acc = len(gen_bot & pred_bot) / len(gen_bot) * 100 if gen_bot else 0
    rho, _ = spearmanr(pred_vals, gen_vals)
    return {'exact': exact, 'adj1': adj1, 'top': top_acc, 'bottom': bot_acc, 'spearman': rho}

GRANS = [(4,'Quartil 25%'), (5,'Quintil 20%'), (10,'Decil 10%'), (20,'Vigintil 5%')]

print(f"\n{'='*140}")
print(f"  VALIDACAO ETHAN - {n_resolved} ANIMAIS USA COM GENOMICA REAL")
print(f"  Analise de Localizacao: PA vs V10 vs V11 vs V12_OLD vs V12_2F vs V12_DAU")
print(f"{'='*140}")

# Per-trait detailed analysis
for trait in TRAITS:
    tdf = pred_df[pred_df['Trait']==trait].copy()
    n = len(tdf)
    if n < 40: continue

    # Spearman
    spears = {}
    for v in VERSIONS:
        if v in tdf.columns and tdf[v].notna().sum() >= 20:
            valid = tdf[v].notna()
            rho, _ = spearmanr(tdf.loc[valid,v].values, tdf.loc[valid,'GENOMIC'].values)
            spears[v] = rho

    best_sp_v = max(spears, key=spears.get) if spears else '?'

    print(f"\n{'='*140}")
    print(f"  TRAIT: {trait}  (N={n}) | Melhor Spearman: {best_sp_v} ({spears.get(best_sp_v,0):.4f})")
    print(f"{'='*140}")

    # Print Spearman row
    print(f"  {'Spearman':>12}", end='')
    for v in VERSIONS:
        if v in spears: print(f" | {spears[v]:>8.4f}", end='')
        else: print(f" | {'---':>8}", end='')
    print()

    for n_groups, gran_name in GRANS:
        random_pct = 100.0 / n_groups
        print(f"\n  {gran_name}  (aleatorio={random_pct:.0f}%)")

        # Exact
        best_v = ''; best_val = -1
        print(f"    {'Exato%':>10}", end='')
        for v in VERSIONS:
            if v in tdf.columns and tdf[v].notna().sum() >= n_groups*2:
                valid = tdf[v].notna()
                res = analyze(tdf.loc[valid,'GENOMIC'].values, tdf.loc[valid,v].values, n_groups)
                if res:
                    print(f" | {res['exact']:>7.1f}%", end='')
                    if res['exact'] > best_val: best_val = res['exact']; best_v = v
                else: print(f" | {'---':>8}", end='')
            else: print(f" | {'---':>8}", end='')
        print(f"  << {best_v}")

        # Top
        best_top_v = ''; best_top = -1
        print(f"    {'Top':>10}", end='')
        for v in VERSIONS:
            if v in tdf.columns and tdf[v].notna().sum() >= n_groups*2:
                valid = tdf[v].notna()
                res = analyze(tdf.loc[valid,'GENOMIC'].values, tdf.loc[valid,v].values, n_groups)
                if res:
                    print(f" | {res['top']:>7.1f}%", end='')
                    if res['top'] > best_top: best_top = res['top']; best_top_v = v
                else: print(f" | {'---':>8}", end='')
            else: print(f" | {'---':>8}", end='')
        print(f"  << {best_top_v}")

# ============================================================
# SUMMARY TABLE
# ============================================================
print(f"\n{'='*140}")
print(f"  MAPA FINAL: MELHOR VERSAO POR TRAIT X GRANULARIDADE (Exato%)")
print(f"{'='*140}")
print(f"  {'Trait':>6} | {'N':>5} | {'Spearman':>10} | {'Quartil':>12} | {'Quintil':>12} | {'Decil':>12} | {'Vigintil':>12}")
print("  " + "-" * 90)

version_wins = {v: 0 for v in VERSIONS}

for trait in TRAITS:
    tdf = pred_df[pred_df['Trait']==trait]
    n = len(tdf)
    if n < 40: continue

    # Best spearman
    best_sp_v = ''; best_sp = -999
    for v in VERSIONS:
        if v in tdf.columns and tdf[v].notna().sum() >= 20:
            valid = tdf[v].notna()
            rho, _ = spearmanr(tdf.loc[valid,v].values, tdf.loc[valid,'GENOMIC'].values)
            if rho > best_sp: best_sp = rho; best_sp_v = v

    cells = [f"{best_sp_v}({best_sp:.3f})"]
    for n_groups, _ in GRANS:
        best_v = ''; best_val = -1
        for v in VERSIONS:
            if v in tdf.columns and tdf[v].notna().sum() >= n_groups*2:
                valid = tdf[v].notna()
                res = analyze(tdf.loc[valid,'GENOMIC'].values, tdf.loc[valid,v].values, n_groups)
                if res and res['exact'] > best_val: best_val = res['exact']; best_v = v
        cells.append(f"{best_v}({best_val:.0f}%)")
        if best_v: version_wins[best_v] = version_wins.get(best_v, 0) + 1

    print(f"  {trait:>6} | {n:>5} | {cells[0]:>12} | {cells[1]:>12} | {cells[2]:>12} | {cells[3]:>12} | {cells[4]:>12}")

print("  " + "-" * 90)
print(f"\n  VITORIAS TOTAIS (Exato% em 4 granularidades x {len(TRAITS)} traits):")
for v in sorted(version_wins, key=version_wins.get, reverse=True):
    if version_wins[v] > 0:
        bar = '#' * version_wins[v]
        print(f"    {v:>10}: {version_wins[v]:>3} {bar}")
print()
