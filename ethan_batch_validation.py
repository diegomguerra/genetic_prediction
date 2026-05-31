"""
ETHAN VALIDATION - BATCH VERSION (fast)
Processa todos os animais em batch por trait
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pickle, csv, re, sys, time
from pathlib import Path
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()

BASE = Path(r"C:\Users\DiegoGuerra\Documents\Projetos\genetic_prediction")
DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")

ETHAN_COL_MAP = {
    'TPI': 'TPI', 'MILK': 'MILK', 'FAT': 'FAT', 'FAT %': 'FAT%',
    'PROT': 'PRO', 'PROT%': 'PRO%', 'PL': 'PL', 'DPR': 'DPR',
    'CCR': 'CCR', 'LIV': 'LIV', 'SCS': 'SCS', 'CDCB_MAST': 'MAST',
    'UDC': 'UDC', 'FLC': 'FLC', 'FI': 'FI',
}
TRAITS = ['TPI', 'MILK', 'FAT', 'FAT%', 'PRO', 'PRO%', 'PL', 'DPR',
          'CCR', 'LIV', 'SCS', 'MAST', 'UDC', 'FLC']
VERSIONS = ['PA', 'V10', 'V11', 'V12_OLD', 'V12_2F', 'V12_DAU']

# === SHARED CONSTANTS ===
STUD_MAP = {'507':'7','509':'9','550':'50','551':'51','559':'59','518':'18','521':'21',
            '523':'23','581':'81','585':'85','501':'1','601':'1','604':'4','614':'14',
            '629':'29','751':'151','752':'152','814':'14','250':'200',
            '001':'1','007':'7','009':'9','011':'11','029':'29','097':'97',
            '100':'100','200':'200','777':'777','796':'796','745':'745'}
GENETIC_SD = {'TPI':250,'MILK':675,'FAT':29,'FAT%':0.05,'PRO':19,'PRO%':0.02,'CFP':40,
    'PL':1.85,'LIV':1.2,'SCS':0.14,'MAST':1.0,'MET':1.0,'RP':0.5,'DA':0.5,'KET':0.8,'MF':0.3,
    'DPR':1.3,'HCR':1.5,'CCR':1.65,'SCE':0.8,'DCE':0.8,'SSB':1.0,'DSB':1.0,
    'FI':1.0,'GFI':3.0,'PTAT':0.70,'UDC':0.75,'FLC':0.65,
    'NM$':200,'CM$':200,'H_LIV':1.0,'F_SAV':50,'RFI':30,'GL':0.8,'EFC':1.0,
    'STA':0.80,'STR':0.70,'DFM':0.80,'BOD':0.60,'RUA':0.60,'RW':0.70,
    'RLS':0.50,'RLR':0.50,'FTA':0.60,'FLS':0.50,'FTL':0.50,
    'FUA':0.80,'RUH':0.80,'RUW':0.80,'UCL':0.60,'UDP':0.80,'FTP':0.70,'RTP':0.60,
    'BWC':0.50}
HERITABILITY = {'TPI':0.30,'MILK':0.25,'FAT':0.25,'FAT%':0.50,'PRO':0.25,'PRO%':0.50,'CFP':0.25,
    'PL':0.08,'LIV':0.05,'SCS':0.12,'MAST':0.04,'MET':0.03,'RP':0.02,'DA':0.04,'KET':0.04,'MF':0.02,
    'DPR':0.04,'HCR':0.04,'CCR':0.04,'SCE':0.08,'DCE':0.06,'SSB':0.05,'DSB':0.04,
    'FI':0.06,'GFI':0.15,'PTAT':0.30,'UDC':0.25,'FLC':0.15,
    'NM$':0.30,'CM$':0.30,'H_LIV':0.05,'F_SAV':0.15,'RFI':0.15,'GL':0.40,'EFC':0.06,
    'STA':0.42,'STR':0.27,'DFM':0.25,'BOD':0.30,'RUA':0.30,'RW':0.25,
    'RLS':0.15,'RLR':0.10,'FTA':0.10,'FLS':0.10,'FTL':0.10,
    'FUA':0.25,'RUH':0.20,'RUW':0.20,'UCL':0.15,'UDP':0.30,'FTP':0.25,'RTP':0.20,
    'BWC':0.20}
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
    ('GL','SCE'):-0.30,('GL','SSB'):-0.25,('CCR','PL'):0.35,('HCR','PL'):0.30}
BULL_COL = {'TPI':'TPI','MILK':'PTAM','FAT':'PTAF','FAT%':'PTAF%','PRO':'PTAP','PRO%':'PTAP%','CFP':'CFP',
    'PL':'PL','LIV':'LIV','H_LIV':'H LIV','SCS':'SCS','MAST':'MAST','MET':'MET','RP':'RP','DA':'DA',
    'KET':'KET','MF':'MF','DPR':'DPR','HCR':'HCR','CCR':'CCR','SCE':'SCE','DCE':'DCE','SSB':'SSB','DSB':'DSB',
    'GL':'GL','EFC':'EFC','FI':'FI','F_SAV':'F SAV','RFI':'RFI','GFI':'GFI',
    'PTAT':'PTAT','UDC':'UDC','FLC':'FLC','STA':'STA','STR':'STR','DFM':'DFM','BOD':'BOD',
    'RUA':'RUA','RW':'RW','RLS':'RLS','RLR':'RLR','FTA':'FTA','FLS':'FLS','FTL':'FTL',
    'FUA':'FUA','RUH':'RUH','RUW':'RUW','UCL':'UCL','UDP':'UDP','FTP':'FTP','RTP':'RTP','TL':None,
    'BWC':'BWC','NM$':'NM$','CM$':'CM$'}
CDCB_COL = {'TPI':None,'MILK':'MLK_PTA','FAT':'FAT_PTA','FAT%':'FAT%','PRO':'PRO_PTA','PRO%':'PRO%',
    'PL':'PL_PTA','LIV':'LIV_PTA','SCS':'SCS_PTA','MAST':'MAS_PTA','DPR':'DPR_PTA','HCR':'HCR_PTA',
    'CCR':'CCR_PTA','PTAT':'TYP_PTA','UDC':'UDC_PTA','FLC':'FLC_PTA','FI':'FS_PTA',
    'NM$':'NM$_PTA','CM$':'CM$_PTA','MET':'MET_PTA','RP':'RPL_PTA','DA':'DAB_PTA','KET':'KET_PTA','MF':'MFV_PTA',
    'CFP':None,'H_LIV':'HLV_PTA','SCE':None,'DCE':None,'SSB':None,'DSB':None,'GL':'GL_PTA','EFC':'EFC_PTA',
    'F_SAV':None,'RFI':'RFI_PTA','GFI':None,'STA':None,'STR':None,'DFM':None,'BOD':None,'RUA':None,'RW':None,
    'RLS':None,'RLR':None,'FTA':None,'FLS':None,'FTL':None,'FUA':None,'RUH':None,'RUW':None,
    'UCL':None,'UDP':None,'FTP':None,'RTP':None,'TL':None,'BWC':None}
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
            k = k[:idx+1] + ''.join(c for c in k[idx+1:] if c.isalnum() or c in ' _').rstrip()
        out[k] = v
    return out

def normalize_naab(naab):
    if not naab or str(naab).strip() in ('','nan','None'): return []
    naab = str(naab).strip()
    m = re.match(r'^(\d+)(HO|BS|H|B)0*(\d+)$', naab)
    if not m: return [naab]
    org, breed_raw, num = m.group(1), m.group(2), m.group(3)
    breeds = ['HO','H'] if breed_raw in ('H','HO') else ['BS','B'] if breed_raw in ('B','BS') else [breed_raw]
    orgs = set([org, org.lstrip('0') or org])
    if org in STUD_MAP: orgs.add(STUD_MAP[org])
    if org.startswith('5'): orgs.add(org[1:])
    nums = [num, num.zfill(5)] if num.zfill(5) != num else [num]
    return list(dict.fromkeys(f'{o}{b}{n}' for o in orgs for b in breeds for n in nums))

# === LOAD BULLS ===
print("Carregando touros...", flush=True)
bulls = {}
with open(BULLS_CSV, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        row = _normalize_bull_row(row)
        naab = row.get('NAAB','').strip()
        if naab: bulls[naab] = row
print(f"  {len(bulls)} touros", flush=True)

cdcb_bulls = {}
try:
    with open(DOWNLOADS / 'Bull_Report (1).csv', 'r', encoding='latin-1') as f:
        for row in csv.DictReader(f):
            naab = row.get('NAAB_CODE','').strip()
            if naab: cdcb_bulls[naab] = row
    print(f"  CDCB: {len(cdcb_bulls)}", flush=True)
except: pass

bulls_by_num = {}
for naab, row in bulls.items():
    m = re.match(r'^(\d+)(HO|BS)(\d+)$', naab)
    if m: bulls_by_num.setdefault(m.group(3), []).append((naab, row))

def get_bull_ptas(bull_row, source='SS'):
    ptas = {}
    col_map = BULL_COL if source == 'SS' else CDCB_COL
    for trait, col in col_map.items():
        if col is None: continue
        v = sf(bull_row.get(col))
        if v is not None: ptas[trait] = v
    return ptas

def lookup_bull_fast(naab_raw):
    for c in normalize_naab(naab_raw):
        if c in bulls: return c, get_bull_ptas(bulls[c], 'SS')
        if c in cdcb_bulls: return c, get_bull_ptas(cdcb_bulls[c], 'CDCB')
    m = re.match(r'^(\d+)(HO|BS|H|B)0*(\d+)$', str(naab_raw).strip())
    if m:
        num = m.group(3)
        if num in bulls_by_num:
            naab, row = bulls_by_num[num][0]
            return naab, get_bull_ptas(row, 'SS')
    return None, {}

# === LOAD MODELS ===
print("Carregando modelos...", flush=True)
with open(BASE/'dsii_v10_results/v10_models.pkl','rb') as f: v10m = pickle.load(f)
with open(BASE/'dsii_v10_results/v10_sire_profiles.pkl','rb') as f: v10sp = pickle.load(f)
with open(BASE/'dsii_v10_results/v10_mgs_profiles.pkl','rb') as f: v10mp = pickle.load(f)
with open(BASE/'dsii_v11_results/v11_models.pkl','rb') as f: v11m = pickle.load(f)
with open(BASE/'dsii_v11_results/v11_sire_profiles.pkl','rb') as f: v11sp = pickle.load(f)
with open(BASE/'dsii_v11_results/v11_mgs_profiles.pkl','rb') as f: v11mp = pickle.load(f)
with open(BASE/'dsii_v12_results/v12_models.pkl','rb') as f: v12m = pickle.load(f)
with open(BASE/'dsii_v12_results/v12_sire_profiles.pkl','rb') as f: v12sp = pickle.load(f)
with open(BASE/'dsii_v12_results/v12_mgs_profiles.pkl','rb') as f: v12mp = pickle.load(f)
print(f"  V10:{len(v10m)} V11:{len(v11m)} V12:{len(v12m)}", flush=True)

# === LOAD & RESOLVE ETHAN ===
print("Carregando Ethan...", flush=True)
ethan = pd.read_excel(DOWNLOADS/'Cliente Ethan.xlsx', engine='openpyxl', header=10)
ethan = ethan.rename(columns=ETHAN_COL_MAP)
ethan = ethan[ethan['Sire of Record NAAB'].notna()].reset_index(drop=True)
print(f"  {len(ethan)} animais com Sire", flush=True)

# Pre-resolve all bulls
print("Resolvendo pedigrees...", flush=True)
animal_data = []  # list of (idx, sire_naab, sire_ptas, mgs_naab, mgs_ptas)
for idx, row in ethan.iterrows():
    sire_raw = str(row['Sire of Record NAAB']).strip()
    mgs_raw = str(row.get('Maternal Grandsire NAAB','')).strip()
    if mgs_raw in ('nan','None',''): mgs_raw = None
    sire_naab, sire_ptas = lookup_bull_fast(sire_raw)
    if not sire_ptas: continue
    mgs_naab, mgs_ptas = (None, {}) if not mgs_raw else lookup_bull_fast(mgs_raw)
    animal_data.append((idx, sire_naab, sire_ptas, mgs_naab, mgs_ptas))
print(f"  {len(animal_data)} pedigrees resolvidos ({time.time()-t0:.0f}s)", flush=True)

# === BATCH PREDICT PER TRAIT ===
def compute_pa(sp, mp, trait):
    s = sp.get(trait)
    if s is None: return None
    pa = s / 2
    mg = mp.get(trait) if mp else None
    if mg is not None: pa += mg / 4
    return pa

def build_feat_batch(animals, trait, version, sire_profiles, mgs_profiles):
    """Build feature matrix for all animals at once"""
    feat_rows = []
    pa_vals = []
    valid_indices = []
    sd = GENETIC_SD.get(trait, 1)
    h2 = HERITABILITY.get(trait, 0.15)
    add_z = (version == 'v12')

    for i, (idx, sire_naab, sp, mgs_naab, mp) in enumerate(animals):
        s = sp.get(trait)
        if s is None: continue
        f = {}
        pa = compute_pa(sp, mp, trait)
        if version != 'v10':
            f['pa'] = pa; f['pa_z'] = pa/sd if sd else 0
        f['sire'] = s; f['sire_z'] = s/sd if sd else 0; f['sire_sq'] = s*s; f['sire_h2'] = s*h2
        if version != 'v10':
            f['sire_dev_from_pa'] = s - pa*2
        mg = mp.get(trait) if mp else None
        f['mgs_val'] = mg if mg is not None else 0
        f['mgs_z'] = (mg/sd if sd else 0) if mg is not None else 0
        f['mgs_sq'] = mg*mg if mg is not None else 0
        f['has_mgs'] = 1 if mg is not None else 0
        if mg is not None:
            f['sire_x_mgs']=s*mg; f['sire_mgs_diff']=s-mg; f['sire_mgs_ratio']=s/mg if mg!=0 else 0
            if version != 'v10':
                f['sire_mgs_mean']=(s+mg)/2; f['mendelian_range']=abs(s-mg)
                f['mendelian_range_z']=abs(s-mg)/sd if sd else 0
        else:
            f['sire_x_mgs']=0; f['sire_mgs_diff']=0; f['sire_mgs_ratio']=0
            if version != 'v10':
                f['sire_mgs_mean']=s/2; f['mendelian_range']=0; f['mendelian_range_z']=0
        f['mmgs_val']=0; f['has_mmgs']=0; f['sire_x_mmgs']=0

        # Cross-traits
        ct_list = CORE_TRAITS if version != 'v10' else [k for k in BULL_COL if k in (v10m if version=='v10' else {})]
        if version == 'v10':
            ct_list = list(set(BULL_COL.keys()) & set(v10m.keys()))
        for ot in ct_list:
            if ot == trait: continue
            ot_sd = GENETIC_SD.get(ot, 1)
            sv = sp.get(ot)
            if sv is not None:
                f[f'sire_{ot}'] = sv
                if add_z: f[f'sire_{ot}_z'] = sv/ot_sd if ot_sd else 0
            mv = mp.get(ot) if mp else None
            if mv is not None:
                f[f'mgs_{ot}'] = mv
                if add_z: f[f'mgs_{ot}_z'] = mv/ot_sd if ot_sd else 0
            if version != 'v10':
                ot_pa = compute_pa(sp, mp, ot)
                if ot_pa is not None:
                    f[f'pa_{ot}'] = ot_pa
                    if add_z: f[f'pa_{ot}_z'] = ot_pa/ot_sd if ot_sd else 0

        # Genetic correlations
        for (t1,t2),corr in GENETIC_CORRELATIONS.items():
            if t1==trait or t2==trait:
                other = t2 if t1==trait else t1
                ot_sd = GENETIC_SD.get(other, 1)
                sv = sp.get(other)
                if sv is not None:
                    f[f'gc_sire_{other}'] = sv*corr
                    if add_z: f[f'gc_sire_{other}_z'] = (sv/ot_sd)*corr if ot_sd else 0
                mv = mp.get(other) if mp else None
                if mv is not None:
                    f[f'gc_mgs_{other}'] = mv*corr
                    if add_z: f[f'gc_mgs_{other}_z'] = (mv/ot_sd)*corr if ot_sd else 0
                if version != 'v10':
                    ot_pa = compute_pa(sp, mp, other)
                    if ot_pa is not None:
                        f[f'gc_pa_{other}'] = ot_pa*corr
                        if add_z: f[f'gc_pa_{other}_z'] = (ot_pa/ot_sd)*corr if ot_sd else 0

        # Profiles
        _sp = v10sp if version=='v10' else v11sp if version=='v11' else v12sp
        _mp = v10mp if version=='v10' else v11mp if version=='v11' else v12mp
        if sire_naab and sire_naab in _sp:
            prof = _sp[sire_naab]; tp = prof.get(trait)
            if tp and tp['n']>=2:
                f['sire_prof_n']=tp['n']; f['sire_prof_mean']=tp['mean']
                f['sire_prof_std']=tp['std']; f['sire_prof_ratio']=tp['trans_ratio']
                f['sire_prof_resid']=tp['residual']
                if version != 'v10' and pa is not None: f['sire_prof_dev_pa'] = tp['mean']-pa
            else:
                f['sire_prof_n']=0; f['sire_prof_mean']=0; f['sire_prof_std']=0
                f['sire_prof_ratio']=1.0; f['sire_prof_resid']=0
                if version != 'v10': f['sire_prof_dev_pa']=0
            for ot in ct_list:
                if ot==trait: continue
                otp = prof.get(ot)
                if otp and otp['n']>=2: f[f'sire_prof_{ot}'] = otp['residual']
        else:
            f['sire_prof_n']=0; f['sire_prof_mean']=0; f['sire_prof_std']=0
            f['sire_prof_ratio']=1.0; f['sire_prof_resid']=0
            if version != 'v10': f['sire_prof_dev_pa']=0
        if mgs_naab and mgs_naab in _mp:
            prof = _mp[mgs_naab]; tp = prof.get(trait)
            if tp and tp['n']>=2:
                f['mgs_prof_n']=tp['n']; f['mgs_prof_mean']=tp['mean']; f['mgs_prof_resid']=tp['residual']
            else:
                f['mgs_prof_n']=0; f['mgs_prof_mean']=0; f['mgs_prof_resid']=0
        else:
            f['mgs_prof_n']=0; f['mgs_prof_mean']=0; f['mgs_prof_resid']=0

        feat_rows.append(f)
        pa_vals.append(pa if pa is not None else 0)
        valid_indices.append(i)

    return feat_rows, np.array(pa_vals), valid_indices

def predict_batch(feat_rows, models, trait, pa_arr):
    """Predict batch using model"""
    info = models.get(trait)
    if not info or not feat_rows: return None
    feat_df = pd.DataFrame(feat_rows)
    for c in info['feature_cols']:
        if c not in feat_df.columns: feat_df[c] = 0
    feat_df = feat_df[info['feature_cols']].fillna(0)
    X = feat_df.values.astype(np.float64)
    if info['type'] == 'stack':
        base_preds = np.column_stack([info['base_models'][mn].predict(X) for mn in info['base_names']])
        return info['meta_model'].predict(base_preds)
    elif info['type'] == 'single':
        return info['model'].predict(X)
    return None

# === MAIN PREDICTION LOOP (per trait, batch) ===
print(f"\nGerando predicoes em batch...", flush=True)

results = {}  # results[trait] = DataFrame with columns: idx, GENOMIC, PA, V10, V11, V12_OLD, V12_2F, V12_DAU

for trait in TRAITS:
    t1 = time.time()
    # Get genomic values
    gen_col = trait
    if gen_col not in ethan.columns:
        print(f"  {trait}: coluna nao encontrada, pulando", flush=True)
        continue

    gen_vals = ethan[gen_col].values
    n_valid_gen = np.sum(~pd.isna(gen_vals))

    # Build features for each version
    trait_results = {'idx': [], 'GENOMIC': [], 'PA': []}
    for v in ['V10','V11','V12_OLD','V12_2F','V12_DAU']:
        trait_results[v] = []

    # Filter animals with genomic value for this trait
    valid_animals = []
    valid_gen = []
    valid_pa = []
    for a_idx, (orig_idx, sire_naab, sp, mgs_naab, mp) in enumerate(animal_data):
        gv = gen_vals[orig_idx] if orig_idx < len(gen_vals) else np.nan
        if pd.isna(gv): continue
        pa = compute_pa(sp, mp, trait)
        if pa is None: continue
        valid_animals.append((orig_idx, sire_naab, sp, mgs_naab, mp))
        valid_gen.append(float(gv))
        valid_pa.append(pa)

    n = len(valid_animals)
    if n < 40:
        print(f"  {trait}: apenas {n} animais validos, pulando", flush=True)
        continue

    gen_arr = np.array(valid_gen)
    pa_arr_trait = np.array(valid_pa)

    # V10 batch
    feat_v10, _, vidx_v10 = build_feat_batch(valid_animals, trait, 'v10', v10sp, v10mp)
    v10_preds = predict_batch(feat_v10, v10m, trait, None)
    v10_full = np.full(n, np.nan)
    if v10_preds is not None:
        for j, vi in enumerate(vidx_v10):
            v10_full[vi] = v10_preds[j]

    # V11 batch
    feat_v11, pa_v11, vidx_v11 = build_feat_batch(valid_animals, trait, 'v11', v11sp, v11mp)
    v11_dev = predict_batch(feat_v11, v11m, trait, pa_v11)
    v11_full = np.full(n, np.nan)
    if v11_dev is not None:
        for j, vi in enumerate(vidx_v11):
            v11_full[vi] = pa_v11[j] + v11_dev[j]

    # V12 batch (3 variants)
    feat_v12, pa_v12, vidx_v12 = build_feat_batch(valid_animals, trait, 'v12', v12sp, v12mp)
    v12_dev = predict_batch(feat_v12, v12m, trait, pa_v12)
    v12_old = np.full(n, np.nan)
    v12_2f = np.full(n, np.nan)
    v12_dau = np.full(n, np.nan)
    if v12_dev is not None:
        info = v12m[trait]
        cal2f = info.get('calibration_2f')
        cal_dau = info.get('calibration_daughter')
        for j, vi in enumerate(vidx_v12):
            pa = pa_v12[j]; dev = v12_dev[j]
            v12_old[vi] = pa + dev
            if cal2f:
                v12_2f[vi] = cal2f['c0'] + cal2f['c1']*pa + cal2f['c2']*dev
            if cal_dau:
                d0,d1,d2 = cal_dau['d0'],cal_dau['d1'],cal_dau['d2']
                ve = cal_dau.get('var_expansion',1.0)
                raw = d0 + d1*pa + d2*dev
                v12_dau[vi] = pa + (raw - pa) * ve

    df_trait = pd.DataFrame({
        'GENOMIC': gen_arr, 'PA': pa_arr_trait,
        'V10': v10_full, 'V11': v11_full,
        'V12_OLD': v12_old, 'V12_2F': v12_2f, 'V12_DAU': v12_dau
    })
    results[trait] = df_trait
    dt = time.time() - t1
    print(f"  {trait}: N={n}, {dt:.1f}s", flush=True)

print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)

# Save
pd.to_pickle(results, BASE / 'dsii_v12_results' / 'ethan_results_by_trait.pkl')
print("Resultados salvos.", flush=True)

# ============================================================
# ANALYSIS
# ============================================================
def pct_group(vals, ng):
    return pd.qcut(pd.Series(vals).rank(method='first'), q=ng, labels=False).values + 1

def loc_acc(gen, pred, ng):
    n = len(gen)
    if n < ng*2: return None
    gg = pct_group(gen, ng); pg = pct_group(pred, ng)
    exact = (gg==pg).mean()*100
    top_g = set(np.where(gg==ng)[0]); top_p = set(np.where(pg==ng)[0])
    top_acc = len(top_g&top_p)/len(top_g)*100 if top_g else 0
    bot_g = set(np.where(gg==1)[0]); bot_p = set(np.where(pg==1)[0])
    bot_acc = len(bot_g&bot_p)/len(bot_g)*100 if bot_g else 0
    return {'exact':exact, 'top':top_acc, 'bottom':bot_acc}

GRANS = [(4,'Quartil 25%'),(5,'Quintil 20%'),(10,'Decil 10%'),(20,'Vigintil 5%')]

print(f"\n{'='*140}")
print(f"  VALIDACAO ETHAN - LOCALIZACAO POR TRAIT E GRANULARIDADE")
print(f"  {len(animal_data)} animais USA com genomica real")
print(f"{'='*140}")

for trait in TRAITS:
    if trait not in results: continue
    df = results[trait]
    n = len(df)

    # Spearman
    spears = {}
    for v in VERSIONS:
        valid = df[v].notna()
        if valid.sum() >= 20:
            rho, _ = spearmanr(df.loc[valid,v], df.loc[valid,'GENOMIC'])
            spears[v] = rho
    best_sp = max(spears, key=spears.get) if spears else '?'

    print(f"\n{'='*140}")
    print(f"  {trait}  N={n}  |  Spearman: ", end='')
    for v in VERSIONS:
        if v in spears: print(f"{v}={spears[v]:.4f}  ", end='')
    print(f" << {best_sp}")

    for ng, gn in GRANS:
        rnd = 100.0/ng
        print(f"  {gn:>14}", end='')
        best_v=''; best_val=-1; best_top_v=''; best_top=-1
        for v in VERSIONS:
            valid = df[v].notna()
            if valid.sum() >= ng*2:
                r = loc_acc(df.loc[valid,'GENOMIC'].values, df.loc[valid,v].values, ng)
                if r:
                    print(f" | {v}={r['exact']:5.1f}%(T{r['top']:4.0f}%)", end='')
                    if r['exact'] > best_val: best_val=r['exact']; best_v=v
                    if r['top'] > best_top: best_top=r['top']; best_top_v=v
        print(f"  << Ex:{best_v} Top:{best_top_v}")

# === MAPA FINAL ===
print(f"\n{'='*140}")
print(f"  MAPA FINAL: MELHOR METODO POR TRAIT x GRANULARIDADE")
print(f"{'='*140}")
print(f"  {'Trait':>6} | {'N':>5} | {'Spearman':>15} | {'Quartil':>15} | {'Quintil':>15} | {'Decil':>15} | {'Vigintil':>15}")
print("  " + "-" * 100)

wins = {v:0 for v in VERSIONS}
for trait in TRAITS:
    if trait not in results: continue
    df = results[trait]; n = len(df)
    cells = []
    # Spearman
    best_sp=''; best_val=-999
    for v in VERSIONS:
        valid = df[v].notna()
        if valid.sum()>=20:
            rho,_ = spearmanr(df.loc[valid,v], df.loc[valid,'GENOMIC'])
            if rho>best_val: best_val=rho; best_sp=v
    cells.append(f"{best_sp}({best_val:.3f})")
    # Granularities
    for ng,_ in GRANS:
        bv=''; bval=-1
        for v in VERSIONS:
            valid = df[v].notna()
            if valid.sum()>=ng*2:
                r = loc_acc(df.loc[valid,'GENOMIC'].values, df.loc[valid,v].values, ng)
                if r and r['exact']>bval: bval=r['exact']; bv=v
        cells.append(f"{bv}({bval:.0f}%)")
        if bv: wins[bv] += 1
    print(f"  {trait:>6} | {n:>5} | {cells[0]:>15} | {cells[1]:>15} | {cells[2]:>15} | {cells[3]:>15} | {cells[4]:>15}")

print("  " + "-" * 100)
print(f"\n  VITORIAS ({len(TRAITS)} traits x 4 granularidades = {len(TRAITS)*4} competicoes):")
for v in sorted(wins, key=wins.get, reverse=True):
    if wins[v]>0:
        bar = '#'*wins[v]
        print(f"    {v:>10}: {wins[v]:>3} {bar}")

print(f"\n  Tempo total: {time.time()-t0:.0f}s")
