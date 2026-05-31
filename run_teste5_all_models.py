"""
Run PA, V8 Oracle, and V9H (no-PA) predictions on Teste5.xls
Teste5 has: ID Animal, Pai, mgs, mmgs (bisavo materno)
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, csv, pickle, sys, re, openpyxl
from pathlib import Path
from collections import defaultdict

DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")
V8_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v8_results")
V9_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v9_results")
OUTPUT_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v9_results")

def sf(v):
    if v is None: return None
    try:
        x = float(v)
        return x if not np.isnan(x) else None
    except: return None

def flush_print(*args, **kwargs):
    print(*args, **kwargs); sys.stdout.flush()

# NAAB normalization
STUD_MAP = {'507':'7','509':'9','550':'50','551':'51','559':'59','518':'18','521':'21',
            '523':'23','581':'81','585':'85','501':'1','601':'1','604':'4','614':'14',
            '629':'29','751':'151','752':'152','814':'14','250':'200'}

def normalize_naab(naab):
    if not naab or str(naab).strip() in ('','nan','None'): return []
    naab = str(naab).strip()
    # Accept both 007HO15977 and 007H15977 formats
    m = re.match(r'^(\d+)(HO|BS|H|B)0*(\d+)$', naab)
    if not m: return [naab]
    org, breed, num = m.group(1), m.group(2), m.group(3)
    # Normalize breed: H -> HO, B -> BS
    if breed == 'H': breed = 'HO'
    elif breed == 'B': breed = 'BS'
    orgs = [org]
    if org.startswith('5'): orgs.append(org[1:])
    if org in STUD_MAP: orgs.append(STUD_MAP[org])
    orgs.append(org.lstrip('0') or org)
    return list(dict.fromkeys(f'{o}{breed}{num}' for o in orgs))

# ============================================================
# DOMAIN KNOWLEDGE
# ============================================================
GENETIC_SD = {'TPI':250,'NM$':275,'CM$':280,'MILK':675,'FAT':29,'FAT%':0.05,'PRO':19,'PRO%':0.02,'CFP':25,
              'PL':1.85,'SCS':0.14,'DPR':1.3,'HCR':1.4,'CCR':1.65,'LIV':1.2,'FI':1.0,'MAST':1.0,'FSAV':50,
              'PTAT':0.70,'UDC':0.75,'FLC':0.65,'SCE':1.5,'DCE':1.2,'SSB':1.0,'DSB':1.0}
HERITABILITY = {'TPI':0.30,'NM$':0.30,'CM$':0.30,'MILK':0.25,'FAT':0.25,'FAT%':0.50,'PRO':0.25,'PRO%':0.50,
                'CFP':0.30,'PL':0.08,'SCS':0.12,'DPR':0.04,'LIV':0.05,'FI':0.06,'HCR':0.04,'CCR':0.04,
                'MAST':0.04,'FSAV':0.15,'PTAT':0.30,'UDC':0.25,'FLC':0.15,'SCE':0.08,'DCE':0.06,'SSB':0.06,'DSB':0.04}
BREED_IDEAL = {'MILK':1800,'FAT':90,'PRO':60,'CFP':70,'DPR':1.5,'CCR':2.0,'HCR':2.0,
               'PL':5.0,'LIV':2.0,'UDC':1.5,'FLC':1.0,'MAST':-1.0,'SCS':2.60,'SCE':2.0}
LOWER_IS_BETTER = {'SCS','SCE','SSB','DSB','DCE','MAST'}
INBREEDING_DEPRESSION = {'NM$':-25.0,'TPI':-20.0,'CM$':-25.0,'MILK':-28.5,'FAT':-1.0,'PRO':-0.9,
                         'PL':-0.35,'DPR':-0.03,'SCS':0.007,'LIV':-0.15}
GENETIC_CORRELATIONS = {('MILK','DPR'):-0.35,('MILK','CCR'):-0.30,('MILK','SCS'):0.10,
    ('MILK','PL'):-0.15,('FAT','PRO'):0.65,('FAT','DPR'):-0.25,('PRO','DPR'):-0.30,
    ('DPR','CCR'):0.55,('DPR','PL'):0.45,('SCS','PL'):-0.30,('SCS','MAST'):0.70,
    ('PTAT','UDC'):0.40,('PTAT','FLC'):0.30,('NM$','TPI'):0.85,('NM$','CM$'):0.95,
    ('MILK','FAT'):0.55,('MILK','PRO'):0.85,('PL','LIV'):0.65,('DPR','FI'):0.70,
    ('SCE','SSB'):0.60,('DCE','DSB'):0.55}
DOMINANCE_RATIOS = {'MILK':0.12,'FAT':0.10,'PRO':0.12,'SCS':0.14,'PL':0.15,'DPR':0.20,
                    'CCR':0.15,'UDC':0.11,'FLC':0.08,'LIV':0.12}

# Bull column mappings
BULL_COL_V8 = {'TPI':'TPI','NM$':'NM$','CM$':'CM$','MILK':'PTAM','FAT':'PTAF','FAT%':'PTAF%',
               'PRO':'PTAP','PRO%':'PTAP%','CFP':'CFP','PL':'PL','SCS':'SCS','DPR':'DPR',
               'LIV':'LIV','FI':'FI','HCR':'HCR','CCR':'CCR','MAST':'MAST','FSAV':'F SAV',
               'PTAT':'PTAT','UDC':'UDC','FLC':'FLC','SCE':'SCE','DCE':'DCE','SSB':'SSB','DSB':'DSB'}
ALL_TRAITS_V8 = list(BULL_COL_V8.keys())

BULL_COL_V9 = {'TPI':'TPI','MILK':'PTAM','FAT':'PTAF','FAT%':'PTAF%','PRO':'PTAP','PRO%':'PTAP%',
               'PL':'PL','SCS':'SCS','DPR':'DPR','LIV':'LIV','FI':'FI','CCR':'CCR',
               'PTAT':'PTAT','UDC':'UDC','FLC':'FLC','MAST':'MAST'}
ALL_TRAITS_V9 = list(BULL_COL_V9.keys())

CDCB_COL = {'TPI':None,'MILK':'MLK_PTA','FAT':'FAT_PTA','FAT%':'FPT_PTA','PRO':'PRO_PTA','PRO%':'PPT_PTA',
            'PL':'PL__PTA','SCS':'SCS_PTA','DPR':'DPR_PTA','LIV':'LIV_PTA','FI':None,'CCR':'CCR_PTA',
            'PTAT':'TYP_PTA','UDC':'UDC_PTA','FLC':'FLC_PTA','MAST':None,
            'NM$':None,'CM$':None,'CFP':None,'HCR':None,'FSAV':None,
            'SCE':None,'DCE':None,'SSB':None,'DSB':None}

# ============================================================
# LOAD DATABASES
# ============================================================
flush_print("=" * 95)
flush_print("  Teste5 — PA vs V8 Oracle vs V9H Predictions")
flush_print("=" * 95)

flush_print("\n  Loading databases...")
bulls = {}
with open(BULLS_CSV, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        naab = row.get('NAAB','').strip()
        if naab: bulls[naab] = row

cdcb_bulls = {}
try:
    with open(DOWNLOADS / "Bull_Report (1).csv", 'r', encoding='latin-1') as f:
        for row in csv.DictReader(f):
            naab = row.get('NAAB_CODE','').strip()
            if naab: cdcb_bulls[naab] = row
except: pass

flush_print(f"    bulls.csv: {len(bulls)}, CDCB: {len(cdcb_bulls)}")

def lookup_bull(naab):
    for c in normalize_naab(naab):
        if c in bulls: return bulls[c], 'SS'
        if c in cdcb_bulls: return cdcb_bulls[c], 'CDCB'
    return None, None

def get_bull_ptas_v8(bull_row, source='SS'):
    ptas = {}
    col_map = BULL_COL_V8 if source == 'SS' else CDCB_COL
    for trait, col in col_map.items():
        if col is None: continue
        v = sf(bull_row.get(col))
        if v is not None: ptas[trait] = v
    return ptas

def get_bull_ptas_v9(bull_row, source='SS'):
    ptas = {}
    col_map = BULL_COL_V9 if source == 'SS' else CDCB_COL
    for trait, col in col_map.items():
        if col is None: continue
        v = sf(bull_row.get(col))
        if v is not None: ptas[trait] = v
    return ptas

# ============================================================
# LOAD MODELS
# ============================================================
flush_print("\n  Loading models...")
with open(V8_DIR / "v8_oracle_models.pkl", 'rb') as f: v8_models = pickle.load(f)
with open(V9_DIR / "v9_holdout_models.pkl", 'rb') as f: v9_models = pickle.load(f)
with open(V9_DIR / "v9_holdout_sire_profiles.pkl", 'rb') as f: sire_profiles = pickle.load(f)
with open(V9_DIR / "v9_holdout_mgs_profiles.pkl", 'rb') as f: mgs_profiles = pickle.load(f)
flush_print(f"    V8 models: {len(v8_models)} traits")
flush_print(f"    V9H models: {len(v9_models)} traits")
flush_print(f"    Sire profiles: {len(sire_profiles)}, MGS profiles: {len(mgs_profiles)}")

# ============================================================
# FEATURE BUILDERS
# ============================================================
def build_features_v8(sire_ptas, dam_ptas, trait):
    """V8 Oracle features (with PA)."""
    s = sire_ptas.get(trait)
    d = dam_ptas.get(trait)
    if s is None or d is None: return None
    f = {}
    sd = GENETIC_SD.get(trait, 1); h2 = HERITABILITY.get(trait, 0.15)
    pa = (s + d) / 2
    f['sire']=s; f['dam']=d; f['pa']=pa
    f['delta_g']=(s-d)/2; f['diff']=s-d
    f['sire_z']=s/sd if sd else 0; f['dam_z']=d/sd if sd else 0
    f['sire_sq']=s*s; f['dam_sq']=d*d; f['sxd']=s*d
    f['h2_pa']=pa*(0.5+h2)
    ideal = BREED_IDEAL.get(trait)
    if ideal is not None:
        defic = max(0, ((d-ideal)/sd) if trait in LOWER_IS_BETTER else ((ideal-d)/sd))
        f['deficiency']=defic; f['dg_x_def']=f['delta_g']*defic
    else:
        f['deficiency']=0; f['dg_x_def']=0
    f['ib_effect']=INBREEDING_DEPRESSION.get(trait,0)*0.085
    dom = DOMINANCE_RATIOS.get(trait,0)
    f['dom_pot']=dom*abs(s-d)/sd if sd else 0
    for ot in ALL_TRAITS_V8:
        if ot==trait: continue
        sv=sire_ptas.get(ot); dv=dam_ptas.get(ot)
        if sv is not None: f[f's_{ot}']=sv
        if dv is not None: f[f'd_{ot}']=dv
    for (t1,t2),corr in GENETIC_CORRELATIONS.items():
        if t1==trait:
            ov=sire_ptas.get(t2); dov=dam_ptas.get(t2)
            if ov is not None and dov is not None: f[f'gc_{t2}']=((ov+dov)/2)*corr
        elif t2==trait:
            ov=sire_ptas.get(t1); dov=dam_ptas.get(t1)
            if ov is not None and dov is not None: f[f'gc_{t1}']=((ov+dov)/2)*corr
    milk_pa=((sire_ptas.get('MILK',0) or 0)+(dam_ptas.get('MILK',0) or 0))/2
    fat_pa=((sire_ptas.get('FAT',0) or 0)+(dam_ptas.get('FAT',0) or 0))/2
    pro_pa=((sire_ptas.get('PRO',0) or 0)+(dam_ptas.get('PRO',0) or 0))/2
    f['prod_press']=milk_pa/675+fat_pa/29+pro_pa/19
    f['sire_nm']=sire_ptas.get('NM$',0) or 0
    f['dam_nm']=dam_ptas.get('NM$',0) or 0
    return f

def build_features_v9(sire_ptas, dam_ptas, trait, sire_naab=None, mgs_naab=None):
    """V9H features (NO PA)."""
    s = sire_ptas.get(trait)
    if s is None: return None
    d = dam_ptas.get(trait, 0)
    f = {}; sd = GENETIC_SD.get(trait, 1); h2 = HERITABILITY.get(trait, 0.15)
    f['sire']=s; f['dam']=d
    f['delta_g']=(s-d)/2; f['diff']=s-d
    f['sire_z']=s/sd if sd else 0; f['dam_z']=d/sd if sd else 0
    f['sire_sq']=s*s; f['dam_sq']=d*d; f['sxd']=s*d
    f['sire_h2']=s*h2; f['dam_h2']=d*h2; f['has_dam']=1 if d!=0 else 0
    for ot in ALL_TRAITS_V9:
        if ot==trait: continue
        sv=sire_ptas.get(ot); dv=dam_ptas.get(ot)
        if sv is not None: f[f's_{ot}']=sv
        if dv is not None: f[f'd_{ot}']=dv
    for (t1,t2),corr in GENETIC_CORRELATIONS.items():
        if t1==trait or t2==trait:
            other=t2 if t1==trait else t1
            sv=sire_ptas.get(other); dv=dam_ptas.get(other,0)
            if sv is not None:
                f[f'gc_s_{other}']=sv*corr
                f[f'gc_d_{other}']=dv*corr
    if sire_naab and sire_naab in sire_profiles:
        sp=sire_profiles[sire_naab]; tp=sp.get(trait)
        if tp and tp['n']>=2:
            f['sire_n_daughters']=tp['n']; f['sire_daughter_mean']=tp['mean']
            f['sire_daughter_std']=tp['std']; f['sire_trans_ratio']=tp['trans_ratio']
            f['sire_residual']=tp['residual']
        else:
            f['sire_n_daughters']=0; f['sire_daughter_mean']=0; f['sire_daughter_std']=0
            f['sire_trans_ratio']=1.0; f['sire_residual']=0
        for ot in ALL_TRAITS_V9:
            if ot==trait: continue
            otp=sp.get(ot)
            if otp and otp['n']>=2: f[f'sire_res_{ot}']=otp['residual']
    else:
        f['sire_n_daughters']=0; f['sire_daughter_mean']=0; f['sire_daughter_std']=0
        f['sire_trans_ratio']=1.0; f['sire_residual']=0
    if mgs_naab and mgs_naab in mgs_profiles:
        mp=mgs_profiles[mgs_naab]; tp=mp.get(trait)
        if tp and tp['n']>=2:
            f['mgs_n_daughters']=tp['n']; f['mgs_daughter_mean']=tp['mean']; f['mgs_residual']=tp['residual']
        else:
            f['mgs_n_daughters']=0; f['mgs_daughter_mean']=0; f['mgs_residual']=0
    else:
        f['mgs_n_daughters']=0; f['mgs_daughter_mean']=0; f['mgs_residual']=0
    return f

# ============================================================
# LOAD TESTE5
# ============================================================
flush_print("\n  Loading Teste5...")
teste5 = pd.read_excel(DOWNLOADS / "Teste5.xls")
flush_print(f"    {len(teste5)} animals")
flush_print(f"    Columns: {list(teste5.columns)}")

# ============================================================
# PREDICT
# ============================================================
flush_print("\n  Running predictions...")

# Common traits for comparison (present in both V8 and V9)
COMPARE_TRAITS = ['TPI','MILK','FAT','FAT%','PRO','PRO%','PL','SCS','DPR','LIV','FI','CCR','PTAT','UDC','FLC','MAST']

results = []
not_found_sires = set()
found = 0

for _, row in teste5.iterrows():
    animal_id = str(row['ID Animal']).strip()
    sire_naab = str(row.get('Pai','')).strip()
    mgs_naab = str(row.get('mgs','')).strip()
    mmgs_naab = str(row.get('mmgs','')).strip()

    # Lookup sire
    sire_row, sire_src = lookup_bull(sire_naab)
    if not sire_row:
        not_found_sires.add(sire_naab)
        continue
    found += 1

    sire_ptas_v8 = get_bull_ptas_v8(sire_row, sire_src)
    sire_ptas_v9 = get_bull_ptas_v9(sire_row, sire_src)

    # Lookup MGS
    mgs_row, mgs_src = lookup_bull(mgs_naab)
    mgs_ptas_v8 = get_bull_ptas_v8(mgs_row, mgs_src) if mgs_row else {}
    mgs_ptas_v9 = get_bull_ptas_v9(mgs_row, mgs_src) if mgs_row else {}

    # Lookup MMGS (bisavo)
    mmgs_row, mmgs_src = lookup_bull(mmgs_naab)
    mmgs_ptas_v8 = get_bull_ptas_v8(mmgs_row, mmgs_src) if mmgs_row else {}

    # Dam estimation
    # V8: Dam = MGS/2 + MMGS/4
    dam_ptas_v8 = {}
    for t in ALL_TRAITS_V8:
        mgs_v = mgs_ptas_v8.get(t)
        mmgs_v = mmgs_ptas_v8.get(t)
        if mgs_v is not None and mmgs_v is not None:
            dam_ptas_v8[t] = mgs_v/2 + mmgs_v/4
        elif mgs_v is not None:
            dam_ptas_v8[t] = mgs_v/2
        elif mmgs_v is not None:
            dam_ptas_v8[t] = mmgs_v/4

    # V9H: Dam = MGS/2 (same as training)
    dam_ptas_v9 = {t: v/2 for t,v in mgs_ptas_v9.items()} if mgs_ptas_v9 else {}

    r = {'Animal_ID': animal_id, 'Sire': sire_naab, 'MGS': mgs_naab, 'MMGS': mmgs_naab}

    for trait in COMPARE_TRAITS:
        s_v9 = sire_ptas_v9.get(trait)
        d_v9 = dam_ptas_v9.get(trait, 0)
        s_v8 = sire_ptas_v8.get(trait)
        d_v8 = dam_ptas_v8.get(trait)

        # PA (using best dam estimate: MGS/2 + MMGS/4)
        if s_v8 is not None and d_v8 is not None:
            pa = (s_v8 + d_v8) / 2
        elif s_v9 is not None:
            pa = (s_v9 + d_v9) / 2 if d_v9 != 0 else s_v9 / 2
        else:
            pa = None

        # V8 Oracle prediction
        v8_pred = None
        if trait in v8_models and s_v8 is not None and d_v8 is not None:
            feat8 = build_features_v8(sire_ptas_v8, dam_ptas_v8, trait)
            if feat8:
                mi8 = v8_models[trait]
                X8 = np.array([[feat8.get(c, 0) for c in mi8['feature_cols']]])
                v8_pred = mi8['model'].predict(X8)[0]

        # V9H prediction
        v9_pred = None
        if trait in v9_models and s_v9 is not None:
            feat9 = build_features_v9(sire_ptas_v9, dam_ptas_v9, trait, sire_naab, mgs_naab)
            if feat9:
                mi9 = v9_models[trait]
                X9 = np.array([[feat9.get(c, 0) for c in mi9['feature_cols']]])
                v9_pred = mi9['model'].predict(X9)[0]

        if pa is not None: r[f'{trait}_PA'] = round(pa, 2)
        if v8_pred is not None: r[f'{trait}_V8'] = round(v8_pred, 2)
        if v9_pred is not None: r[f'{trait}_V9H'] = round(v9_pred, 2)

    results.append(r)

flush_print(f"    Found: {found}/{len(teste5)}")
flush_print(f"    Sires not found: {len(not_found_sires)}")
if not_found_sires:
    flush_print(f"    Missing: {sorted(not_found_sires)[:15]}")

# ============================================================
# COMPARISON TABLE
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  PREDICTIONS COMPARISON: PA vs V8 vs V9H")
flush_print(f"{'='*95}")

res_df = pd.DataFrame(results)

flush_print(f"\n  {'Trait':>6} | {'N':>4} | {'PA mean':>8} | {'V8 mean':>8} | {'V9H mean':>8} | {'PA std':>7} | {'V8 std':>7} | {'V9H std':>7}")
flush_print(f"  {'-'*80}")

for trait in COMPARE_TRAITS:
    pa_c = f'{trait}_PA'; v8_c = f'{trait}_V8'; v9_c = f'{trait}_V9H'
    cols_present = [c for c in [pa_c, v8_c, v9_c] if c in res_df.columns]
    if len(cols_present) < 2: continue

    valid = pd.Series(True, index=res_df.index)
    for c in cols_present:
        valid &= res_df[c].notna()
    sub = res_df[valid]
    if len(sub) < 5: continue

    parts = [f"  {trait:>6} | {len(sub):>4}"]
    for c in [pa_c, v8_c, v9_c]:
        if c in sub.columns:
            parts.append(f"{sub[c].astype(float).mean():>8.2f}")
        else:
            parts.append(f"{'N/A':>8}")
    for c in [pa_c, v8_c, v9_c]:
        if c in sub.columns:
            parts.append(f"{sub[c].astype(float).std():>7.2f}")
        else:
            parts.append(f"{'N/A':>7}")
    flush_print(" | ".join(parts))

# Sign agreement between V8 and V9H
flush_print(f"\n  Sign agreement (V8 vs V9H):")
flush_print(f"  {'Trait':>6} | {'N':>4} | {'Same sign':>10} | {'%':>5}")
flush_print(f"  {'-'*40}")
for trait in COMPARE_TRAITS:
    v8_c = f'{trait}_V8'; v9_c = f'{trait}_V9H'
    if v8_c not in res_df.columns or v9_c not in res_df.columns: continue
    valid = res_df[v8_c].notna() & res_df[v9_c].notna()
    sub = res_df[valid]
    if len(sub) < 5: continue
    v8 = sub[v8_c].astype(float).values
    v9 = sub[v9_c].astype(float).values
    same = np.sum((v8 * v9) >= 0)
    flush_print(f"  {trait:>6} | {len(sub):>4} | {same:>10} | {same/len(sub)*100:>4.0f}%")

# ============================================================
# SAVE
# ============================================================
out_path = OUTPUT_DIR / "Teste5_PA_V8_V9H_Predictions.xlsx"
res_df.to_excel(out_path, index=False, engine='openpyxl')
flush_print(f"\n  Saved: {out_path}")
flush_print(f"  Done!")
