"""
DSII V10 — Prediction Engine
Input: Sire NAAB + MGS NAAB + MMGS NAAB
Output: Predicted genomic PTAs
Rule: ALL 3 lineages must be found, otherwise skip.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, csv, pickle, sys, re
from pathlib import Path
from collections import defaultdict

DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")
V10_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v10_results")

def sf(v):
    if v is None: return None
    try:
        x = float(v)
        return x if not np.isnan(x) else None
    except: return None

def flush_print(*args, **kwargs):
    print(*args, **kwargs); sys.stdout.flush()

STUD_MAP = {'507':'7','509':'9','550':'50','551':'51','559':'59','518':'18','521':'21',
            '523':'23','581':'81','585':'85','501':'1','601':'1','604':'4','614':'14',
            '629':'29','751':'151','752':'152','814':'14','250':'200'}

def normalize_naab(naab):
    if not naab or str(naab).strip() in ('','nan','None'): return []
    naab = str(naab).strip()
    m = re.match(r'^(\d+)(HO|BS|H|B)0*(\d+)$', naab)
    if not m: return [naab]
    org, breed, num = m.group(1), m.group(2), m.group(3)
    if breed == 'H': breed = 'HO'
    elif breed == 'B': breed = 'BS'
    orgs = [org]
    if org.startswith('5'): orgs.append(org[1:])
    if org in STUD_MAP: orgs.append(STUD_MAP[org])
    orgs.append(org.lstrip('0') or org)
    return list(dict.fromkeys(f'{o}{breed}{num}' for o in orgs))

GENETIC_SD = {'TPI':250,'MILK':675,'FAT':29,'FAT%':0.05,'PRO':19,'PRO%':0.02,
              'PL':1.85,'SCS':0.14,'DPR':1.3,'LIV':1.2,'FI':1.0,
              'CCR':1.65,'PTAT':0.70,'UDC':0.75,'FLC':0.65,'MAST':1.0}
HERITABILITY = {'TPI':0.30,'MILK':0.25,'FAT':0.25,'FAT%':0.50,'PRO':0.25,'PRO%':0.50,
                'PL':0.08,'SCS':0.12,'DPR':0.04,'LIV':0.05,'FI':0.06,
                'CCR':0.04,'PTAT':0.30,'UDC':0.25,'FLC':0.15,'MAST':0.04}
GENETIC_CORRELATIONS = {
    ('MILK','DPR'):-0.35,('MILK','CCR'):-0.30,('MILK','SCS'):0.10,('MILK','PL'):-0.15,
    ('FAT','PRO'):0.65,('FAT','DPR'):-0.25,('PRO','DPR'):-0.30,('DPR','CCR'):0.55,
    ('DPR','PL'):0.45,('SCS','PL'):-0.30,('SCS','MAST'):0.70,('PTAT','UDC'):0.40,
    ('PTAT','FLC'):0.30,('MILK','FAT'):0.55,('MILK','PRO'):0.85,('PL','LIV'):0.65,('DPR','FI'):0.70}

BULL_COL = {'TPI':'TPI','MILK':'PTAM','FAT':'PTAF','FAT%':'PTAF%','PRO':'PTAP','PRO%':'PTAP%',
            'PL':'PL','SCS':'SCS','DPR':'DPR','LIV':'LIV','FI':'FI','CCR':'CCR',
            'PTAT':'PTAT','UDC':'UDC','FLC':'FLC','MAST':'MAST'}
CDCB_COL = {'TPI':None,'MILK':'MLK_PTA','FAT':'FAT_PTA','FAT%':'FPT_PTA','PRO':'PRO_PTA',
            'PRO%':'PPT_PTA','PL':'PL__PTA','SCS':'SCS_PTA','DPR':'DPR_PTA','LIV':'LIV_PTA',
            'FI':None,'CCR':'CCR_PTA','PTAT':'TYP_PTA','UDC':'UDC_PTA','FLC':'FLC_PTA','MAST':None}
ALL_TRAITS = list(BULL_COL.keys())

# ============================================================
# LOAD
# ============================================================
flush_print("=" * 95)
flush_print("  DSII V10 — Prediction Engine")
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

def get_bull_ptas(bull_row, source='SS'):
    ptas = {}
    col_map = BULL_COL if source == 'SS' else CDCB_COL
    for trait, col in col_map.items():
        if col is None: continue
        v = sf(bull_row.get(col))
        if v is not None: ptas[trait] = v
    return ptas

flush_print("  Loading V10 models...")
with open(V10_DIR / "v10_models.pkl", 'rb') as f: v10_models = pickle.load(f)
with open(V10_DIR / "v10_sire_profiles.pkl", 'rb') as f: sire_profiles = pickle.load(f)
with open(V10_DIR / "v10_mgs_profiles.pkl", 'rb') as f: mgs_profiles = pickle.load(f)
flush_print(f"    V10 models: {len(v10_models)} traits")

# ============================================================
# FEATURE BUILDER (same as training)
# ============================================================
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
        f['sire_x_mgs'] = s*mg; f['sire_mgs_diff'] = s-mg
        f['sire_mgs_ratio'] = s/mg if mg != 0 else 0
    else:
        f['sire_x_mgs'] = 0; f['sire_mgs_diff'] = 0; f['sire_mgs_ratio'] = 0

    mmg = mmgs_ptas.get(trait) if mmgs_ptas else None
    f['mmgs_val'] = mmg if mmg is not None else 0
    f['has_mmgs'] = 1 if mmg is not None else 0
    f['sire_x_mmgs'] = s*mmg if mmg is not None else 0

    for ot in ALL_TRAITS:
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

    if sire_naab and sire_naab in sire_profiles:
        sp = sire_profiles[sire_naab]; tp = sp.get(trait)
        if tp and tp['n'] >= 2:
            f['sire_prof_n']=tp['n']; f['sire_prof_mean']=tp['mean']
            f['sire_prof_std']=tp['std']; f['sire_prof_ratio']=tp['trans_ratio']
            f['sire_prof_resid']=tp['residual']
        else:
            f['sire_prof_n']=0; f['sire_prof_mean']=0; f['sire_prof_std']=0
            f['sire_prof_ratio']=1.0; f['sire_prof_resid']=0
        for ot in ALL_TRAITS:
            if ot==trait: continue
            otp = sp.get(ot)
            if otp and otp['n']>=2: f[f'sire_prof_{ot}'] = otp['residual']
    else:
        f['sire_prof_n']=0; f['sire_prof_mean']=0; f['sire_prof_std']=0
        f['sire_prof_ratio']=1.0; f['sire_prof_resid']=0

    if mgs_naab and mgs_naab in mgs_profiles:
        mp = mgs_profiles[mgs_naab]; tp = mp.get(trait)
        if tp and tp['n']>=2:
            f['mgs_prof_n']=tp['n']; f['mgs_prof_mean']=tp['mean']; f['mgs_prof_resid']=tp['residual']
        else:
            f['mgs_prof_n']=0; f['mgs_prof_mean']=0; f['mgs_prof_resid']=0
    else:
        f['mgs_prof_n']=0; f['mgs_prof_mean']=0; f['mgs_prof_resid']=0

    return f

def predict_v10(feat, trait):
    if trait not in v10_models: return None
    mi = v10_models[trait]
    X = np.array([[feat.get(c, 0) for c in mi['feature_cols']]])

    if mi['type'] == 'single':
        return mi['model'].predict(X)[0]
    elif mi['type'] == 'stack':
        base_preds = []
        for mn in mi['base_names']:
            base_preds.append(mi['base_models'][mn].predict(X)[0])
        stack_X = np.array([base_preds])
        return mi['meta_model'].predict(stack_X)[0]

# ============================================================
# RUN ON TESTE5
# ============================================================
flush_print("\n  Loading Teste5...")
teste5 = pd.read_excel(DOWNLOADS / "Teste5.xls")
flush_print(f"    {len(teste5)} animals")

# Load genomic for comparison
gen = pd.read_excel(DOWNLOADS / "daniel andre da silva sms 040526.xls")
gen['Brinco'] = gen['Brinco'].astype(str).str.strip()
gen_map = gen.set_index('Brinco').to_dict('index')

GEN_TRAIT_MAP = {'TPI':'TPI','MILK':'Leite lbs','FAT':'Gordura lbs','SCS':'CCS','PL':'VP','DPR':'DPR'}
# Fix Proteina col
for c in gen.columns:
    if 'rote' in c.lower() and 'lbs' in c.lower():
        GEN_TRAIT_MAP['PRO'] = c
        break

flush_print("\n  Running V10 predictions...")
results = []
skipped_sire = set()
skipped_mgs = set()
skipped_mmgs = set()

for _, row in teste5.iterrows():
    animal_id = str(row['ID Animal']).strip()
    sire_naab = str(row.get('Pai','')).strip()
    mgs_naab = str(row.get('mgs','')).strip()
    mmgs_naab = str(row.get('mmgs','')).strip()

    # ALL 3 must be found
    sire_row, sire_src = lookup_bull(sire_naab)
    if not sire_row:
        skipped_sire.add(sire_naab)
        continue
    sire_ptas = get_bull_ptas(sire_row, sire_src)

    mgs_row, mgs_src = lookup_bull(mgs_naab)
    if not mgs_row:
        skipped_mgs.add(mgs_naab)
        continue
    mgs_ptas = get_bull_ptas(mgs_row, mgs_src)

    mmgs_row, mmgs_src = lookup_bull(mmgs_naab)
    if not mmgs_row:
        skipped_mmgs.add(mmgs_naab)
        continue
    mmgs_ptas = get_bull_ptas(mmgs_row, mmgs_src)

    r = {'Animal_ID': animal_id, 'Sire': sire_naab, 'MGS': mgs_naab, 'MMGS': mmgs_naab}

    # Get genomic if available
    gen_data = gen_map.get(animal_id, {})

    for trait in ALL_TRAITS:
        feat = build_features_v10(sire_ptas, mgs_ptas, trait, sire_naab, mgs_naab, mmgs_ptas)
        if feat is None: continue
        pred = predict_v10(feat, trait)
        if pred is not None:
            r[f'{trait}_V10'] = round(pred, 2)

        # Genomic for comparison
        gen_col = GEN_TRAIT_MAP.get(trait)
        if gen_col and gen_col in gen_data:
            gv = sf(gen_data[gen_col])
            if gv is not None:
                r[f'{trait}_Genomic'] = round(gv, 2)

    results.append(r)

flush_print(f"    Predicted: {len(results)}/{len(teste5)}")
flush_print(f"    Skipped - Sire not found: {len(skipped_sire)} ({sorted(skipped_sire)})")
flush_print(f"    Skipped - MGS not found: {len(skipped_mgs)} ({sorted(skipped_mgs)})")
flush_print(f"    Skipped - MMGS not found: {len(skipped_mmgs)} ({sorted(skipped_mmgs)})")

# ============================================================
# ACCURACY vs GENOMIC
# ============================================================
res_df = pd.DataFrame(results)

flush_print(f"\n{'='*95}")
flush_print(f"  V10 vs Genomic — Teste5 Validation")
flush_print(f"{'='*95}")
flush_print(f"\n  {'Trait':>5} | {'N':>4} | {'V10 dist':>8} | {'V10 corr':>8} | {'Sign inv':>8}")
flush_print(f"  {'-'*50}")

for trait in ALL_TRAITS:
    v10_c = f'{trait}_V10'
    gen_c = f'{trait}_Genomic'
    if v10_c not in res_df.columns or gen_c not in res_df.columns: continue
    valid = res_df[v10_c].notna() & res_df[gen_c].notna()
    sub = res_df[valid]
    if len(sub) < 5: continue
    actual = sub[gen_c].astype(float).values
    pred = sub[v10_c].astype(float).values
    dist = np.mean(np.abs(actual - pred))
    corr = np.corrcoef(actual, pred)[0,1]
    inv = np.sum((actual * pred) < 0)
    flush_print(f"  {trait:>5} | {len(sub):>4} | {dist:>8.2f} | {corr:>8.4f} | {inv:>3} ({inv/len(sub)*100:.0f}%)")

# Save
out_path = V10_DIR / "Teste5_V10_vs_Genomic.xlsx"
res_df.to_excel(out_path, index=False, engine='openpyxl')
flush_print(f"\n  Saved: {out_path}")
flush_print(f"  Done!")
