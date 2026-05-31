"""
DSII V9 Lineage — Proper Holdout Validation.
Excludes Teste1 animals from training, recomputes sire/MGS profiles without them,
retrains models, then predicts and compares vs genomic gold standard.
This eliminates data leakage and gives honest accuracy numbers.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, csv, pickle, sys, re, openpyxl
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")
OUTPUT_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v9_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    m = re.match(r'^(\d+)(HO|BS)0*(\d+)$', naab)
    if not m: return [naab]
    org, breed, num = m.group(1), m.group(2), m.group(3)
    orgs = [org]
    if org.startswith('5'): orgs.append(org[1:])
    if org in STUD_MAP: orgs.append(STUD_MAP[org])
    orgs.append(org.lstrip('0') or org)
    return list(dict.fromkeys(f'{o}{breed}{num}' for o in orgs))

GENETIC_SD = {'TPI':250,'MILK':675,'FAT':29,'FAT%':0.05,'PRO':19,'PRO%':0.02,'PL':1.85,'SCS':0.14,'DPR':1.3,'LIV':1.2,'FI':1.0,'CCR':1.65,'PTAT':0.70,'UDC':0.75,'FLC':0.65,'MAST':1.0}
HERITABILITY = {'TPI':0.30,'MILK':0.25,'FAT':0.25,'FAT%':0.50,'PRO':0.25,'PRO%':0.50,'PL':0.08,'SCS':0.12,'DPR':0.04,'LIV':0.05,'FI':0.06,'CCR':0.04,'PTAT':0.30,'UDC':0.25,'FLC':0.15,'MAST':0.04}
GENETIC_CORRELATIONS = {('MILK','DPR'):-0.35,('MILK','CCR'):-0.30,('MILK','SCS'):0.10,('MILK','PL'):-0.15,('FAT','PRO'):0.65,('FAT','DPR'):-0.25,('PRO','DPR'):-0.30,('DPR','CCR'):0.55,('DPR','PL'):0.45,('SCS','PL'):-0.30,('SCS','MAST'):0.70,('PTAT','UDC'):0.40,('PTAT','FLC'):0.30,('MILK','FAT'):0.55,('MILK','PRO'):0.85,('PL','LIV'):0.65,('DPR','FI'):0.70}
BULL_COL = {'TPI':'TPI','MILK':'PTAM','FAT':'PTAF','FAT%':'PTAF%','PRO':'PTAP','PRO%':'PTAP%','PL':'PL','SCS':'SCS','DPR':'DPR','LIV':'LIV','FI':'FI','CCR':'CCR','PTAT':'PTAT','UDC':'UDC','FLC':'FLC','MAST':'MAST'}
CDCB_COL = {'TPI':None,'MILK':'MLK_PTA','FAT':'FAT_PTA','FAT%':'FPT_PTA','PRO':'PRO_PTA','PRO%':'PPT_PTA','PL':'PL__PTA','SCS':'SCS_PTA','DPR':'DPR_PTA','LIV':'LIV_PTA','FI':None,'CCR':'CCR_PTA','PTAT':'TYP_PTA','UDC':'UDC_PTA','FLC':'FLC_PTA','MAST':None}
BANCO_COL = {'TPI':'TPI','MILK':'MILK','FAT':'FAT','FAT%':'FAT %','PRO':'PROT','PRO%':'PROT%','PL':'PL','SCS':'SCS','DPR':'DPR','LIV':'LIV','FI':'FI','CCR':'CCR','PTAT':'TYPE FS','UDC':'UDC','FLC':'FLC','MAST':'CDCB_MAST'}
ALL_TRAITS = list(BULL_COL.keys())

def get_bull_ptas(bull_row, source='SS'):
    ptas = {}
    col_map = BULL_COL if source == 'SS' else CDCB_COL
    for trait, col in col_map.items():
        if col is None: continue
        v = sf(bull_row.get(col))
        if v is not None: ptas[trait] = v
    return ptas

# ============================================================
# LOAD
# ============================================================
flush_print("=" * 95)
flush_print("  DSII V9 Lineage — HOLDOUT VALIDATION (No Data Leakage)")
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

def lookup_bull(naab):
    for c in normalize_naab(naab):
        if c in bulls: return bulls[c], 'SS'
        if c in cdcb_bulls: return cdcb_bulls[c], 'CDCB'
    return None, None

flush_print(f"    bulls.csv: {len(bulls)}, CDCB: {len(cdcb_bulls)}")

# Load banco21052026
flush_print("\n  Loading banco21052026...")
raw = pd.read_excel(DOWNLOADS / "banco21052026.xlsx", header=None, engine='openpyxl')
hdr = [str(c).strip() if c else f'col_{i}' for i, c in enumerate(raw.iloc[10])]
banco = raw.iloc[11:].copy()
banco.columns = hdr
banco = banco.reset_index(drop=True)
banco['Animal ID'] = banco['Animal ID'].astype(str).str.strip()

# Load Teste1
flush_print("  Loading Teste1...")
teste1 = pd.read_excel(DOWNLOADS / "Teste1.xlsx", engine='openpyxl')
teste1['Animal ID'] = teste1['Animal ID'].astype(str).str.strip()

# Load genomic gold standard
flush_print("  Loading ReportCoreTraits (genomic)...")
gen_raw = pd.read_excel(DOWNLOADS / "ReportCoreTraits2026-05-21 (1).xlsx", header=None, engine='openpyxl')
gen_hdr = [str(c).strip() if c else f'col_{i}' for i, c in enumerate(gen_raw.iloc[10])]
gen_df = gen_raw.iloc[11:].copy()
gen_df.columns = gen_hdr
gen_df = gen_df.reset_index(drop=True)
gen_df['Animal ID'] = gen_df['Animal ID'].astype(str).str.strip()

# ============================================================
# SPLIT: exclude Teste1 animals from training
# ============================================================
teste1_ids = set(teste1['Animal ID'])
banco_train = banco[~banco['Animal ID'].isin(teste1_ids)].copy()
banco_test = banco[banco['Animal ID'].isin(teste1_ids)].copy()

flush_print(f"\n  banco total: {len(banco)}")
flush_print(f"  Teste1 animals: {len(teste1_ids)}")
flush_print(f"  TRAINING set (banco - Teste1): {len(banco_train)}")
flush_print(f"  HOLDOUT set (Teste1 in banco): {len(banco_test)}")

# ============================================================
# ASSEMBLE TRAINING RECORDS (excluding Teste1)
# ============================================================
flush_print("\n  Assembling training records (WITHOUT Teste1)...")

records = []
sire_daughters = defaultdict(list)
mgs_daughters = defaultdict(list)

for _, row in banco_train.iterrows():
    sire_naab_raw = str(row.get('Sire of Record NAAB', '')).strip()
    mgs_naab_raw = str(row.get('Maternal Grandsire NAAB', '')).strip()
    animal_id = str(row.get('Animal ID', '')).strip()

    if sire_naab_raw in ('', 'nan', 'None'): continue

    sire_row, sire_src = lookup_bull(sire_naab_raw)
    if not sire_row: continue
    sire_ptas = get_bull_ptas(sire_row, sire_src)

    mgs_ptas = {}
    has_mgs = mgs_naab_raw not in ('', 'nan', 'None')
    if has_mgs:
        mgs_row, mgs_src = lookup_bull(mgs_naab_raw)
        if mgs_row:
            mgs_ptas = get_bull_ptas(mgs_row, mgs_src)

    dam_ptas = {t: v/2 for t,v in mgs_ptas.items()} if mgs_ptas else {}

    daughter_ptas = {}
    for trait, bcol in BANCO_COL.items():
        if bcol and bcol in row.index:
            v = sf(row[bcol])
            if v is not None: daughter_ptas[trait] = v

    if not sire_ptas or not daughter_ptas: continue

    for t, v in daughter_ptas.items():
        sire_daughters[sire_naab_raw].append((t, v))
        if has_mgs:
            mgs_daughters[mgs_naab_raw].append((t, v))

    records.append({
        'animal_id': animal_id,
        'sire_naab': sire_naab_raw,
        'mgs_naab': mgs_naab_raw if has_mgs else None,
        'sire_ptas': sire_ptas,
        'dam_ptas': dam_ptas,
        'daughter_ptas': daughter_ptas,
    })

flush_print(f"    Training records: {len(records)}")

# ============================================================
# COMPUTE SIRE/MGS PROFILES (from training only)
# ============================================================
flush_print("\n  Computing sire transmission profiles (training only)...")

sire_profiles = {}
for sire_naab, daughter_vals in sire_daughters.items():
    sire_row, src = lookup_bull(sire_naab)
    if not sire_row: continue
    sire_ptas = get_bull_ptas(sire_row, src)
    profile = {}
    trait_vals = defaultdict(list)
    for t, v in daughter_vals:
        trait_vals[t].append(v)
    for t, vals in trait_vals.items():
        if len(vals) >= 2 and t in sire_ptas:
            sp = sire_ptas[t]
            mean_d = np.mean(vals)
            std_d = np.std(vals)
            trans_ratio = mean_d / (sp/2) if sp != 0 else 1.0
            profile[t] = {'n':len(vals),'mean':mean_d,'std':std_d,'sire_pta':sp,
                          'trans_ratio':trans_ratio,'residual':mean_d - sp/2}
    if profile: sire_profiles[sire_naab] = profile

mgs_profiles = {}
for mgs_naab, daughter_vals in mgs_daughters.items():
    mgs_row, src = lookup_bull(mgs_naab)
    if not mgs_row: continue
    mgs_ptas = get_bull_ptas(mgs_row, src)
    profile = {}
    trait_vals = defaultdict(list)
    for t, v in daughter_vals:
        trait_vals[t].append(v)
    for t, vals in trait_vals.items():
        if len(vals) >= 2 and t in mgs_ptas:
            mp = mgs_ptas[t]
            mean_d = np.mean(vals)
            profile[t] = {'n':len(vals),'mean':mean_d,'std':np.std(vals),
                          'mgs_pta':mp,'residual':mean_d - mp/4}
    if profile: mgs_profiles[mgs_naab] = profile

flush_print(f"    Sire profiles (train only): {len(sire_profiles)}")
flush_print(f"    MGS profiles (train only): {len(mgs_profiles)}")

# ============================================================
# BUILD FEATURES
# ============================================================
def build_features_v9(sire_ptas, dam_ptas, trait, sire_naab=None, mgs_naab=None):
    s = sire_ptas.get(trait)
    if s is None: return None
    d = dam_ptas.get(trait, 0)
    f = {}; sd = GENETIC_SD.get(trait, 1); h2 = HERITABILITY.get(trait, 0.15)
    # NO PA features — model learns its own weighting
    f['sire']=s; f['dam']=d
    f['delta_g']=(s-d)/2; f['diff']=s-d
    f['sire_z']=s/sd if sd else 0; f['dam_z']=d/sd if sd else 0
    f['sire_sq']=s*s; f['dam_sq']=d*d; f['sxd']=s*d
    f['sire_h2']=s*h2; f['dam_h2']=d*h2; f['has_dam']=1 if d!=0 else 0
    for ot in ALL_TRAITS:
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
        for ot in ALL_TRAITS:
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
# TRAIN MODELS (on training set only)
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  TRAINING V9 MODELS (excluding Teste1)")
flush_print(f"{'='*95}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

ORACLE_MAP = {
    'TPI':'GBR','MILK':'GBR','FAT':'GBR','FAT%':'GBR',
    'PRO':'LGBM','PRO%':'GBR','PL':'LGBM','SCS':'GBR',
    'DPR':'LGBM','LIV':'GBR','FI':'LGBM','CCR':'GBR',
    'PTAT':'XGB','UDC':'GBR','FLC':'LGBM','MAST':'LGBM',
}

def get_model(name):
    if name=='GBR': return GradientBoostingRegressor(n_estimators=500,max_depth=5,learning_rate=0.03,min_samples_leaf=5,subsample=0.8,random_state=42)
    elif name=='LGBM': return LGBMRegressor(n_estimators=500,max_depth=6,learning_rate=0.03,subsample=0.8,colsample_bytree=0.7,min_child_samples=5,reg_alpha=0.1,reg_lambda=1.0,random_state=42,verbose=-1)
    elif name=='XGB': return XGBRegressor(n_estimators=500,max_depth=6,learning_rate=0.03,subsample=0.8,colsample_bytree=0.7,reg_alpha=0.1,reg_lambda=1.0,min_child_weight=5,random_state=42,verbosity=0)
    elif name=='RF': return RandomForestRegressor(n_estimators=500,max_depth=12,min_samples_leaf=5,random_state=42,n_jobs=-1)

flush_print(f"\n  {'Trait':>6} | {'N':>5} | {'Model':>5} | {'CV R2':>7} | {'CV MAE':>8} | {'PA R2':>7} | {'PA MAE':>8}")
flush_print(f"  {'-'*70}")

saved_models = {}
cv_results = []

for trait in ALL_TRAITS:
    feat_rows, y_vals, pa_vals = [], [], []
    for rec in records:
        dv = rec['daughter_ptas'].get(trait)
        if dv is None: continue
        feat = build_features_v9(rec['sire_ptas'], rec['dam_ptas'], trait, rec['sire_naab'], rec.get('mgs_naab'))
        if feat is None: continue
        feat_rows.append(feat)
        y_vals.append(dv)
        s = rec['sire_ptas'].get(trait, 0)
        d = rec['dam_ptas'].get(trait, 0)
        pa_vals.append((s+d)/2 if d!=0 else s/2)

    if len(feat_rows) < 50: continue

    feat_df = pd.DataFrame(feat_rows)
    feat_df = feat_df.dropna(axis=1, thresh=int(len(feat_df)*0.3))
    feature_cols = list(feat_df.columns)
    for c in feature_cols:
        if feat_df[c].isna().any(): feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    X = feat_df.values
    y = np.array(y_vals)
    pa = np.array(pa_vals)

    model_name = ORACLE_MAP.get(trait, 'GBR')
    r2s, maes = [], []
    for tr, te in kf.split(X):
        m = get_model(model_name)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        r2s.append(r2_score(y[te], pred))
        maes.append(mean_absolute_error(y[te], pred))

    v9_r2 = np.mean(r2s); v9_mae = np.mean(maes)
    pa_r2 = r2_score(y, pa); pa_mae = mean_absolute_error(y, pa)

    flush_print(f"  {trait:>6} | {len(y):>5} | {model_name:>5} | {v9_r2:>7.4f} | {v9_mae:>8.2f} | {pa_r2:>7.4f} | {pa_mae:>8.2f}")

    cv_results.append({'Trait':trait,'N':len(y),'Model':model_name,'CV_R2':round(v9_r2,4),'CV_MAE':round(v9_mae,2),'PA_R2':round(pa_r2,4),'PA_MAE':round(pa_mae,2)})

    final_model = get_model(model_name)
    final_model.fit(X, y)
    saved_models[trait] = {'model':final_model,'model_name':model_name,'feature_cols':feature_cols}

# ============================================================
# HOLDOUT VALIDATION: predict Teste1 animals
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  HOLDOUT VALIDATION: Teste1 vs Genomic (NO DATA LEAKAGE)")
flush_print(f"{'='*95}")

# Merge Teste1 with genomic
GEN_COL = {'TPI':'TPI','MILK':'MILK','FAT':'FAT','FAT%':'FAT %','PRO':'PROT','PRO%':'PROT%','PL':'PL','SCS':'SCS','DPR':'DPR','LIV':'LIV','FI':'FI','CCR':'CCR','PTAT':'TYPE FS','UDC':'UDC','FLC':'FLC','MAST':'CDCB_MAST'}

merged = teste1.merge(gen_df, on='Animal ID', how='inner', suffixes=('','_gen'))
flush_print(f"\n  Teste1: {len(teste1)}, Genomic: {len(gen_df)}, Matched: {len(merged)}")

results_rows = []
for _, row in merged.iterrows():
    animal_id = row['Animal ID']
    sire_naab = str(row.get('Sire of Record NAAB','')).strip()
    mgs_naab = str(row.get('Maternal Grandsire NAAB','')).strip()

    sire_row, sire_src = lookup_bull(sire_naab)
    if not sire_row: continue
    sire_ptas = get_bull_ptas(sire_row, sire_src)

    mgs_row, mgs_src = lookup_bull(mgs_naab)
    mgs_ptas = get_bull_ptas(mgs_row, mgs_src) if mgs_row else {}
    dam_ptas = {t: v/2 for t,v in mgs_ptas.items()} if mgs_ptas else {}

    r = {'Animal_ID': animal_id}
    for trait in ALL_TRAITS:
        gen_col = GEN_COL.get(trait)
        if not gen_col or gen_col not in row.index: continue
        gen_val = sf(row[gen_col])
        if gen_val is None: continue

        s = sire_ptas.get(trait)
        d = dam_ptas.get(trait, 0)
        if s is None: continue
        pa = (s+d)/2 if d!=0 else s/2

        feat = build_features_v9(sire_ptas, dam_ptas, trait, sire_naab, mgs_naab)
        if feat and trait in saved_models:
            mi = saved_models[trait]
            X = np.array([[feat.get(c,0) for c in mi['feature_cols']]])
            v9_pred = mi['model'].predict(X)[0]
        else:
            v9_pred = s / 2  # fallback: sire contribution only, no PA

        r[f'{trait}_PA'] = round(pa, 2)
        r[f'{trait}_V9'] = round(v9_pred, 2)
        r[f'{trait}_Genomic'] = round(gen_val, 2)
    results_rows.append(r)

flush_print(f"  Predictions: {len(results_rows)}")

# ============================================================
# ACCURACY TABLE
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  ACCURACY: PA vs V9 vs Genomic (HOLDOUT — no leakage)")
flush_print(f"{'='*95}")
flush_print(f"\n  {'Trait':>6} | {'N':>4} | {'PA dist':>8} | {'V9 dist':>8} | {'V9 closer':>12} | {'Corr PA':>7} | {'Corr V9':>7} | Winner")
flush_print(f"  {'-'*85}")

res_df = pd.DataFrame(results_rows)
summary = []
for trait in ALL_TRAITS:
    pa_c, v9_c, gen_c = f'{trait}_PA', f'{trait}_V9', f'{trait}_Genomic'
    if pa_c not in res_df.columns or v9_c not in res_df.columns or gen_c not in res_df.columns: continue
    valid = res_df[pa_c].notna() & res_df[v9_c].notna() & res_df[gen_c].notna()
    sub = res_df[valid]
    if len(sub) < 10: continue

    actual = sub[gen_c].values.astype(float)
    pa_v = sub[pa_c].values.astype(float)
    v9_v = sub[v9_c].values.astype(float)

    pa_dist = np.mean(np.abs(actual - pa_v))
    v9_dist = np.mean(np.abs(actual - v9_v))
    v9_closer = np.sum(np.abs(actual - v9_v) < np.abs(actual - pa_v))
    corr_pa = np.corrcoef(actual, pa_v)[0,1]
    corr_v9 = np.corrcoef(actual, v9_v)[0,1]
    winner = 'V9' if v9_dist < pa_dist else 'PA'

    flush_print(f"  {trait:>6} | {len(sub):>4} | {pa_dist:>8.2f} | {v9_dist:>8.2f} | "
                f"{v9_closer:>4}/{len(sub)} ({v9_closer/len(sub)*100:.0f}%) | {corr_pa:>7.4f} | {corr_v9:>7.4f} | {winner}")

    summary.append({'Trait':trait,'N':len(sub),'PA_dist':round(pa_dist,2),'V9_dist':round(v9_dist,2),
                    'V9_closer':v9_closer,'V9_closer_pct':round(v9_closer/len(sub)*100,1),
                    'Corr_PA':round(corr_pa,4),'Corr_V9':round(corr_v9,4),'Winner':winner})

# ============================================================
# SUMMARY
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  SUMMARY (HOLDOUT — honest numbers)")
flush_print(f"{'='*95}")

v9_wins = sum(1 for s in summary if s['Winner']=='V9')
pa_wins = len(summary) - v9_wins
flush_print(f"  Traits evaluated: {len(summary)}")
flush_print(f"  V9 wins (lower dist): {v9_wins}/{len(summary)}")
flush_print(f"  PA wins: {pa_wins}/{len(summary)}")
if summary:
    flush_print(f"  PA avg dist: {np.mean([s['PA_dist'] for s in summary]):.2f}")
    flush_print(f"  V9 avg dist: {np.mean([s['V9_dist'] for s in summary]):.2f}")
    flush_print(f"  V9 closer (animal avg): {np.mean([s['V9_closer_pct'] for s in summary]):.1f}%")
    flush_print(f"  Avg Corr PA: {np.mean([s['Corr_PA'] for s in summary]):.4f}")
    flush_print(f"  Avg Corr V9: {np.mean([s['Corr_V9'] for s in summary]):.4f}")

# Save
out_path = OUTPUT_DIR / "Teste1_HOLDOUT_PA_vs_V9_vs_Genomic.xlsx"
res_df.to_excel(out_path, index=False, engine='openpyxl')
flush_print(f"\n  Saved: {out_path}")

pd.DataFrame(summary).to_csv(OUTPUT_DIR / "holdout_validation_summary.csv", index=False)
pd.DataFrame(cv_results).to_csv(OUTPUT_DIR / "holdout_cv_results.csv", index=False)

# Save holdout models and profiles
with open(OUTPUT_DIR / "v9_holdout_models.pkl", 'wb') as fpk: pickle.dump(saved_models, fpk)
with open(OUTPUT_DIR / "v9_holdout_sire_profiles.pkl", 'wb') as fpk: pickle.dump(sire_profiles, fpk)
with open(OUTPUT_DIR / "v9_holdout_mgs_profiles.pkl", 'wb') as fpk: pickle.dump(mgs_profiles, fpk)

flush_print(f"\n  Done! This is the HONEST validation — no data leakage.")
