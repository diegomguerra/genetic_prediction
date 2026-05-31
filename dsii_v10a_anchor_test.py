"""
DSII V10-A — Anchor Test
Ensaio: traits problema (R2 < 0.6) recebem predicoes das traits boas como features adicionais.
Cascata de ancoragem baseada em correlacoes geneticas da literatura.

Compara V10 original vs V10-A (anchored) em: R2, MAE, SD erro, inversao de sinal.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, pickle, sys
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v10_results")

# Load V10 models and profiles
with open(RESULTS_DIR / 'v10_models.pkl', 'rb') as f:
    v10_models = pickle.load(f)
with open(RESULTS_DIR / 'v10_sire_profiles.pkl', 'rb') as f:
    sire_profiles = pickle.load(f)
with open(RESULTS_DIR / 'v10_mgs_profiles.pkl', 'rb') as f:
    mgs_profiles = pickle.load(f)

# ============================================================
# GENETIC CORRELATION MATRIX (literature consolidated)
# ============================================================
GC = {
    ('MILK','FAT'):0.55, ('MILK','PRO'):0.85, ('MILK','PL'):-0.05, ('MILK','SCS'):0.10,
    ('MILK','DPR'):-0.35, ('MILK','CCR'):-0.30, ('MILK','LIV'):-0.10, ('MILK','PTAT'):0.15,
    ('MILK','UDC'):-0.15, ('MILK','FLC'):-0.05, ('MILK','MAST'):-0.30, ('MILK','FI'):0.30,
    ('MILK','FAT%'):-0.55, ('MILK','PRO%'):-0.40,
    ('FAT','PRO'):0.65, ('FAT','PL'):0.05, ('FAT','SCS'):0.05, ('FAT','DPR'):-0.25,
    ('FAT','CCR'):-0.20, ('FAT','MAST'):-0.20, ('FAT','FI'):0.25, ('FAT','FAT%'):0.50,
    ('PRO','PL'):-0.05, ('PRO','SCS'):0.10, ('PRO','DPR'):-0.35, ('PRO','CCR'):-0.25,
    ('PRO','MAST'):-0.25, ('PRO','FI'):0.30, ('PRO','FAT%'):-0.20, ('PRO','PRO%'):0.10,
    ('PL','SCS'):-0.35, ('PL','DPR'):0.45, ('PL','CCR'):0.35, ('PL','LIV'):0.85,
    ('PL','PTAT'):0.10, ('PL','UDC'):0.25, ('PL','FLC'):0.15, ('PL','MAST'):0.39,
    ('SCS','DPR'):-0.15, ('SCS','CCR'):-0.10, ('SCS','LIV'):-0.25, ('SCS','UDC'):-0.30,
    ('SCS','MAST'):-0.68,
    ('DPR','CCR'):0.55, ('DPR','LIV'):0.40, ('DPR','PTAT'):-0.15, ('DPR','FI'):-0.20,
    ('CCR','LIV'):0.30, ('CCR','MAST'):0.21,
    ('LIV','UDC'):0.20, ('LIV','MAST'):0.22,
    ('PTAT','UDC'):0.45, ('PTAT','FLC'):0.35,
    ('UDC','FLC'):0.20, ('UDC','MAST'):0.15,
    ('FAT%','PRO%'):0.50,
}

def get_gc(t1, t2):
    if (t1, t2) in GC: return GC[(t1, t2)]
    if (t2, t1) in GC: return GC[(t2, t1)]
    return 0.0

# ============================================================
# DEFINE ANCHORS: which good traits help each problem trait
# ============================================================
GOOD_TRAITS = ['TPI', 'FAT', 'PRO', 'PTAT', 'PL', 'FLC', 'MILK', 'UDC', 'FI']  # R2 >= 0.6
PROBLEM_TRAITS = ['DPR', 'CCR', 'SCS', 'MAST', 'LIV', 'FAT%', 'PRO%']

# For each problem trait, list anchors with |corr| >= 0.15
ANCHORS = {}
for pt in PROBLEM_TRAITS:
    anc = []
    for gt in GOOD_TRAITS:
        c = get_gc(pt, gt)
        if abs(c) >= 0.15:
            anc.append((gt, c))
    anc.sort(key=lambda x: -abs(x[1]))
    ANCHORS[pt] = anc

print("=" * 90)
print("  DSII V10-A — Anchor Test: Correlacoes Geneticas como Features")
print("=" * 90)
print()
for pt, ancs in ANCHORS.items():
    r2_orig = v10_models[pt]['r2_cv'] if pt in v10_models else 0
    anc_str = ', '.join([f'{g}({c:+.2f})' for g, c in ancs])
    print(f"  {pt:>5} (R2={r2_orig:.4f}) <- {anc_str}")

# ============================================================
# REBUILD TRAINING DATA (same as V10 engine)
# We need to rerun training with anchor features
# ============================================================
print()
print("  Rebuilding training data...")

# Import functions from v10 engine context
import csv, re, glob
from collections import defaultdict

DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")

GENETIC_SD = {'TPI':250,'MILK':675,'FAT':29,'FAT%':0.05,'PRO':19,'PRO%':0.02,
              'PL':1.85,'SCS':0.14,'DPR':1.3,'LIV':1.2,'FI':1.0,
              'CCR':1.65,'PTAT':0.70,'UDC':0.75,'FLC':0.65,'MAST':1.0}
HERITABILITY = {'TPI':0.30,'MILK':0.25,'FAT':0.25,'FAT%':0.50,'PRO':0.25,'PRO%':0.50,
                'PL':0.08,'SCS':0.12,'DPR':0.04,'LIV':0.05,'FI':0.06,
                'CCR':0.04,'PTAT':0.30,'UDC':0.25,'FLC':0.15,'MAST':0.04}
ALL_TRAITS = ['TPI','MILK','FAT','FAT%','PRO','PRO%','PL','SCS','DPR','LIV','FI','CCR','PTAT','UDC','FLC','MAST']
BULL_COL = {'TPI':'TPI','MILK':'PTAM','FAT':'PTAF','FAT%':'PTAF%','PRO':'PTAP','PRO%':'PTAP%',
            'PL':'PL','SCS':'SCS','DPR':'DPR','LIV':'LIV','FI':'FI','CCR':'CCR',
            'PTAT':'PTAT','UDC':'UDC','FLC':'FLC','MAST':'MAST'}
CDCB_COL = {'TPI':None,'MILK':'MLK_PTA','FAT':'FAT_PTA','FAT%':'FPT_PTA','PRO':'PRO_PTA',
            'PRO%':'PPT_PTA','PL':'PL__PTA','SCS':'SCS_PTA','DPR':'DPR_PTA','LIV':'LIV_PTA',
            'FI':None,'CCR':'CCR_PTA','PTAT':'TYP_PTA','UDC':'UDC_PTA','FLC':'FLC_PTA','MAST':None}
BANCO_COL = {'TPI':'TPI','MILK':'MILK','FAT':'FAT','FAT%':'FAT %','PRO':'PROT','PRO%':'PROT%',
             'PL':'PL','SCS':'SCS','DPR':'DPR','LIV':'LIV','FI':'FI','CCR':'CCR',
             'PTAT':'TYPE FS','UDC':'UDC','FLC':'FLC','MAST':'CDCB_MAST'}
MAY_COL = {'TPI':'GTPI','MILK':'MILK','FAT':'FAT','FAT%':'%F','PRO':'PRO','PRO%':'%P',
           'PL':'PL','SCS':'SCS','DPR':'DPR','LIV':'LIV','FI':'FI','CCR':'CCR',
           'PTAT':'PTAT','UDC':'UDC','FLC':'FLC','MAST':'MAST'}
DAM_COL = {'TPI':'TPI','MILK':'PTA Milk','FAT':'PTA Fat','PRO':'PTA Pro','SCS':'SCS',
           'DPR':'PTA DPR','CCR':'CCR','PL':'PL','LIV':'PTA LIV','PTAT':'PTA Type',
           'UDC':'UDC','FLC':'FLC','MAST':'Mastitis','FI':'Feed Saved'}

STUD_MAP = {'507':'7','509':'9','550':'50','551':'51','559':'59','518':'18','521':'21',
            '523':'23','581':'81','585':'85','501':'1','601':'1','604':'4','614':'14',
            '629':'29','751':'151','752':'152','814':'14','250':'200'}

def sf(v):
    if v is None: return None
    try:
        x = float(v)
        return x if not np.isnan(x) else None
    except: return None

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

# Load bulls
bulls = {}
name_to_naab = {}
with open(BULLS_CSV, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        naab = row.get('NAAB','').strip()
        if naab:
            bulls[naab] = row
            name = row.get('Registration Name','').strip().upper()
            if name: name_to_naab[name] = naab
            name2 = row.get('Name','').strip().upper()
            if name2: name_to_naab[name2] = naab

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

def lookup_bull_by_name(name):
    if not name or str(name).strip() in ('','nan','None'): return None, None
    key = str(name).strip().upper()
    if key in name_to_naab:
        return bulls.get(name_to_naab[key]), 'SS'
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

# Load training data
print("    Loading banco21052026...")
raw = pd.read_excel(DOWNLOADS / "banco21052026.xlsx", header=None, engine='openpyxl')
hdr = [str(c).strip() if c else f'col_{i}' for i, c in enumerate(raw.iloc[10])]
banco = raw.iloc[11:].copy()
banco.columns = hdr
banco = banco.reset_index(drop=True)

print("    Loading May 2026...")
may_df = pd.read_excel(DOWNLOADS / "May 2026.xlsx", engine='openpyxl')

print("    Loading DAM1-20...")
dam_frames = []
for fp in sorted(glob.glob(str(DOWNLOADS / "DAM*.xlsx"))):
    dam_frames.append(pd.read_excel(fp, engine='openpyxl'))
dam_df = pd.concat(dam_frames, ignore_index=True) if dam_frames else pd.DataFrame()

# Assemble records (same as V10)
all_records = []
sire_daughters = defaultdict(list)
mgs_daughters = defaultdict(list)

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
    for t, v in daughter_ptas.items():
        sire_daughters[sire_naab].append((t, v))
        if mgs_ptas: mgs_daughters[mgs_naab].append((t, v))
    all_records.append({'source':'banco','sire_naab':sire_naab,'mgs_naab':mgs_naab if mgs_ptas else None,
                        'sire_ptas':sire_ptas,'mgs_ptas':mgs_ptas,'daughter_ptas':daughter_ptas})

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
    if sire_naab:
        for t, v in daughter_ptas.items():
            sire_daughters[sire_naab].append((t, v))
    if mgs_naab and mgs_ptas:
        for t, v in daughter_ptas.items():
            mgs_daughters[mgs_naab].append((t, v))
    all_records.append({'source':'may2026','sire_naab':sire_naab,'mgs_naab':mgs_naab if mgs_ptas else None,
                        'sire_ptas':sire_ptas,'mgs_ptas':mgs_ptas,'daughter_ptas':daughter_ptas})

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
    if sire_naab_d:
        for t, v in daughter_ptas_d.items():
            sire_daughters[sire_naab_d].append((t, v))
    if mgs_naab_d and mgs_ptas_d:
        for t, v in daughter_ptas_d.items():
            mgs_daughters[mgs_naab_d].append((t, v))
    all_records.append({'source':'dam','sire_naab':sire_naab_d,'mgs_naab':mgs_naab_d if mgs_ptas_d else None,
                        'sire_ptas':sire_ptas_d,'mgs_ptas':mgs_ptas_d,'mmgs_ptas':mmgs_ptas_d,
                        'daughter_ptas':daughter_ptas_d})

print(f"    Total: {len(all_records)} training records")

# ============================================================
# STEP 1: Generate OOF predictions for GOOD traits (to use as anchors)
# ============================================================
print()
print("=" * 90)
print("  STEP 1: Generating OOF predictions for GOOD traits (anchors)")
print("=" * 90)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def build_features_v10(sire_ptas, mgs_ptas, trait, sire_naab=None, mgs_naab=None, mmgs_ptas=None):
    s = sire_ptas.get(trait)
    if s is None: return None
    f = {}
    sd = GENETIC_SD.get(trait, 1)
    h2 = HERITABILITY.get(trait, 0.15)
    f['sire'] = s
    f['sire_z'] = s / sd if sd else 0
    f['sire_sq'] = s * s
    f['sire_h2'] = s * h2
    mg = mgs_ptas.get(trait) if mgs_ptas else None
    f['mgs_val'] = mg if mg is not None else 0
    f['mgs_z'] = (mg / sd if sd else 0) if mg is not None else 0
    f['mgs_sq'] = mg * mg if mg is not None else 0
    f['has_mgs'] = 1 if mg is not None else 0
    if mg is not None:
        f['sire_x_mgs'] = s * mg
        f['sire_mgs_diff'] = s - mg
        f['sire_mgs_ratio'] = s / mg if mg != 0 else 0
    else:
        f['sire_x_mgs'] = 0; f['sire_mgs_diff'] = 0; f['sire_mgs_ratio'] = 0
    mmg = mmgs_ptas.get(trait) if mmgs_ptas else None
    f['mmgs_val'] = mmg if mmg is not None else 0
    f['has_mmgs'] = 1 if mmg is not None else 0
    f['sire_x_mmgs'] = s * mmg if mmg is not None else 0
    for ot in ALL_TRAITS:
        if ot == trait: continue
        sv = sire_ptas.get(ot)
        if sv is not None: f[f'sire_{ot}'] = sv
        mv = mgs_ptas.get(ot) if mgs_ptas else None
        if mv is not None: f[f'mgs_{ot}'] = mv
    if sire_naab and sire_naab in sire_profiles:
        sp = sire_profiles[sire_naab]
        tp = sp.get(trait)
        if tp and tp['n'] >= 2:
            f['sire_prof_n'] = tp['n']; f['sire_prof_mean'] = tp['mean']
            f['sire_prof_std'] = tp['std']; f['sire_prof_ratio'] = tp['trans_ratio']
            f['sire_prof_resid'] = tp['residual']
        else:
            f['sire_prof_n'] = 0; f['sire_prof_mean'] = 0; f['sire_prof_std'] = 0
            f['sire_prof_ratio'] = 1.0; f['sire_prof_resid'] = 0
    else:
        f['sire_prof_n'] = 0; f['sire_prof_mean'] = 0; f['sire_prof_std'] = 0
        f['sire_prof_ratio'] = 1.0; f['sire_prof_resid'] = 0
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

def get_models():
    return {
        'GBR': GradientBoostingRegressor(n_estimators=600, max_depth=5, learning_rate=0.025,
                                          min_samples_leaf=5, subsample=0.8, random_state=42),
        'LGBM': LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.025,
                               subsample=0.8, colsample_bytree=0.7, min_child_samples=5,
                               reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1),
        'XGB': XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.025,
                             subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                             reg_lambda=1.0, min_child_weight=5, random_state=42, verbosity=0),
        'RF': RandomForestRegressor(n_estimators=500, max_depth=14, min_samples_leaf=4,
                                     random_state=42, n_jobs=-1),
        'MLP': Pipeline([('scaler', StandardScaler()),
                         ('mlp', MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu',
                                              solver='adam', max_iter=500, early_stopping=True,
                                              validation_fraction=0.15, learning_rate='adaptive',
                                              learning_rate_init=0.001, random_state=42))]),
    }

# For each good trait, generate OOF predictions
oof_predictions = {}  # trait -> array of OOF predictions aligned with all_records

for gt in GOOD_TRAITS:
    feat_rows, y_vals, indices = [], [], []
    for i, rec in enumerate(all_records):
        dv = rec['daughter_ptas'].get(gt)
        if dv is None: continue
        feat = build_features_v10(rec['sire_ptas'], rec['mgs_ptas'], gt,
                                   rec['sire_naab'], rec.get('mgs_naab'), rec.get('mmgs_ptas'))
        if feat is None: continue
        feat_rows.append(feat)
        y_vals.append(dv)
        indices.append(i)

    feat_df = pd.DataFrame(feat_rows)
    feat_df = feat_df.dropna(axis=1, thresh=int(len(feat_df) * 0.3))
    for c in feat_df.columns:
        if feat_df[c].isna().any():
            feat_df[c] = feat_df[c].fillna(feat_df[c].median())
    X = feat_df.values.astype(np.float64)
    y = np.array(y_vals, dtype=np.float64)

    # Use best model from V10
    mi = v10_models[gt]
    best_model_name = mi['model_name'] if mi['type'] == 'single' else 'LGBM'

    preds_oof = np.full(len(all_records), np.nan)
    for tr, te in kf.split(X):
        m = get_models()[best_model_name]
        m.fit(X[tr], y[tr])
        preds_oof_fold = m.predict(X[te])
        for k, idx in enumerate(te):
            preds_oof[indices[idx]] = preds_oof_fold[k]

    oof_predictions[gt] = preds_oof
    valid = np.sum(~np.isnan(preds_oof))
    r2_oof = r2_score(y, [preds_oof[i] for i in indices])
    print(f"    {gt:>6} | OOF R2={r2_oof:.4f} | {valid}/{len(all_records)} predictions")

# ============================================================
# STEP 2: Train problem traits with and without anchors
# ============================================================
print()
print("=" * 90)
print("  STEP 2: Training PROBLEM traits — V10 original vs V10-A (anchored)")
print("=" * 90)
print()

# Also add: Bayesian shrinkage approach
# pred_final = pred * confidence + pop_mean * (1-confidence)
# confidence based on sire_prof_n and agreement between sire/MGS

results_comparison = []

print(f"  {'Trait':>6} | {'V10 R2':>7} | {'V10-A R2':>8} | {'Delta':>7} | {'V10 MAE':>8} | {'V10-A MAE':>9} | "
      f"{'V10 SD_err':>10} | {'V10-A SD_err':>12} | {'V10 Inv%':>8} | {'V10-A Inv%':>10}")
print(f"  {'-'*120}")

for pt in PROBLEM_TRAITS:
    # Build V10 features (original)
    feat_rows_v10, y_vals, indices = [], [], []
    for i, rec in enumerate(all_records):
        dv = rec['daughter_ptas'].get(pt)
        if dv is None: continue
        feat = build_features_v10(rec['sire_ptas'], rec['mgs_ptas'], pt,
                                   rec['sire_naab'], rec.get('mgs_naab'), rec.get('mmgs_ptas'))
        if feat is None: continue
        feat_rows_v10.append(feat)
        y_vals.append(dv)
        indices.append(i)

    if len(feat_rows_v10) < 50: continue

    feat_df_v10 = pd.DataFrame(feat_rows_v10)
    feat_df_v10 = feat_df_v10.dropna(axis=1, thresh=int(len(feat_df_v10) * 0.3))
    for c in feat_df_v10.columns:
        if feat_df_v10[c].isna().any():
            feat_df_v10[c] = feat_df_v10[c].fillna(feat_df_v10[c].median())

    # Build V10-A features (with anchor predictions)
    feat_df_v10a = feat_df_v10.copy()
    anchors_for_trait = ANCHORS.get(pt, [])
    for gt, corr in anchors_for_trait:
        anchor_vals = []
        for idx in indices:
            oof_val = oof_predictions.get(gt, np.full(len(all_records), np.nan))[idx]
            anchor_vals.append(oof_val if not np.isnan(oof_val) else 0)
        feat_df_v10a[f'anchor_{gt}'] = anchor_vals
        feat_df_v10a[f'anchor_{gt}_corr'] = [v * corr for v in anchor_vals]

    # Also add z-score target for better numeric resolution on small-scale traits
    y = np.array(y_vals, dtype=np.float64)
    y_mean, y_std = np.mean(y), np.std(y)

    X_v10 = feat_df_v10.values.astype(np.float64)
    X_v10a = feat_df_v10a.values.astype(np.float64)

    # Train both with 5-fold CV using best architecture (STACK of top3)
    def train_cv(X, y_target, label):
        model_scores = {}
        model_preds_all = {}
        for mname, m in get_models().items():
            try:
                preds = np.zeros(len(y_target))
                for tr, te in kf.split(X):
                    mdl = get_models()[mname]
                    mdl.fit(X[tr], y_target[tr])
                    preds[te] = mdl.predict(X[te])
                r2 = r2_score(y_target, preds)
                model_scores[mname] = r2
                model_preds_all[mname] = preds
            except:
                pass

        # Stack top 3
        sorted_m = sorted(model_scores.items(), key=lambda x: -x[1])
        top3 = [m[0] for m in sorted_m[:3]]
        stack_X = np.column_stack([model_preds_all[m] for m in top3])
        stack_preds = np.zeros(len(y_target))
        for tr, te in kf.split(stack_X):
            ridge = Ridge(alpha=1.0)
            ridge.fit(stack_X[tr], y_target[tr])
            stack_preds[te] = ridge.predict(stack_X[te])
        stack_r2 = r2_score(y_target, stack_preds)

        best_single_name = sorted_m[0][0]
        best_single_r2 = sorted_m[0][1]

        if stack_r2 > best_single_r2:
            return stack_preds, stack_r2, f'STACK({",".join(top3)})'
        else:
            return model_preds_all[best_single_name], best_single_r2, best_single_name

    preds_v10, r2_v10, name_v10 = train_cv(X_v10, y, 'V10')
    preds_v10a, r2_v10a, name_v10a = train_cv(X_v10a, y, 'V10-A')

    # ---- Apply Bayesian shrinkage to V10-A ----
    # Shrink predictions toward population mean when confidence is low
    preds_v10a_shrunk = preds_v10a.copy()
    for k, idx in enumerate(indices):
        rec = all_records[idx]
        sire_naab = rec['sire_naab']
        # Confidence based on sire profile
        has_sire_prof = (sire_naab in sire_profiles and pt in sire_profiles.get(sire_naab, {}))
        n_daughters = sire_profiles.get(sire_naab, {}).get(pt, {}).get('n', 0) if has_sire_prof else 0
        # More daughters = more confidence, plateau at 20
        conf = min(n_daughters / 20, 1.0) if n_daughters > 0 else 0.3
        # Shrink toward population mean
        preds_v10a_shrunk[k] = preds_v10a[k] * conf + y_mean * (1 - conf)

    r2_v10a_s = r2_score(y, preds_v10a_shrunk)

    # Use shrunk if better
    if r2_v10a_s > r2_v10a:
        final_preds_a = preds_v10a_shrunk
        final_r2_a = r2_v10a_s
        name_v10a += '+SHRINK'
    else:
        final_preds_a = preds_v10a
        final_r2_a = r2_v10a

    # Metrics
    mae_v10 = mean_absolute_error(y, preds_v10)
    mae_v10a = mean_absolute_error(y, final_preds_a)
    sd_err_v10 = np.std(y - preds_v10)
    sd_err_v10a = np.std(y - final_preds_a)

    # Sign inversion rate
    inv_v10 = np.mean(np.sign(y) != np.sign(preds_v10)) * 100
    inv_v10a = np.mean(np.sign(y) != np.sign(final_preds_a)) * 100

    delta = final_r2_a - r2_v10
    arrow = '+' if delta > 0 else ''

    print(f"  {pt:>6} | {r2_v10:.4f}  | {final_r2_a:.4f}   | {arrow}{delta:.4f} | {mae_v10:.4f}  | "
          f"{mae_v10a:.4f}    | {sd_err_v10:.4f}     | {sd_err_v10a:.4f}       | "
          f"{inv_v10:.1f}%    | {inv_v10a:.1f}%")

    results_comparison.append({
        'Trait': pt,
        'V10_R2': r2_v10, 'V10A_R2': final_r2_a, 'Delta_R2': delta,
        'V10_MAE': mae_v10, 'V10A_MAE': mae_v10a,
        'V10_SD_err': sd_err_v10, 'V10A_SD_err': sd_err_v10a,
        'V10_Inv%': inv_v10, 'V10A_Inv%': inv_v10a,
        'V10A_method': name_v10a,
        'Anchors': ', '.join([f'{g}({c:+.2f})' for g, c in anchors_for_trait]),
    })

print()
print()
print("=" * 90)
print("  RESUMO DETALHADO")
print("=" * 90)
print()
for r in results_comparison:
    pt = r['Trait']
    delta = r['Delta_R2']
    inv_delta = r['V10A_Inv%'] - r['V10_Inv%']
    sd_delta = r['V10A_SD_err'] - r['V10_SD_err']
    print(f"  {pt}:")
    print(f"    Metodo V10-A: {r['V10A_method']}")
    print(f"    Ancoras: {r['Anchors']}")
    print(f"    R2:        {r['V10_R2']:.4f} -> {r['V10A_R2']:.4f} ({'+' if delta>0 else ''}{delta:.4f})")
    print(f"    MAE:       {r['V10_MAE']:.4f} -> {r['V10A_MAE']:.4f}")
    print(f"    SD erro:   {r['V10_SD_err']:.4f} -> {r['V10A_SD_err']:.4f} ({'+' if sd_delta>0 else ''}{sd_delta:.4f})")
    print(f"    Inversao:  {r['V10_Inv%']:.1f}% -> {r['V10A_Inv%']:.1f}% ({'+' if inv_delta>0 else ''}{inv_delta:.1f}pp)")
    print()

# Extra: what if we use sign-constrained for fertility traits?
print("=" * 90)
print("  BONUS: Sign-Constrained Prediction (DPR, CCR)")
print("  Regra: se Pai e MGS concordam no sinal, modelo NAO pode inverter")
print("=" * 90)
print()

for pt in ['DPR', 'CCR']:
    feat_rows, y_vals, rec_data = [], [], []
    for i, rec in enumerate(all_records):
        dv = rec['daughter_ptas'].get(pt)
        if dv is None: continue
        feat = build_features_v10(rec['sire_ptas'], rec['mgs_ptas'], pt,
                                   rec['sire_naab'], rec.get('mgs_naab'), rec.get('mmgs_ptas'))
        if feat is None: continue
        feat_rows.append(feat)
        y_vals.append(dv)
        rec_data.append(rec)

    feat_df = pd.DataFrame(feat_rows)
    feat_df = feat_df.dropna(axis=1, thresh=int(len(feat_df) * 0.3))
    for c in feat_df.columns:
        if feat_df[c].isna().any():
            feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    # Add anchor features
    for gt, corr in ANCHORS.get(pt, []):
        anchor_vals = []
        for i, rec in enumerate(rec_data):
            # Direct calculation from sire/mgs PTAs
            sv = rec['sire_ptas'].get(gt, 0)
            mv = rec['mgs_ptas'].get(gt, 0) if rec['mgs_ptas'] else 0
            anchor_vals.append((sv + mv) * corr)
        feat_df[f'anchor_{gt}'] = anchor_vals

    X = feat_df.values.astype(np.float64)
    y = np.array(y_vals, dtype=np.float64)
    y_mean_pt = np.mean(y)

    preds, r2_base, _ = train_cv(X, y, pt)

    # Apply sign constraint
    preds_constrained = preds.copy()
    constrained_count = 0
    for k, rec in enumerate(rec_data):
        sire_val = rec['sire_ptas'].get(pt, 0)
        mgs_val = rec['mgs_ptas'].get(pt, 0) if rec['mgs_ptas'] else 0
        # If both agree on sign
        if sire_val * mgs_val > 0:
            expected_sign = np.sign(sire_val)
            if np.sign(preds[k]) != expected_sign:
                # Force sign but use small magnitude
                preds_constrained[k] = expected_sign * abs(preds[k]) * 0.3
                constrained_count += 1

    r2_const = r2_score(y, preds_constrained)
    inv_base = np.mean(np.sign(y) != np.sign(preds)) * 100
    inv_const = np.mean(np.sign(y) != np.sign(preds_constrained)) * 100

    print(f"  {pt}:")
    print(f"    Predicoes corrigidas: {constrained_count}/{len(y)} ({constrained_count/len(y)*100:.1f}%)")
    print(f"    R2:       {r2_base:.4f} -> {r2_const:.4f}")
    print(f"    Inversao: {inv_base:.1f}% -> {inv_const:.1f}%")
    print()
