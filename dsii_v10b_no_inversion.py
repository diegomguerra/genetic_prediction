"""
DSII V10-B — Anti-Inversion Engine
Objetivo: gerar PTAs reais sem inversao de sinal para traits problema.

Abordagens:
  FAT%, PRO%  -> Derivacao aritmetica de FAT/MILK e PRO/MILK
  LIV         -> Regressao ancorada em PL (corr=0.85)
  DPR,CCR,SCS,MAST -> Modelo 2-estagios: Sign(classificacao) + Magnitude(regressao)
                       + Shrinkage bayesiano por confianca do perfil
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pickle, sys, csv, re, glob
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier

sys.stdout.reconfigure(encoding='utf-8')

RESULTS_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v10_results")
DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")

with open(RESULTS_DIR / 'v10_models.pkl', 'rb') as f: v10_models = pickle.load(f)
with open(RESULTS_DIR / 'v10_sire_profiles.pkl', 'rb') as f: sire_profiles = pickle.load(f)
with open(RESULTS_DIR / 'v10_mgs_profiles.pkl', 'rb') as f: mgs_profiles = pickle.load(f)

# ============================================================
# INFRASTRUCTURE (same as V10)
# ============================================================
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

print("Loading bulls..."); sys.stdout.flush()
bulls = {}; name_to_naab = {}
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
    if key in name_to_naab: return bulls.get(name_to_naab[key]), 'SS'
    for k, naab in name_to_naab.items():
        if key in k or k in key: return bulls.get(naab), 'SS'
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
print("Loading training data..."); sys.stdout.flush()
raw = pd.read_excel(DOWNLOADS / "banco21052026.xlsx", header=None, engine='openpyxl')
hdr = [str(c).strip() if c else f'col_{i}' for i, c in enumerate(raw.iloc[10])]
banco = raw.iloc[11:].copy(); banco.columns = hdr; banco = banco.reset_index(drop=True)
may_df = pd.read_excel(DOWNLOADS / "May 2026.xlsx", engine='openpyxl')
dam_frames = []
for fp in sorted(glob.glob(str(DOWNLOADS / "DAM*.xlsx"))):
    dam_frames.append(pd.read_excel(fp, engine='openpyxl'))
dam_df = pd.concat(dam_frames, ignore_index=True) if dam_frames else pd.DataFrame()

# Assemble records
all_records = []
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
        mr, ms = lookup_bull(mgs_naab)
        if mr: mgs_ptas = get_bull_ptas(mr, ms)
    daughter_ptas = {}
    for trait, bcol in BANCO_COL.items():
        if bcol and bcol in row.index:
            v = sf(row[bcol])
            if v is not None: daughter_ptas[trait] = v
    if not daughter_ptas: continue
    all_records.append({'sire_naab':sire_naab,'mgs_naab':mgs_naab if mgs_ptas else None,
                        'sire_ptas':sire_ptas,'mgs_ptas':mgs_ptas,'daughter_ptas':daughter_ptas})

for _, row in may_df.iterrows():
    sire_name = str(row.get('SIRENAME', '')).strip()
    sire_row, sire_src = lookup_bull_by_name(sire_name)
    if not sire_row:
        sire_row, sire_src = lookup_bull(str(row.get('SIREREGNUM2', '')).strip())
    if not sire_row: continue
    sire_ptas = get_bull_ptas(sire_row, sire_src)
    if not sire_ptas: continue
    sire_naab = sire_row.get('NAAB', '') if sire_src == 'SS' else ''
    mgs_name = str(row.get('MGSNAME2', '')).strip()
    mgs_ptas = {}; mgs_naab = ''
    if mgs_name not in ('', 'nan', 'None'):
        mr, ms = lookup_bull_by_name(mgs_name)
        if mr: mgs_ptas = get_bull_ptas(mr, ms); mgs_naab = mr.get('NAAB','') if ms=='SS' else ''
    daughter_ptas = {}
    for trait, mcol in MAY_COL.items():
        if mcol and mcol in row.index:
            v = sf(row[mcol])
            if v is not None: daughter_ptas[trait] = v
    if not daughter_ptas: continue
    all_records.append({'sire_naab':sire_naab,'mgs_naab':mgs_naab if mgs_ptas else None,
                        'sire_ptas':sire_ptas,'mgs_ptas':mgs_ptas,'daughter_ptas':daughter_ptas})

for _, row in dam_df.iterrows():
    sire_name = str(row.get('Sire', '')).strip()
    mgs_name = str(row.get('MGS', '')).strip()
    mggs_name = str(row.get('MGGS', '')).strip()
    sr, ss = lookup_bull_by_name(sire_name)
    if not sr: continue
    sp = get_bull_ptas(sr, ss)
    if not sp: continue
    sn = sr.get('NAAB','') if ss=='SS' else ''
    mp = {}; mn = ''
    if mgs_name not in ('','nan','None'):
        mr2, ms2 = lookup_bull_by_name(mgs_name)
        if mr2: mp = get_bull_ptas(mr2, ms2); mn = mr2.get('NAAB','') if ms2=='SS' else ''
    mmp = {}
    if mggs_name not in ('','nan','None'):
        mr3, ms3 = lookup_bull_by_name(mggs_name)
        if mr3: mmp = get_bull_ptas(mr3, ms3)
    dp = {}
    for trait, dcol in DAM_COL.items():
        if dcol and dcol in row.index:
            v = sf(row[dcol])
            if v is not None: dp[trait] = v
    if not dp: continue
    all_records.append({'sire_naab':sn,'mgs_naab':mn if mp else None,
                        'sire_ptas':sp,'mgs_ptas':mp,'mmgs_ptas':mmp,'daughter_ptas':dp})

print(f"Total: {len(all_records)} records"); sys.stdout.flush()

# Feature builder
def build_features(sire_ptas, mgs_ptas, trait, sire_naab=None, mgs_naab=None, mmgs_ptas=None):
    s = sire_ptas.get(trait)
    if s is None: return None
    f = {}
    sd = GENETIC_SD.get(trait, 1); h2 = HERITABILITY.get(trait, 0.15)
    f['sire'] = s; f['sire_z'] = s/sd if sd else 0; f['sire_sq'] = s*s; f['sire_h2'] = s*h2
    mg = mgs_ptas.get(trait) if mgs_ptas else None
    f['mgs_val'] = mg if mg is not None else 0
    f['mgs_z'] = (mg/sd if sd else 0) if mg is not None else 0
    f['has_mgs'] = 1 if mg is not None else 0
    if mg is not None:
        f['sire_x_mgs'] = s*mg; f['sire_mgs_diff'] = s-mg
        f['sire_mgs_ratio'] = s/mg if mg != 0 else 0
    else:
        f['sire_x_mgs'] = 0; f['sire_mgs_diff'] = 0; f['sire_mgs_ratio'] = 0
    mmg = (mmgs_ptas or {}).get(trait)
    f['mmgs_val'] = mmg if mmg is not None else 0
    f['has_mmgs'] = 1 if mmg is not None else 0
    f['sire_x_mmgs'] = s*mmg if mmg is not None else 0
    for ot in ALL_TRAITS:
        if ot == trait: continue
        sv = sire_ptas.get(ot)
        if sv is not None: f[f'sire_{ot}'] = sv
        mv = mgs_ptas.get(ot) if mgs_ptas else None
        if mv is not None: f[f'mgs_{ot}'] = mv
    if sire_naab and sire_naab in sire_profiles:
        sp = sire_profiles[sire_naab]; tp = sp.get(trait)
        if tp and tp['n'] >= 2:
            f['sire_prof_n'] = tp['n']; f['sire_prof_mean'] = tp['mean']
            f['sire_prof_std'] = tp['std']; f['sire_prof_ratio'] = tp['trans_ratio']
            f['sire_prof_resid'] = tp['residual']
        else:
            f['sire_prof_n']=0;f['sire_prof_mean']=0;f['sire_prof_std']=0;f['sire_prof_ratio']=1.0;f['sire_prof_resid']=0
    else:
        f['sire_prof_n']=0;f['sire_prof_mean']=0;f['sire_prof_std']=0;f['sire_prof_ratio']=1.0;f['sire_prof_resid']=0
    if mgs_naab and mgs_naab in mgs_profiles:
        mp2 = mgs_profiles[mgs_naab]; tp2 = mp2.get(trait)
        if tp2 and tp2['n'] >= 2:
            f['mgs_prof_n']=tp2['n'];f['mgs_prof_mean']=tp2['mean'];f['mgs_prof_resid']=tp2['residual']
        else:
            f['mgs_prof_n']=0;f['mgs_prof_mean']=0;f['mgs_prof_resid']=0
    else:
        f['mgs_prof_n']=0;f['mgs_prof_mean']=0;f['mgs_prof_resid']=0
    return f

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def get_reg_models():
    return {
        'GBR': GradientBoostingRegressor(n_estimators=600, max_depth=5, learning_rate=0.025,
                                          min_samples_leaf=5, subsample=0.8, random_state=42),
        'LGBM': LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.025,
                               subsample=0.8, colsample_bytree=0.7, min_child_samples=5,
                               reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1),
        'XGB': XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.025,
                             subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                             reg_lambda=1.0, min_child_weight=5, random_state=42, verbosity=0),
        'RF': RandomForestRegressor(n_estimators=500, max_depth=14, min_samples_leaf=4, random_state=42, n_jobs=-1),
        'MLP': Pipeline([('scaler', StandardScaler()),
                         ('mlp', MLPRegressor(hidden_layer_sizes=(256,128,64), activation='relu',
                                              solver='adam', max_iter=500, early_stopping=True,
                                              validation_fraction=0.15, random_state=42))]),
    }

def get_clf_models():
    return {
        'GBC': GradientBoostingClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                                           min_samples_leaf=10, subsample=0.8, random_state=42),
        'LGBC': LGBMClassifier(n_estimators=400, max_depth=5, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
                                random_state=42, verbose=-1),
        'RFC': RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5,
                                       random_state=42, n_jobs=-1),
    }

def prepare_trait_data(trait):
    feat_rows, y_vals, recs = [], [], []
    for rec in all_records:
        dv = rec['daughter_ptas'].get(trait)
        if dv is None: continue
        feat = build_features(rec['sire_ptas'], rec['mgs_ptas'], trait,
                               rec['sire_naab'], rec.get('mgs_naab'), rec.get('mmgs_ptas'))
        if feat is None: continue
        feat_rows.append(feat)
        y_vals.append(dv)
        recs.append(rec)
    feat_df = pd.DataFrame(feat_rows)
    feat_df = feat_df.dropna(axis=1, thresh=int(len(feat_df)*0.3))
    for c in feat_df.columns:
        if feat_df[c].isna().any(): feat_df[c] = feat_df[c].fillna(feat_df[c].median())
    return feat_df.values.astype(np.float64), np.array(y_vals, dtype=np.float64), recs

def train_regression_cv(X, y):
    best_r2, best_preds, best_name = -999, None, ''
    all_model_preds = {}
    for mname in get_reg_models():
        preds = np.zeros(len(y))
        for tr, te in kf.split(X):
            m = get_reg_models()[mname]
            m.fit(X[tr], y[tr]); preds[te] = m.predict(X[te])
        r2 = r2_score(y, preds)
        all_model_preds[mname] = preds
        if r2 > best_r2: best_r2 = r2; best_preds = preds; best_name = mname
    # Stack top 3
    sorted_m = sorted(all_model_preds.items(), key=lambda x: -r2_score(y, x[1]))
    top3 = [m[0] for m in sorted_m[:3]]
    sX = np.column_stack([all_model_preds[m] for m in top3])
    stack_preds = np.zeros(len(y))
    for tr, te in kf.split(sX):
        ridge = Ridge(alpha=1.0); ridge.fit(sX[tr], y[tr]); stack_preds[te] = ridge.predict(sX[te])
    stack_r2 = r2_score(y, stack_preds)
    if stack_r2 > best_r2:
        return stack_preds, stack_r2, f'STACK({",".join(top3)})'
    return best_preds, best_r2, best_name

def metrics(y_true, y_pred, label=''):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    sd_err = np.std(y_true - y_pred)
    # Sign inversion: only where y_true != 0
    mask = y_true != 0
    inv = np.mean(np.sign(y_true[mask]) != np.sign(y_pred[mask])) * 100 if mask.any() else 0
    return r2, mae, sd_err, inv

print()
print("=" * 100)
print("  DSII V10-B — Anti-Inversion Engine")
print("=" * 100)

# ============================================================
# 1) FAT% and PRO% — Arithmetic derivation
# ============================================================
print()
print("=" * 100)
print("  PARTE 1: FAT% e PRO% por derivacao aritmetica")
print("=" * 100)

for pct_trait, abs_trait, base_trait in [('FAT%', 'FAT', 'MILK'), ('PRO%', 'PRO', 'MILK')]:
    # Get V10 original predictions
    X, y, recs = prepare_trait_data(pct_trait)
    preds_v10, r2_v10, name_v10 = train_regression_cv(X, y)

    # Arithmetic derivation: use V10 predictions of FAT and MILK
    X_abs, y_abs, recs_abs = prepare_trait_data(abs_trait)
    preds_abs, _, _ = train_regression_cv(X_abs, y_abs)
    X_base, y_base, recs_base = prepare_trait_data(base_trait)
    preds_base, _, _ = train_regression_cv(X_base, y_base)

    # For each animal in pct_trait data, find matching abs and base predictions
    # Build index by daughter_ptas identity
    preds_derived = np.zeros(len(y))
    derived_count = 0
    for i, rec in enumerate(recs):
        # Find this animal's FAT and MILK predictions
        fat_val = rec['daughter_ptas'].get(abs_trait)
        milk_val = rec['daughter_ptas'].get(base_trait)
        sire_fat = rec['sire_ptas'].get(abs_trait, 0)
        sire_milk = rec['sire_ptas'].get(base_trait, 0)
        mgs_fat = rec['mgs_ptas'].get(abs_trait, 0) if rec['mgs_ptas'] else 0
        mgs_milk = rec['mgs_ptas'].get(base_trait, 0) if rec['mgs_ptas'] else 0

        # Use sire profile means if available
        sn = rec['sire_naab']
        sp = sire_profiles.get(sn, {})
        mn = rec.get('mgs_naab', '')
        mp = mgs_profiles.get(mn, {}) if mn else {}

        # Best estimate of daughter's abs trait
        if abs_trait in sp and sp[abs_trait]['n'] >= 3:
            est_abs = sp[abs_trait]['mean']
        else:
            est_abs = sire_fat / 2

        if base_trait in sp and sp[base_trait]['n'] >= 3:
            est_base = sp[base_trait]['mean']
        else:
            est_base = sire_milk / 2

        # Derive percentage
        if est_base != 0:
            # PTA FAT% ~ (PTA FAT / PTA MILK) adjusted by breed average
            # The actual relationship: FAT% PTA = FAT_PTA / (MILK_mean + MILK_PTA) * 100 - base_FAT%
            # Simplified linear approximation calibrated from training data
            preds_derived[i] = est_abs / (abs(est_base) + 1) * 3.5  # scaling factor
            derived_count += 1
        else:
            preds_derived[i] = 0

    # Calibrate derived predictions using linear regression on training data
    from sklearn.linear_model import LinearRegression
    # Use OOF to avoid leakage
    preds_calibrated = np.zeros(len(y))
    for tr, te in kf.split(preds_derived.reshape(-1, 1)):
        lr = LinearRegression()
        lr.fit(preds_derived[tr].reshape(-1, 1), y[tr])
        preds_calibrated[te] = lr.predict(preds_derived[te].reshape(-1, 1))

    r2_v10_m, mae_v10, sd_v10, inv_v10 = metrics(y, preds_v10)
    r2_cal, mae_cal, sd_cal, inv_cal = metrics(y, preds_calibrated)

    # Hybrid: blend V10 + calibrated derivation
    # OOF blend
    preds_hybrid = np.zeros(len(y))
    blend_X = np.column_stack([preds_v10, preds_calibrated])
    for tr, te in kf.split(blend_X):
        ridge = Ridge(alpha=1.0); ridge.fit(blend_X[tr], y[tr]); preds_hybrid[te] = ridge.predict(blend_X[te])
    r2_hyb, mae_hyb, sd_hyb, inv_hyb = metrics(y, preds_hybrid)

    print(f"\n  {pct_trait}:")
    print(f"    V10 original:  R2={r2_v10_m:.4f}  MAE={mae_v10:.4f}  SD_err={sd_v10:.4f}  Inv={inv_v10:.1f}%  [{name_v10}]")
    print(f"    Derivacao cal: R2={r2_cal:.4f}  MAE={mae_cal:.4f}  SD_err={sd_cal:.4f}  Inv={inv_cal:.1f}%")
    print(f"    Hibrido:       R2={r2_hyb:.4f}  MAE={mae_hyb:.4f}  SD_err={sd_hyb:.4f}  Inv={inv_hyb:.1f}%")
    sys.stdout.flush()

# ============================================================
# 2) LIV — Anchored regression from PL
# ============================================================
print()
print("=" * 100)
print("  PARTE 2: LIV ancorado em PL (correlacao genetica = 0.85)")
print("=" * 100)

X_liv, y_liv, recs_liv = prepare_trait_data('LIV')
preds_v10_liv, r2_v10_liv, name_v10_liv = train_regression_cv(X_liv, y_liv)

# Get PL predictions for same animals
X_pl, y_pl, recs_pl = prepare_trait_data('PL')
preds_pl, _, _ = train_regression_cv(X_pl, y_pl)

# Build PL-anchored features for LIV
# For each LIV record, find the PL prediction of same animal
# Since records may not align perfectly, use sire_naab + mgs_naab as key
pl_by_key = {}
for i, rec in enumerate(recs_pl):
    key = (rec['sire_naab'], rec.get('mgs_naab',''))
    if key not in pl_by_key:
        pl_by_key[key] = []
    pl_by_key[key].append((y_pl[i], preds_pl[i]))

# Add PL features to LIV
X_liv_aug = np.copy(X_liv)
pl_pred_col = np.zeros(len(y_liv))
pl_real_col = np.zeros(len(y_liv))
for i, rec in enumerate(recs_liv):
    key = (rec['sire_naab'], rec.get('mgs_naab',''))
    if key in pl_by_key:
        # Use mean of PL predictions for animals with same lineage
        pl_vals = pl_by_key[key]
        pl_pred_col[i] = np.mean([p[1] for p in pl_vals])
        pl_real_col[i] = np.mean([p[0] for p in pl_vals])
    else:
        # Estimate PL from sire
        pl_pred_col[i] = rec['sire_ptas'].get('PL', 0) / 2

X_liv_aug2 = np.column_stack([X_liv, pl_pred_col, pl_pred_col**2, pl_pred_col * pl_real_col])
preds_liv_anc, r2_liv_anc, name_liv_anc = train_regression_cv(X_liv_aug2, y_liv)

# Also try simple: LIV = f(PL) linear
preds_liv_linear = np.zeros(len(y_liv))
for tr, te in kf.split(pl_pred_col.reshape(-1,1)):
    lr = LinearRegression()
    lr.fit(pl_pred_col[tr].reshape(-1,1), y_liv[tr])
    preds_liv_linear[te] = lr.predict(pl_pred_col[te].reshape(-1,1))

# Hybrid
blend_liv = np.column_stack([preds_v10_liv, preds_liv_anc, preds_liv_linear])
preds_liv_hyb = np.zeros(len(y_liv))
for tr, te in kf.split(blend_liv):
    ridge = Ridge(alpha=1.0); ridge.fit(blend_liv[tr], y_liv[tr]); preds_liv_hyb[te] = ridge.predict(blend_liv[te])

r2m, maem, sdm, invm = metrics(y_liv, preds_v10_liv)
r2a, maea, sda, inva = metrics(y_liv, preds_liv_anc)
r2l, mael, sdl, invl = metrics(y_liv, preds_liv_linear)
r2h, maeh, sdh, invh = metrics(y_liv, preds_liv_hyb)

print(f"\n  LIV:")
print(f"    V10 original:     R2={r2m:.4f}  MAE={maem:.4f}  SD_err={sdm:.4f}  Inv={invm:.1f}%  [{name_v10_liv}]")
print(f"    + PL features:    R2={r2a:.4f}  MAE={maea:.4f}  SD_err={sda:.4f}  Inv={inva:.1f}%  [{name_liv_anc}]")
print(f"    PL linear only:   R2={r2l:.4f}  MAE={mael:.4f}  SD_err={sdl:.4f}  Inv={invl:.1f}%")
print(f"    Hibrido (3-way):  R2={r2h:.4f}  MAE={maeh:.4f}  SD_err={sdh:.4f}  Inv={invh:.1f}%")
sys.stdout.flush()

# ============================================================
# 3) DPR, CCR, SCS, MAST — Two-stage: Sign + Magnitude + Shrinkage
# ============================================================
print()
print("=" * 100)
print("  PARTE 3: Modelo 2-estagios (SINAL + MAGNITUDE + SHRINKAGE)")
print("  Para: DPR, CCR, SCS, MAST")
print("=" * 100)

for trait in ['DPR', 'CCR', 'SCS', 'MAST']:
    X, y, recs = prepare_trait_data(trait)
    N = len(y)
    y_mean = np.mean(y)

    # --- V10 ORIGINAL ---
    preds_v10, r2_v10, name_v10 = train_regression_cv(X, y)

    # --- STAGE 1: Classify sign (positive vs negative) ---
    # Exclude zeros (neutral) from sign classification
    y_sign = np.sign(y)  # -1, 0, +1
    y_binary = (y_sign >= 0).astype(int)  # 0=negative, 1=zero_or_positive

    sign_preds = np.zeros(N, dtype=float)  # probability of positive

    # Check if both classes exist (SCS is always positive -> skip sign classification)
    n_classes = len(np.unique(y_binary))
    if n_classes < 2:
        # All same sign — skip 2-stage, use V10 + anchored only
        sign_preds[:] = 1.0 if y_binary[0] == 1 else 0.0
        best_clf = 'SKIP(1class)'
        best_acc = 1.0
    else:
        sign_scores = {}
        for cname in get_clf_models():
            proba = np.zeros(N)
            for tr, te in kf.split(X):
                clf = get_clf_models()[cname]
                clf.fit(X[tr], y_binary[tr])
                proba[te] = clf.predict_proba(X[te])[:, 1]
            acc = np.mean((proba >= 0.5) == y_binary)
            sign_scores[cname] = acc

        best_clf = max(sign_scores, key=sign_scores.get)
        best_acc = sign_scores[best_clf]

        # Retrain best classifier for OOF probabilities
        for tr, te in kf.split(X):
            clf = get_clf_models()[best_clf]
            clf.fit(X[tr], y_binary[tr])
            sign_preds[te] = clf.predict_proba(X[te])[:, 1]

    # --- STAGE 2: Predict absolute magnitude ---
    y_abs = np.abs(y)
    preds_mag, r2_mag, name_mag = train_regression_cv(X, y_abs)
    preds_mag = np.maximum(preds_mag, 0)  # magnitude can't be negative

    # --- COMBINE: sign * magnitude ---
    # Use probability: if prob > 0.5 -> positive, else negative
    # Confidence = distance from 0.5
    pred_sign_hard = np.where(sign_preds >= 0.5, 1.0, -1.0)
    sign_confidence = np.abs(sign_preds - 0.5) * 2  # 0 to 1

    preds_2stage_raw = pred_sign_hard * preds_mag

    # --- SHRINKAGE: reduce magnitude based on confidence ---
    # Low confidence -> shrink toward 0 (conservative)
    # High confidence -> keep full magnitude
    # Also use sire profile reliability
    preds_2stage = np.zeros(N)
    for i, rec in enumerate(recs):
        sn = rec['sire_naab']
        has_prof = sn in sire_profiles and trait in sire_profiles.get(sn, {})
        n_daughters = sire_profiles.get(sn, {}).get(trait, {}).get('n', 0) if has_prof else 0

        # Profile confidence (0 to 1, plateau at 30)
        prof_conf = min(n_daughters / 30, 1.0) if n_daughters > 0 else 0.2

        # Combined confidence
        total_conf = 0.6 * sign_confidence[i] + 0.4 * prof_conf

        # Shrink: pred * conf + 0 * (1-conf)
        # But keep sign from classifier
        preds_2stage[i] = preds_2stage_raw[i] * total_conf

    # --- APPROACH B: Sire-profile-anchored regression ---
    # Use sire profile mean directly as strong prior, adjust with features
    preds_anchored = np.zeros(N)
    for i, rec in enumerate(recs):
        sn = rec['sire_naab']
        sp = sire_profiles.get(sn, {})
        tp = sp.get(trait)

        if tp and tp['n'] >= 5:
            # Strong prior: use sire profile mean as base
            base = tp['mean']
            # Adjust slightly based on MGS
            mgs_val = rec['mgs_ptas'].get(trait, 0) if rec['mgs_ptas'] else 0
            mgs_contrib = mgs_val * 0.15  # MGS contributes ~15%
            preds_anchored[i] = base * 0.85 + mgs_contrib
        elif tp and tp['n'] >= 2:
            base = tp['mean']
            mgs_val = rec['mgs_ptas'].get(trait, 0) if rec['mgs_ptas'] else 0
            # Less confidence, shrink more
            preds_anchored[i] = base * 0.6 + (rec['sire_ptas'].get(trait, 0)/2) * 0.25 + mgs_val * 0.15
        else:
            # No profile: conservative estimate from sire PTA
            sire_val = rec['sire_ptas'].get(trait, 0)
            mgs_val = rec['mgs_ptas'].get(trait, 0) if rec['mgs_ptas'] else 0
            preds_anchored[i] = sire_val * 0.35 + mgs_val * 0.15

    # --- HYBRID: blend V10 + 2-stage + anchored ---
    blend = np.column_stack([preds_v10, preds_2stage, preds_anchored])
    preds_hybrid = np.zeros(N)
    for tr, te in kf.split(blend):
        ridge = Ridge(alpha=1.0); ridge.fit(blend[tr], y[tr]); preds_hybrid[te] = ridge.predict(blend[te])

    # Metrics
    r2_orig, mae_orig, sd_orig, inv_orig = metrics(y, preds_v10)
    r2_2s, mae_2s, sd_2s, inv_2s = metrics(y, preds_2stage)
    r2_anc, mae_anc, sd_anc, inv_anc = metrics(y, preds_anchored)
    r2_hyb, mae_hyb, sd_hyb, inv_hyb = metrics(y, preds_hybrid)

    print(f"\n  {trait} (N={N}):")
    print(f"    Classificador sinal: {best_clf} acc={best_acc:.3f}")
    print(f"    Magnitude R2={r2_mag:.4f}")
    print(f"    ---------------------------------------------------------------")
    print(f"    {'Metodo':<25} | {'R2':>7} | {'MAE':>7} | {'SD_err':>7} | {'Inv%':>6}")
    print(f"    {'-'*63}")
    print(f"    {'V10 original':<25} | {r2_orig:.4f} | {mae_orig:.4f} | {sd_orig:.4f} | {inv_orig:.1f}%")
    print(f"    {'2-estagios+shrinkage':<25} | {r2_2s:.4f} | {mae_2s:.4f} | {sd_2s:.4f} | {inv_2s:.1f}%")
    print(f"    {'Perfil ancorado':<25} | {r2_anc:.4f} | {mae_anc:.4f} | {sd_anc:.4f} | {inv_anc:.1f}%")
    print(f"    {'HIBRIDO (3-way)':<25} | {r2_hyb:.4f} | {mae_hyb:.4f} | {sd_hyb:.4f} | {inv_hyb:.1f}%")
    sys.stdout.flush()

print()
print("=" * 100)
print("  RESUMO FINAL — Melhor abordagem por trait")
print("=" * 100)
print("  Concluido.")
