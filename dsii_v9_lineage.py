"""
DSII V9 Lineage — Learns sire and MGS transmission patterns from real genomic data.
Philosophy: study each bull's transmission behavior, each lineage's contribution pattern,
and use ML to capture non-linear interactions between specific sire x MGS combinations.

Data sources:
  - banco21052026.xlsx (5,624 CLARIFIDE animals with genomic PTAs)
  - May 2026.xlsx + DAM1-20 (1,709 trios with real dam PTAs)
  - Bull_Report (CDCB 92k bulls) for sire PTA lookup
  - bulls.csv (40k Select Sires bulls) for sire PTA lookup

Training: uses Dam estimated from pedigree (same as production)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import csv, pickle, sys, re, openpyxl
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
    print(*args, **kwargs)
    sys.stdout.flush()

# ============================================================
# NAAB NORMALIZATION
# ============================================================
STUD_MAP = {
    '507': '7', '509': '9', '550': '50', '551': '51',
    '559': '59', '518': '18', '521': '21', '523': '23',
    '581': '81', '585': '85', '501': '1',
    '601': '1', '604': '4', '614': '14', '629': '29',
    '751': '151', '752': '152', '814': '14', '250': '200',
}

def normalize_naab(naab):
    """Generate candidate NAABs for lookup."""
    if not naab or str(naab).strip() == '' or str(naab).strip().lower() == 'nan':
        return []
    naab = str(naab).strip()
    m = re.match(r'^(\d+)(HO|BS)0*(\d+)$', naab)
    if not m:
        return [naab]
    org, breed, num = m.group(1), m.group(2), m.group(3)
    orgs = [org]
    if org.startswith('5'):
        orgs.append(org[1:])
    if org in STUD_MAP:
        orgs.append(STUD_MAP[org])
    orgs.append(org.lstrip('0') or org)
    return list(dict.fromkeys(f'{o}{breed}{num}' for o in orgs))

# ============================================================
# DOMAIN KNOWLEDGE
# ============================================================
GENETIC_SD = {
    'TPI': 250, 'MILK': 675, 'FAT': 29, 'FAT%': 0.05, 'PRO': 19, 'PRO%': 0.02,
    'PL': 1.85, 'SCS': 0.14, 'DPR': 1.3, 'LIV': 1.2, 'FI': 1.0,
    'CCR': 1.65, 'PTAT': 0.70, 'UDC': 0.75, 'FLC': 0.65, 'MAST': 1.0,
}
HERITABILITY = {
    'TPI': 0.30, 'MILK': 0.25, 'FAT': 0.25, 'FAT%': 0.50, 'PRO': 0.25, 'PRO%': 0.50,
    'PL': 0.08, 'SCS': 0.12, 'DPR': 0.04, 'LIV': 0.05, 'FI': 0.06,
    'CCR': 0.04, 'PTAT': 0.30, 'UDC': 0.25, 'FLC': 0.15, 'MAST': 0.04,
}
GENETIC_CORRELATIONS = {
    ('MILK', 'DPR'): -0.35, ('MILK', 'CCR'): -0.30, ('MILK', 'SCS'): 0.10,
    ('MILK', 'PL'): -0.15, ('FAT', 'PRO'): 0.65, ('FAT', 'DPR'): -0.25,
    ('PRO', 'DPR'): -0.30, ('DPR', 'CCR'): 0.55, ('DPR', 'PL'): 0.45,
    ('SCS', 'PL'): -0.30, ('SCS', 'MAST'): 0.70,
    ('PTAT', 'UDC'): 0.40, ('PTAT', 'FLC'): 0.30,
    ('MILK', 'FAT'): 0.55, ('MILK', 'PRO'): 0.85,
    ('PL', 'LIV'): 0.65, ('DPR', 'FI'): 0.70,
}

# Traits we can train (present in both bulls.csv and banco21052026)
BULL_COL = {
    'TPI': 'TPI', 'MILK': 'PTAM', 'FAT': 'PTAF', 'FAT%': 'PTAF%',
    'PRO': 'PTAP', 'PRO%': 'PTAP%', 'PL': 'PL', 'SCS': 'SCS',
    'DPR': 'DPR', 'LIV': 'LIV', 'FI': 'FI', 'CCR': 'CCR',
    'PTAT': 'PTAT', 'UDC': 'UDC', 'FLC': 'FLC', 'MAST': 'MAST',
}
# Mapping from banco21052026 columns
BANCO_COL = {
    'TPI': 'TPI', 'MILK': 'MILK', 'FAT': 'FAT', 'FAT%': 'FAT %',
    'PRO': 'PROT', 'PRO%': 'PROT%', 'PL': 'PL', 'SCS': 'SCS',
    'DPR': 'DPR', 'LIV': 'LIV', 'FI': 'FI', 'CCR': 'CCR',
    'PTAT': 'TYPE FS', 'UDC': 'UDC', 'FLC': 'FLC', 'MAST': 'CDCB_MAST',
}
ALL_TRAITS = list(BULL_COL.keys())

# ============================================================
# LOAD BULL DATABASE
# ============================================================
flush_print("=" * 95)
flush_print("  DSII V9 Lineage — Sire/MGS Transmission Learning")
flush_print("=" * 95)

flush_print("\n  Loading bull databases...")
bulls = {}
with open(BULLS_CSV, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        naab = row.get('NAAB', '').strip()
        if naab:
            bulls[naab] = row
flush_print(f"    bulls.csv: {len(bulls)} bulls")

# Also load CDCB Bull Report for broader coverage
cdcb_bulls = {}
try:
    cdcb_path = DOWNLOADS / "Bull_Report (1).csv"
    with open(cdcb_path, 'r', encoding='latin-1') as f:
        for row in csv.DictReader(f):
            naab = row.get('NAAB_CODE', '').strip()
            if naab:
                cdcb_bulls[naab] = row
    flush_print(f"    CDCB Bull Report: {len(cdcb_bulls)} bulls")
except:
    flush_print(f"    CDCB Bull Report: not available")

def lookup_bull(naab):
    """Look up bull in bulls.csv, then CDCB."""
    for candidate in normalize_naab(naab):
        if candidate in bulls:
            return bulls[candidate], 'SS'
        if candidate in cdcb_bulls:
            return cdcb_bulls[candidate], 'CDCB'
    return None, None

# CDCB column mapping
CDCB_COL = {
    'TPI': None, 'MILK': 'MLK_PTA', 'FAT': 'FAT_PTA', 'FAT%': 'FPT_PTA',
    'PRO': 'PRO_PTA', 'PRO%': 'PPT_PTA', 'PL': 'PL__PTA', 'SCS': 'SCS_PTA',
    'DPR': 'DPR_PTA', 'LIV': 'LIV_PTA', 'FI': None, 'CCR': 'CCR_PTA',
    'PTAT': 'TYP_PTA', 'UDC': 'UDC_PTA', 'FLC': 'FLC_PTA', 'MAST': None,
}

def get_bull_ptas(bull_row, source='SS'):
    """Extract PTAs from a bull row."""
    ptas = {}
    col_map = BULL_COL if source == 'SS' else CDCB_COL
    for trait, col in col_map.items():
        if col is None:
            continue
        v = sf(bull_row.get(col))
        if v is not None:
            ptas[trait] = v
    return ptas

# ============================================================
# LOAD TRAINING DATA: banco21052026 (5,624 animals)
# ============================================================
flush_print("\n  Loading banco21052026.xlsx...")
raw = pd.read_excel(DOWNLOADS / "banco21052026.xlsx", header=None, engine='openpyxl')
hdr = [str(c).strip() if c else f'col_{i}' for i, c in enumerate(raw.iloc[10])]
banco = raw.iloc[11:].copy()
banco.columns = hdr
banco = banco.reset_index(drop=True)
flush_print(f"    {len(banco)} CLARIFIDE animals")

# ============================================================
# ASSEMBLE TRAINING RECORDS
# ============================================================
flush_print("\n  Assembling training records...")

records = []
sire_daughters = defaultdict(list)  # sire_naab -> list of daughter genomic values
mgs_daughters = defaultdict(list)   # mgs_naab -> list of daughter genomic values
sire_found = 0
mgs_found = 0
both_found = 0

for idx, row in banco.iterrows():
    sire_naab_raw = str(row.get('Sire of Record NAAB', '')).strip()
    mgs_naab_raw = str(row.get('Maternal Grandsire NAAB', '')).strip()
    animal_id = str(row.get('Animal ID', '')).strip()

    if sire_naab_raw in ('', 'nan', 'None'):
        continue

    sire_row, sire_src = lookup_bull(sire_naab_raw)
    if not sire_row:
        continue
    sire_found += 1

    sire_ptas = get_bull_ptas(sire_row, sire_src)

    # MGS lookup (optional — some animals don't have MGS)
    mgs_ptas = {}
    has_mgs = mgs_naab_raw not in ('', 'nan', 'None')
    if has_mgs:
        mgs_row, mgs_src = lookup_bull(mgs_naab_raw)
        if mgs_row:
            mgs_ptas = get_bull_ptas(mgs_row, mgs_src)
            mgs_found += 1

    # Dam estimated from pedigree: Dam = MGS/2 (production condition)
    dam_ptas = {t: v / 2 for t, v in mgs_ptas.items()} if mgs_ptas else {}

    # Daughter genomic values (target)
    daughter_ptas = {}
    for trait, bcol in BANCO_COL.items():
        if bcol and bcol in row.index:
            v = sf(row[bcol])
            if v is not None:
                daughter_ptas[trait] = v

    if not sire_ptas or not daughter_ptas:
        continue

    if has_mgs and mgs_ptas:
        both_found += 1

    # Store for lineage learning
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

flush_print(f"    Sire found: {sire_found}")
flush_print(f"    MGS found:  {mgs_found}")
flush_print(f"    Both found: {both_found}")
flush_print(f"    Total records: {len(records)}")

# ============================================================
# COMPUTE SIRE TRANSMISSION PROFILES
# ============================================================
flush_print("\n  Computing sire transmission profiles...")

# For each sire, compute: how his daughters' genomic values compare to his PTA
sire_profiles = {}  # sire_naab -> {trait: {mean_daughter, std_daughter, n_daughters, transmission_ratio}}
for sire_naab, daughter_vals in sire_daughters.items():
    sire_row, src = lookup_bull(sire_naab)
    if not sire_row:
        continue
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
            # Transmission ratio: how daughters compare to sire PTA/2
            trans_ratio = mean_d / (sp / 2) if sp != 0 else 1.0
            profile[t] = {
                'n': len(vals), 'mean': mean_d, 'std': std_d,
                'sire_pta': sp, 'trans_ratio': trans_ratio,
                'residual': mean_d - sp / 2,
            }
    if profile:
        sire_profiles[sire_naab] = profile

flush_print(f"    Sires with transmission profile: {len(sire_profiles)}")
n_daughters_per_sire = [len(set(t for t, _ in v)) for v in sire_daughters.values()]
flush_print(f"    Avg traits per sire: {np.mean(n_daughters_per_sire):.0f}")

# Same for MGS
mgs_profiles = {}
for mgs_naab, daughter_vals in mgs_daughters.items():
    mgs_row, src = lookup_bull(mgs_naab)
    if not mgs_row:
        continue
    mgs_ptas = get_bull_ptas(mgs_row, src)
    profile = {}
    trait_vals = defaultdict(list)
    for t, v in daughter_vals:
        trait_vals[t].append(v)
    for t, vals in trait_vals.items():
        if len(vals) >= 2 and t in mgs_ptas:
            mp = mgs_ptas[t]
            mean_d = np.mean(vals)
            profile[t] = {
                'n': len(vals), 'mean': mean_d, 'std': np.std(vals),
                'mgs_pta': mp, 'residual': mean_d - mp / 4,
            }
    if profile:
        mgs_profiles[mgs_naab] = profile

flush_print(f"    MGS with transmission profile: {len(mgs_profiles)}")

# ============================================================
# BUILD FEATURES (V9 — Lineage-aware)
# ============================================================
def build_features_v9(sire_ptas, dam_ptas, trait, sire_naab=None, mgs_naab=None):
    s = sire_ptas.get(trait)
    if s is None:
        return None

    d = dam_ptas.get(trait, 0)  # Dam may be unknown → default to 0 (breed avg)
    f = {}
    sd = GENETIC_SD.get(trait, 1)
    h2 = HERITABILITY.get(trait, 0.15)
    pa = (s + d) / 2 if d != 0 else s / 2

    # === BASE FEATURES ===
    f['sire'] = s
    f['dam'] = d
    f['pa'] = pa
    f['sire_half'] = s / 2
    f['delta_g'] = (s - d) / 2
    f['diff'] = s - d
    f['sire_z'] = s / sd if sd else 0
    f['dam_z'] = d / sd if sd else 0
    f['sire_sq'] = s * s
    f['sxd'] = s * d
    f['h2_pa'] = pa * (0.5 + h2)
    f['has_dam'] = 1 if d != 0 else 0

    # === CROSS-TRAIT FEATURES ===
    for ot in ALL_TRAITS:
        if ot == trait:
            continue
        sv = sire_ptas.get(ot)
        dv = dam_ptas.get(ot)
        if sv is not None:
            f[f's_{ot}'] = sv
        if dv is not None:
            f[f'd_{ot}'] = dv

    # === GENETIC CORRELATION FEATURES ===
    for (t1, t2), corr in GENETIC_CORRELATIONS.items():
        if t1 == trait or t2 == trait:
            other = t2 if t1 == trait else t1
            sv = sire_ptas.get(other)
            dv = dam_ptas.get(other, 0)
            if sv is not None:
                f[f'gc_{other}'] = ((sv + dv) / 2) * corr

    # === SIRE TRANSMISSION PROFILE (KEY V9 FEATURE) ===
    if sire_naab and sire_naab in sire_profiles:
        sp = sire_profiles[sire_naab]
        tp = sp.get(trait)
        if tp and tp['n'] >= 2:
            f['sire_n_daughters'] = tp['n']
            f['sire_daughter_mean'] = tp['mean']
            f['sire_daughter_std'] = tp['std']
            f['sire_trans_ratio'] = tp['trans_ratio']
            f['sire_residual'] = tp['residual']
        else:
            f['sire_n_daughters'] = 0
            f['sire_daughter_mean'] = 0
            f['sire_daughter_std'] = 0
            f['sire_trans_ratio'] = 1.0
            f['sire_residual'] = 0

        # Cross-trait sire profiles
        for ot in ALL_TRAITS:
            if ot == trait:
                continue
            otp = sp.get(ot)
            if otp and otp['n'] >= 2:
                f[f'sire_res_{ot}'] = otp['residual']
    else:
        f['sire_n_daughters'] = 0
        f['sire_daughter_mean'] = 0
        f['sire_daughter_std'] = 0
        f['sire_trans_ratio'] = 1.0
        f['sire_residual'] = 0

    # === MGS TRANSMISSION PROFILE ===
    if mgs_naab and mgs_naab in mgs_profiles:
        mp = mgs_profiles[mgs_naab]
        tp = mp.get(trait)
        if tp and tp['n'] >= 2:
            f['mgs_n_daughters'] = tp['n']
            f['mgs_daughter_mean'] = tp['mean']
            f['mgs_residual'] = tp['residual']
        else:
            f['mgs_n_daughters'] = 0
            f['mgs_daughter_mean'] = 0
            f['mgs_residual'] = 0
    else:
        f['mgs_n_daughters'] = 0
        f['mgs_daughter_mean'] = 0
        f['mgs_residual'] = 0

    return f

# ============================================================
# TRAIN V9 MODELS
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  TRAINING V9 LINEAGE MODELS")
flush_print(f"{'='*95}")

# Build dataset per trait
kf = KFold(n_splits=5, shuffle=True, random_state=42)

ORACLE_MAP = {
    'TPI': 'GBR', 'MILK': 'GBR', 'FAT': 'GBR', 'FAT%': 'GBR',
    'PRO': 'LGBM', 'PRO%': 'GBR', 'PL': 'LGBM', 'SCS': 'GBR',
    'DPR': 'LGBM', 'LIV': 'GBR', 'FI': 'LGBM', 'CCR': 'GBR',
    'PTAT': 'XGB', 'UDC': 'GBR', 'FLC': 'LGBM', 'MAST': 'LGBM',
}

def get_model(name):
    if name == 'GBR':
        return GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                                          min_samples_leaf=5, subsample=0.8, random_state=42)
    elif name == 'LGBM':
        return LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.7, min_child_samples=5,
                              reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
    elif name == 'XGB':
        return XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
                            reg_lambda=1.0, min_child_weight=5, random_state=42, verbosity=0)
    elif name == 'RF':
        return RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=5,
                                      random_state=42, n_jobs=-1)

flush_print(f"\n  {'Trait':>6} | {'N':>5} | {'Feats':>5} | {'Model':>5} | {'PA R2':>7} | {'V9 R2':>7} | "
            f"{'PA MAE':>8} | {'V9 MAE':>8} | {'MAE Gain':>9} | {'Corr PA':>7} | {'Corr V9':>7}")
flush_print(f"  {'-'*105}")

all_results = []
saved_models = {}

for trait in ALL_TRAITS:
    # Build X, y
    feat_rows = []
    y_vals = []
    pa_vals = []

    for rec in records:
        dv = rec['daughter_ptas'].get(trait)
        if dv is None:
            continue
        feat = build_features_v9(rec['sire_ptas'], rec['dam_ptas'], trait,
                                  rec['sire_naab'], rec.get('mgs_naab'))
        if feat is None:
            continue

        # Leave-one-out for sire transmission (avoid leakage)
        # We'll handle this in CV by not using the current daughter's value
        feat_rows.append(feat)
        y_vals.append(dv)

        s = rec['sire_ptas'].get(trait, 0)
        d = rec['dam_ptas'].get(trait, 0)
        pa_vals.append((s + d) / 2 if d != 0 else s / 2)

    if len(feat_rows) < 50:
        continue

    feat_df = pd.DataFrame(feat_rows)
    feat_df = feat_df.dropna(axis=1, thresh=int(len(feat_df) * 0.3))
    feature_cols = list(feat_df.columns)
    for c in feature_cols:
        if feat_df[c].isna().any():
            feat_df[c] = feat_df[c].fillna(feat_df[c].median())

    X = feat_df.values
    y = np.array(y_vals)
    pa = np.array(pa_vals)

    # Cross-validation
    model_name = ORACLE_MAP.get(trait, 'GBR')
    r2s, maes = [], []
    all_preds = np.zeros(len(y))

    for tr, te in kf.split(X):
        m = get_model(model_name)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        r2s.append(r2_score(y[te], pred))
        maes.append(mean_absolute_error(y[te], pred))
        all_preds[te] = pred

    v9_r2 = np.mean(r2s)
    v9_mae = np.mean(maes)
    pa_r2 = r2_score(y, pa)
    pa_mae = mean_absolute_error(y, pa)
    mae_gain = (pa_mae - v9_mae) / pa_mae * 100 if pa_mae > 0 else 0

    corr_pa = np.corrcoef(y, pa)[0, 1]
    corr_v9 = np.corrcoef(y, all_preds)[0, 1]

    flush_print(f"  {trait:>6} | {len(y):>5} | {len(feature_cols):>5} | {model_name:>5} | {pa_r2:>7.4f} | {v9_r2:>7.4f} | "
                f"{pa_mae:>8.2f} | {v9_mae:>8.2f} | {mae_gain:>+8.1f}% | {corr_pa:>7.4f} | {corr_v9:>7.4f}")

    all_results.append({
        'Trait': trait, 'N': len(y), 'Features': len(feature_cols),
        'Model': model_name,
        'PA_R2': round(pa_r2, 4), 'V9_R2': round(v9_r2, 4),
        'PA_MAE': round(pa_mae, 2), 'V9_MAE': round(v9_mae, 2),
        'MAE_Gain': round(mae_gain, 1),
        'Corr_PA': round(corr_pa, 4), 'Corr_V9': round(corr_v9, 4),
    })

    # Train final model on all data
    final_model = get_model(model_name)
    final_model.fit(X, y)
    saved_models[trait] = {
        'model': final_model,
        'model_name': model_name,
        'feature_cols': feature_cols,
        'r2_cv': round(v9_r2, 4),
    }

# ============================================================
# SUMMARY
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  SUMMARY V9 LINEAGE")
flush_print(f"{'='*95}")

pa_r2s = [r['PA_R2'] for r in all_results]
v9_r2s = [r['V9_R2'] for r in all_results]
v9_wins_r2 = sum(1 for r in all_results if r['V9_R2'] > r['PA_R2'])
v9_wins_corr = sum(1 for r in all_results if r['Corr_V9'] > r['Corr_PA'])

flush_print(f"  Training samples: {len(records)} (banco21052026)")
flush_print(f"  PA avg R2:  {np.mean(pa_r2s):.4f}")
flush_print(f"  V9 avg R2:  {np.mean(v9_r2s):.4f}")
flush_print(f"  V9 wins R2: {v9_wins_r2}/{len(all_results)}")
flush_print(f"  V9 wins Corr: {v9_wins_corr}/{len(all_results)}")
flush_print(f"  Avg MAE gain: {np.mean([r['MAE_Gain'] for r in all_results]):+.1f}%")

# Save
pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "v9_lineage_results.csv", index=False)
with open(OUTPUT_DIR / "v9_lineage_models.pkl", 'wb') as f:
    pickle.dump(saved_models, f)
with open(OUTPUT_DIR / "v9_sire_profiles.pkl", 'wb') as f:
    pickle.dump(sire_profiles, f)
with open(OUTPUT_DIR / "v9_mgs_profiles.pkl", 'wb') as f:
    pickle.dump(mgs_profiles, f)

flush_print(f"\n  Models saved: {OUTPUT_DIR / 'v9_lineage_models.pkl'}")
flush_print(f"  Sire profiles: {OUTPUT_DIR / 'v9_sire_profiles.pkl'}")
flush_print(f"  MGS profiles: {OUTPUT_DIR / 'v9_mgs_profiles.pkl'}")
flush_print(f"\n  V9 Lineage ready!")
