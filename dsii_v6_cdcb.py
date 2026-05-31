"""
DSII v6 — CDCB-Powered Model
=============================
Uses 121k sire-daughter pairs from CDCB official reports.
Bull_Report = sire PTAs (features)
Cow_Report = daughter genomic proofs (targets)

Also tests combined model: CDCB 121k + May2026 1.7k trios.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import csv
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

DOWNLOADS = Path("C:/Users/DiegoGuerra/Downloads")
OUTPUT_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v6_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Trait mapping: CDCB column -> our trait name
CDCB_TRAITS = {
    'NM$':  'NM$_PTA',
    'CM$':  'CM$_PTA',
    'FM$':  'FM$_PTA',
    'GM$':  'GM$_PTA',
    'MILK': 'MLK_PTA',
    'FAT':  'FAT_PTA',
    'PRO':  'PRO_PTA',
    'PL':   'PL_PTA',
    'SCS':  'SCS_PTA',
    'DPR':  'DPR_PTA',
    'HCR':  'HCR_PTA',
    'CCR':  'CCR_PTA',
    'LIV':  'LIV_PTA',
    'MAST': 'MAS_PTA',
    'MET':  'MET_PTA',
    'KET':  'KET_PTA',
    'DA':   'DAB_PTA',
    'MF':   'MFV_PTA',
    'RP':   'RPL_PTA',
    'FSAV': 'FS_PTA',
    'RFI':  'RFI_PTA',
    'EFC':  'EFC_PTA',
    'HLIV': 'HLV_PTA',
    'GL':   'GL_PTA',
}

ALL_TRAITS = list(CDCB_TRAITS.keys())

# Domain knowledge
GENETIC_SD = {
    'NM$': 275, 'CM$': 280, 'FM$': 270, 'GM$': 260,
    'MILK': 675, 'FAT': 29, 'PRO': 19,
    'PL': 1.85, 'SCS': 0.14, 'DPR': 1.3, 'HCR': 1.4, 'CCR': 1.65,
    'LIV': 1.2, 'MAST': 1.0, 'MET': 0.5, 'KET': 0.5, 'DA': 0.5,
    'MF': 0.5, 'RP': 0.5, 'FSAV': 50, 'RFI': 50, 'EFC': 5.0,
    'HLIV': 0.8, 'GL': 1.0,
}

HERITABILITY = {
    'NM$': 0.30, 'CM$': 0.30, 'FM$': 0.30, 'GM$': 0.30,
    'MILK': 0.25, 'FAT': 0.25, 'PRO': 0.25,
    'PL': 0.08, 'SCS': 0.12, 'DPR': 0.04, 'HCR': 0.04, 'CCR': 0.04,
    'LIV': 0.05, 'MAST': 0.04, 'MET': 0.03, 'KET': 0.03, 'DA': 0.03,
    'MF': 0.03, 'RP': 0.03, 'FSAV': 0.15, 'RFI': 0.15, 'EFC': 0.06,
    'HLIV': 0.04, 'GL': 0.10,
}

BREED_IDEAL = {
    'MILK': 1800, 'FAT': 90, 'PRO': 60,
    'DPR': 1.5, 'CCR': 2.0, 'HCR': 2.0,
    'PL': 5.0, 'LIV': 2.0, 'MAST': -1.0, 'SCS': 2.60,
}

LOWER_IS_BETTER = {'SCS', 'MAST', 'MET', 'KET', 'DA', 'MF', 'RP'}

GENETIC_CORRELATIONS = {
    ('MILK', 'DPR'): -0.35, ('MILK', 'CCR'): -0.30, ('MILK', 'SCS'): 0.10,
    ('MILK', 'PL'): -0.15, ('FAT', 'PRO'): 0.65, ('FAT', 'DPR'): -0.25,
    ('PRO', 'DPR'): -0.30, ('DPR', 'CCR'): 0.55, ('DPR', 'PL'): 0.45,
    ('SCS', 'PL'): -0.30, ('SCS', 'MAST'): 0.70, ('PL', 'LIV'): 0.65,
    ('NM$', 'CM$'): 0.95, ('MILK', 'FAT'): 0.55, ('MILK', 'PRO'): 0.85,
}


def sf(val):
    if val is None: return None
    val = str(val).strip().strip('"')
    if not val: return None
    try:
        v = float(val)
        return v if not np.isnan(v) else None
    except: return None


def read_csv_safe(path):
    for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            with open(path, 'r', encoding=enc) as f:
                return list(csv.DictReader(f))
        except: continue
    return None


def build_features(sire_ptas, trait_name, all_trait_names):
    """Build feature vector from sire PTAs for a target trait."""
    features = {}
    s = sire_ptas.get(trait_name)
    if s is None:
        return None

    sd = GENETIC_SD.get(trait_name, 1)
    h2 = HERITABILITY.get(trait_name, 0.15)

    # Core sire features
    features['sire'] = s
    features['sire_half'] = s / 2  # expected daughter = sire/2 + dam/2
    features['sire_z'] = s / sd if sd > 0 else 0
    features['sire_sq'] = s * s
    features['h2_weighted'] = s * h2

    # Deficiency from ideal
    ideal = BREED_IDEAL.get(trait_name)
    if ideal is not None:
        if trait_name in LOWER_IS_BETTER:
            features['sire_vs_ideal'] = (s - ideal) / sd
        else:
            features['sire_vs_ideal'] = (ideal - s) / sd
    else:
        features['sire_vs_ideal'] = 0

    # All other sire traits (cross-trait learning)
    for other in all_trait_names:
        if other == trait_name:
            continue
        ov = sire_ptas.get(other)
        if ov is not None:
            osd = GENETIC_SD.get(other, 1)
            features[f's_{other}'] = ov
            features[f's_{other}_z'] = ov / osd if osd > 0 else 0

    # Genetic correlation features
    for (t1, t2), corr in GENETIC_CORRELATIONS.items():
        if t1 == trait_name:
            ov = sire_ptas.get(t2)
            if ov is not None:
                features[f'corr_{t2}_weighted'] = ov * corr
        elif t2 == trait_name:
            ov = sire_ptas.get(t1)
            if ov is not None:
                features[f'corr_{t1}_weighted'] = ov * corr

    # Production pressure (for health/fertility traits)
    milk = sire_ptas.get('MILK', 0) or 0
    fat = sire_ptas.get('FAT', 0) or 0
    pro = sire_ptas.get('PRO', 0) or 0
    features['prod_pressure'] = milk / 675 + fat / 29 + pro / 19

    # Economic index components
    nm = sire_ptas.get('NM$', 0) or 0
    features['sire_nm'] = nm

    return features


def run_cdcb_pipeline():
    print("=" * 90)
    print("  DSII v6 — CDCB-Powered Model (121k trios)")
    print("=" * 90)

    # Load data
    print("\nCarregando Bull_Report (1)...")
    bull_rows = read_csv_safe(DOWNLOADS / "Bull_Report (1).csv")
    print(f"  {len(bull_rows)} touros")

    print("Carregando Cow_Report (4)...")
    cow_rows = read_csv_safe(DOWNLOADS / "Cow_Report (4).csv")
    print(f"  {len(cow_rows)} femeas")

    # Index bulls by ANIMAL
    bull_idx = {}
    for r in bull_rows:
        aid = r['ANIMAL'].strip('"').upper()
        bull_idx[aid] = r

    # Build sire-daughter pairs
    print("\nMontando pares sire-filha...")
    records = []
    sire_daughters = defaultdict(list)

    for r in cow_rows:
        sire_id = r['SIRE'].strip('"').upper()
        sire = bull_idx.get(sire_id)
        if not sire:
            continue

        rec = {'sire_id': sire_id}
        sire_ptas = {}
        valid = True

        for trait, cdcb_col in CDCB_TRAITS.items():
            d_val = sf(r.get(cdcb_col))
            s_val = sf(sire.get(cdcb_col))
            s_rel = sf(sire.get(cdcb_col.replace('_PTA', '_REL')))
            d_rel = sf(r.get(cdcb_col.replace('_PTA', '_REL')))

            rec[f'd_{trait}'] = d_val
            rec[f's_{trait}'] = s_val
            rec[f's_rel_{trait}'] = s_rel
            rec[f'd_rel_{trait}'] = d_rel

            if s_val is not None:
                sire_ptas[trait] = s_val
                rec[f'pa_{trait}'] = s_val / 2.0  # sire contribution only

        rec['_sire_ptas'] = sire_ptas
        records.append(rec)
        sire_daughters[sire_id].append(len(records) - 1)

    print(f"  Pares completos: {len(records)}")
    print(f"  Sires unicos: {len(sire_daughters)}")
    print(f"  Media filhas/sire: {np.mean([len(v) for v in sire_daughters.values()]):.1f}")

    df = pd.DataFrame(records)

    # Build sire consistency features (leave-one-out)
    print("\nCalculando sire consistency features...")
    for trait in ALL_TRAITS:
        target = f'd_{trait}'
        sire_mean_loo = np.full(len(df), np.nan)
        sire_std = np.full(len(df), np.nan)
        sire_n = np.zeros(len(df))

        for sire_id, indices in sire_daughters.items():
            vals = [(i, df.at[i, target]) for i in indices
                    if pd.notna(df.at[i, target])]
            if len(vals) < 2:
                continue
            all_vals = [v for _, v in vals]
            for idx, val in vals:
                others = [v for i, v in vals if i != idx]
                sire_mean_loo[idx] = np.mean(others)
                sire_std[idx] = np.std(all_vals)
                sire_n[idx] = len(others)

        df[f'sire_mean_loo_{trait}'] = sire_mean_loo
        df[f'sire_std_{trait}'] = sire_std
        df[f'sire_n_{trait}'] = sire_n

    # ============================================================
    # TRAIN PER TRAIT
    # ============================================================
    print("\n" + "=" * 90)
    print(f"  {'Trait':>6} | {'N':>7} | {'PA MAE':>8} {'PA R2':>7} | "
          f"{'V6 MAE':>8} {'V6 R2':>7} | {'Gain MAE':>9} {'Gain R2':>8} | Best")
    print("-" * 100)

    all_results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for trait in ALL_TRAITS:
        target = f'd_{trait}'
        s_col = f's_{trait}'

        # Need at least sire and daughter values
        mask = df[[target, s_col]].notna().all(axis=1)
        sub = df[mask].reset_index(drop=True)

        if len(sub) < 100:
            continue

        y = sub[target].values

        # PA baseline: sire/2 (best we can do without dam)
        # Actually, for genomic animals, the expected daughter value
        # is approximately sire_PTA * weight, where weight varies by reliability
        # With full data, simple sire/2 is the baseline
        s_vals = sub[s_col].values

        # But PA in CDCB context = (sire + population_mean_dam) / 2
        # Population mean is ~0 on the base, so PA ~ sire/2
        pa_pred = s_vals / 2.0
        pa_mae = mean_absolute_error(y, pa_pred)
        pa_r2 = r2_score(y, pa_pred)

        # Build feature matrix
        feat_rows = []
        for idx, row in sub.iterrows():
            sire_ptas = row['_sire_ptas']
            feats = build_features(sire_ptas, trait, ALL_TRAITS)
            if feats is None:
                feat_rows.append(None)
                continue

            # Add sire consistency features
            feats['sire_mean_loo'] = row.get(f'sire_mean_loo_{trait}', np.nan)
            feats['sire_std'] = row.get(f'sire_std_{trait}', np.nan)
            feats['sire_n'] = row.get(f'sire_n_{trait}', 0)

            # Add reliability features
            s_rel = row.get(f's_rel_{trait}', np.nan)
            d_rel = row.get(f'd_rel_{trait}', np.nan)
            if pd.notna(s_rel):
                feats['sire_rel'] = s_rel
            if pd.notna(d_rel):
                feats['daughter_rel'] = d_rel

            feat_rows.append(feats)

        valid_mask = [f is not None for f in feat_rows]
        valid_feats = [f for f in feat_rows if f is not None]

        if len(valid_feats) < 100:
            continue

        feat_df = pd.DataFrame(valid_feats)
        y_valid = y[valid_mask]

        # Drop sparse columns
        feat_df = feat_df.dropna(axis=1, thresh=int(len(feat_df) * 0.3))
        for c in feat_df.columns:
            if feat_df[c].isna().any():
                feat_df[c] = feat_df[c].fillna(feat_df[c].median())

        X = feat_df.values
        n_feats = X.shape[1]

        # Recalculate PA for valid subset
        pa_valid = s_vals[valid_mask] / 2.0
        pa_mae_v = mean_absolute_error(y_valid, pa_valid)
        pa_r2_v = r2_score(y_valid, pa_valid)

        # Train models
        lgbm = LGBMRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1
        )

        ridge = Ridge(alpha=1.0)

        best_mae, best_r2, best_name = pa_mae_v, pa_r2_v, 'PA'

        for name, model in [('Ridge', ridge), ('LGBM', lgbm)]:
            maes, r2s = [], []
            for tr, te in kf.split(X):
                model.fit(X[tr], y_valid[tr])
                pred = model.predict(X[te])
                maes.append(mean_absolute_error(y_valid[te], pred))
                r2s.append(r2_score(y_valid[te], pred))

            avg_mae = np.mean(maes)
            avg_r2 = np.mean(r2s)

            if avg_r2 > best_r2:
                best_mae = avg_mae
                best_r2 = avg_r2
                best_name = name

        gain_mae = (pa_mae_v - best_mae) / pa_mae_v * 100
        gain_r2 = (best_r2 - pa_r2_v) / max(abs(pa_r2_v), 0.001) * 100

        print(f"  {trait:>6} | {len(y_valid):>7} | {pa_mae_v:>8.3f} {pa_r2_v:>7.4f} | "
              f"{best_mae:>8.3f} {best_r2:>7.4f} | {gain_mae:>+8.1f}% {gain_r2:>+7.1f}% | {best_name}")

        all_results.append({
            'Trait': trait, 'N': len(y_valid), 'N_Features': n_feats,
            'PA_MAE': round(pa_mae_v, 4), 'PA_R2': round(pa_r2_v, 4),
            'V6_MAE': round(best_mae, 4), 'V6_R2': round(best_r2, 4),
            'Best': best_name,
            'Gain_MAE_%': round(gain_mae, 2), 'Gain_R2_%': round(gain_r2, 2),
        })

    # Summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "v6_cdcb_results.csv", index=False)

    print("\n" + "=" * 90)
    print("  RESUMO")
    print("=" * 90)

    avg_pa_r2 = np.mean([r['PA_R2'] for r in all_results])
    avg_v6_r2 = np.mean([r['V6_R2'] for r in all_results])
    avg_gain_mae = np.mean([r['Gain_MAE_%'] for r in all_results])
    avg_gain_r2 = np.mean([r['Gain_R2_%'] for r in all_results])

    print(f"  Traits: {len(all_results)}")
    print(f"  PA  R2 medio: {avg_pa_r2:.4f}")
    print(f"  V6  R2 medio: {avg_v6_r2:.4f}")
    print(f"  Gain MAE medio: {avg_gain_mae:+.1f}%")
    print(f"  Gain R2 medio: {avg_gain_r2:+.1f}%")

    v6_wins = sum(1 for r in all_results if r['V6_R2'] > r['PA_R2'])
    print(f"  V6 vence PA: {v6_wins}/{len(all_results)} traits")

    pa_wins = [r['Trait'] for r in all_results if r['PA_R2'] >= r['V6_R2']]
    if pa_wins:
        print(f"  PA vence: {pa_wins}")

    print(f"\n  Resultados: {OUTPUT_DIR / 'v6_cdcb_results.csv'}")

    return results_df


if __name__ == '__main__':
    run_cdcb_pipeline()
