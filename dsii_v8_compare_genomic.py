"""
DSII V8 Oracle — Compare predictions vs genomic gold standard.
Merge Teste1 predictions with ReportCoreTraits (actual genomic PTAs).
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import openpyxl, sys
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

PREDICTIONS = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v8_results/Teste1_DSII_V8_Predictions.xlsx")
GENOMIC = Path("C:/Users/DiegoGuerra/Downloads/ReportCoreTraits2026-05-21 (1).xlsx")
OUTPUT_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v8_results")

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ============================================================
# LOAD DATA
# ============================================================
flush_print("=" * 95)
flush_print("  DSII V8 Oracle — Genomic Validation (Gold Standard)")
flush_print("=" * 95)

# Load predictions
flush_print("\n  Loading predictions...", end=' ')
pred_df = pd.read_excel(PREDICTIONS, engine='openpyxl')
flush_print(f"{len(pred_df)} animals.")

# Load genomic gold standard
flush_print("  Loading genomic data...", end=' ')
gen_df = pd.read_excel(GENOMIC, engine='openpyxl')
flush_print(f"{len(gen_df)} animals.")

# Normalize Animal ID to string
pred_df['Animal ID'] = pred_df['Animal ID'].astype(str).str.strip()
gen_df['Animal ID'] = gen_df['Animal ID'].astype(str).str.strip()

# Merge
merged = pred_df.merge(gen_df, on='Animal ID', how='inner', suffixes=('_pred', '_gen'))
flush_print(f"  Matched: {len(merged)} animals.")

# ============================================================
# TRAIT MAPPING: prediction columns -> genomic columns
# ============================================================
# Prediction cols: TPI_PA, TPI_DSII, TPI_Diff
# Genomic cols: TPI, DPR, FAT, FAT %, LIV, PROT, PROT%, PL, UDC, PTAT, MILK, FI, FLC, SCS, CCR
TRAIT_COMPARE = {
    'TPI':  'TPI',
    'DPR':  'DPR',
    'FAT':  'FAT',
    'FAT%': 'FAT %',
    'LIV':  'LIV',
    'PRO':  'PROT',
    'PRO%': 'PROT%',
    'PL':   'PL',
    'UDC':  'UDC',
    'PTAT': 'PTAT',
    'MILK': 'MILK',
    'FI':   'FI',
    'FLC':  'FLC',
    'SCS':  'SCS',
    'CCR':  'CCR',
}

# ============================================================
# COMPUTE ACCURACY
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  ACCURACY: DSII V8 vs Parent Average vs Genomic (Gold Standard)")
flush_print(f"{'='*95}")
flush_print(f"\n  {'Trait':>6} | {'N':>5} | {'PA R2':>7} | {'DSII R2':>8} | {'PA MAE':>8} | {'DSII MAE':>9} | "
            f"{'MAE Gain':>9} | {'R2 Gain':>8} | Winner")
flush_print(f"  {'-'*95}")

results = []
for trait, gen_col in TRAIT_COMPARE.items():
    pa_col = f'{trait}_PA'
    dsii_col = f'{trait}_DSII'

    if pa_col not in merged.columns or dsii_col not in merged.columns:
        continue
    if gen_col not in merged.columns:
        continue

    # Get valid rows
    mask = merged[pa_col].notna() & merged[dsii_col].notna() & merged[gen_col].notna()
    sub = merged[mask]
    if len(sub) < 10:
        continue

    actual = pd.to_numeric(sub[gen_col], errors='coerce')
    pa_pred = pd.to_numeric(sub[pa_col], errors='coerce')
    dsii_pred = pd.to_numeric(sub[dsii_col], errors='coerce')

    # Drop any remaining NaN
    valid = actual.notna() & pa_pred.notna() & dsii_pred.notna()
    actual = actual[valid].values
    pa_pred = pa_pred[valid].values
    dsii_pred = dsii_pred[valid].values

    if len(actual) < 10:
        continue

    pa_r2 = r2_score(actual, pa_pred)
    dsii_r2 = r2_score(actual, dsii_pred)
    pa_mae = mean_absolute_error(actual, pa_pred)
    dsii_mae = mean_absolute_error(actual, dsii_pred)

    mae_gain = (pa_mae - dsii_mae) / pa_mae * 100 if pa_mae > 0 else 0
    r2_gain = (dsii_r2 - pa_r2) / abs(pa_r2) * 100 if pa_r2 != 0 else 0
    winner = "DSII" if dsii_r2 > pa_r2 else "PA"

    flush_print(f"  {trait:>6} | {len(actual):>5} | {pa_r2:>7.4f} | {dsii_r2:>8.4f} | "
                f"{pa_mae:>8.2f} | {dsii_mae:>9.2f} | {mae_gain:>+8.1f}% | {r2_gain:>+7.1f}% | {winner}")

    results.append({
        'Trait': trait, 'N': len(actual),
        'PA_R2': round(pa_r2, 4), 'DSII_R2': round(dsii_r2, 4),
        'PA_MAE': round(pa_mae, 2), 'DSII_MAE': round(dsii_mae, 2),
        'MAE_Gain_pct': round(mae_gain, 1), 'R2_Gain_pct': round(r2_gain, 1),
        'Winner': winner,
    })

# ============================================================
# SUMMARY
# ============================================================
flush_print(f"\n{'='*95}")
flush_print(f"  SUMMARY")
flush_print(f"{'='*95}")

if results:
    pa_r2s = [r['PA_R2'] for r in results]
    dsii_r2s = [r['DSII_R2'] for r in results]
    dsii_wins = sum(1 for r in results if r['Winner'] == 'DSII')
    pa_wins = sum(1 for r in results if r['Winner'] == 'PA')
    mae_gains = [r['MAE_Gain_pct'] for r in results]

    flush_print(f"  Traits compared:    {len(results)}")
    flush_print(f"  Animals matched:    {len(merged)}")
    flush_print(f"  PA avg R2:          {np.mean(pa_r2s):.4f}")
    flush_print(f"  DSII avg R2:        {np.mean(dsii_r2s):.4f}")
    flush_print(f"  DSII wins:          {dsii_wins}/{len(results)}")
    flush_print(f"  PA wins:            {pa_wins}/{len(results)}")
    flush_print(f"  Avg MAE reduction:  {np.mean(mae_gains):+.1f}%")

    # Correlation analysis
    flush_print(f"\n  Correlation (Pearson) between predicted and actual:")
    for r in results:
        trait = r['Trait']
        gen_col = TRAIT_COMPARE[trait]
        pa_col = f'{trait}_PA'
        dsii_col = f'{trait}_DSII'
        mask = merged[pa_col].notna() & merged[dsii_col].notna() & merged[gen_col].notna()
        sub = merged[mask]
        actual = pd.to_numeric(sub[gen_col], errors='coerce')
        pa_v = pd.to_numeric(sub[pa_col], errors='coerce')
        dsii_v = pd.to_numeric(sub[dsii_col], errors='coerce')
        valid = actual.notna() & pa_v.notna() & dsii_v.notna()
        if valid.sum() > 10:
            corr_pa = np.corrcoef(actual[valid], pa_v[valid])[0, 1]
            corr_dsii = np.corrcoef(actual[valid], dsii_v[valid])[0, 1]
            flush_print(f"    {trait:>6}: PA r={corr_pa:.4f}  DSII r={corr_dsii:.4f}  "
                        f"{'DSII better' if corr_dsii > corr_pa else 'PA better'}")

# Save results
res_df = pd.DataFrame(results)
res_path = OUTPUT_DIR / "genomic_validation_results.csv"
res_df.to_csv(res_path, index=False)
flush_print(f"\n  Results saved: {res_path}")

# Save merged comparison Excel
comp_rows = []
for _, row in merged.iterrows():
    r = {'Animal ID': row['Animal ID']}
    for trait, gen_col in TRAIT_COMPARE.items():
        pa_col = f'{trait}_PA'
        dsii_col = f'{trait}_DSII'
        gen_val = pd.to_numeric(row.get(gen_col), errors='coerce') if gen_col in row.index else None
        pa_val = row.get(pa_col)
        dsii_val = row.get(dsii_col)
        r[f'{trait}_Actual'] = gen_val
        r[f'{trait}_PA'] = pa_val
        r[f'{trait}_DSII'] = dsii_val
        if gen_val is not None and dsii_val is not None and pa_val is not None:
            try:
                r[f'{trait}_PA_err'] = round(abs(float(gen_val) - float(pa_val)), 2)
                r[f'{trait}_DSII_err'] = round(abs(float(gen_val) - float(dsii_val)), 2)
            except:
                pass
    comp_rows.append(r)

comp_df = pd.DataFrame(comp_rows)
comp_path = OUTPUT_DIR / "Teste1_Genomic_Comparison.xlsx"
comp_df.to_excel(comp_path, index=False, engine='openpyxl')
flush_print(f"  Comparison Excel: {comp_path}")
flush_print(f"\n  Done!")
