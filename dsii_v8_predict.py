"""
DSII V8 Oracle - Production Prediction Engine
Input: NAAB do pai, avo materno (MGS) e bisavo materno (MGGS)
Output: Predicted PTAs for 25 traits

Usage:
  Single:  python dsii_v8_predict.py <sire_naab> <mgs_naab> <mggs_naab>
  Batch:   python dsii_v8_predict.py --batch input.csv
           (CSV with columns: sire_naab, mgs_naab, mggs_naab)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import csv, pickle, sys
from pathlib import Path
from collections import defaultdict

BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")
MODELS_PKL = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v8_results/v8_oracle_models.pkl")
OUTPUT_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v8_results")

# ============================================================
# DOMAIN KNOWLEDGE (same as V8 training)
# ============================================================
GENETIC_SD = {
    'TPI': 250, 'NM$': 275, 'CM$': 280,
    'MILK': 675, 'FAT': 29, 'FAT%': 0.05, 'PRO': 19, 'PRO%': 0.02, 'CFP': 25,
    'PL': 1.85, 'SCS': 0.14, 'DPR': 1.3, 'HCR': 1.4, 'CCR': 1.65,
    'LIV': 1.2, 'FI': 1.0, 'MAST': 1.0, 'FSAV': 50,
    'PTAT': 0.70, 'UDC': 0.75, 'FLC': 0.65,
    'SCE': 1.5, 'DCE': 1.2, 'SSB': 1.0, 'DSB': 1.0,
}
HERITABILITY = {
    'TPI': 0.30, 'NM$': 0.30, 'CM$': 0.30,
    'MILK': 0.25, 'FAT': 0.25, 'FAT%': 0.50, 'PRO': 0.25, 'PRO%': 0.50,
    'CFP': 0.30, 'PL': 0.08, 'SCS': 0.12, 'DPR': 0.04, 'LIV': 0.05,
    'FI': 0.06, 'HCR': 0.04, 'CCR': 0.04, 'MAST': 0.04, 'FSAV': 0.15,
    'PTAT': 0.30, 'UDC': 0.25, 'FLC': 0.15,
    'SCE': 0.08, 'DCE': 0.06, 'SSB': 0.06, 'DSB': 0.04,
}
BREED_IDEAL = {
    'MILK': 1800, 'FAT': 90, 'PRO': 60, 'CFP': 70,
    'DPR': 1.5, 'CCR': 2.0, 'HCR': 2.0,
    'PL': 5.0, 'LIV': 2.0, 'UDC': 1.5, 'FLC': 1.0,
    'MAST': -1.0, 'SCS': 2.60, 'SCE': 2.0,
}
LOWER_IS_BETTER = {'SCS', 'SCE', 'SSB', 'DSB', 'DCE', 'MAST'}
INBREEDING_DEPRESSION = {
    'NM$': -25.0, 'TPI': -20.0, 'CM$': -25.0,
    'MILK': -28.5, 'FAT': -1.0, 'PRO': -0.9,
    'PL': -0.35, 'DPR': -0.03, 'SCS': 0.007, 'LIV': -0.15,
}
GENETIC_CORRELATIONS = {
    ('MILK', 'DPR'): -0.35, ('MILK', 'CCR'): -0.30, ('MILK', 'SCS'): 0.10,
    ('MILK', 'PL'): -0.15, ('FAT', 'PRO'): 0.65, ('FAT', 'DPR'): -0.25,
    ('PRO', 'DPR'): -0.30, ('DPR', 'CCR'): 0.55, ('DPR', 'PL'): 0.45,
    ('SCS', 'PL'): -0.30, ('SCS', 'MAST'): 0.70,
    ('PTAT', 'UDC'): 0.40, ('PTAT', 'FLC'): 0.30,
    ('NM$', 'TPI'): 0.85, ('NM$', 'CM$'): 0.95,
    ('MILK', 'FAT'): 0.55, ('MILK', 'PRO'): 0.85,
    ('PL', 'LIV'): 0.65, ('DPR', 'FI'): 0.70,
    ('SCE', 'SSB'): 0.60, ('DCE', 'DSB'): 0.55,
}
DOMINANCE_RATIOS = {
    'MILK': 0.12, 'FAT': 0.10, 'PRO': 0.12,
    'SCS': 0.14, 'PL': 0.15, 'DPR': 0.20, 'CCR': 0.15,
    'UDC': 0.11, 'FLC': 0.08, 'LIV': 0.12,
}

# Trait name -> bulls.csv column name
BULL_COL = {
    'TPI': 'TPI', 'NM$': 'NM$', 'CM$': 'CM$',
    'MILK': 'PTAM', 'FAT': 'PTAF', 'FAT%': 'PTAF%',
    'PRO': 'PTAP', 'PRO%': 'PTAP%', 'CFP': 'CFP',
    'PL': 'PL', 'SCS': 'SCS', 'DPR': 'DPR',
    'LIV': 'LIV', 'FI': 'FI', 'HCR': 'HCR',
    'CCR': 'CCR', 'MAST': 'MAST', 'FSAV': 'F SAV',
    'PTAT': 'PTAT', 'UDC': 'UDC', 'FLC': 'FLC',
    'SCE': 'SCE', 'DCE': 'DCE', 'SSB': 'SSB', 'DSB': 'DSB',
}
ALL_TRAITS = list(BULL_COL.keys())


def sf(v):
    if v is None: return None
    try:
        x = float(v)
        return x if not np.isnan(x) else None
    except: return None


def load_bulls():
    """Load bulls.csv indexed by NAAB code."""
    bulls = {}
    with open(BULLS_CSV, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            naab = row.get('NAAB', '').strip()
            if naab:
                bulls[naab] = row
    return bulls


def get_bull_ptas(bull_row):
    """Extract PTAs dict from a bulls.csv row."""
    ptas = {}
    for trait, col in BULL_COL.items():
        v = sf(bull_row.get(col))
        if v is not None:
            ptas[trait] = v
    return ptas


def estimate_dam_ptas(mgs_ptas, mggs_ptas):
    """
    Estimate dam PTAs from pedigree:
      Dam_PTA = MGS/2 + MGGS/4
    The remaining 1/4 (unknown great-great-grandam) is assumed breed average (0).
    """
    dam = {}
    for trait in ALL_TRAITS:
        mgs_v = mgs_ptas.get(trait)
        mggs_v = mggs_ptas.get(trait)
        if mgs_v is not None and mggs_v is not None:
            dam[trait] = mgs_v / 2 + mggs_v / 4
        elif mgs_v is not None:
            dam[trait] = mgs_v / 2
        elif mggs_v is not None:
            dam[trait] = mggs_v / 4
    return dam


def build_features(sire_ptas, dam_ptas, trait):
    """Build feature vector for a single trait (same as V8 training)."""
    s = sire_ptas.get(trait)
    d = dam_ptas.get(trait)
    if s is None or d is None:
        return None
    f = {}
    sd = GENETIC_SD.get(trait, 1)
    h2 = HERITABILITY.get(trait, 0.15)
    pa = (s + d) / 2

    f['sire'] = s; f['dam'] = d; f['pa'] = pa
    f['delta_g'] = (s - d) / 2; f['diff'] = s - d
    f['sire_z'] = s / sd if sd else 0; f['dam_z'] = d / sd if sd else 0
    f['sire_sq'] = s * s; f['dam_sq'] = d * d; f['sxd'] = s * d
    f['h2_pa'] = pa * (0.5 + h2)

    ideal = BREED_IDEAL.get(trait)
    if ideal is not None:
        defic = max(0, ((d - ideal) / sd) if trait in LOWER_IS_BETTER else ((ideal - d) / sd))
        f['deficiency'] = defic; f['dg_x_def'] = f['delta_g'] * defic
    else:
        f['deficiency'] = 0; f['dg_x_def'] = 0

    f['ib_effect'] = INBREEDING_DEPRESSION.get(trait, 0) * 0.085
    dom = DOMINANCE_RATIOS.get(trait, 0)
    f['dom_pot'] = dom * abs(s - d) / sd if sd else 0

    for ot in ALL_TRAITS:
        if ot == trait: continue
        sv = sire_ptas.get(ot); dv = dam_ptas.get(ot)
        if sv is not None: f[f's_{ot}'] = sv
        if dv is not None: f[f'd_{ot}'] = dv

    for (t1, t2), corr in GENETIC_CORRELATIONS.items():
        if t1 == trait:
            ov = sire_ptas.get(t2); dov = dam_ptas.get(t2)
            if ov is not None and dov is not None:
                f[f'gc_{t2}'] = ((ov + dov) / 2) * corr
        elif t2 == trait:
            ov = sire_ptas.get(t1); dov = dam_ptas.get(t1)
            if ov is not None and dov is not None:
                f[f'gc_{t1}'] = ((ov + dov) / 2) * corr

    milk_pa = ((sire_ptas.get('MILK', 0) or 0) + (dam_ptas.get('MILK', 0) or 0)) / 2
    fat_pa = ((sire_ptas.get('FAT', 0) or 0) + (dam_ptas.get('FAT', 0) or 0)) / 2
    pro_pa = ((sire_ptas.get('PRO', 0) or 0) + (dam_ptas.get('PRO', 0) or 0)) / 2
    f['prod_press'] = milk_pa / 675 + fat_pa / 29 + pro_pa / 19
    f['sire_nm'] = sire_ptas.get('NM$', 0) or 0
    f['dam_nm'] = dam_ptas.get('NM$', 0) or 0

    return f


def predict_mating(sire_naab, mgs_naab, mggs_naab, bulls, models):
    """
    Predict daughter PTAs for a single mating.
    Returns dict with predictions, PA, and dam estimates.
    """
    sire_row = bulls.get(sire_naab)
    mgs_row = bulls.get(mgs_naab)
    mggs_row = bulls.get(mggs_naab)

    errors = []
    if not sire_row: errors.append(f"Sire NAAB '{sire_naab}' not found")
    if not mgs_row: errors.append(f"MGS NAAB '{mgs_naab}' not found")
    if not mggs_row: errors.append(f"MGGS NAAB '{mggs_naab}' not found")
    if errors:
        return {'error': '; '.join(errors)}

    sire_ptas = get_bull_ptas(sire_row)
    mgs_ptas = get_bull_ptas(mgs_row)
    mggs_ptas = get_bull_ptas(mggs_row)
    dam_ptas = estimate_dam_ptas(mgs_ptas, mggs_ptas)

    sire_name = sire_row.get('Name', '') or sire_row.get('Registration Name', '')
    mgs_name = mgs_row.get('Name', '') or mgs_row.get('Registration Name', '')
    mggs_name = mggs_row.get('Name', '') or mggs_row.get('Registration Name', '')

    result = {
        'sire_naab': sire_naab, 'sire_name': sire_name,
        'mgs_naab': mgs_naab, 'mgs_name': mgs_name,
        'mggs_naab': mggs_naab, 'mggs_name': mggs_name,
    }

    for trait in ALL_TRAITS:
        model_info = models.get(trait)
        if not model_info:
            continue

        s = sire_ptas.get(trait)
        d = dam_ptas.get(trait)
        if s is None or d is None:
            continue

        pa = (s + d) / 2
        result[f'{trait}_PA'] = round(pa, 2)
        result[f'{trait}_Sire'] = round(s, 2)
        result[f'{trait}_Dam_est'] = round(d, 2)

        # Build features
        feat = build_features(sire_ptas, dam_ptas, trait)
        if feat is None:
            result[f'{trait}_DSII'] = round(pa, 2)
            continue

        # Align features with training columns
        feat_cols = model_info['feature_cols']
        X = np.array([[feat.get(c, 0) for c in feat_cols]])

        model = model_info['model']
        pred = model.predict(X)[0]
        result[f'{trait}_DSII'] = round(pred, 2)
        result[f'{trait}_diff'] = round(pred - pa, 2)

    return result


def print_single_result(result):
    """Pretty print a single mating prediction."""
    if 'error' in result:
        print(f"\n  ERRO: {result['error']}")
        return

    print(f"\n{'='*80}")
    print(f"  DSII V8 Oracle - Predicao de Acasalamento")
    print(f"{'='*80}")
    print(f"  Pai (Sire):     {result['sire_naab']:>12}  {result.get('sire_name', '')}")
    print(f"  Avo Mat (MGS):  {result['mgs_naab']:>12}  {result.get('mgs_name', '')}")
    print(f"  Bisavo (MGGS):  {result['mggs_naab']:>12}  {result.get('mggs_name', '')}")
    print(f"{'='*80}")
    print(f"  {'Trait':>6} | {'Sire':>8} | {'Dam(est)':>8} | {'PA':>8} | {'DSII V8':>8} | {'Diff':>7}")
    print(f"  {'-'*60}")

    for trait in ALL_TRAITS:
        sire = result.get(f'{trait}_Sire')
        dam = result.get(f'{trait}_Dam_est')
        pa = result.get(f'{trait}_PA')
        dsii = result.get(f'{trait}_DSII')
        diff = result.get(f'{trait}_diff', 0)
        if dsii is None:
            continue
        arrow = '+' if diff > 0 else ''
        print(f"  {trait:>6} | {sire:>8} | {dam:>8.1f} | {pa:>8.1f} | {dsii:>8.1f} | {arrow}{diff:.1f}")

    print(f"{'='*80}")


def run_batch(csv_path, bulls, models):
    """Process a batch CSV file with columns: sire_naab, mgs_naab, mggs_naab."""
    inp = pd.read_csv(csv_path)
    # Normalize column names
    cols = {c.strip().lower().replace(' ', '_'): c for c in inp.columns}

    sire_col = cols.get('sire_naab') or cols.get('sire') or cols.get('pai') or inp.columns[0]
    mgs_col = cols.get('mgs_naab') or cols.get('mgs') or cols.get('avo_materno') or inp.columns[1]
    mggs_col = cols.get('mggs_naab') or cols.get('mggs') or cols.get('bisavo_materno') or inp.columns[2]

    print(f"  Batch: {len(inp)} matings from {csv_path}")
    print(f"  Columns: sire={sire_col}, mgs={mgs_col}, mggs={mggs_col}")

    all_results = []
    errors = 0
    for idx, row in inp.iterrows():
        sire = str(row[sire_col]).strip()
        mgs = str(row[mgs_col]).strip()
        mggs = str(row[mggs_col]).strip()
        result = predict_mating(sire, mgs, mggs, bulls, models)
        if 'error' in result:
            errors += 1
            result['row'] = idx + 1
        all_results.append(result)

    # Build output DataFrame
    out_rows = []
    for r in all_results:
        if 'error' in r:
            out_rows.append({'sire_naab': r.get('sire_naab', ''), 'error': r['error']})
            continue
        row = {
            'sire_naab': r['sire_naab'], 'sire_name': r.get('sire_name', ''),
            'mgs_naab': r['mgs_naab'], 'mgs_name': r.get('mgs_name', ''),
            'mggs_naab': r['mggs_naab'], 'mggs_name': r.get('mggs_name', ''),
        }
        for trait in ALL_TRAITS:
            row[f'{trait}_Sire'] = r.get(f'{trait}_Sire')
            row[f'{trait}_Dam'] = r.get(f'{trait}_Dam_est')
            row[f'{trait}_PA'] = r.get(f'{trait}_PA')
            row[f'{trait}_DSII'] = r.get(f'{trait}_DSII')
            row[f'{trait}_Diff'] = r.get(f'{trait}_diff')
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_path = OUTPUT_DIR / "v8_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  OK: {len(all_results) - errors} predictions, {errors} errors")
    print(f"  Saved: {out_path}")
    return out_df


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("  Loading bulls database...", end=' ')
    sys.stdout.flush()
    bulls = load_bulls()
    print(f"{len(bulls)} bulls loaded.")

    print("  Loading V8 Oracle models...", end=' ')
    sys.stdout.flush()
    with open(MODELS_PKL, 'rb') as f:
        models = pickle.load(f)
    print(f"{len(models)} trait models loaded.")

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Single:  python dsii_v8_predict.py <sire_naab> <mgs_naab> <mggs_naab>")
        print("  Batch:   python dsii_v8_predict.py --batch input.csv")
        sys.exit(0)

    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("  ERROR: Provide CSV path. Ex: python dsii_v8_predict.py --batch matings.csv")
            sys.exit(1)
        run_batch(sys.argv[2], bulls, models)
    else:
        if len(sys.argv) < 4:
            print("  ERROR: Provide 3 NAABs. Ex: python dsii_v8_predict.py 7HO18596 7HO15708 7HO12345")
            sys.exit(1)
        sire_naab = sys.argv[1].strip()
        mgs_naab = sys.argv[2].strip()
        mggs_naab = sys.argv[3].strip()
        result = predict_mating(sire_naab, mgs_naab, mggs_naab, bulls, models)
        print_single_result(result)
