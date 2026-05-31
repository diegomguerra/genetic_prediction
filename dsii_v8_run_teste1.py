"""
DSII V8 Oracle — Run predictions on Teste1.xlsx
Input: Sire NAAB + MGS NAAB (no MGGS available)
Dam estimated as: Dam = MGS/2 (remaining unknown = breed avg)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import csv, pickle, sys, openpyxl
from pathlib import Path
from collections import defaultdict

BULLS_CSV = Path("C:/Users/DiegoGuerra/gen-genie-advisor/sms-engine/data/bulls.csv")
MODELS_PKL = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v8_results/v8_oracle_models.pkl")
INPUT_FILE = Path("C:/Users/DiegoGuerra/Downloads/Teste1.xlsx")
OUTPUT_DIR = Path("C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/dsii_v8_results")

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

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def get_bull_ptas(bull_row):
    ptas = {}
    for trait, col in BULL_COL.items():
        v = sf(bull_row.get(col))
        if v is not None:
            ptas[trait] = v
    return ptas

def estimate_dam_ptas(mgs_ptas, mggs_ptas=None):
    """Dam = MGS/2 + MGGS/4. If no MGGS, Dam = MGS/2."""
    dam = {}
    for trait in ALL_TRAITS:
        mgs_v = mgs_ptas.get(trait)
        mggs_v = mggs_ptas.get(trait) if mggs_ptas else None
        if mgs_v is not None and mggs_v is not None:
            dam[trait] = mgs_v / 2 + mggs_v / 4
        elif mgs_v is not None:
            dam[trait] = mgs_v / 2
        elif mggs_v is not None:
            dam[trait] = mggs_v / 4
    return dam

def build_features(sire_ptas, dam_ptas, trait):
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

# ============================================================
# MAIN
# ============================================================
flush_print("=" * 90)
flush_print("  DSII V8 Oracle — Teste1.xlsx (300 animals)")
flush_print("=" * 90)

# Load bulls
flush_print("\n  Loading bulls database...", end=' ')
bulls = {}
with open(BULLS_CSV, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        naab = row.get('NAAB', '').strip()
        if naab:
            bulls[naab] = row
flush_print(f"{len(bulls)} bulls loaded.")

# Load models
flush_print("  Loading V8 Oracle models...", end=' ')
with open(MODELS_PKL, 'rb') as f:
    models = pickle.load(f)
flush_print(f"{len(models)} trait models loaded.")

# Load Teste1
flush_print("  Loading Teste1.xlsx...", end=' ')
wb = openpyxl.load_workbook(INPUT_FILE, read_only=True, data_only=True)
ws = wb.active
rows = list(ws.iter_rows(values_only=True))
wb.close()
hdr = [str(c).strip() if c else f'col_{i}' for i, c in enumerate(rows[0])]
flush_print(f"{len(rows)-1} animals.")

# Process
flush_print(f"\n  Processing predictions...\n")

all_results = []
sire_not_found = set()
mgs_not_found = set()
ok_count = 0
err_count = 0

for r in rows[1:]:
    rd = {hdr[i]: r[i] for i in range(min(len(hdr), len(r)))}
    animal_id = str(rd.get('Animal ID', '')).strip()
    sire_naab = str(rd.get('Sire of Record NAAB', '')).strip()
    mgs_naab = str(rd.get('Maternal Grandsire NAAB', '')).strip()

    # Stud code mappings: sexed semen (5xx) and known aliases
    STUD_MAP = {
        '507': '7',    '509': '9',
        '550': '50',   '551': '51',
        '559': '59',   '518': '18',
        '521': '21',   '523': '23',
        '581': '81',   '585': '85',
        '501': '1',
        '601': '1',    '604': '4',
        '614': '14',   '629': '29',
        '751': '151',  '752': '152',
        '814': '14',   '250': '200',
    }

    def lookup_bull(naab):
        """Try multiple NAAB normalization strategies."""
        import re
        row = bulls.get(naab)
        if row:
            return row, naab

        m = re.match(r'^(\d+)(HO|BS)0*(\d+)$', naab)
        if not m:
            return None, naab

        org, breed, num = m.group(1), m.group(2), m.group(3)

        # Build candidates from stud mapping and zero-stripping
        orgs_to_try = [org]
        # Strip leading 5 (sexed semen)
        if org.startswith('5'):
            orgs_to_try.append(org[1:])
        # Known stud mappings
        if org in STUD_MAP:
            orgs_to_try.append(STUD_MAP[org])
        # Strip leading zeros
        orgs_to_try.append(org.lstrip('0') or org)

        for o in orgs_to_try:
            candidate = f'{o}{breed}{num}'
            row = bulls.get(candidate)
            if row:
                return row, candidate

        return None, naab

    sire_row, sire_matched = lookup_bull(sire_naab)
    mgs_row, mgs_matched = lookup_bull(mgs_naab)

    result = {'Animal_ID': animal_id, 'Sire_NAAB': sire_naab, 'MGS_NAAB': mgs_naab}

    if not sire_row:
        sire_not_found.add(sire_naab)
        result['Error'] = f'Sire {sire_naab} not found'
        err_count += 1
        all_results.append(result)
        continue
    if not mgs_row:
        mgs_not_found.add(mgs_naab)
        result['Error'] = f'MGS {mgs_naab} not found'
        err_count += 1
        all_results.append(result)
        continue

    sire_ptas = get_bull_ptas(sire_row)
    mgs_ptas = get_bull_ptas(mgs_row)
    dam_ptas = estimate_dam_ptas(mgs_ptas)  # Dam = MGS/2 only

    result['Sire_Name'] = sire_row.get('Name', '') or ''
    result['MGS_Name'] = mgs_row.get('Name', '') or ''

    for trait in ALL_TRAITS:
        model_info = models.get(trait)
        if not model_info:
            continue

        s = sire_ptas.get(trait)
        d = dam_ptas.get(trait)
        if s is None or d is None:
            continue

        pa = (s + d) / 2
        result[f'{trait}_Sire'] = round(s, 2)
        result[f'{trait}_Dam_est'] = round(d, 2)
        result[f'{trait}_PA'] = round(pa, 2)

        feat = build_features(sire_ptas, dam_ptas, trait)
        if feat is None:
            result[f'{trait}_DSII'] = round(pa, 2)
            continue

        feat_cols = model_info['feature_cols']
        X = np.array([[feat.get(c, 0) for c in feat_cols]])
        pred = model_info['model'].predict(X)[0]
        result[f'{trait}_DSII'] = round(pred, 2)
        result[f'{trait}_Diff'] = round(pred - pa, 2)

    ok_count += 1
    all_results.append(result)

# Summary
flush_print(f"  {'='*90}")
flush_print(f"  RESULTS SUMMARY")
flush_print(f"  {'='*90}")
flush_print(f"  Total animals: {len(rows)-1}")
flush_print(f"  Predicted OK:  {ok_count}")
flush_print(f"  Errors:        {err_count}")

if sire_not_found:
    flush_print(f"\n  Sires not found ({len(sire_not_found)} unique):")
    for s in sorted(sire_not_found):
        flush_print(f"    - {s}")

if mgs_not_found:
    flush_print(f"\n  MGS not found ({len(mgs_not_found)} unique):")
    for s in sorted(mgs_not_found):
        flush_print(f"    - {s}")

# Save only successful predictions (exclude errors)
ok_results = [r for r in all_results if 'Error' not in r]

excel_rows = []
for r in ok_results:
    row = {
        'Animal ID': r['Animal_ID'],
        'Sire NAAB': r['Sire_NAAB'],
        'Sire Name': r.get('Sire_Name', ''),
        'MGS NAAB': r['MGS_NAAB'],
        'MGS Name': r.get('MGS_Name', ''),
    }
    for t in ALL_TRAITS:
        row[f'{t}_PA'] = r.get(f'{t}_PA')
        row[f'{t}_DSII'] = r.get(f'{t}_DSII')
        row[f'{t}_Diff'] = r.get(f'{t}_Diff')
    excel_rows.append(row)

excel_df = pd.DataFrame(excel_rows)
excel_path = OUTPUT_DIR / "Teste1_DSII_V8_Predictions.xlsx"
excel_df.to_excel(excel_path, index=False, engine='openpyxl')
flush_print(f"\n  Excel saved ({len(ok_results)} animals): {excel_path}")

# Show first 5 examples
key_traits = ['TPI', 'NM$', 'CM$', 'MILK', 'FAT', 'PRO', 'PL', 'DPR', 'SCS', 'PTAT', 'UDC', 'FLC']
flush_print(f"\n  {'='*90}")
flush_print(f"  FIRST 5 PREDICTIONS (key traits)")
flush_print(f"  {'='*90}")
shown = 0
for r in all_results:
    if shown >= 5: break
    if 'Error' in r:
        continue
    shown += 1
    flush_print(f"\n  Animal {r['Animal_ID']} | Sire: {r['Sire_NAAB']} ({r.get('Sire_Name','')}) | MGS: {r['MGS_NAAB']} ({r.get('MGS_Name','')})")
    flush_print(f"  {'Trait':>6} | {'PA':>8} | {'DSII':>8} | {'Diff':>7}")
    flush_print(f"  {'-'*40}")
    for t in key_traits:
        pa = r.get(f'{t}_PA')
        dsii = r.get(f'{t}_DSII')
        diff = r.get(f'{t}_Diff', 0)
        if pa is not None and dsii is not None:
            arrow = '+' if diff and diff > 0 else ''
            flush_print(f"  {t:>6} | {pa:>8.1f} | {dsii:>8.1f} | {arrow}{diff:.1f}")

flush_print(f"\n  Done! Ready for genomic comparison.")
