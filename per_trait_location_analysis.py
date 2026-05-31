"""
Analise DETALHADA por trait - Granularidade de localizacao
PA vs V10 vs V11 vs V12_OLD vs V12_2F vs V12_DAU
"""
import pandas as pd, numpy as np, pickle, sys
from scipy.stats import spearmanr, pearsonr
sys.stdout.reconfigure(encoding='utf-8')

pred_df = pd.read_pickle('dsii_v12_results/multi_version_predictions.pkl')
TRAITS = ['TPI', 'NM$', 'MILK', 'FAT', 'PRO', 'FAT%', 'PRO%', 'PL', 'DPR', 'HCR']
VERSIONS = ['PA', 'V10', 'V11', 'V12_OLD', 'V12_2F', 'V12_DAU']

# Also load V12 model info for ML method details
with open('dsii_v12_results/v12_models.pkl', 'rb') as f: v12_models = pickle.load(f)
with open('dsii_v11_results/v11_models.pkl', 'rb') as f: v11_models = pickle.load(f)
with open('dsii_v10_results/v10_models.pkl', 'rb') as f: v10_models = pickle.load(f)

def percentile_group(vals, n_groups):
    return pd.qcut(pd.Series(vals).rank(method='first'), q=n_groups, labels=False).values + 1

def analyze_version(gen_vals, pred_vals, n_groups):
    n = len(gen_vals)
    if n < n_groups * 2: return None
    gen_g = percentile_group(gen_vals, n_groups)
    pred_g = percentile_group(pred_vals, n_groups)
    exact = (gen_g == pred_g).mean() * 100
    adj1 = (np.abs(gen_g - pred_g) <= 1).mean() * 100
    gen_top = set(np.where(gen_g == n_groups)[0])
    pred_top = set(np.where(pred_g == n_groups)[0])
    top_acc = len(gen_top & pred_top) / len(gen_top) * 100 if gen_top else 0
    gen_bot = set(np.where(gen_g == 1)[0])
    pred_bot = set(np.where(pred_g == 1)[0])
    bot_acc = len(gen_bot & pred_bot) / len(gen_bot) * 100 if gen_bot else 0
    return {'exact': exact, 'adj1': adj1, 'top': top_acc, 'bottom': bot_acc}

# ML method info per trait
def get_ml_info(trait):
    info = {}
    # V10
    v10i = v10_models.get(trait, {})
    v10_name = v10i.get('model_name','?') if v10i.get('type') == 'single' else 'STACK'
    info['V10'] = v10_name
    # V11
    v11i = v11_models.get(trait, {})
    v11_name = v11i.get('model_name','?') if v11i.get('type') == 'single' else 'STACK'
    if v11i.get('v10b') == 'hybrid_3way': v11_name = 'HYBRID'
    info['V11'] = v11_name
    # V12
    v12i = v12_models.get(trait, {})
    bases = v12i.get('base_names', [])
    v12_name = '+'.join(bases)
    if v12i.get('hybrid'): v12_name += '+HYB'
    info['V12'] = v12_name
    # R2 values
    info['V10_R2'] = v10i.get('r2_cv', -999)
    info['V11_R2'] = v11i.get('r2_cv', v11i.get('r2_dev', -999))
    info['V12_R2'] = v12i.get('r2_cv', v12i.get('r2_dev', -999))
    info['V12_PA_R2'] = v12i.get('pa_r2', -999)
    return info

# ============================================================
print("=" * 130)
print("  ANALISE DETALHADA POR TRAIT - LOCALIZACAO EM DIFERENTES GRANULARIDADES")
print("  Jose - 92 animais com genomica real")
print("=" * 130)

GRANS = [(4, 'Quartil 25%'), (5, 'Quintil 20%'), (10, 'Decil 10%'), (20, 'Vigintil 5%')]

for trait in TRAITS:
    tdf = pred_df[pred_df['Trait'] == trait].copy()
    gen_vals = tdf['GENOMIC'].values.astype(float)
    n = len(gen_vals)

    ml_info = get_ml_info(trait)

    print(f"\n{'='*130}")
    print(f"  TRAIT: {trait}  (N={n} animais)")
    print(f"  ML: V10={ml_info['V10']}(R2={ml_info['V10_R2']:.3f}) | V11={ml_info['V11']}(R2={ml_info['V11_R2']:.3f}) | V12={ml_info['V12']}(R2={ml_info['V12_R2']:.3f}) | PA_R2={ml_info['V12_PA_R2']:.3f}")
    print(f"{'='*130}")

    # Correlations
    print(f"\n  Correlacoes com Genomica Real:")
    print(f"  {'':>12}", end='')
    for v in VERSIONS:
        print(f" | {v:>8}", end='')
    print()
    print(f"  {'':>12}", end='')
    for v in VERSIONS:
        if v in tdf.columns and tdf[v].notna().sum() >= 10:
            valid = tdf[v].notna()
            rho, _ = spearmanr(tdf.loc[valid, v].values.astype(float), tdf.loc[valid, 'GENOMIC'].values.astype(float))
            r, _ = pearsonr(tdf.loc[valid, v].values.astype(float), tdf.loc[valid, 'GENOMIC'].values.astype(float))
            print(f" | {rho:>8.3f}", end='')
        else:
            print(f" | {'---':>8}", end='')
    print("  (Spearman)")

    # Bias
    print(f"  {'Vies medio':>12}", end='')
    for v in VERSIONS:
        if v in tdf.columns and tdf[v].notna().sum() >= 10:
            valid = tdf[v].notna()
            bias = (tdf.loc[valid, v].values.astype(float) - tdf.loc[valid, 'GENOMIC'].values.astype(float)).mean()
            print(f" | {bias:>+8.1f}", end='')
        else:
            print(f" | {'---':>8}", end='')
    print()

    # Per-granularity analysis
    for n_groups, gran_name in GRANS:
        random_pct = 100.0 / n_groups

        print(f"\n  {gran_name} ({n_groups} grupos, aleatorio={random_pct:.0f}%):")
        # Exact
        print(f"    {'Exato%':>10}", end='')
        best_v = ''; best_val = -1
        for v in VERSIONS:
            if v in tdf.columns and tdf[v].notna().sum() >= n_groups * 2:
                valid = tdf[v].notna()
                res = analyze_version(tdf.loc[valid, 'GENOMIC'].values.astype(float),
                                      tdf.loc[valid, v].values.astype(float), n_groups)
                if res:
                    marker = ''
                    if res['exact'] > best_val:
                        best_val = res['exact']; best_v = v
                    print(f" | {res['exact']:>7.1f}%", end='')
                else:
                    print(f" | {'---':>8}", end='')
            else:
                print(f" | {'---':>8}", end='')
        print(f"  << {best_v}")

        # Adjacent
        print(f"    {'Adj +-1':>10}", end='')
        for v in VERSIONS:
            if v in tdf.columns and tdf[v].notna().sum() >= n_groups * 2:
                valid = tdf[v].notna()
                res = analyze_version(tdf.loc[valid, 'GENOMIC'].values.astype(float),
                                      tdf.loc[valid, v].values.astype(float), n_groups)
                if res:
                    print(f" | {res['adj1']:>7.1f}%", end='')
                else:
                    print(f" | {'---':>8}", end='')
            else:
                print(f" | {'---':>8}", end='')
        print()

        # Top group
        print(f"    {'Top':>10}", end='')
        best_top_v = ''; best_top = -1
        for v in VERSIONS:
            if v in tdf.columns and tdf[v].notna().sum() >= n_groups * 2:
                valid = tdf[v].notna()
                res = analyze_version(tdf.loc[valid, 'GENOMIC'].values.astype(float),
                                      tdf.loc[valid, v].values.astype(float), n_groups)
                if res:
                    if res['top'] > best_top:
                        best_top = res['top']; best_top_v = v
                    print(f" | {res['top']:>7.1f}%", end='')
                else:
                    print(f" | {'---':>8}", end='')
            else:
                print(f" | {'---':>8}", end='')
        print(f"  << {best_top_v}")

        # Bottom group
        print(f"    {'Bottom':>10}", end='')
        for v in VERSIONS:
            if v in tdf.columns and tdf[v].notna().sum() >= n_groups * 2:
                valid = tdf[v].notna()
                res = analyze_version(tdf.loc[valid, 'GENOMIC'].values.astype(float),
                                      tdf.loc[valid, v].values.astype(float), n_groups)
                if res:
                    print(f" | {res['top']:>7.1f}%", end='')
                else:
                    print(f" | {'---':>8}", end='')
            else:
                print(f" | {'---':>8}", end='')
        print()

# ============================================================
# FINAL SUMMARY: Best method per trait
# ============================================================
print(f"\n{'='*130}")
print(f"  MAPA FINAL: MELHOR VERSAO POR TRAIT X GRANULARIDADE")
print(f"{'='*130}")
print(f"  {'Trait':>6} | {'Spearman':>10} | {'Quartil':>10} | {'Quintil':>10} | {'Decil':>10} | {'Vigintil':>10} | {'Top25%':>10} | {'Top10%':>10} | {'ML V12':>12}")
print("  " + "-" * 105)

for trait in TRAITS:
    tdf = pred_df[pred_df['Trait'] == trait].copy()
    ml_info = get_ml_info(trait)

    # Best Spearman
    best_sp_v = ''; best_sp = -999
    for v in VERSIONS:
        if v in tdf.columns and tdf[v].notna().sum() >= 10:
            valid = tdf[v].notna()
            rho, _ = spearmanr(tdf.loc[valid, v].values.astype(float), tdf.loc[valid, 'GENOMIC'].values.astype(float))
            if rho > best_sp: best_sp = rho; best_sp_v = v

    # Best per granularity
    results = {}
    for n_groups, gname in [(4,'Q4'),(5,'Q5'),(10,'D10'),(20,'V20')]:
        best_v = ''; best_val = -1
        for v in VERSIONS:
            if v in tdf.columns and tdf[v].notna().sum() >= n_groups * 2:
                valid = tdf[v].notna()
                res = analyze_version(tdf.loc[valid, 'GENOMIC'].values.astype(float),
                                      tdf.loc[valid, v].values.astype(float), n_groups)
                if res and res['exact'] > best_val:
                    best_val = res['exact']; best_v = v
        results[gname] = best_v

    # Best Top25 and Top10
    for n_groups, gname in [(4,'T25'),(10,'T10')]:
        best_v = ''; best_val = -1
        for v in VERSIONS:
            if v in tdf.columns and tdf[v].notna().sum() >= n_groups * 2:
                valid = tdf[v].notna()
                res = analyze_version(tdf.loc[valid, 'GENOMIC'].values.astype(float),
                                      tdf.loc[valid, v].values.astype(float), n_groups)
                if res and res['top'] > best_val:
                    best_val = res['top']; best_v = v
        results[gname] = best_v

    print(f"  {trait:>6} | {best_sp_v:>10} | {results.get('Q4',''):>10} | {results.get('Q5',''):>10} | {results.get('D10',''):>10} | {results.get('V20',''):>10} | {results.get('T25',''):>10} | {results.get('T10',''):>10} | {ml_info['V12']:>12}")

# Count wins
print(f"\n  CONTAGEM DE VITORIAS:")
for v in VERSIONS:
    wins = {'Spearman': 0, 'Quartil': 0, 'Quintil': 0, 'Decil': 0, 'Vigintil': 0, 'Top25': 0, 'Top10': 0}
    for trait in TRAITS:
        tdf = pred_df[pred_df['Trait'] == trait].copy()
        # Spearman
        best_sp = -999; best_v = ''
        for vv in VERSIONS:
            if vv in tdf.columns and tdf[vv].notna().sum() >= 10:
                valid = tdf[vv].notna()
                rho, _ = spearmanr(tdf.loc[valid, vv].values.astype(float), tdf.loc[valid, 'GENOMIC'].values.astype(float))
                if rho > best_sp: best_sp = rho; best_v = vv
        if best_v == v: wins['Spearman'] += 1

        for n_groups, gname in [(4,'Quartil'),(5,'Quintil'),(10,'Decil'),(20,'Vigintil')]:
            best_val = -1; best_v = ''
            for vv in VERSIONS:
                if vv in tdf.columns and tdf[vv].notna().sum() >= n_groups * 2:
                    valid = tdf[vv].notna()
                    res = analyze_version(tdf.loc[valid, 'GENOMIC'].values.astype(float),
                                          tdf.loc[valid, vv].values.astype(float), n_groups)
                    if res and res['exact'] > best_val: best_val = res['exact']; best_v = vv
            if best_v == v: wins[gname] += 1

        for n_groups, gname in [(4,'Top25'),(10,'Top10')]:
            best_val = -1; best_v = ''
            for vv in VERSIONS:
                if vv in tdf.columns and tdf[vv].notna().sum() >= n_groups * 2:
                    valid = tdf[vv].notna()
                    res = analyze_version(tdf.loc[valid, 'GENOMIC'].values.astype(float),
                                          tdf.loc[valid, vv].values.astype(float), n_groups)
                    if res and res['top'] > best_val: best_val = res['top']; best_v = vv
            if best_v == v: wins[gname] += 1

    total = sum(wins.values())
    if total > 0:
        print(f"    {v:>10}: Sp={wins['Spearman']} Q4={wins['Quartil']} Q5={wins['Quintil']} D10={wins['Decil']} V20={wins['Vigintil']} T25={wins['Top25']} T10={wins['Top10']} | TOTAL={total}/70")

print()
