"""
Analise de Acuracia de Localizacao (Quartil/Decil) - V12 vs Genomica Real
"""
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(r"C:\Users\DiegoGuerra\Documents\Projetos\genetic_prediction\dsii_v12_results")

# Predicoes OLD (melhores correlacoes)
pred_old = pd.read_excel(RESULTS_DIR / "jose_predictions_v12_BEFORE.xlsx")

# Resultados genomicos reais
genomic = pd.read_csv(r"C:\Users\DiegoGuerra\Documents\Projetos\genetic_prediction\jose_genomic_results.csv")

TRAITS = ['TPI', 'NM$', 'MILK', 'FAT', 'PRO', 'FAT%', 'PRO%', 'PL', 'DPR', 'HCR']

# Merge por ID - colunas iguais vao ganhar sufixos
merged = pred_old[['ID'] + [t for t in TRAITS if t in pred_old.columns]].merge(
    genomic[['ID'] + [t for t in TRAITS if t in genomic.columns]],
    on='ID', how='inner', suffixes=('_v12', '_gen')
)
print(f"Animais com match: {len(merged)}")
print(f"Colunas merged: {list(merged.columns)}")
print()

def quartile_rank(series):
    return pd.qcut(series.rank(method='first'), q=4, labels=False) + 1

def decile_rank(series):
    return pd.qcut(series.rank(method='first'), q=10, labels=False) + 1

print("=" * 90)
print("  ANALISE DE ACURACIA DE LOCALIZACAO - V12 (OLD) vs GENOMICA REAL (Jose, 93 animais)")
print("=" * 90)

results_quartile = []
results_decile = []
results_extreme = []

for trait in TRAITS:
    v12_col = f"{trait}_v12"
    gen_col = f"{trait}_gen"

    if v12_col not in merged.columns or gen_col not in merged.columns:
        print(f"  {trait}: colunas nao encontradas, pulando...")
        continue

    v12_vals = merged[v12_col].dropna().values.astype(float)
    gen_vals = merged[gen_col].dropna().values.astype(float)

    # Alinhar (remover NaN de ambos)
    valid = merged[v12_col].notna() & merged[gen_col].notna()
    v12_vals = merged.loc[valid, v12_col].values.astype(float)
    gen_vals = merged.loc[valid, gen_col].values.astype(float)
    n = len(gen_vals)

    if n < 20:
        continue

    # --- QUARTIS ---
    gen_q = quartile_rank(pd.Series(gen_vals)).values
    v12_q = quartile_rank(pd.Series(v12_vals)).values

    v12_q_exact = (gen_q == v12_q).mean() * 100
    v12_q_adj = (np.abs(gen_q - v12_q) <= 1).mean() * 100

    results_quartile.append({
        'Trait': trait, 'N': n,
        'V12_Exato%': v12_q_exact, 'V12_Adj%': v12_q_adj,
    })

    # --- DECIS ---
    gen_d = decile_rank(pd.Series(gen_vals)).values
    v12_d = decile_rank(pd.Series(v12_vals)).values

    v12_d_exact = (gen_d == v12_d).mean() * 100
    v12_d_1 = (np.abs(gen_d - v12_d) <= 1).mean() * 100
    v12_d_2 = (np.abs(gen_d - v12_d) <= 2).mean() * 100

    results_decile.append({
        'Trait': trait,
        'V12_Exato%': v12_d_exact, 'V12_+-1%': v12_d_1, 'V12_+-2%': v12_d_2,
    })

    # --- EXTREMOS ---
    gen_top25 = set(np.where(gen_q == 4)[0])
    v12_top25 = set(np.where(v12_q == 4)[0])
    gen_bot25 = set(np.where(gen_q == 1)[0])
    v12_bot25 = set(np.where(v12_q == 1)[0])
    gen_top10 = set(np.where(gen_d == 10)[0])
    v12_top10 = set(np.where(v12_d == 10)[0])

    v12_top25_acc = len(gen_top25 & v12_top25) / len(gen_top25) * 100 if gen_top25 else 0
    v12_bot25_acc = len(gen_bot25 & v12_bot25) / len(gen_bot25) * 100 if gen_bot25 else 0
    v12_top10_acc = len(gen_top10 & v12_top10) / len(gen_top10) * 100 if gen_top10 else 0

    # Inversoes de sinal
    v12_sign_inv = ((v12_vals > 0) & (gen_vals < 0) | (v12_vals < 0) & (gen_vals > 0)).sum()

    # Correlacao de Spearman (rank)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(v12_vals, gen_vals)

    results_extreme.append({
        'Trait': trait,
        'V12_Top25%': v12_top25_acc,
        'V12_Bot25%': v12_bot25_acc,
        'V12_Top10%': v12_top10_acc,
        'V12_SignInv': v12_sign_inv,
        'Spearman': rho,
    })

# === IMPRIMIR ===
print("\n" + "=" * 90)
print("  1. ACURACIA DE QUARTIL")
print("     (Aleatorio = 25% exato, 50% adjacente)")
print("=" * 90)
print(f"  {'Trait':>6} | {'N':>3} | {'Exato%':>7} | {'vs Random':>9} | {'Adj +-1':>8} | {'vs Random':>9}")
print("  " + "-" * 60)
for r in results_quartile:
    print(f"  {r['Trait']:>6} | {r['N']:>3} | {r['V12_Exato%']:>6.1f}% | {r['V12_Exato%']-25:>+8.1f}pp | {r['V12_Adj%']:>7.1f}% | {r['V12_Adj%']-50:>+8.1f}pp")
avg_ex = np.mean([r['V12_Exato%'] for r in results_quartile])
avg_adj = np.mean([r['V12_Adj%'] for r in results_quartile])
print("  " + "-" * 60)
print(f"  {'MEDIA':>6} |     | {avg_ex:>6.1f}% | {avg_ex-25:>+8.1f}pp | {avg_adj:>7.1f}% | {avg_adj-50:>+8.1f}pp")

print("\n" + "=" * 90)
print("  2. ACURACIA DE DECIL")
print("     (Aleatorio = 10% exato, 30% +-1, 50% +-2)")
print("=" * 90)
print(f"  {'Trait':>6} | {'Exato%':>7} | {'vs Rand':>7} | {'+-1 Decil':>9} | {'vs Rand':>7} | {'+-2 Decis':>9} | {'vs Rand':>7}")
print("  " + "-" * 75)
for r in results_decile:
    print(f"  {r['Trait']:>6} | {r['V12_Exato%']:>6.1f}% | {r['V12_Exato%']-10:>+6.1f}pp | {r['V12_+-1%']:>8.1f}% | {r['V12_+-1%']-30:>+6.1f}pp | {r['V12_+-2%']:>8.1f}% | {r['V12_+-2%']-50:>+6.1f}pp")
avg_dex = np.mean([r['V12_Exato%'] for r in results_decile])
avg_d1 = np.mean([r['V12_+-1%'] for r in results_decile])
avg_d2 = np.mean([r['V12_+-2%'] for r in results_decile])
print("  " + "-" * 75)
print(f"  {'MEDIA':>6} | {avg_dex:>6.1f}% | {avg_dex-10:>+6.1f}pp | {avg_d1:>8.1f}% | {avg_d1-30:>+6.1f}pp | {avg_d2:>8.1f}% | {avg_d2-50:>+6.1f}pp")

print("\n" + "=" * 90)
print("  3. EXTREMOS + RANKING")
print("=" * 90)
print(f"  {'Trait':>6} | {'Top25%':>7} | {'Bot25%':>7} | {'Top10%':>7} | {'Inv.Sinal':>9} | {'Spearman':>8}")
print("  " + "-" * 60)
for r in results_extreme:
    print(f"  {r['Trait']:>6} | {r['V12_Top25%']:>6.1f}% | {r['V12_Bot25%']:>6.1f}% | {r['V12_Top10%']:>6.1f}% | {r['V12_SignInv']:>9d} | {r['Spearman']:>8.3f}")
avg_t25 = np.mean([r['V12_Top25%'] for r in results_extreme])
avg_b25 = np.mean([r['V12_Bot25%'] for r in results_extreme])
avg_t10 = np.mean([r['V12_Top10%'] for r in results_extreme])
avg_spear = np.mean([r['Spearman'] for r in results_extreme])
total_inv = sum(r['V12_SignInv'] for r in results_extreme)
print("  " + "-" * 60)
print(f"  {'MEDIA':>6} | {avg_t25:>6.1f}% | {avg_b25:>6.1f}% | {avg_t10:>6.1f}% | {total_inv:>9d} | {avg_spear:>8.3f}")

# === RESUMO ===
print("\n" + "=" * 90)
print("  RESUMO GERAL")
print("=" * 90)
print(f"  Quartil exato:    {avg_ex:.1f}% (aleatorio=25%, ganho={avg_ex-25:+.1f}pp)")
print(f"  Quartil adj +-1:  {avg_adj:.1f}% (aleatorio=50%, ganho={avg_adj-50:+.1f}pp)")
print(f"  Decil exato:      {avg_dex:.1f}% (aleatorio=10%, ganho={avg_dex-10:+.1f}pp)")
print(f"  Decil +-1:        {avg_d1:.1f}% (aleatorio=30%, ganho={avg_d1-30:+.1f}pp)")
print(f"  Decil +-2:        {avg_d2:.1f}% (aleatorio=50%, ganho={avg_d2-50:+.1f}pp)")
print(f"  Top 25% acerto:   {avg_t25:.1f}% (aleatorio=25%)")
print(f"  Bottom 25% acerto:{avg_b25:.1f}% (aleatorio=25%)")
print(f"  Top 10% acerto:   {avg_t10:.1f}% (aleatorio=10%)")
print(f"  Spearman medio:   {avg_spear:.3f}")
print(f"  Inversoes sinal:  {total_inv} de {len(merged)*len(TRAITS)} valores")
print()

# === DETALHAMENTO: quais animais erram mais? ===
print("=" * 90)
print("  4. ANIMAIS COM MAIOR ERRO DE LOCALIZACAO (Quartil)")
print("=" * 90)
err_by_animal = {}
for trait in TRAITS:
    v12_col = f"{trait}_v12"
    gen_col = f"{trait}_gen"
    if v12_col not in merged.columns or gen_col not in merged.columns:
        continue
    valid = merged[v12_col].notna() & merged[gen_col].notna()
    v12_vals = merged.loc[valid, v12_col].values.astype(float)
    gen_vals = merged.loc[valid, gen_col].values.astype(float)
    gen_q = quartile_rank(pd.Series(gen_vals)).values
    v12_q = quartile_rank(pd.Series(v12_vals)).values
    ids = merged.loc[valid, 'ID'].values
    for i in range(len(ids)):
        aid = ids[i]
        if aid not in err_by_animal:
            err_by_animal[aid] = {'total_err': 0, 'n_traits': 0, 'big_miss': []}
        err = abs(int(gen_q[i]) - int(v12_q[i]))
        err_by_animal[aid]['total_err'] += err
        err_by_animal[aid]['n_traits'] += 1
        if err >= 2:
            err_by_animal[aid]['big_miss'].append(f"{trait}(Q{v12_q[i]}->Q{gen_q[i]})")

# Top 10 piores
worst = sorted(err_by_animal.items(), key=lambda x: -x[1]['total_err'])[:15]
print(f"  {'ID':>6} | {'Erro_Total':>10} | {'N_Traits':>8} | {'Err_Med':>7} | Erros grandes (>=2Q)")
print("  " + "-" * 80)
for aid, info in worst:
    avg_e = info['total_err'] / info['n_traits']
    misses = ', '.join(info['big_miss'][:5])
    print(f"  {aid:>6} | {info['total_err']:>10} | {info['n_traits']:>8} | {avg_e:>7.2f} | {misses}")
print()
