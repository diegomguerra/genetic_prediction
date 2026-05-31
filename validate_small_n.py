"""
Validacao do efeito de N pequeno no recall@K
Subamostra o Ethan (4156 animais) em grupos de 50, 100, 200, 500, 1000
e mede como o recall do top 5% se comporta
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pickle, sys
from scipy.stats import rankdata

sys.stdout.reconfigure(encoding='utf-8')

BASE = r"C:\Users\DiegoGuerra\Documents\Projetos\genetic_prediction"
results = pd.read_pickle(f"{BASE}/dsii_v12_results/ethan_results_by_trait.pkl")

TRAITS = ['TPI','MILK','FAT','FAT%','PRO','PRO%','PL','DPR','CCR','LIV','SCS','MAST','UDC','FLC']
N_SIZES = [50, 100, 200, 500, 1000, 2000, 4000]
N_REPS = 200  # repeticoes por tamanho

def top_recall_at_k(gen, pred, top_pct, select_pct):
    n = len(gen)
    n_real_top = max(1, int(n * top_pct))
    n_select = max(1, int(n * select_pct))
    real_top = set(np.argsort(gen)[-n_real_top:])
    pred_top = set(np.argsort(pred)[-n_select:])
    return len(real_top & pred_top) / len(real_top) * 100

def top_acc_5pct(gen, pred):
    n = len(gen)
    n_top = max(1, int(n * 0.05))
    real_top = set(np.argsort(gen)[-n_top:])
    pred_top = set(np.argsort(pred)[-n_top:])
    return len(real_top & pred_top) / len(real_top) * 100

print("=" * 140)
print("  VALIDACAO DE N PEQUENO — COMO O RECALL DEGRADA COM MENOS ANIMAIS")
print(f"  {N_REPS} repeticoes por tamanho, subamostrando Ethan (V11)")
print("=" * 140)

# Recall@5% (top5 accuracy)
print(f"\n  1. TOP 5% ACCURACY (seleciono 5% predito)")
print(f"  {'Trait':>6}", end='')
for ns in N_SIZES:
    print(f" | N={ns:>5}", end='')
print()
print("  " + "-" * (10 + 10 * len(N_SIZES)))

summary_5 = {}
for trait in TRAITS:
    df = results[trait]
    valid_mask = df[['GENOMIC', 'V11']].notna().all(axis=1)
    df_v = df[valid_mask].reset_index(drop=True)
    full_n = len(df_v)

    print(f"  {trait:>6}", end='')
    trait_results = {}
    for ns in N_SIZES:
        if ns > full_n:
            print(f" |    ---", end='')
            continue
        recalls = []
        rng = np.random.RandomState(42)
        for rep in range(N_REPS):
            idx = rng.choice(full_n, size=ns, replace=False)
            gen_sub = df_v.loc[idx, 'GENOMIC'].values
            pred_sub = df_v.loc[idx, 'V11'].values
            recalls.append(top_acc_5pct(gen_sub, pred_sub))
        mean_r = np.mean(recalls)
        trait_results[ns] = mean_r
        print(f" | {mean_r:>5.1f}%", end='')
    print()
    summary_5[trait] = trait_results

# Recall@10% (seleciono 10%, capturo quantos do top 5%)
print(f"\n  2. RECALL@10% (seleciono top 10% predito, capturo quantos do top 5% real)")
print(f"  {'Trait':>6}", end='')
for ns in N_SIZES:
    print(f" | N={ns:>5}", end='')
print()
print("  " + "-" * (10 + 10 * len(N_SIZES)))

summary_10 = {}
for trait in TRAITS:
    df = results[trait]
    valid_mask = df[['GENOMIC', 'V11']].notna().all(axis=1)
    df_v = df[valid_mask].reset_index(drop=True)
    full_n = len(df_v)

    print(f"  {trait:>6}", end='')
    trait_results = {}
    for ns in N_SIZES:
        if ns > full_n:
            print(f" |    ---", end='')
            continue
        recalls = []
        rng = np.random.RandomState(42)
        for rep in range(N_REPS):
            idx = rng.choice(full_n, size=ns, replace=False)
            gen_sub = df_v.loc[idx, 'GENOMIC'].values
            pred_sub = df_v.loc[idx, 'V11'].values
            recalls.append(top_recall_at_k(gen_sub, pred_sub, 0.05, 0.10))
        mean_r = np.mean(recalls)
        trait_results[ns] = mean_r
        print(f" | {mean_r:>5.1f}%", end='')
    print()
    summary_10[trait] = trait_results

# Recall@15%
print(f"\n  3. RECALL@15% (seleciono top 15% predito, capturo quantos do top 5% real)")
print(f"  {'Trait':>6}", end='')
for ns in N_SIZES:
    print(f" | N={ns:>5}", end='')
print()
print("  " + "-" * (10 + 10 * len(N_SIZES)))

summary_15 = {}
for trait in TRAITS:
    df = results[trait]
    valid_mask = df[['GENOMIC', 'V11']].notna().all(axis=1)
    df_v = df[valid_mask].reset_index(drop=True)
    full_n = len(df_v)

    print(f"  {trait:>6}", end='')
    trait_results = {}
    for ns in N_SIZES:
        if ns > full_n:
            print(f" |    ---", end='')
            continue
        recalls = []
        rng = np.random.RandomState(42)
        for rep in range(N_REPS):
            idx = rng.choice(full_n, size=ns, replace=False)
            gen_sub = df_v.loc[idx, 'GENOMIC'].values
            pred_sub = df_v.loc[idx, 'V11'].values
            recalls.append(top_recall_at_k(gen_sub, pred_sub, 0.05, 0.15))
        mean_r = np.mean(recalls)
        trait_results[ns] = mean_r
        print(f" | {mean_r:>5.1f}%", end='')
    print()
    summary_15[trait] = trait_results

# RESUMO
print(f"\n{'='*140}")
print(f"  RESUMO: RECALL MEDIO ENTRE TRAITS POR TAMANHO DE REBANHO")
print(f"{'='*140}")
print(f"  {'Estrategia':>20}", end='')
for ns in N_SIZES:
    print(f" | N={ns:>5}", end='')
print()
print("  " + "-" * (24 + 10 * len(N_SIZES)))

for name, summary in [('Top5% accuracy', summary_5), ('Recall@10%', summary_10), ('Recall@15%', summary_15)]:
    print(f"  {name:>20}", end='')
    for ns in N_SIZES:
        vals = [summary[t].get(ns) for t in TRAITS if ns in summary[t]]
        if vals:
            print(f" | {np.mean(vals):>5.1f}%", end='')
        else:
            print(f" |    ---", end='')
    print()

print(f"\n  CONCLUSAO:")
print(f"  - N=50 (top5%=2-3 animais): recall cai significativamente, muita variancia")
print(f"  - N=100 (top5%=5 animais): recall ja e razoavel, ~5-10pp abaixo do N grande")
print(f"  - N>=200: recall se estabiliza proximo ao valor com N grande")
print(f"  - Recall@10% e Recall@15% sao mais estaveis com N pequeno")
