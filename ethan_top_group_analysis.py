"""
ANALISE PROFUNDA - MAXIMIZAR ACERTO NO TOP GROUP (VIGINTIL 5%)
Objetivo: 75% dos animais verdadeiramente top 5% devem ser identificados corretamente

Estrategias testadas:
1. V11 puro (baseline)
2. Ensemble PA + V11 (media ponderada)
3. Ensemble PA + V10 + V11 + V12 (votacao / media)
4. Top-group focused: usar ranking medio de multiplos metodos
5. Threshold otimizado: selecionar top N% predito para capturar top 5% real
6. Combinacao otimizada com pesos por trait
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pickle, sys, time
from scipy.stats import spearmanr, rankdata
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()

BASE = r"C:\Users\DiegoGuerra\Documents\Projetos\genetic_prediction"
results = pd.read_pickle(f"{BASE}/dsii_v12_results/ethan_results_by_trait.pkl")

VERSIONS = ['PA', 'V10', 'V11', 'V12_OLD', 'V12_2F', 'V12_DAU']
TRAITS = [t for t in ['TPI','MILK','FAT','FAT%','PRO','PRO%','PL','DPR','CCR','LIV','SCS','MAST','UDC','FLC'] if t in results]

# Load V11 model info
with open(f"{BASE}/dsii_v11_results/v11_models.pkl", 'rb') as f: v11m = pickle.load(f)
with open(f"{BASE}/dsii_v10_results/v10_models.pkl", 'rb') as f: v10m = pickle.load(f)
with open(f"{BASE}/dsii_v12_results/v12_models.pkl", 'rb') as f: v12m = pickle.load(f)

def pct_group(vals, ng):
    return pd.qcut(pd.Series(vals).rank(method='first'), q=ng, labels=False).values + 1

def top_acc(gen, pred, ng):
    """% dos animais realmente top que foram preditos como top"""
    gg = pct_group(gen, ng); pg = pct_group(pred, ng)
    top_g = set(np.where(gg == ng)[0])
    top_p = set(np.where(pg == ng)[0])
    return len(top_g & top_p) / len(top_g) * 100 if top_g else 0

def top_recall_at_k(gen, pred, top_pct, select_pct):
    """Se selecionarmos os top select_pct% preditos, quantos % dos top_pct% reais capturamos?"""
    n = len(gen)
    n_real_top = max(1, int(n * top_pct))
    n_select = max(1, int(n * select_pct))
    real_top = set(np.argsort(gen)[-n_real_top:])
    pred_top = set(np.argsort(pred)[-n_select:])
    return len(real_top & pred_top) / len(real_top) * 100

def exact_pct(gen, pred, ng):
    gg = pct_group(gen, ng); pg = pct_group(pred, ng)
    return (gg == pg).mean() * 100

# ============================================================
# 1. BASELINE: V11 e cada versao individual
# ============================================================
print("=" * 140)
print("  ANALISE TOP GROUP - ETHAN DATASET")
print("  Objetivo: maximizar recall do top 5% (vigintil)")
print("=" * 140)

print(f"\n{'='*140}")
print(f"  1. BASELINE - TOP GROUP ACCURACY POR VERSAO")
print(f"{'='*140}")
print(f"  {'Trait':>6} | {'N':>5}", end='')
for v in VERSIONS:
    print(f" | {v+' T5%':>10}", end='')
print(f" | {'Best':>8}")
print("  " + "-" * 100)

baseline_top5 = {}
for trait in TRAITS:
    df = results[trait]
    valid_mask = df[VERSIONS].notna().all(axis=1)
    df_v = df[valid_mask].reset_index(drop=True)
    n = len(df_v)
    gen = df_v['GENOMIC'].values

    print(f"  {trait:>6} | {n:>5}", end='')
    best_v = ''; best_val = -1
    trait_tops = {}
    for v in VERSIONS:
        pred = df_v[v].values
        ta = top_acc(gen, pred, 20)
        trait_tops[v] = ta
        if ta > best_val: best_val = ta; best_v = v
        print(f" | {ta:>9.1f}%", end='')
    print(f" | {best_v:>8}")
    baseline_top5[trait] = trait_tops

# ============================================================
# 2. ENSEMBLE STRATEGIES - MEDIA PONDERADA DE RANKINGS
# ============================================================
print(f"\n{'='*140}")
print(f"  2. ENSEMBLE STRATEGIES - COMBINACOES DE VERSOES")
print(f"     Tecnica: media dos rankings normalizados (0-1)")
print(f"{'='*140}")

def ensemble_rank_mean(df_v, versions_to_combine):
    """Combine predictions by averaging their normalized ranks"""
    ranks = []
    for v in versions_to_combine:
        r = rankdata(df_v[v].values) / len(df_v)
        ranks.append(r)
    return np.mean(ranks, axis=0)

def ensemble_weighted_rank(df_v, versions_weights):
    """Combine predictions by weighted average of normalized ranks"""
    ranks = []
    weights = []
    for v, w in versions_weights:
        r = rankdata(df_v[v].values) / len(df_v)
        ranks.append(r * w)
        weights.append(w)
    return np.sum(ranks, axis=0) / sum(weights)

# Strategies to test
strategies = {
    'PA+V11': ['PA', 'V11'],
    'PA+V10+V11': ['PA', 'V10', 'V11'],
    'ALL6': VERSIONS,
    'V10+V11': ['V10', 'V11'],
    'V10+V11+V12OLD': ['V10', 'V11', 'V12_OLD'],
    'PA+V11+V12OLD': ['PA', 'V11', 'V12_OLD'],
}

# Weighted strategies
weighted_strategies = {
    'PA1+V11_2': [('PA',1),('V11',2)],
    'PA1+V11_3': [('PA',1),('V11',3)],
    'PA2+V11_1': [('PA',2),('V11',1)],
    'PA1+V10_1+V11_2': [('PA',1),('V10',1),('V11',2)],
    'PA1+V10_1+V11_3': [('PA',1),('V10',1),('V11',3)],
    'V10_1+V11_2': [('V10',1),('V11',2)],
    'V10_1+V11_3': [('V10',1),('V11',3)],
}

all_strats = list(strategies.keys()) + list(weighted_strategies.keys())

print(f"\n  {'Trait':>6}", end='')
for s in all_strats:
    print(f" | {s:>16}", end='')
print(f" | {'BEST':>16} | {'vs V11':>7}")
print("  " + "-" * (20 + 19*len(all_strats)))

strat_wins = {s:0 for s in all_strats + ['V11_solo']}
best_per_trait = {}

for trait in TRAITS:
    df = results[trait]
    valid_mask = df[VERSIONS].notna().all(axis=1)
    df_v = df[valid_mask].reset_index(drop=True)
    gen = df_v['GENOMIC'].values
    n = len(df_v)

    v11_base = top_acc(gen, df_v['V11'].values, 20)

    print(f"  {trait:>6}", end='')
    best_s = 'V11_solo'; best_val = v11_base

    for sname in all_strats:
        if sname in strategies:
            ens = ensemble_rank_mean(df_v, strategies[sname])
        else:
            ens = ensemble_weighted_rank(df_v, weighted_strategies[sname])
        ta = top_acc(gen, ens, 20)
        if ta > best_val:
            best_val = ta; best_s = sname
        print(f" | {ta:>15.1f}%", end='')

    delta = best_val - v11_base
    print(f" | {best_s:>16} | {delta:>+6.1f}pp")
    strat_wins[best_s] += 1
    best_per_trait[trait] = (best_s, best_val, v11_base)

print(f"\n  Vitorias por estrategia:")
for s in sorted(strat_wins, key=strat_wins.get, reverse=True):
    if strat_wins[s] > 0:
        print(f"    {s:>20}: {strat_wins[s]}")

# ============================================================
# 3. RECALL@K - SE SELECIONARMOS TOP 10% PREDITO, QUANTOS TOP 5% REAIS CAPTURAMOS?
# ============================================================
print(f"\n{'='*140}")
print(f"  3. RECALL@K - EXPANDINDO A SELECAO PARA CAPTURAR MAIS TOP 5%")
print(f"     'Se seleciono os top K% preditos, quantos dos top 5% reais capturo?'")
print(f"{'='*140}")

select_pcts = [0.05, 0.10, 0.15, 0.20, 0.25]

for trait in TRAITS:
    df = results[trait]
    valid_mask = df[VERSIONS].notna().all(axis=1)
    df_v = df[valid_mask].reset_index(drop=True)
    gen = df_v['GENOMIC'].values
    n = len(df_v)

    best_s, _, _ = best_per_trait[trait]
    if best_s == 'V11_solo':
        pred_best = df_v['V11'].values
    elif best_s in strategies:
        pred_best = ensemble_rank_mean(df_v, strategies[best_s])
    else:
        pred_best = ensemble_weighted_rank(df_v, weighted_strategies[best_s])

    # Also test V11 for comparison
    pred_v11 = df_v['V11'].values

    print(f"\n  {trait} (N={n}, best={best_s}):")
    print(f"    {'Seleciono':>10} | {'V11 Recall':>12} | {'Best Recall':>12} | {'# reais em top5%':>16} | {'# selecionados':>14}")
    for sp in select_pcts:
        r_v11 = top_recall_at_k(gen, pred_v11, 0.05, sp)
        r_best = top_recall_at_k(gen, pred_best, 0.05, sp)
        n_real = int(n * 0.05)
        n_sel = int(n * sp)
        print(f"    {'Top '+str(int(sp*100))+'%':>10} | {r_v11:>11.1f}% | {r_best:>11.1f}% | {n_real:>16} | {n_sel:>14}")

# ============================================================
# 4. OTIMIZACAO DE PESOS - GRID SEARCH POR TRAIT
# ============================================================
print(f"\n{'='*140}")
print(f"  4. OTIMIZACAO DE PESOS PA/V10/V11/V12 POR TRAIT (Grid Search)")
print(f"     Maximizando TOP 5% accuracy (vigintil)")
print(f"{'='*140}")

def grid_search_weights(df_v, gen, versions_to_try, ng=20, steps=11):
    """Grid search weights for rank ensemble, maximizing top group accuracy"""
    n_v = len(versions_to_try)
    # Precompute ranks
    ranks = {v: rankdata(df_v[v].values) / len(df_v) for v in versions_to_try}

    best_w = None; best_ta = -1
    # For 2 versions: simple grid
    if n_v == 2:
        for w0 in np.linspace(0, 1, steps):
            w1 = 1 - w0
            ens = ranks[versions_to_try[0]] * w0 + ranks[versions_to_try[1]] * w1
            ta = top_acc(gen, ens, ng)
            if ta > best_ta: best_ta = ta; best_w = {versions_to_try[0]: w0, versions_to_try[1]: w1}
    elif n_v == 3:
        for w0 in np.linspace(0, 1, steps):
            for w1 in np.linspace(0, 1-w0, steps):
                w2 = 1 - w0 - w1
                if w2 < 0: continue
                ens = ranks[versions_to_try[0]]*w0 + ranks[versions_to_try[1]]*w1 + ranks[versions_to_try[2]]*w2
                ta = top_acc(gen, ens, ng)
                if ta > best_ta: best_ta = ta; best_w = dict(zip(versions_to_try, [w0,w1,w2]))
    elif n_v == 4:
        # Coarser grid for 4 vars
        for w0 in np.linspace(0, 1, 6):
            for w1 in np.linspace(0, 1-w0, 6):
                for w2 in np.linspace(0, 1-w0-w1, 6):
                    w3 = 1 - w0 - w1 - w2
                    if w3 < -0.01: continue
                    w3 = max(0, w3)
                    ens = ranks[versions_to_try[0]]*w0 + ranks[versions_to_try[1]]*w1 + \
                          ranks[versions_to_try[2]]*w2 + ranks[versions_to_try[3]]*w3
                    ta = top_acc(gen, ens, ng)
                    if ta > best_ta: best_ta = ta; best_w = dict(zip(versions_to_try, [w0,w1,w2,w3]))
    return best_w, best_ta

print(f"\n  {'Trait':>6} | {'V11 solo':>10} | {'PA+V11 opt':>12} | {'PA+V10+V11':>12} | {'PA+V10+V11+V12':>15} | {'BEST':>10} | {'Pesos':>30}")
print("  " + "-" * 120)

optimal_configs = {}

for trait in TRAITS:
    df = results[trait]
    valid_mask = df[VERSIONS].notna().all(axis=1)
    df_v = df[valid_mask].reset_index(drop=True)
    gen = df_v['GENOMIC'].values

    v11_ta = top_acc(gen, df_v['V11'].values, 20)

    # Grid search for different combos
    combos = [
        ['PA', 'V11'],
        ['PA', 'V10', 'V11'],
        ['PA', 'V10', 'V11', 'V12_OLD'],
    ]

    best_overall = v11_ta; best_cfg = 'V11 solo'; best_weights = {'V11': 1.0}
    combo_results = [v11_ta]

    for combo in combos:
        w, ta = grid_search_weights(df_v, gen, combo, ng=20, steps=11)
        combo_results.append(ta)
        if ta > best_overall:
            best_overall = ta; best_cfg = '+'.join(combo); best_weights = w

    w_str = ' '.join(f"{k}={v:.2f}" for k,v in best_weights.items() if v > 0.01)
    print(f"  {trait:>6} | {combo_results[0]:>9.1f}% | {combo_results[1]:>11.1f}% | {combo_results[2]:>11.1f}% | {combo_results[3]:>14.1f}% | {best_overall:>9.1f}% | {w_str}")
    optimal_configs[trait] = {'config': best_cfg, 'weights': best_weights, 'top5': best_overall, 'v11_base': v11_ta}

# ============================================================
# 5. MESMA ANALISE PARA TODOS OS GRUPOS (top 25%, 10%, 5%)
# ============================================================
print(f"\n{'='*140}")
print(f"  5. MELHOR CONFIGURACAO OTIMIZADA POR TRAIT - TODOS OS GRANULARIDADES")
print(f"     Usando pesos otimizados do grid search")
print(f"{'='*140}")

print(f"\n  {'Trait':>6} | {'Config':>20} | {'Q25% Top':>10} | {'D10% Top':>10} | {'V5% Top':>10} | {'Q25% Ex':>10} | {'D10% Ex':>10} | {'V5% Ex':>10} | {'Spearman':>10}")
print("  " + "-" * 130)

for trait in TRAITS:
    df = results[trait]
    valid_mask = df[VERSIONS].notna().all(axis=1)
    df_v = df[valid_mask].reset_index(drop=True)
    gen = df_v['GENOMIC'].values
    n = len(df_v)

    cfg = optimal_configs[trait]
    weights = cfg['weights']

    # Build ensemble
    ranks = {v: rankdata(df_v[v].values) / n for v in weights}
    ens = np.sum([ranks[v] * w for v, w in weights.items()], axis=0)

    # Also compute for V11 solo
    t25_ens = top_acc(gen, ens, 4); t10_ens = top_acc(gen, ens, 10); t5_ens = top_acc(gen, ens, 20)
    e25_ens = exact_pct(gen, ens, 4); e10_ens = exact_pct(gen, ens, 10); e5_ens = exact_pct(gen, ens, 20)
    rho_ens, _ = spearmanr(ens, gen)

    print(f"  {trait:>6} | {cfg['config']:>20} | {t25_ens:>9.1f}% | {t10_ens:>9.1f}% | {t5_ens:>9.1f}% | {e25_ens:>9.1f}% | {e10_ens:>9.1f}% | {e5_ens:>9.1f}% | {rho_ens:>10.4f}")

# ============================================================
# 6. ANALISE CROSS-VALIDATED DOS PESOS (para evitar overfitting)
# ============================================================
print(f"\n{'='*140}")
print(f"  6. VALIDACAO CRUZADA (5-fold) - PESOS OTIMIZADOS SAO ROBUSTOS?")
print(f"     Treina pesos em 4 folds, testa no 5o")
print(f"{'='*140}")

from sklearn.model_selection import KFold

def cv_top_acc(df_v, gen, versions_to_try, ng=20, n_folds=5):
    """Cross-validated top group accuracy with weight optimization"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for train_idx, test_idx in kf.split(gen):
        gen_train = gen[train_idx]; gen_test = gen[test_idx]
        df_train = df_v.iloc[train_idx]; df_test = df_v.iloc[test_idx]

        # Optimize weights on train
        ranks_train = {v: rankdata(df_train[v].values) / len(df_train) for v in versions_to_try}

        if len(versions_to_try) == 1:
            best_w = {versions_to_try[0]: 1.0}
        else:
            best_w, _ = grid_search_weights(df_train, gen_train, versions_to_try, ng=ng, steps=11)

        # Apply on test
        ranks_test = {v: rankdata(df_test[v].values) / len(df_test) for v in versions_to_try}
        ens_test = np.sum([ranks_test[v] * w for v, w in best_w.items()], axis=0)
        ta = top_acc(gen_test, ens_test, ng)
        fold_results.append(ta)

    return np.mean(fold_results), np.std(fold_results)

print(f"\n  {'Trait':>6} | {'V11 CV':>12} | {'PA+V11 CV':>14} | {'PA+V10+V11 CV':>16} | {'4-way CV':>14} | {'Best CV':>12} | {'Overfit?':>10}")
print("  " + "-" * 110)

for trait in TRAITS:
    df = results[trait]
    valid_mask = df[VERSIONS].notna().all(axis=1)
    df_v = df[valid_mask].reset_index(drop=True)
    gen = df_v['GENOMIC'].values

    v11_cv, v11_std = cv_top_acc(df_v, gen, ['V11'], ng=20)
    pv11_cv, pv11_std = cv_top_acc(df_v, gen, ['PA','V11'], ng=20)
    pv10v11_cv, pv10v11_std = cv_top_acc(df_v, gen, ['PA','V10','V11'], ng=20)
    four_cv, four_std = cv_top_acc(df_v, gen, ['PA','V10','V11','V12_OLD'], ng=20)

    cvs = {'V11': v11_cv, 'PA+V11': pv11_cv, 'PA+V10+V11': pv10v11_cv, '4-way': four_cv}
    best_cv_name = max(cvs, key=cvs.get)
    best_cv_val = cvs[best_cv_name]

    # Compare with non-CV optimized
    non_cv = optimal_configs[trait]['top5']
    overfit = non_cv - best_cv_val
    overfit_str = f"{overfit:+.1f}pp" if overfit > 2 else "OK"

    print(f"  {trait:>6} | {v11_cv:>5.1f}+-{v11_std:>4.1f}% | {pv11_cv:>6.1f}+-{pv11_std:>4.1f}% | {pv10v11_cv:>8.1f}+-{pv10v11_std:>4.1f}% | {four_cv:>5.1f}+-{four_std:>4.1f}% | {best_cv_name:>12} | {overfit_str:>10}")

# ============================================================
# 7. V11 ARCHITECTURE DETAIL
# ============================================================
print(f"\n{'='*140}")
print(f"  7. V11 ARCHITECTURE - MODELO ML POR TRAIT")
print(f"{'='*140}")
print(f"  {'Trait':>6} | {'Type':>8} | {'Model':>15} | {'R2 dev':>8} | {'PA R2':>8} | {'Target':>12} | {'N features':>10}")
print("  " + "-" * 85)

for trait in TRAITS:
    info = v11m.get(trait, {})
    mtype = info.get('type', '?')
    mname = info.get('model_name', '?')
    r2 = info.get('r2_dev', info.get('r2_cv', -1))
    pa_r2 = info.get('pa_r2', -1)
    target = info.get('target', '?')
    n_feat = len(info.get('feature_cols', []))

    # Check for hybrid/v10b
    if info.get('v10b'): mname += f"({info['v10b']})"

    print(f"  {trait:>6} | {mtype:>8} | {mname:>15} | {r2:>8.3f} | {pa_r2:>8.3f} | {target:>12} | {n_feat:>10}")

# V10 models for comparison
print(f"\n  V10 MODELS:")
print(f"  {'Trait':>6} | {'Type':>8} | {'Model':>15} | {'R2 CV':>8} | {'N features':>10}")
print("  " + "-" * 60)
for trait in TRAITS:
    info = v10m.get(trait, {})
    mtype = info.get('type', '?')
    mname = info.get('model_name', '?')
    r2 = info.get('r2_cv', -1)
    n_feat = len(info.get('feature_cols', []))
    print(f"  {trait:>6} | {mtype:>8} | {mname:>15} | {r2:>8.3f} | {n_feat:>10}")

# V12 models
print(f"\n  V12 MODELS:")
print(f"  {'Trait':>6} | {'Type':>8} | {'Bases':>25} | {'R2 dev':>8} | {'PA R2':>8} | {'N features':>10}")
print("  " + "-" * 80)
for trait in TRAITS:
    info = v12m.get(trait, {})
    mtype = info.get('type', '?')
    bases = '+'.join(info.get('base_names', ['?']))
    r2 = info.get('r2_dev', info.get('r2_cv', -1))
    pa_r2 = info.get('pa_r2', -1)
    n_feat = len(info.get('feature_cols', []))
    print(f"  {trait:>6} | {mtype:>8} | {bases:>25} | {r2:>8.3f} | {pa_r2:>8.3f} | {n_feat:>10}")

# ============================================================
# 8. RESUMO FINAL EXECUTIVO
# ============================================================
print(f"\n{'='*140}")
print(f"  8. RESUMO FINAL - LIMITACOES POR TRAIT")
print(f"     Para cada trait: ate onde vamos bem na granularidade")
print(f"{'='*140}")

GRAN_THRESHOLDS = {
    'Quartil 25%': (4, 50),    # >50% top acc = BOM
    'Quintil 20%': (5, 45),    # >45% = BOM
    'Decil 10%': (10, 35),     # >35% = BOM
    'Vigintil 5%': (20, 25),   # >25% = BOM
}

print(f"\n  {'Trait':>6} | {'Best Method':>18} |", end='')
for gname in GRAN_THRESHOLDS:
    print(f" {gname:>14}", end='')
print(f" | {'Limite':>12} | {'Nota':>30}")
print("  " + "-" * 140)

for trait in TRAITS:
    df = results[trait]
    valid_mask = df[VERSIONS].notna().all(axis=1)
    df_v = df[valid_mask].reset_index(drop=True)
    gen = df_v['GENOMIC'].values
    n = len(df_v)

    # Use V11 (proven best)
    pred = df_v['V11'].values

    vals = []
    limite = 'Vigintil 5%'
    for gname, (ng, threshold) in GRAN_THRESHOLDS.items():
        ta = top_acc(gen, pred, ng)
        vals.append(ta)
        if ta < threshold:
            limite = list(GRAN_THRESHOLDS.keys())[list(GRAN_THRESHOLDS.keys()).index(gname) - 1] if list(GRAN_THRESHOLDS.keys()).index(gname) > 0 else 'Nenhum'

    # Note
    h2 = {'TPI':0.30,'MILK':0.25,'FAT':0.25,'FAT%':0.50,'PRO':0.25,'PRO%':0.50,
           'PL':0.08,'LIV':0.05,'SCS':0.12,'MAST':0.04,'DPR':0.04,'HCR':0.04,
           'CCR':0.04,'UDC':0.25,'FLC':0.15}.get(trait, 0.10)

    nota = f"h2={h2:.2f}"
    if h2 >= 0.25: nota += " ALTA-h2"
    elif h2 >= 0.10: nota += " MED-h2"
    else: nota += " BAIXA-h2"

    rho, _ = spearmanr(pred, gen)
    nota += f" rho={rho:.3f}"

    print(f"  {trait:>6} | {'V11':>18} |", end='')
    for v in vals:
        marker = '*' if v >= 50 else '+' if v >= 35 else '-' if v >= 25 else '!'
        print(f" {v:>12.1f}%{marker}", end='')
    print(f" | {limite:>12} | {nota}")

print(f"\n  Legenda: * >= 50% (excelente) | + >= 35% (bom) | - >= 25% (aceitavel) | ! < 25% (insuficiente)")
print(f"  Top group accuracy = % dos animais realmente no grupo top que foram preditos no grupo top")
print(f"\n  Tempo total: {time.time()-t0:.0f}s")
