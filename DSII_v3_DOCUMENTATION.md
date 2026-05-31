# DSII v3 — Dam-Specific Improvement Index

## Motor de Predicao Genomica Avancada para a Raca Holandesa

**Autor:** VYR Labs / Diego Guerra
**Data de criacao:** 23/04/2026
**Versao:** 3.0
**Arquivos-fonte:**
- `predicao_genomica_holstein.py` — Motor base (GenomicPredictionEngine)
- `rodar_predicao_v3.py` — Motor v3 completo (DSII + Multi-Trait)

---

## 1. Visao Geral

O DSII v3 e um motor de predicao genomica que vai alem do Parent Average (PA) simples `(PTA_touro + PTA_femea) / 2`. A premissa central e que **cada femea tem um perfil unico de deficiencias**, e o melhor touro para uma femea nao e necessariamente o melhor para outra.

### Resultado comprovado
- **96% das combinacoes** mudaram 5+ posicoes no ranking vs PA simples
- **41% das femeas** tiveram touro #1 diferente do que o PA simples indicaria

---

## 2. Fundamento Teorico

| Conceito | Referencia |
|---|---|
| ssGBLUP (Single-Step Genomic BLUP) | Aguilar & Misztal (2010) |
| Metricas de confiabilidade genomica | VanRaden (2008) |
| Pesos economicos NM$ 2025 | CDCB/USDA-AGIL ARR-NM8 |
| Correcao de depressao endogamica | Pryce et al. (2014), VanRaden (2005) |
| Variancia de Mendelian Sampling | Bijma (2012) |
| Mate Allocation Index | Kinghorn (1998, 2011) |
| Razoes de dominancia | Sun et al. (2014) J. Dairy Sci. 97:3852 |
| Correlacoes geneticas | VanRaden et al. (2004, 2014, 2018), CDCB |

---

## 3. Arquitetura do Motor

### 3.1 Pipeline de Calculo (por combinacao touro x femea)

```
Para cada combinacao (touro, femea):
  1. Parent Average (PA) por trait
  2. Estimativa de endogamia esperada (F) da cria
  3. Depressao endogamica por trait e em NM$
  4. NM$ corrigido = PA_NM$ + depressao
  5. TPI corrigido com IC 95%
  6. Predicao de todos os traits (PA + correcao endogamica)
  7. DeltaG por trait = PA_predito - PTA_mae (melhoria sobre a mae)
  8. Perfil de deficiencias da femea
  9. DSII = indice de melhoria especifico por femea
 10. Variancia de Mendelian Sampling e IC 95% para NM$
 11. Probabilidades: P(acima da media), P(top 25%), P(top 10%)
 12. Penalidade de limites criticos
 13. Penalidade de antagonismo (producao x funcionalidade)
 14. Bonus de dominancia (heterose intra-raca)
 15. Distancia ao perfil ideal
 16. Verificacao de haplotipos letais e caseinas
 17. Score final composto (normalizacao global)
```

### 3.2 Normalizacao Global (pos-calculo)

Apos calcular todas as combinacoes, os scores sao normalizados globalmente:
- DSII normalizado para escala 0-100
- NM$ corrigido normalizado para escala 0-100
- Distancia ao ideal normalizada
- Antagonismo normalizado
- Composicao ponderada final

---

## 4. Componentes Detalhados

### 4.1 Perfil de Deficiencias da Femea

Funcao: `compute_dam_deficiency_profile(dam)`

Para cada trait, calcula quanto a femea esta abaixo do ideal da raca:

```
Para traits "maior = melhor":
  deficit = (ideal - valor_femea) / sigma_a

Para traits "menor = melhor":
  deficit = (valor_femea - ideal) / sigma_a

deficiency = max(0, deficit)
```

**Alvos ideais (BREED_IDEAL) — Holstein elite:**

| Trait | Ideal | Trait | Ideal |
|---|---|---|---|
| MILK | 1800 lbs | DPR | +1.5% |
| FAT | 90 lbs | CCR | +2.0% |
| PROT | 60 lbs | HCR | +2.0% |
| CFP | 70 lbs | PL | +5.0 meses |
| UDC | +1.50 | LIV | +2.0% |
| FLC | +1.00 | RFI | -50 lbs/dia |

### 4.2 DSII — Dam-Specific Improvement Index

Funcao: `compute_dsii(sire, dam, deficiency_profile)`

O nucleo do motor. Para cada trait:

```python
DeltaG = (PTA_touro - PTA_femea) / 2.0    # Melhoria esperada sobre a mae

# Amplificacao por deficiencia:
# Se a femea e deficiente, o peso economico sobe ate 3x
amplification = 1.0 + min(2.0, deficiency * 0.8)

# Contribuicao ao DSII:
Para traits "maior = melhor":
  contribution = DeltaG * peso_econ * amplification

Para traits "menor = melhor":
  contribution = -DeltaG * |peso_econ| * amplification

DSII = soma(contributions)
```

**Efeito pratico:** O mesmo touro gera DSII diferentes para femeas diferentes. Um touro com UDC alto beneficia mais uma femea com ubere fraco do que uma com ubere bom. Isso faz o ranking de touros **mudar conforme a femea**.

### 4.3 Pesos Economicos (NM$ 2024/2025)

Funcao: `compute_nm_dollar(trait_values)`

Baseados no CDCB/USDA-AGIL ARR-NM8:

| Trait | Peso ($/unidade) | Enfase NM$ |
|---|---|---|
| FAT | +$2.60/lb | ~27% |
| PROT | +$1.40/lb | ~17% |
| MILK | -$0.02/lb | ~-1% |
| PL | +$35.00/mes | ~13% |
| SCS | -$120.00/unit | ~7% |
| DPR | +$28.00/% | ~7% |
| LIV | +$25.00/% | ~7% |
| UDC | +$30.00/pt | ~4% |
| CCR | +$11.00/% | ~2% |
| BWC | -$6.00/pt | ~-5% |
| SCE | -$3.50/% | ~2% |
| RFI | -$1.00/lb | ~3% |
| MAST | -$25.00/unit | Saude |
| MF | -$18.00/unit | Saude |
| DA | -$18.00/unit | Saude |
| KET | -$18.00/unit | Saude |
| MET | -$12.00/unit | Saude |
| RP | -$12.00/unit | Saude |
| FLC | +$5.00/pt | ~3% |
| HCR | +$6.00/% | ~1% |
| HLIV | +$6.00/% | - |
| EFC | +$2.00/unit | - |

### 4.4 Endogamia e Depressao Endogamica

#### 4.4.1 Estimativa de F da cria

Funcao: `estimate_inbreeding(sire, dam, all_sire_names)`

```
F_base = 3.125% (background Holstein)

Verificacoes de pedigree:
  Mesmo pai         -> F = 12.5%
  Pai do touro = MGS da vaca -> F = 6.25%
  MGS do touro = pai da vaca -> F = 6.25%
  MGGS do touro = pai da vaca -> F = 3.125%

Ajuste por concentracao genetica:
  Se pai do touro aparece em outros touros -> +0.5%

Ajuste por F parental:
  F += (F_pai + F_mae) * 3%
```

#### 4.4.2 Depressao endogamica por trait

Funcao: `compute_inbreeding_depression(expected_f)`

```
Depressao = taxa_por_1%_F * F_em_%
```

| Trait | Depressao por 1% de F |
|---|---|
| NM$ | -$25.00 |
| MILK | -28.5 lbs |
| FAT | -1.0 lb |
| PROT | -0.9 lb |
| PL | -0.35 meses |
| DPR | -0.03% |
| SCS | +0.007 (pior) |
| LIV | -0.15% |
| CCR | -0.025% |
| HCR | -0.02% |

**Fontes:** VanRaden (2005), Bjelland et al. (2013), Pryce et al. (2014), CDCB

### 4.5 Variancia de Mendelian Sampling

```python
sigma2_a = desvio_padrao_genetico ** 2

# Componente A: Recombinacao genetica (irredutivel)
var_mendelian = 0.5 * (1.0 - F_esperada) * sigma2_a

# Componente B: Incerteza dos GPTAs parentais
var_uncertainty = 0.20 * sigma2_a  # Media estimada

# Variancia total
total_var = var_mendelian + var_uncertainty

# Intervalo de confianca 95%
IC = PA_corrigido +/- 1.96 * sqrt(total_var)
```

**Desvios-padrao geneticos aditivos (sigma_a) — Holstein:**

| Trait | sigma_a | Trait | sigma_a |
|---|---|---|---|
| MILK | 675 lbs | DPR | 1.30% |
| FAT | 29 lbs | CCR | 1.65% |
| PROT | 19 lbs | PL | 1.85 meses |
| NM$ | $275 | UDC | 0.75 |
| TPI | 250 | FLC | 0.65 |
| SCS | 0.14 | LIV | 1.20% |

**Fonte:** Cole & VanRaden (2018) J. Dairy Sci. 101:5227-5236, CDCB PTA SDs

### 4.6 Analise Probabilistica

Funcao: `compute_risk_metrics(expected_nm, variance_nm, ...)`

Usando distribuicao normal:

```
P(cria > media) = Phi(z_avg)
  onde z_avg = (NM$_corrigido - media_pop) / sd_pred

P(cria no top 25%) = 1 - Phi(z_top25)
  onde z_top25 = (media_pop + 0.674 * sd_pop - NM$_corrigido) / sd_pred

P(cria no top 10%) = 1 - Phi(z_top10)
  onde z_top10 = (media_pop + 1.282 * sd_pop - NM$_corrigido) / sd_pred
```

CDF normal aproximada via Abramowitz & Stegun (1964).

### 4.7 Penalidade de Limites Criticos

Funcao: `compute_critical_penalty(trait_pred)`

"A corrente e tao forte quanto o elo mais fraco."

| Trait | Limite Critico | Condicao |
|---|---|---|
| DPR | -2.0% | Abaixo = fertilidade critica |
| PL | -1.0 meses | Abaixo = longevidade critica |
| SCS | 3.30 | Acima = mastite cronica |
| UDC | -1.0 | Abaixo = ubere muito ruim |
| FLC | -1.0 | Abaixo = pernas muito ruins |
| SCE | 5.0% | Acima = parto dificil |

```
Se trait predito viola o limite:
  penalty += |diferenca| * 20
```

### 4.8 Antagonismo Producao x Funcionalidade

Funcao: `compute_antagonism(sire, dam)`

Usa correlacoes geneticas antagonicas conhecidas para penalizar combinacoes onde producao alta puxa funcionalidade para baixo.

**Correlacoes geneticas utilizadas:**

| Trait 1 | Trait 2 | r_g |
|---|---|---|
| MILK | DPR | -0.35 |
| MILK | CCR | -0.30 |
| MILK | PL | -0.15 |
| MILK | UDC | -0.15 |
| MILK | SCS | +0.10 |
| FAT | DPR | -0.25 |
| FAT | CCR | -0.20 |
| PROT | DPR | -0.30 |
| PROT | CCR | -0.25 |
| DPR | PL | +0.45 |
| SCS | MAST | +0.70 |
| SCS | PL | -0.30 |
| UDC | PL | +0.25 |

```
Se z_trait1 > 1.0 E z_trait2 < -0.5 (antagonismo realizado):
  penalty += |r_g| * (z1 - z2) * 3
```

**Fonte:** VanRaden et al. (2004, 2014, 2018), CDCB multi-trait model

### 4.9 Bonus de Dominancia (Heterose Intra-Raca)

Funcao: `compute_dominance_bonus(expected_f)`

Pais menos aparentados geram crias com mais heterozigosidade, o que produz desvio de dominancia positivo (vigor hibrido parcial, mesmo dentro da raca).

```
heterozygosity = 1.0 - F_esperada

Para cada trait com efeito de dominancia:
  dom_sd = sqrt(d_ratio) * sigma_a
  bonus = heterozygosity * dom_sd * 0.15  # estimativa conservadora
  total += bonus * |peso_econ| / 100
```

**Razoes dominancia/aditivo (d²/sigma²_a):**

| Trait | d²/sigma²_a |
|---|---|
| DPR | 0.20 |
| PL | 0.15 |
| CCR | 0.15 |
| SCS | 0.14 |
| MILK | 0.12 |
| PROT | 0.12 |
| LIV | 0.12 |
| UDC | 0.11 |
| FAT | 0.10 |
| FLC | 0.08 |

**Fonte:** Sun et al. (2014) J. Dairy Sci. 97:3852-3861

### 4.10 Distancia ao Perfil Ideal

Funcao: `compute_distance_to_ideal(trait_pred)`

Distancia euclidiana ponderada do perfil predito ao perfil ideal:

```
Para cada trait:
  diff = max(0, ideal - predito) / sigma_a   # normalizado
  dist_sq += diff^2 * (peso_econ / 30)       # ponderado

distancia = sqrt(dist_sq)
```

Menor distancia = combinacao mais proxima do animal ideal.

### 4.11 Verificacao de Haplotipos Letais

Funcao: `check_haplotypes(sire, dam)`

Verifica se touro E femea sao portadores do mesmo haplotipo letal:

**Haplotipos verificados:** HH1, HH2, HH3, HH4, HH5, HH6, HCD, HHR, HMW3, HMW4, HHP

Se ambos portadores do mesmo haplotipo: penalidade de **-15 pontos** no score final por conflito.

### 4.12 Analise de Caseinas

Funcao: `check_caseins(sire, dam)`

Avalia combinacao de beta-caseina e kappa-caseina:

| Combinacao Beta | Resultado | Score |
|---|---|---|
| A2A2 x A2A2 | 100% A2 | +3 |
| A2A2 x A1A2 ou A1A2 x A2A2 | 50% A2 | +1 |
| A1A2 x A1A2 | 25% A2 | 0 |
| A1A1 x qualquer | risco A1 | -2 |

| Combinacao Kappa | Resultado | Score |
|---|---|---|
| BB x BB | 100% BB | +2 |
| BB x AB ou AB x BB | 50% BB | +1 |
| BB x AA ou AA x BB | 100% AB | +1 |

---

## 5. Score Final Composto

Funcao: `compute_final_scores(predictions, pop_avg_nm, pop_sd_nm)`

```python
Score_Final = (
    0.35 * DSII_normalizado +             # Melhoria especifica da femea
    0.20 * NM$_normalizado +               # Merito economico bruto
    0.15 * Distancia_ideal_norm +           # Proximidade ao perfil ideal
    0.10 * Antagonismo_norm +               # Ausencia de antagonismo
    0.10 * max(0, 100 - Penalidade_critica) + # Ausencia de traits criticos
    0.05 * max(0, 100 - F_esperada * 800) +   # Penalidade endogamia
    0.05 * min(100, P_top25 * 100)           # Probabilidade top 25%
    - n_haplo_conflitos * 15                 # Haplotipos letais
    + casein_score * 1.5                     # Bonus caseinas
    + dominance_bonus                        # Bonus dominancia
)

Score_Final = clamp(0, 100)
```

### Pesos do Score Final

| Componente | Peso | Descricao |
|---|---|---|
| **DSII** | **35%** | Melhoria especifica — o que diferencia do PA |
| NM$ corrigido | 20% | Merito economico bruto |
| Distancia ao ideal | 15% | Quao perto do animal ideal |
| Ausencia de antagonismo | 10% | Sem conflito producao x funcionalidade |
| Ausencia de traits criticos | 10% | Nenhum trait abaixo do limite |
| Endogamia baixa | 5% | F < 12.5% |
| P(top 25%) | 5% | Probabilidade de cria elite |
| Haplotipos letais | -15/conflito | Penalidade por risco |
| Caseinas | +1.5/ponto | Bonus A2/BB |
| Dominancia | variavel | Bonus heterose |

---

## 6. Traits Avaliados (30+)

### Producao
MILK, FAT, PROT, CFP

### Fertilidade
DPR, CCR, HCR, EFC

### Saude e Longevidade
SCS, PL, LIV, HLIV, MAST (mastite), MET (metrite), RP (retencao de placenta), DA (deslocamento de abomaso), KET (cetose), MF (febre do leite)

### Tipo (Compostos)
UDC, FLC, BWC

### Tipo (Lineares — 18 traits)
STA (estatura), STR (forca), BD (profundidade corporal), DF (forma leiteira), RA (angulo de garupa), TW (largura de garupa/thurls width), RLS (pernas vista lateral), RLR (pernas vista posterior), FA (angulo de casco), FLS (pernas score), FUA (insercao anterior ubere), RUH (altura ubere posterior), RUW (largura ubere posterior), UC (ligamento central), UD (profundidade ubere), FTP (colocacao tetas anteriores), RTP (colocacao tetas posteriores), TL (comprimento de tetas)

### Eficiencia Alimentar
RFI (consumo alimentar residual)

### Parto
SCE (facilidade de parto), SSB (natimortos do touro), DSB (natimortos da filha)

---

## 7. Dados de Entrada

### Touros
Arquivo Excel com 70+ colunas por touro:
- NAAB, Nome, Registro, TPI, NM$
- Todos os PTAs (producao, saude, tipo, lineares, parto)
- Beta-caseina, Kappa-caseina, Haplotipos
- Pai, MGS, MGGS

### Femeas
Arquivo Excel com 68+ colunas por femea:
- Numero, NAAB, Registro, TPI, NM$
- Todos os PTAs (producao, saude, tipo, lineares, parto)
- Beta-caseina, Kappa-caseina, Haplotipos
- Pai, MGS, MGGS

---

## 8. Saidas

### CSV de Resultados (`predicao_v3_dsii.csv`)

Colunas por combinacao:

| Grupo | Colunas |
|---|---|
| Ranking | Rank_v3, Rank_PA_Simples, Diferenca_Ranking |
| Touro | NAAB, Nome, TPI, NM$ |
| Femea | NAAB, Num, Reg, TPI, NM$ |
| Economico | NM$_PA, NM$_Corrigido, NM$_IC_Lower, NM$_IC_Upper |
| DeltaG | Delta_NM$_sobre_Mae |
| TPI | TPI_PA, TPI_Corrigido, TPI_IC_Lower, TPI_IC_Upper, Delta_TPI |
| DSII | DSII, Score_Final_v3 |
| Endogamia | F_Esperada_%, Depressao_$ |
| Penalidades | Penalidade_Critica, Penalidade_Antagonismo |
| Bonus | Bonus_Dominancia, Distancia_Ideal |
| Probabilidades | P_Acima_Media_%, P_Top25_%, P_Top10_% |
| Genomica | Haplo_Conflitos, Beta_Caseina, Kappa_Caseina |
| Pedigree | Pai_da_Vaca |
| Traits | PTA_predito e DeltaG para cada um dos 30+ traits |

### Relatorios

1. **Comparacao v3 vs PA Simples** — ranking lado a lado
2. **Melhor Touro por Femea** — com indicacao quando difere do PA
3. **Relatorio por Touro** — DeltaG detalhado por femea
4. **Analise de Diversificacao** — concentracao por grupo de pai-da-vaca

---

## 9. Predicoes Realizadas

| Dataset | Touros | Femeas | Combinacoes | Data |
|---|---|---|---|---|
| Demo | 3 | 3 | 9 | 23/04/2026 |
| Demo expandido | 6 | 46 | 276 | 24/04/2026 |
| Felipe Santana (IA) | 6 | ~52 | ~313 | 25/04/2026 |
| Felipe Santana (Embriao) | 6 | ~5 | ~27 | 25/04/2026 |

---

## 10. O que o DSII v3 FAZ DIFERENTE do PA Simples

1. **Rankings mudam por femea** — O mesmo touro pode ser #1 para uma femea e #5 para outra
2. **Corrige deficiencias especificas** — Amplifica o peso economico de traits onde a femea e fraca
3. **Penaliza riscos** — Traits criticos, antagonismos, haplotipos
4. **Calcula incerteza real** — IC 95% via Mendelian Sampling, nao apenas um ponto
5. **Avalia melhoria sobre a mae** — DeltaG mostra o ganho real por geracao
6. **Considera heterose** — Bonus de dominancia para cruzamentos menos aparentados
7. **Verifica caseinas** — Bonus para A2A2 e kappa BB
8. **Monitora diversidade** — Alerta quando um touro domina um grupo de meio-irmas

---

## 11. Integracao com SMS Engine

O motor DSII v3 foi posteriormente incorporado ao **SMS Mating Engine** (`sms-engine/mating_engine.py`), onde:

- O DSII se tornou o componente `score_trait_correction` com prioridades de traits
- O controle de endogamia evoluiu para usar o grafo de pedigree real (292k animais do BULLMAS.DBF)
- Os pesos do score final foram reorganizados em: trait_weight (40%) + economic_weight (40%) + hhp_weight (20%)
- Adicionou controle operacional: doses por touro, restricoes vaca/novilha, facilidade de parto, Build Index customizavel
- Manteve 100% de compatibilidade com os resultados do SMS original da Select Sires (verificado com 137 vacas x 3 touros)
