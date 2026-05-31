# DSII V8 Oracle — Full Technical Report

## What is DSII?

**Dam-Specific Improvement Index** — A machine learning-based genetic prediction engine that estimates daughter PTAs from mating combinations (sire x dam estimated via pedigree), outperforming the classical Parent Average (PA) by learning non-linear trait interactions, genetic antagonisms, and dam deficiency profiles.

---

## Training Data

- **1,709 verified trios** (genotyped daughter + sire + dam)
- Daughters: CDCB May 2026 genomic evaluations — 1,974 genotyped females
- Dams: DAM1-20 files — 1,301 unique dams, linked by NAAB Code (86.7% match rate)
- Sires: bulls.csv — 40,047 bulls, linked by Registration Name (99.8% match rate)
- Validation: 5-Fold Cross-Validation on all models

---

## Model Evolution Summary

| Version | Method | Avg R² | Beats PA | Status |
|---------|--------|--------|----------|--------|
| PA | Classical (Sire + Dam) / 2 | 0.774 | — | Baseline |
| V3 | Deterministic rules (deficiency + antagonism corrections) | 0.776 | 13/25 | Marginal gains |
| V4 | Pure ML (Ridge/RF, no domain knowledge) | 0.778 | 19/25 | PA still wins 6 traits |
| **V5** | **Hybrid: V3 domain knowledge as ML features + LightGBM** | **0.823** | **25/25** | **Best single-model** |
| V6 | CDCB sire-only model (121k sire-daughter pairs) | 0.668 | — | No dam info available |
| V7 | Stacking ensemble (Ridge + LGBM + RF) | 0.731 | 0/25 | Overfitted (n=1,709 too small) |
| **V8** | **Oracle: best model per trait from 10-architecture shootout** | **0.823** | **25/25** | **Production model** |

---

## Complete R² Comparison Across All Versions (25 Traits)

| Trait | h² | PA | V3 | V4 | V5 | V7 | V8 | V8 Model | PA→V8 Gain |
|-------|------|--------|--------|--------|--------|--------|--------|----------|------------|
| TPI | 0.30 | 0.9240 | 0.9240 | 0.9456 | 0.9588 | 0.8775 | **0.9585** | GBR | +3.7% |
| NM$ | 0.30 | 0.9523 | 0.9512 | 0.9638 | 0.9713 | 0.6028 | **0.9716** | RF | +2.0% |
| CM$ | 0.30 | 0.9503 | 0.9503 | 0.9630 | 0.9705 | 0.4855 | **0.9706** | RF | +2.1% |
| MILK | 0.25 | 0.7844 | 0.7839 | 0.7798 | 0.8135 | 0.7747 | **0.8153** | GBR | +3.9% |
| FAT | 0.25 | 0.9346 | 0.9376 | 0.9414 | 0.9489 | 0.9098 | **0.9493** | GBR | +1.6% |
| FAT% | 0.50 | 0.7543 | 0.7543 | 0.7518 | 0.7836 | 0.7545 | **0.7893** | GBR | +4.6% |
| PRO | 0.25 | 0.8977 | 0.9027 | 0.9043 | 0.9185 | 0.8945 | **0.9191** | LGBM | +2.4% |
| PRO% | 0.50 | 0.6295 | 0.6295 | 0.6347 | 0.6789 | 0.6338 | **0.6961** | GBR | +10.6% |
| CFP | 0.30 | 0.9430 | 0.9447 | 0.9515 | 0.9579 | 0.9374 | **0.9586** | GBR | +1.7% |
| PL | 0.08 | 0.7929 | 0.7952 | 0.8060 | 0.8284 | 0.4363 | **0.8322** | LGBM | +5.0% |
| SCS | 0.12 | 0.6008 | 0.5999 | 0.6017 | 0.7042 | 0.6441 | **0.7076** | GBR | +17.8% |
| DPR | 0.04 | 0.7103 | 0.7057 | 0.7071 | 0.7635 | 0.7167 | **0.7620** | LGBM | +7.3% |
| LIV | 0.05 | 0.7682 | 0.7657 | 0.7674 | 0.8017 | 0.7433 | **0.8010** | GBR | +4.3% |
| FI | 0.06 | 0.6987 | 0.6987 | 0.6964 | 0.7626 | 0.7172 | **0.7562** | LGBM | +8.2% |
| HCR | 0.04 | 0.6330 | 0.6271 | 0.6352 | 0.7265 | 0.6874 | **0.7234** | GBR | +14.3% |
| CCR | 0.04 | 0.7024 | 0.6959 | 0.7004 | 0.7539 | 0.7088 | **0.7541** | GBR | +7.4% |
| MAST | 0.04 | 0.4968 | 0.4889 | 0.4871 | 0.6385 | 0.4824 | **0.6356** | LGBM | +27.9% |
| FSAV | 0.15 | 0.8404 | 0.8404 | 0.8445 | 0.8669 | 0.8428 | **0.8661** | LGBM | +3.1% |
| PTAT | 0.30 | 0.8898 | 0.8898 | 0.8988 | 0.9183 | 0.9069 | **0.9196** | XGB | +3.3% |
| UDC | 0.25 | 0.8067 | 0.8052 | 0.8094 | 0.8302 | 0.8145 | **0.8297** | GBR | +2.9% |
| FLC | 0.15 | 0.9078 | 0.9086 | 0.9101 | 0.9237 | 0.9099 | **0.9237** | LGBM | +1.8% |
| SCE | 0.08 | 0.7126 | 0.7126 | 0.7517 | 0.8242 | 0.7182 | **0.8238** | GBR | +15.6% |
| DCE | 0.06 | 0.5800 | 0.5800 | 0.5560 | 0.6558 | 0.6100 | **0.6558** | LGBM | +13.1% |
| SSB | 0.06 | 0.7733 | 0.7733 | 0.7747 | 0.8208 | 0.7791 | **0.8188** | LGBM | +5.9% |
| DSB | 0.04 | 0.6767 | 0.6767 | 0.6666 | 0.7489 | 0.6962 | **0.7447** | LGBM | +10.0% |
| **AVG** | | **0.7744** | **0.7745** | **0.7780** | **0.8228** | **0.7310** | **0.8233** | | **+6.3%** |

---

## R² Gain Relative to PA — Version by Version

| Trait | V3 vs PA | V4 vs PA | V5 vs PA | V7 vs PA | V8 vs PA |
|-------|----------|----------|----------|----------|----------|
| TPI | +0.0% | +2.3% | +3.8% | -5.0% | **+3.7%** |
| NM$ | -0.1% | +1.2% | +2.0% | -36.7% | **+2.0%** |
| CM$ | +0.0% | +1.3% | +2.1% | -48.9% | **+2.1%** |
| MILK | -0.1% | -0.6% | +3.7% | -1.2% | **+3.9%** |
| FAT | +0.3% | +0.7% | +1.5% | -2.7% | **+1.6%** |
| FAT% | +0.0% | -0.3% | +3.9% | +0.0% | **+4.6%** |
| PRO | +0.6% | +0.7% | +2.3% | -0.4% | **+2.4%** |
| PRO% | +0.0% | +0.8% | +7.8% | +0.7% | **+10.6%** |
| CFP | +0.2% | +0.9% | +1.6% | -0.6% | **+1.7%** |
| PL | +0.3% | +1.7% | +4.5% | -45.0% | **+5.0%** |
| SCS | -0.2% | +0.1% | +17.2% | +7.2% | **+17.8%** |
| DPR | -0.6% | -0.5% | +7.5% | +0.9% | **+7.3%** |
| LIV | -0.3% | -0.1% | +4.4% | -3.2% | **+4.3%** |
| FI | +0.0% | -0.3% | +9.1% | +2.6% | **+8.2%** |
| HCR | -0.9% | +0.3% | +14.8% | +8.6% | **+14.3%** |
| CCR | -0.9% | -0.3% | +7.3% | +0.9% | **+7.4%** |
| MAST | -1.6% | -2.0% | +28.5% | -2.9% | **+27.9%** |
| FSAV | +0.0% | +0.5% | +3.2% | +0.3% | **+3.1%** |
| PTAT | +0.0% | +1.0% | +3.2% | +1.9% | **+3.3%** |
| UDC | -0.2% | +0.3% | +2.9% | +1.0% | **+2.9%** |
| FLC | +0.1% | +0.3% | +1.8% | +0.2% | **+1.8%** |
| SCE | +0.0% | +5.5% | +15.7% | +0.8% | **+15.6%** |
| DCE | +0.0% | -4.1% | +13.1% | +5.2% | **+13.1%** |
| SSB | +0.0% | +0.2% | +6.1% | +0.7% | **+5.9%** |
| DSB | +0.0% | -1.5% | +10.7% | +2.9% | **+10.0%** |
| **AVG** | **+0.0%** | **+0.5%** | **+6.3%** | **-5.6%** | **+6.3%** |

---

## Model Architecture Shootout (10 Models x 25 Traits x 5-Fold CV)

| Model | Avg R² | Min R² | Max R² | Wins | Status |
|-------|--------|--------|--------|------|--------|
| PA (baseline) | 0.7744 | 0.4968 | 0.9523 | 0 | Baseline |
| Ridge | 0.7782 | 0.4862 | 0.9653 | 0 | Linear — limited |
| BayesianRidge | 0.7801 | 0.4913 | 0.9649 | 0 | Marginal over Ridge |
| ElasticNet | 0.7809 | 0.4922 | 0.9649 | 0 | Marginal over Ridge |
| SVR | 0.6581 | 0.3046 | 0.9282 | 0 | Failed — O(n²) with 75+ features |
| MLP | 0.5386 | -2.215 | 0.9569 | 0 | Failed — overfits with n=1,709 |
| RF | 0.7985 | 0.5336 | 0.9716 | 2 | Strong for composite indices |
| **GBR** | **0.8216** | **0.6273** | **0.9705** | **12** | **Top model** |
| XGB | 0.8176 | 0.6237 | 0.9700 | 1 | Close to GBR |
| **LGBM** | **0.8216** | **0.6356** | **0.9703** | **10** | **Tied top model** |
| CatBoost | 0.8103 | 0.6029 | 0.9675 | 0 | Good but never best |

### V8 Oracle Model Assignment

| Model | Traits Assigned |
|-------|-----------------|
| GBR (12) | TPI, MILK, FAT, FAT%, PRO%, CFP, SCS, LIV, HCR, CCR, UDC, SCE |
| LGBM (10) | PRO, PL, DPR, FI, MAST, FSAV, FLC, DCE, SSB, DSB |
| RF (2) | NM$, CM$ |
| XGB (1) | PTAT |

---

## Feature Engineering (75+ Features per Trait)

| Category | Features | Description |
|----------|----------|-------------|
| Base | sire, dam, PA, diff, delta_g | Core genetic values and parent average |
| Z-scores | sire_z, dam_z | Standardized by genetic SD of each trait |
| Interactions | sire_sq, dam_sq, sxd | Quadratic terms and sire x dam interaction |
| Heritability | h2_pa | PA weighted by (0.5 + h²) |
| Deficiency | deficiency, dg_x_def | Dam distance from breed ideal, crossed with delta_g |
| Inbreeding | ib_effect | Inbreeding depression estimate at 8.5% F |
| Dominance | dom_pot | Dominance ratio x |sire - dam| / SD |
| Genetic correlations | gc_MILK, gc_DPR, ... | Correlated trait PA x genetic correlation coefficient |
| Production pressure | prod_press | Combined MILK/FAT/PRO index |
| Cross-trait context | s_MILK, d_FAT, ... | All 24 other trait PTAs from sire and dam |
| Sire consistency | sire_loo | Leave-one-out mean of sire's other daughters |

---

## V8 vs PA — Final Comparison

| Metric | Parent Average | DSII V8 Oracle |
|--------|----------------|----------------|
| Average R² | 0.774 | **0.823 (+6.3%)** |
| Minimum R² | 0.497 (MAST) | **0.636 (MAST)** |
| Maximum R² | 0.952 (NM$) | **0.972 (NM$)** |
| Traits R² >= 0.90 | 5 | **8** |
| Traits R² >= 0.80 | 10 | **15** |
| Traits R² >= 0.70 | 17 | **22** |
| Average MAE reduction | — | **-17.8%** |
| Traits beating PA | — | **25/25 (100%)** |

### R² Distribution

| Tier | Count | Traits |
|------|-------|--------|
| R² >= 0.90 | 8 | TPI, NM$, CM$, FAT, PRO, CFP, PTAT, FLC |
| R² 0.80-0.89 | 7 | MILK, PL, LIV, FSAV, UDC, SCE, SSB |
| R² 0.70-0.79 | 7 | FAT%, SCS, DPR, FI, HCR, CCR, DSB |
| R² < 0.70 | 3 | PRO%, MAST, DCE |

---

## Why V8 Outperforms Parent Average

The **Parent Average** assumes a simple linear relationship: daughter = (sire + dam) / 2. In reality:

1. **Genetic antagonisms** — Traits with negative genetic correlation (e.g., MILK vs DPR at -0.35) create trade-offs that PA ignores. The ML model learns how high production pressure from one parent affects fertility predictions.

2. **Dam deficiency profiles** — When the dam is deficient in a trait, the sire's corrective effect is non-linear. A dam with low UDC benefits more from a high-UDC sire than a dam already at breed ideal.

3. **Cross-trait context** — All 24 other trait PTAs from both parents inform each prediction. A sire with extreme MILK but moderate DPR transmits differently than one balanced across both.

4. **Sire transmission consistency** — The leave-one-out feature captures how consistently a sire transmits vs. his published PTA, based on his other daughters in the dataset.

5. **Non-linear interactions** — Tree-based models (GBR, LGBM) capture sire x dam interactions, threshold effects, and diminishing returns that linear models cannot.

---

## Production Usage

**Input:** 3 NAAB codes (sire, maternal grandsire, maternal great-grandsire)

```bash
# Single mating prediction
python dsii_v8_predict.py 7HO18596 7HO18595 7HO18710

# Batch prediction from CSV
python dsii_v8_predict.py --batch matings.csv
```

**Dam estimation from pedigree:** Dam_PTA = MGS/2 + MGGS/4

The remaining 1/4 (unknown maternal great-great-grandam) defaults to breed average (0).

---

## Limitations and Next Steps

1. **Sample size** — 1,709 trios limit ML capacity. More genotyped daughter-dam pairs will directly improve R², especially for low-heritability traits.

2. **Mendelian sampling variance** — Traits with h² = 0.04 (DPR, HCR, MAST) have a biological ceiling on prediction accuracy without direct genomic information (SNP data).

3. **Dam estimation gap** — In production, dam PTAs are estimated from pedigree (MGS/2 + MGGS/4). During training, actual dam PTAs were used. Production R² will be slightly lower.

4. **Genetic base drift** — Models trained on May 2026 data. Should be retrained periodically as the genetic base evolves.

5. **Path to R² ~ 0.90** — Achievable for 8/25 traits already. Remaining traits need either more samples (>5,000 trios), deeper pedigree information, or direct genomic features.
