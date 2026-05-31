"""
================================================================================
PREDIÇÃO GENÔMICA AVANÇADA — RAÇA HOLANDESA
Motor de Predição de Progênie (Offspring Merit Prediction Engine)
================================================================================

Metodologia baseada em:
- ssGBLUP (Aguilar/Misztal 2010) — conceitos de variância genômica
- NM$ 2025 (CDCB/USDA-AGIL) — pesos econômicos oficiais
- VanRaden (2008) — métricas de confiabilidade genômica
- Correção de depressão endogâmica (Pryce et al. 2014)
- Mendelian Sampling Variance (Bijma 2012)

Autor: VYR Labs / Diego Guerra
Data: 2026-04-23
================================================================================
"""

import math
import itertools
from dataclasses import dataclass, field
from typing import Optional
import json
import csv
import os
from datetime import datetime

# =============================================================================
# 1. CONFIGURAÇÃO DOS TRAITS E PESOS ECONÔMICOS (NM$ 2025)
# =============================================================================

@dataclass
class TraitConfig:
    """Configuração de cada trait no sistema de avaliação."""
    name: str
    unit: str
    nm_weight: float          # Peso relativo no NM$ 2025 (%)
    economic_value: float     # Valor econômico por unidade de PTA ($/unidade)
    heritability: float       # Herdabilidade (h²)
    genetic_sd: float         # Desvio padrão genético aditivo
    inbreeding_depression: float  # Depressão por 1% de endogamia (unid. do trait)
    direction: int            # +1 = maior é melhor, -1 = menor é melhor
    description: str = ""

# Traits oficiais com pesos NM$ 2025 e parâmetros genéticos para Holstein
# Fontes: CDCB, USDA-AGIL ARR-NM9, Journal of Dairy Science
TRAITS_CONFIG = {
    "MILK": TraitConfig(
        name="Milk", unit="lbs", nm_weight=3.2,
        economic_value=0.00009, heritability=0.20,
        genetic_sd=1100.0, inbreeding_depression=-29.6,
        direction=1, description="Produção de leite 305d"
    ),
    "FAT": TraitConfig(
        name="Fat", unit="lbs", nm_weight=31.8,
        economic_value=2.84, heritability=0.20,
        genetic_sd=42.0, inbreeding_depression=-1.41,
        direction=1, description="Produção de gordura 305d"
    ),
    "PROT": TraitConfig(
        name="Protein", unit="lbs", nm_weight=13.0,
        economic_value=1.84, heritability=0.20,
        genetic_sd=30.0, inbreeding_depression=-1.13,
        direction=1, description="Produção de proteína 305d"
    ),
    "SCS": TraitConfig(
        name="SCS", unit="score", nm_weight=2.6,
        economic_value=-152.0, heritability=0.12,
        genetic_sd=0.18, inbreeding_depression=0.008,
        direction=-1, description="Escore de células somáticas"
    ),
    "PL": TraitConfig(
        name="Productive Life", unit="months", nm_weight=13.0,
        economic_value=36.0, heritability=0.08,
        genetic_sd=1.7, inbreeding_depression=-0.26,
        direction=1, description="Vida produtiva"
    ),
    "LIV": TraitConfig(
        name="Cow Livability", unit="%", nm_weight=5.9,
        economic_value=24.0, heritability=0.02,
        genetic_sd=1.2, inbreeding_depression=-0.10,
        direction=1, description="Livabilidade da vaca"
    ),
    "HLIV": TraitConfig(
        name="Heifer Livability", unit="%", nm_weight=0.8,
        economic_value=9.0, heritability=0.01,
        genetic_sd=0.8, inbreeding_depression=-0.05,
        direction=1, description="Livabilidade da novilha"
    ),
    "DPR": TraitConfig(
        name="DPR", unit="%", nm_weight=2.1,
        economic_value=11.0, heritability=0.04,
        genetic_sd=1.3, inbreeding_depression=-0.24,
        direction=1, description="Taxa de prenhez das filhas"
    ),
    "CCR": TraitConfig(
        name="CCR", unit="%", nm_weight=1.8,
        economic_value=8.0, heritability=0.02,
        genetic_sd=1.5, inbreeding_depression=-0.18,
        direction=1, description="Taxa de concepção da vaca"
    ),
    "HCR": TraitConfig(
        name="HCR", unit="%", nm_weight=0.5,
        economic_value=3.0, heritability=0.01,
        genetic_sd=1.4, inbreeding_depression=-0.12,
        direction=1, description="Taxa de concepção da novilha"
    ),
    "EFC": TraitConfig(
        name="Early First Calving", unit="days", nm_weight=1.0,
        economic_value=0.4, heritability=0.06,
        genetic_sd=5.0, inbreeding_depression=0.0,
        direction=1, description="Precocidade ao primeiro parto"
    ),
    "RFI": TraitConfig(
        name="Feed Saved (RFI)", unit="lbs/day", nm_weight=6.8,
        economic_value=-42.0, heritability=0.15,
        genetic_sd=0.6, inbreeding_depression=0.0,
        direction=-1, description="Consumo alimentar residual"
    ),
    "BWC": TraitConfig(
        name="Body Weight Composite", unit="score", nm_weight=11.0,
        economic_value=-5.0, heritability=0.40,
        genetic_sd=1.0, inbreeding_depression=0.0,
        direction=-1, description="Composição de peso corporal"
    ),
    "UDC": TraitConfig(
        name="Udder Composite", unit="score", nm_weight=1.3,
        economic_value=18.0, heritability=0.25,
        genetic_sd=0.8, inbreeding_depression=0.0,
        direction=1, description="Composto de úbere"
    ),
    "FLC": TraitConfig(
        name="Feet & Leg Composite", unit="score", nm_weight=0.4,
        economic_value=4.0, heritability=0.10,
        genetic_sd=0.7, inbreeding_depression=0.0,
        direction=1, description="Composto de pernas e pés"
    ),
    "CA": TraitConfig(
        name="Calving Ability", unit="$", nm_weight=3.3,
        economic_value=1.0, heritability=0.05,
        genetic_sd=15.0, inbreeding_depression=0.0,
        direction=1, description="Habilidade de parto"
    ),
    "HTH": TraitConfig(
        name="Health$", unit="$", nm_weight=1.5,
        economic_value=1.0, heritability=0.03,
        genetic_sd=10.0, inbreeding_depression=0.0,
        direction=1, description="Índice de saúde"
    ),
}


# =============================================================================
# 2. ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class Animal:
    """Representa um animal (touro ou vaca) com seus GPTAs."""
    id: str
    name: str
    sex: str  # "M" ou "F"
    ptas: dict  # {trait_code: valor_pta}
    reliabilities: dict  # {trait_code: confiabilidade 0-1}
    inbreeding_coef: float = 0.0  # Coeficiente de endogamia genômico (FROH)
    sire_id: Optional[str] = None
    dam_id: Optional[str] = None
    generation: int = 0


@dataclass
class MatingPrediction:
    """Resultado da predição de um acasalamento específico."""
    sire: Animal
    dam: Animal
    # Parent Average ajustado por trait
    parent_avg: dict = field(default_factory=dict)
    # Variância do Mendelian Sampling por trait
    mendelian_var: dict = field(default_factory=dict)
    # Intervalo de confiança (95%) por trait
    ci_lower: dict = field(default_factory=dict)
    ci_upper: dict = field(default_factory=dict)
    # Endogamia esperada da cria
    expected_inbreeding: float = 0.0
    # Depressão endogâmica total ($)
    inbreeding_depression_dollars: float = 0.0
    # Correção por trait
    inbreeding_correction: dict = field(default_factory=dict)
    # Mérito econômico
    nm_dollar: float = 0.0
    nm_dollar_corrected: float = 0.0  # Após correção de endogamia
    # Métricas de risco
    prob_top25: float = 0.0  # P(cria no top 25%)
    prob_above_avg: float = 0.0  # P(cria acima da média)
    risk_score: float = 0.0  # Score de risco (0-100)
    # Score de complementaridade
    complementarity_score: float = 0.0
    # Score final composto
    final_score: float = 0.0
    # Ranking
    rank: int = 0


# =============================================================================
# 3. MOTOR DE PREDIÇÃO
# =============================================================================

class GenomicPredictionEngine:
    """
    Motor de predição genômica avançada para Holstein.

    Vai além do PA simples incorporando:
    1. Ponderação por confiabilidade (REL)
    2. Variância de Mendelian Sampling genômica
    3. Correção de depressão endogâmica
    4. Índice econômico NM$ 2025
    5. Análise probabilística de risco
    6. Score de complementaridade de traits
    """

    def __init__(self, traits_config=None):
        self.traits = traits_config or TRAITS_CONFIG
        self.population_avg_nm = 0.0  # Média NM$ da população
        self.population_sd_nm = 200.0  # SD do NM$ na população (estimativa Holstein)

    # -------------------------------------------------------------------------
    # 3.1 PARENT AVERAGE PONDERADO POR CONFIABILIDADE
    # -------------------------------------------------------------------------
    def compute_weighted_parent_average(self, sire: Animal, dam: Animal, trait_code: str) -> tuple:
        """
        Calcula o Parent Average ponderado pela confiabilidade de cada progenitor.

        Diferente do PA simples (PTA_s + PTA_d)/2, este método:
        - Pondera pela REL de cada progenitor
        - Quando RELs são iguais, reduz ao PA simples
        - Quando um progenitor tem REL muito baixa, o outro domina

        Fórmula (derivada do índice de seleção multi-fonte):
        PA_w = w_s * PTA_s + w_d * PTA_d

        Onde:
        w_s = REL_s / (REL_s + REL_d)
        w_d = REL_d / (REL_s + REL_d)

        Mas o ESPERADO da cria é sempre (PTA_s + PTA_d)/2 em termos de
        valor genético VERDADEIRO. A ponderação por REL afeta a VARIÂNCIA
        da predição, não o valor esperado.

        Retorna: (expected_value, prediction_variance)
        """
        pta_s = sire.ptas.get(trait_code, 0.0)
        pta_d = dam.ptas.get(trait_code, 0.0)
        rel_s = sire.reliabilities.get(trait_code, 0.30)
        rel_d = dam.reliabilities.get(trait_code, 0.30)

        tc = self.traits[trait_code]
        sigma2_a = tc.genetic_sd ** 2  # Variância genética aditiva

        # Valor esperado da cria = PA (isso é correto mesmo com RELs diferentes)
        # Porque E[BV_offspring] = (TBV_sire + TBV_dam) / 2
        # E o melhor estimador de TBV é o próprio GPTA
        expected = (pta_s + pta_d) / 2.0

        # Variância da predição (incorpora incerteza dos GPTAs dos pais)
        # Var(BV_offspring - PA_predicted) tem 2 componentes:
        #
        # A) Variância do Mendelian Sampling: (1/2)(1 - (F_s + F_d)/2) * σ²_a
        #    Esta é IRREDUTÍVEL — é a recombinação genética real
        #
        # B) Variância da incerteza dos GPTAs parentais:
        #    (1/4) * [(1 - REL_s) * σ²_a + (1 - REL_d) * σ²_a]
        #    Quanto menor a REL, maior a incerteza

        # Componente A: Mendelian Sampling
        f_avg = (sire.inbreeding_coef + dam.inbreeding_coef) / 2.0
        var_mendelian = 0.5 * (1.0 - f_avg) * sigma2_a

        # Componente B: Incerteza dos GPTAs parentais
        var_uncertainty_sire = 0.25 * (1.0 - rel_s) * sigma2_a
        var_uncertainty_dam = 0.25 * (1.0 - rel_d) * sigma2_a
        var_uncertainty = var_uncertainty_sire + var_uncertainty_dam

        # Variância total da predição
        total_var = var_mendelian + var_uncertainty

        return expected, total_var

    # -------------------------------------------------------------------------
    # 3.2 ENDOGAMIA ESPERADA DA CRIA
    # -------------------------------------------------------------------------
    def estimate_expected_inbreeding(self, sire: Animal, dam: Animal,
                                      known_relationship: Optional[float] = None) -> float:
        """
        Estima o coeficiente de endogamia esperado da cria.

        F_offspring = a_SD / 2

        Onde a_SD é o coeficiente de parentesco entre touro e vaca.

        Se não temos a relação genômica direta, estimamos por:
        1. Relação conhecida (se fornecida)
        2. Média da raça Holstein atual + incremento por geração

        Em populações Holstein atuais:
        - F médio genômico (FROH): ~10-12% (novilhas nascidas 2024)
        - ΔF por geração: ~0.5-1.0%
        """
        if known_relationship is not None:
            return known_relationship / 2.0

        # Se temos os pais em comum (mesmo pai = meio-irmãos)
        if sire.sire_id and dam.sire_id and sire.sire_id == dam.sire_id:
            # Meio-irmãos paternos: a_SD ≈ 0.25
            # + endogamia base da raça
            base_relationship = 0.25
        elif sire.id == dam.sire_id:
            # Touro é pai da vaca: a_SD = 0.50
            base_relationship = 0.50
        else:
            # Relação desconhecida — usa média da raça Holstein
            # Holstein atual: F médio ≈ 10%, portanto relação média ≈ 20%
            # Mas entre touro e vaca aleatórios: ~6-8% (EFI médio)
            base_relationship = 0.125  # Estimativa conservadora

        # Ajusta pela endogamia dos próprios pais
        f_sire = sire.inbreeding_coef
        f_dam = dam.inbreeding_coef

        # F_offspring = a_SD / 2 (Wright, 1922)
        expected_f = base_relationship / 2.0

        # Adiciona contribuição da endogamia parental
        # Se os pais já são endogâmicos, a cria tende a ser mais
        expected_f += (f_sire + f_dam) * 0.10  # Contribuição incremental

        return min(expected_f, 0.30)  # Cap em 30%

    # -------------------------------------------------------------------------
    # 3.3 DEPRESSÃO ENDOGÂMICA
    # -------------------------------------------------------------------------
    def compute_inbreeding_depression(self, expected_f: float) -> dict:
        """
        Calcula a depressão endogâmica esperada por trait.

        Depressão = F * b_ID (por 1% de endogamia * F em %)

        Valores de b_ID baseados em:
        - CDCB/USDA (VanRaden, Nani et al. 2025)
        - Meta-análise de Leroy (2014)
        """
        corrections = {}
        f_percent = expected_f * 100.0  # Converte para %

        for code, tc in self.traits.items():
            # Depressão = taxa por 1% × F%
            depression = tc.inbreeding_depression * f_percent
            corrections[code] = depression

        return corrections

    # -------------------------------------------------------------------------
    # 3.4 ÍNDICE ECONÔMICO NM$ 2025
    # -------------------------------------------------------------------------
    def compute_nm_dollar(self, trait_values: dict) -> float:
        """
        Calcula o Net Merit $ 2025 a partir dos PTAs preditos.

        NM$ = Σ (PTA_trait × economic_value_trait)

        Pesos econômicos do USDA-AGIL ARR-NM9 (2025).
        """
        nm = 0.0
        for code, tc in self.traits.items():
            pta = trait_values.get(code, 0.0)
            nm += pta * tc.economic_value
        return nm

    # -------------------------------------------------------------------------
    # 3.5 ANÁLISE PROBABILÍSTICA DE RISCO
    # -------------------------------------------------------------------------
    def compute_risk_metrics(self, expected_nm: float, variance_nm: float,
                              population_avg: float = 0.0,
                              population_sd: float = 200.0) -> dict:
        """
        Calcula métricas probabilísticas usando distribuição normal.

        Como o valor genético da cria segue:
        BV_offspring ~ N(PA, σ²_mendelian + σ²_uncertainty)

        Podemos calcular:
        - P(cria > média da população)
        - P(cria no top 25%)
        - P(cria no top 10%)
        - Valor em risco (VaR genético) — pior cenário a 5%
        """
        sd = math.sqrt(variance_nm) if variance_nm > 0 else 1.0

        # Z-scores
        z_avg = (expected_nm - population_avg) / sd if sd > 0 else 0.0
        z_top25 = (population_avg + 0.674 * population_sd - expected_nm) / sd
        z_top10 = (population_avg + 1.282 * population_sd - expected_nm) / sd

        # Probabilidades usando aproximação da CDF normal
        prob_above_avg = self._normal_cdf(z_avg)
        prob_top25 = 1.0 - self._normal_cdf(z_top25)
        prob_top10 = 1.0 - self._normal_cdf(z_top10)

        # Value at Risk genético (5% inferior)
        var_5pct = expected_nm - 1.645 * sd

        # Score de risco (0-100, onde 100 = máxima certeza de alto mérito)
        # Combina nível esperado + consistência (baixa variância)
        level_score = min(max(self._normal_cdf(z_avg) * 100, 0), 100)
        consistency = max(0, 100 - (sd / population_sd) * 50)
        risk_score = 0.7 * level_score + 0.3 * consistency

        return {
            "prob_above_avg": prob_above_avg,
            "prob_top25": prob_top25,
            "prob_top10": prob_top10,
            "var_5pct": var_5pct,
            "risk_score": risk_score,
            "prediction_sd": sd,
        }

    # -------------------------------------------------------------------------
    # 3.6 COMPLEMENTARIDADE DE TRAITS
    # -------------------------------------------------------------------------
    def compute_complementarity(self, sire: Animal, dam: Animal) -> float:
        """
        Avalia a complementaridade entre touro e vaca.

        Princípio: o melhor acasalamento corrige as fraquezas de um progenitor
        com as forças do outro, sem criar desequilíbrios extremos.

        Score = penaliza quando AMBOS são fracos no mesmo trait
               bonifica quando um compensa a fraqueza do outro

        Baseado em Mate Allocation Index (Kinghorn 1998).
        """
        score = 0.0
        n_traits = 0

        for code, tc in self.traits.items():
            pta_s = sire.ptas.get(code, 0.0)
            pta_d = dam.ptas.get(code, 0.0)
            sd = tc.genetic_sd

            if sd <= 0:
                continue

            # Normaliza PTAs em unidades de desvio padrão
            z_s = (pta_s * tc.direction) / sd
            z_d = (pta_d * tc.direction) / sd

            # Penaliza se ambos negativos (ponderado pela importância do trait)
            weight = tc.nm_weight / 100.0
            if z_s < -0.5 and z_d < -0.5:
                # Ambos fracos — penalidade severa
                penalty = abs(z_s + z_d) * weight * 2.0
                score -= penalty
            elif (z_s < -0.5 and z_d > 0.5) or (z_s > 0.5 and z_d < -0.5):
                # Um compensa o outro — bônus
                bonus = min(abs(z_s), abs(z_d)) * weight
                score += bonus
            elif z_s > 0.5 and z_d > 0.5:
                # Ambos fortes — bônus moderado
                bonus = min(z_s, z_d) * weight * 0.5
                score += bonus

            n_traits += 1

        # Normaliza para escala 0-100
        if n_traits > 0:
            score = max(0, min(100, 50 + score * 20))

        return score

    # -------------------------------------------------------------------------
    # 3.7 PREDIÇÃO COMPLETA DE UM ACASALAMENTO
    # -------------------------------------------------------------------------
    def predict_mating(self, sire: Animal, dam: Animal,
                        known_relationship: Optional[float] = None) -> MatingPrediction:
        """
        Executa a predição completa para um acasalamento touro × vaca.
        """
        pred = MatingPrediction(sire=sire, dam=dam)

        # 1. Endogamia esperada da cria
        pred.expected_inbreeding = self.estimate_expected_inbreeding(
            sire, dam, known_relationship
        )

        # 2. Depressão endogâmica por trait
        pred.inbreeding_correction = self.compute_inbreeding_depression(
            pred.expected_inbreeding
        )

        # 3. Parent Average + Variância por trait
        total_nm_variance = 0.0

        for code, tc in self.traits.items():
            # PA ponderado e variância
            expected, variance = self.compute_weighted_parent_average(
                sire, dam, code
            )

            # Aplica correção de endogamia
            depression = pred.inbreeding_correction.get(code, 0.0)
            corrected = expected + depression  # Depression já é negativo para traits favoráveis

            pred.parent_avg[code] = corrected
            pred.mendelian_var[code] = variance

            # Intervalo de confiança 95%
            sd = math.sqrt(variance) if variance > 0 else 0.0
            pred.ci_lower[code] = corrected - 1.96 * sd
            pred.ci_upper[code] = corrected + 1.96 * sd

            # Acumula variância para NM$
            total_nm_variance += (tc.economic_value ** 2) * variance

        # 4. NM$ esperado (com correção de endogamia)
        pred.nm_dollar = self.compute_nm_dollar(
            {code: (sire.ptas.get(code, 0) + dam.ptas.get(code, 0)) / 2.0
             for code in self.traits}
        )
        pred.nm_dollar_corrected = self.compute_nm_dollar(pred.parent_avg)

        # Depressão endogâmica total em $
        pred.inbreeding_depression_dollars = pred.nm_dollar - pred.nm_dollar_corrected

        # 5. Métricas de risco
        risk = self.compute_risk_metrics(
            pred.nm_dollar_corrected, total_nm_variance,
            self.population_avg_nm, self.population_sd_nm
        )
        pred.prob_above_avg = risk["prob_above_avg"]
        pred.prob_top25 = risk["prob_top25"]
        pred.risk_score = risk["risk_score"]

        # 6. Complementaridade
        pred.complementarity_score = self.compute_complementarity(sire, dam)

        # 7. Score final composto
        # Pesos: 50% mérito econômico + 20% risco + 15% complementaridade + 15% endogamia
        merit_normalized = min(max(
            (pred.nm_dollar_corrected - self.population_avg_nm) /
            self.population_sd_nm * 25 + 50, 0), 100)

        inbreeding_penalty = max(0, 100 - pred.expected_inbreeding * 500)
        # 6.25% = penalidade total, >12.5% = score 0

        pred.final_score = (
            0.50 * merit_normalized +
            0.20 * pred.risk_score +
            0.15 * pred.complementarity_score +
            0.15 * inbreeding_penalty
        )

        return pred

    # -------------------------------------------------------------------------
    # 3.8 BATCH: TODAS AS COMBINAÇÕES
    # -------------------------------------------------------------------------
    def predict_all_matings(self, sires: list, dams: list,
                             relationships: Optional[dict] = None) -> list:
        """
        Prediz todas as combinações touro × vaca e retorna ranking.

        relationships: dict opcional {(sire_id, dam_id): coef_parentesco}
        """
        predictions = []

        for sire in sires:
            for dam in dams:
                rel = None
                if relationships:
                    rel = relationships.get((sire.id, dam.id))

                pred = self.predict_mating(sire, dam, rel)
                predictions.append(pred)

        # Ordena por score final (decrescente)
        predictions.sort(key=lambda p: p.final_score, reverse=True)

        # Atribui ranking
        for i, pred in enumerate(predictions):
            pred.rank = i + 1

        return predictions

    # -------------------------------------------------------------------------
    # 3.9 RELATÓRIO
    # -------------------------------------------------------------------------
    def generate_report(self, predictions: list, top_n: int = None) -> str:
        """Gera relatório textual dos resultados."""
        lines = []
        lines.append("=" * 100)
        lines.append("RELATÓRIO DE PREDIÇÃO GENÔMICA — RAÇA HOLANDESA")
        lines.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Total de combinações: {len(predictions)}")
        lines.append("=" * 100)
        lines.append("")

        display = predictions[:top_n] if top_n else predictions

        # Cabeçalho
        header = (
            f"{'Rank':>4} | {'Touro':<20} | {'Vaca':<20} | "
            f"{'NM$ Pred':>9} | {'NM$ Corr':>9} | "
            f"{'F(%) Cria':>9} | {'Dep.Endo$':>9} | "
            f"{'P(>Méd)':>7} | {'P(Top25)':>8} | "
            f"{'Compl.':>6} | {'Score':>6}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for pred in display:
            line = (
                f"{pred.rank:>4} | "
                f"{pred.sire.name:<20.20} | "
                f"{pred.dam.name:<20.20} | "
                f"${pred.nm_dollar:>8.0f} | "
                f"${pred.nm_dollar_corrected:>8.0f} | "
                f"{pred.expected_inbreeding*100:>8.2f}% | "
                f"${pred.inbreeding_depression_dollars:>8.0f} | "
                f"{pred.prob_above_avg*100:>6.1f}% | "
                f"{pred.prob_top25*100:>7.1f}% | "
                f"{pred.complementarity_score:>5.1f} | "
                f"{pred.final_score:>5.1f}"
            )
            lines.append(line)

        lines.append("")
        lines.append("=" * 100)

        # Detalhamento do melhor acasalamento
        if predictions:
            best = predictions[0]
            lines.append("")
            lines.append(f">>> MELHOR ACASALAMENTO: {best.sire.name} × {best.dam.name}")
            lines.append(f"    Score Final: {best.final_score:.1f}/100")
            lines.append(f"    NM$ Predito (corrigido): ${best.nm_dollar_corrected:.0f}")
            lines.append(f"    Endogamia Esperada: {best.expected_inbreeding*100:.2f}%")
            lines.append(f"    Depressão Endogâmica: ${best.inbreeding_depression_dollars:.0f}")
            lines.append("")
            lines.append("    Predição por Trait (com IC 95%):")
            lines.append(f"    {'Trait':<25} | {'PTA Pred':>9} | {'IC Inferior':>11} | {'IC Superior':>11} | {'Unidade':<10}")
            lines.append("    " + "-" * 80)

            for code in self.traits:
                tc = self.traits[code]
                pa = best.parent_avg.get(code, 0)
                lo = best.ci_lower.get(code, 0)
                hi = best.ci_upper.get(code, 0)
                lines.append(
                    f"    {tc.description:<25} | "
                    f"{pa:>9.2f} | "
                    f"{lo:>11.2f} | "
                    f"{hi:>11.2f} | "
                    f"{tc.unit:<10}"
                )

        lines.append("")
        lines.append("=" * 100)
        lines.append("Metodologia: ssGBLUP-derived PA + REL-weighted variance + ")
        lines.append("             Inbreeding depression (CDCB/EFI) + NM$ 2025 + ")
        lines.append("             Mendelian Sampling Variance + Risk Analysis")
        lines.append("Referências: VanRaden(2008), Aguilar/Misztal(2010), Cole/VanRaden(2025)")
        lines.append("=" * 100)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # 3.10 EXPORTAR CSV
    # -------------------------------------------------------------------------
    def export_csv(self, predictions: list, filepath: str):
        """Exporta resultados para CSV."""
        fieldnames = [
            "Rank", "Touro_ID", "Touro_Nome", "Vaca_ID", "Vaca_Nome",
            "NM_Dollar", "NM_Dollar_Corrigido", "F_Esperada_%",
            "Depressao_Endo_Dollar", "Prob_Acima_Media_%", "Prob_Top25_%",
            "Complementaridade", "Risk_Score", "Score_Final"
        ]

        # Adiciona colunas de cada trait
        for code in self.traits:
            fieldnames.extend([
                f"PTA_{code}", f"IC_Lower_{code}", f"IC_Upper_{code}"
            ])

        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()

            for pred in predictions:
                row = {
                    "Rank": pred.rank,
                    "Touro_ID": pred.sire.id,
                    "Touro_Nome": pred.sire.name,
                    "Vaca_ID": pred.dam.id,
                    "Vaca_Nome": pred.dam.name,
                    "NM_Dollar": f"{pred.nm_dollar:.0f}",
                    "NM_Dollar_Corrigido": f"{pred.nm_dollar_corrected:.0f}",
                    "F_Esperada_%": f"{pred.expected_inbreeding*100:.2f}",
                    "Depressao_Endo_Dollar": f"{pred.inbreeding_depression_dollars:.0f}",
                    "Prob_Acima_Media_%": f"{pred.prob_above_avg*100:.1f}",
                    "Prob_Top25_%": f"{pred.prob_top25*100:.1f}",
                    "Complementaridade": f"{pred.complementarity_score:.1f}",
                    "Risk_Score": f"{pred.risk_score:.1f}",
                    "Score_Final": f"{pred.final_score:.1f}",
                }

                for code in self.traits:
                    row[f"PTA_{code}"] = f"{pred.parent_avg.get(code, 0):.2f}"
                    row[f"IC_Lower_{code}"] = f"{pred.ci_lower.get(code, 0):.2f}"
                    row[f"IC_Upper_{code}"] = f"{pred.ci_upper.get(code, 0):.2f}"

                writer.writerow(row)

        print(f"[OK] Exportado: {filepath}")

    # -------------------------------------------------------------------------
    # UTILIDADES
    # -------------------------------------------------------------------------
    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Aproximação da CDF normal padrão (Abramowitz & Stegun)."""
        if z > 6:
            return 1.0
        if z < -6:
            return 0.0

        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        p = 0.2316419

        a = abs(z)
        t = 1.0 / (1.0 + p * a)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        pdf = math.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
        cdf = 1.0 - pdf * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5)

        if z < 0:
            cdf = 1.0 - cdf

        return cdf


# =============================================================================
# 4. DEMONSTRAÇÃO COM DADOS DE EXEMPLO
# =============================================================================

def create_demo_data():
    """
    Cria dados de demonstração realistas para Holstein.

    Estes valores são baseados em touros e vacas com GPTAs típicos
    publicados pelo CDCB (Abril 2025). Substitua pelos seus dados reais.
    """

    # --- TOUROS (exemplos baseados em perfis reais) ---
    sires = [
        Animal(
            id="HO840003247878341", name="PEAK ALTAZENITH-ET",
            sex="M",
            ptas={
                "MILK": 1847, "FAT": 108, "PROT": 78,
                "SCS": 2.72, "PL": 7.2, "LIV": 4.1, "HLIV": 1.8,
                "DPR": 1.5, "CCR": 2.1, "HCR": 1.3, "EFC": 3.2,
                "RFI": -0.8, "BWC": -0.4, "UDC": 2.14, "FLC": 1.52,
                "CA": 5.0, "HTH": 180,
            },
            reliabilities={
                "MILK": 0.81, "FAT": 0.81, "PROT": 0.81,
                "SCS": 0.75, "PL": 0.70, "LIV": 0.60, "HLIV": 0.55,
                "DPR": 0.65, "CCR": 0.60, "HCR": 0.55, "EFC": 0.60,
                "RFI": 0.55, "BWC": 0.75, "UDC": 0.78, "FLC": 0.70,
                "CA": 0.60, "HTH": 0.55,
            },
            inbreeding_coef=0.085,
        ),
        Animal(
            id="HO840003234512789", name="S-S-I FLAGSHIP-ET",
            sex="M",
            ptas={
                "MILK": 1205, "FAT": 132, "PROT": 62,
                "SCS": 2.65, "PL": 8.5, "LIV": 5.2, "HLIV": 2.1,
                "DPR": 2.8, "CCR": 3.5, "HCR": 2.0, "EFC": 4.5,
                "RFI": -1.2, "BWC": -0.8, "UDC": 1.85, "FLC": 1.90,
                "CA": 8.0, "HTH": 210,
            },
            reliabilities={
                "MILK": 0.78, "FAT": 0.78, "PROT": 0.78,
                "SCS": 0.72, "PL": 0.68, "LIV": 0.58, "HLIV": 0.52,
                "DPR": 0.62, "CCR": 0.58, "HCR": 0.52, "EFC": 0.58,
                "RFI": 0.52, "BWC": 0.72, "UDC": 0.75, "FLC": 0.68,
                "CA": 0.58, "HTH": 0.52,
            },
            inbreeding_coef=0.072,
        ),
        Animal(
            id="HO840003210987654", name="DENOVO RESOLVER-ET",
            sex="M",
            ptas={
                "MILK": 2150, "FAT": 85, "PROT": 92,
                "SCS": 2.80, "PL": 5.8, "LIV": 3.5, "HLIV": 1.5,
                "DPR": 0.8, "CCR": 1.2, "HCR": 0.9, "EFC": 2.1,
                "RFI": -0.5, "BWC": 0.2, "UDC": 2.45, "FLC": 1.10,
                "CA": 3.0, "HTH": 145,
            },
            reliabilities={
                "MILK": 0.83, "FAT": 0.83, "PROT": 0.83,
                "SCS": 0.78, "PL": 0.72, "LIV": 0.62, "HLIV": 0.57,
                "DPR": 0.67, "CCR": 0.62, "HCR": 0.57, "EFC": 0.62,
                "RFI": 0.57, "BWC": 0.78, "UDC": 0.80, "FLC": 0.72,
                "CA": 0.62, "HTH": 0.57,
            },
            inbreeding_coef=0.095,
        ),
    ]

    # --- VACAS (exemplos realistas) ---
    dams = [
        Animal(
            id="BR00001234", name="FAZENDA BOA VISTA TULIPA",
            sex="F",
            ptas={
                "MILK": 650, "FAT": 45, "PROT": 28,
                "SCS": 2.90, "PL": 3.2, "LIV": 2.0, "HLIV": 1.0,
                "DPR": 1.0, "CCR": 1.5, "HCR": 0.8, "EFC": 1.5,
                "RFI": -0.3, "BWC": 0.1, "UDC": 1.20, "FLC": 0.80,
                "CA": 2.0, "HTH": 85,
            },
            reliabilities={
                "MILK": 0.55, "FAT": 0.55, "PROT": 0.55,
                "SCS": 0.50, "PL": 0.45, "LIV": 0.35, "HLIV": 0.30,
                "DPR": 0.40, "CCR": 0.35, "HCR": 0.30, "EFC": 0.35,
                "RFI": 0.30, "BWC": 0.50, "UDC": 0.52, "FLC": 0.45,
                "CA": 0.35, "HTH": 0.30,
            },
            inbreeding_coef=0.068,
        ),
        Animal(
            id="BR00005678", name="SITIO ESTRELA FORTUNA",
            sex="F",
            ptas={
                "MILK": 980, "FAT": 62, "PROT": 40,
                "SCS": 2.78, "PL": 4.8, "LIV": 3.0, "HLIV": 1.4,
                "DPR": 2.2, "CCR": 2.8, "HCR": 1.6, "EFC": 3.0,
                "RFI": -0.6, "BWC": -0.3, "UDC": 1.50, "FLC": 1.20,
                "CA": 4.0, "HTH": 120,
            },
            reliabilities={
                "MILK": 0.60, "FAT": 0.60, "PROT": 0.60,
                "SCS": 0.55, "PL": 0.50, "LIV": 0.40, "HLIV": 0.35,
                "DPR": 0.45, "CCR": 0.40, "HCR": 0.35, "EFC": 0.40,
                "RFI": 0.35, "BWC": 0.55, "UDC": 0.57, "FLC": 0.50,
                "CA": 0.40, "HTH": 0.35,
            },
            inbreeding_coef=0.075,
        ),
        Animal(
            id="BR00009012", name="GRANJA SOL MARGARIDA",
            sex="F",
            ptas={
                "MILK": 420, "FAT": 55, "PROT": 22,
                "SCS": 3.05, "PL": 2.5, "LIV": 1.5, "HLIV": 0.8,
                "DPR": -0.5, "CCR": 0.2, "HCR": 0.3, "EFC": 0.5,
                "RFI": 0.1, "BWC": 0.5, "UDC": 0.90, "FLC": 0.60,
                "CA": 1.0, "HTH": 60,
            },
            reliabilities={
                "MILK": 0.52, "FAT": 0.52, "PROT": 0.52,
                "SCS": 0.48, "PL": 0.42, "LIV": 0.32, "HLIV": 0.28,
                "DPR": 0.38, "CCR": 0.33, "HCR": 0.28, "EFC": 0.33,
                "RFI": 0.28, "BWC": 0.48, "UDC": 0.50, "FLC": 0.43,
                "CA": 0.33, "HTH": 0.28,
            },
            inbreeding_coef=0.062,
        ),
    ]

    return sires, dams


# =============================================================================
# 5. EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    print("Iniciando Predição Genômica Avançada — Holstein")
    print("=" * 60)

    # Cria motor
    engine = GenomicPredictionEngine()

    # Carrega dados de demonstração
    sires, dams = create_demo_data()

    print(f"Touros carregados: {len(sires)}")
    print(f"Vacas carregadas: {len(dams)}")
    print(f"Combinações possíveis: {len(sires) * len(dams)}")
    print()

    # Roda predição para todas as combinações
    predictions = engine.predict_all_matings(sires, dams)

    # Relatório
    report = engine.generate_report(predictions)
    print(report)

    # Exporta CSV
    csv_path = os.path.join(os.path.dirname(__file__), "resultado_predicao.csv")
    engine.export_csv(predictions, csv_path)

    # Salva relatório em texto
    report_path = os.path.join(os.path.dirname(__file__), "relatorio_predicao.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Relatório salvo: {report_path}")
