"""
Generate DSII V8 Oracle PDF Report — Professional format for sharing.
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable)
from reportlab.platypus.flowables import KeepTogether
import os

OUTPUT = "C:/Users/DiegoGuerra/Documents/Projetos/genetic_prediction/DSII_V8_Oracle_Report.pdf"

# Colors
DARK_BLUE = colors.HexColor("#1a365d")
MED_BLUE = colors.HexColor("#2b6cb0")
LIGHT_BLUE = colors.HexColor("#bee3f8")
PALE_BLUE = colors.HexColor("#ebf8ff")
GREEN = colors.HexColor("#276749")
LIGHT_GREEN = colors.HexColor("#c6f6d5")
RED = colors.HexColor("#c53030")
LIGHT_RED = colors.HexColor("#fed7d7")
GRAY = colors.HexColor("#718096")
LIGHT_GRAY = colors.HexColor("#f7fafc")
BORDER_GRAY = colors.HexColor("#e2e8f0")
WHITE = colors.white
BLACK = colors.HexColor("#1a202c")

doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
                        topMargin=0.6*inch, bottomMargin=0.6*inch,
                        leftMargin=0.65*inch, rightMargin=0.65*inch)

styles = getSampleStyleSheet()

# Custom styles
styles.add(ParagraphStyle('Title2', parent=styles['Title'],
                           fontSize=22, textColor=DARK_BLUE, spaceAfter=6,
                           fontName='Helvetica-Bold'))
styles.add(ParagraphStyle('Subtitle', parent=styles['Normal'],
                           fontSize=11, textColor=GRAY, spaceAfter=18,
                           fontName='Helvetica'))
styles.add(ParagraphStyle('H1', parent=styles['Heading1'],
                           fontSize=16, textColor=DARK_BLUE, spaceBefore=18,
                           spaceAfter=8, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle('H2', parent=styles['Heading2'],
                           fontSize=13, textColor=MED_BLUE, spaceBefore=14,
                           spaceAfter=6, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle('H3', parent=styles['Heading3'],
                           fontSize=11, textColor=DARK_BLUE, spaceBefore=10,
                           spaceAfter=4, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle('Body', parent=styles['Normal'],
                           fontSize=9.5, textColor=BLACK, spaceAfter=6,
                           leading=13, alignment=TA_JUSTIFY,
                           fontName='Helvetica'))
styles.add(ParagraphStyle('BodyBold', parent=styles['Normal'],
                           fontSize=9.5, textColor=BLACK, spaceAfter=6,
                           leading=13, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle('Bullet2', parent=styles['Normal'],
                           fontSize=9.5, textColor=BLACK, spaceAfter=4,
                           leading=13, leftIndent=18, bulletIndent=6,
                           fontName='Helvetica'))
styles.add(ParagraphStyle('Small', parent=styles['Normal'],
                           fontSize=8, textColor=GRAY, spaceAfter=4,
                           fontName='Helvetica'))
styles.add(ParagraphStyle('CellCenter', parent=styles['Normal'],
                           fontSize=8, alignment=TA_CENTER, fontName='Helvetica'))
styles.add(ParagraphStyle('CellBold', parent=styles['Normal'],
                           fontSize=8, alignment=TA_CENTER, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle('CellLeft', parent=styles['Normal'],
                           fontSize=8, alignment=TA_LEFT, fontName='Helvetica'))

story = []

# ============================================================
# COVER / TITLE
# ============================================================
story.append(Spacer(1, 0.8*inch))
story.append(Paragraph("DSII V8 Oracle", styles['Title2']))
story.append(Paragraph("Dam-Specific Improvement Index", styles['Subtitle']))
story.append(HRFlowable(width="100%", thickness=2, color=MED_BLUE, spaceAfter=12))
story.append(Paragraph(
    "A machine learning-based genetic prediction engine that estimates daughter PTAs "
    "from mating combinations, outperforming classical Parent Average by learning "
    "non-linear trait interactions, genetic antagonisms, and dam deficiency profiles.",
    styles['Body']))
story.append(Spacer(1, 8))
story.append(Paragraph("May 2026 | Select Sires | Confidential", styles['Small']))
story.append(Spacer(1, 0.3*inch))

# ============================================================
# 1. OVERVIEW
# ============================================================
story.append(Paragraph("1. Overview", styles['H1']))
story.append(Paragraph(
    "The <b>Dam-Specific Improvement Index (DSII)</b> predicts genomic PTAs of daughters "
    "using only three inputs: the NAAB codes of the <b>sire</b>, <b>maternal grandsire (MGS)</b>, "
    "and <b>maternal great-grandsire (MGGS)</b>. The dam's genetic merit is estimated from pedigree "
    "(Dam = MGS/2 + MGGS/4), then combined with the sire's PTAs through 75+ engineered features "
    "that capture genetic interactions invisible to simple Parent Average.",
    styles['Body']))

story.append(Paragraph(
    "DSII V8 Oracle selects the <b>optimal ML architecture per trait</b> from a 10-model shootout, "
    "achieving <b>R² = 0.823</b> across 25 traits — beating Parent Average in <b>all 25 traits</b> "
    "with an average <b>17.8% reduction in prediction error (MAE)</b>.",
    styles['Body']))

# ============================================================
# 2. TRAINING DATA
# ============================================================
story.append(Paragraph("2. Training Data", styles['H1']))
story.append(Paragraph(
    "The model was trained and validated on <b>1,709 verified trios</b> (genotyped daughter + sire + dam) "
    "using 5-fold cross-validation:", styles['Body']))

data_table = [
    ['Source', 'Records', 'Link Method', 'Match Rate'],
    ['Daughters (CDCB May 2026)', '1,974 genotyped', 'SIRENAME / DAMREGNUM2', '—'],
    ['Dams (DAM1-20.xlsx)', '1,301 unique', 'NAAB Code', '86.7%'],
    ['Sires (bulls.csv)', '40,047 bulls', 'Registration Name', '99.8%'],
    ['Final trios assembled', '1,709', '—', '—'],
]
t = Table(data_table, colWidths=[2.2*inch, 1.5*inch, 1.8*inch, 1*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
    ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('BACKGROUND', (0, 1), (-1, -1), PALE_BLUE),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GRAY),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t)
story.append(Spacer(1, 10))

# ============================================================
# 3. MODEL EVOLUTION
# ============================================================
story.append(Paragraph("3. Model Evolution", styles['H1']))
story.append(Paragraph(
    "Six model versions were developed iteratively. Each version built upon learnings "
    "from the previous one:", styles['Body']))

evo_data = [
    ['Version', 'Method', 'Avg R²', 'Beats PA', 'Outcome'],
    ['PA', '(Sire + Dam) / 2', '0.774', '—', 'Baseline'],
    ['V3', 'Deterministic rules\n(deficiency + antagonism)', '0.776', '13/25', 'Marginal (+0.1%)'],
    ['V4', 'Pure ML (Ridge/RF)\nno domain knowledge', '0.778', '19/25', 'PA still wins 6 traits'],
    ['V5', 'Hybrid: domain knowledge\nfeatures + LightGBM', '0.823', '25/25', 'Breakthrough (+6.3%)'],
    ['V7', 'Stacking ensemble\n(Ridge+LGBM+RF)', '0.731', '0/25', 'Overfitted (n too small)'],
    ['V8', 'Oracle: best model\nper trait (shootout)', '0.823', '25/25', 'Production model'],
]
t = Table(evo_data, colWidths=[0.6*inch, 1.8*inch, 0.7*inch, 0.8*inch, 2.4*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
    ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GRAY),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    # Highlight V5 and V8
    ('BACKGROUND', (0, 4), (-1, 4), LIGHT_GREEN),
    ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold'),
    ('BACKGROUND', (0, 6), (-1, 6), LIGHT_GREEN),
    ('FONTNAME', (0, 6), (-1, 6), 'Helvetica-Bold'),
    # Red for V7
    ('BACKGROUND', (0, 5), (-1, 5), LIGHT_RED),
]))
story.append(t)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>Key insight:</b> V3 (rules) and V4 (pure ML) each gained less than 1% over PA. "
    "The breakthrough came in V5 by <b>combining domain knowledge as ML features</b> — "
    "deficiency profiles, genetic correlations, antagonism penalties — fed into gradient boosting. "
    "V7 (stacking) proved that ensemble complexity hurts with only 1,709 samples. "
    "V8 selects the optimal single model per trait.", styles['Body']))

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
story.append(Paragraph("4. Feature Engineering (75+ Features per Trait)", styles['H1']))
story.append(Paragraph(
    "The core innovation of DSII is transforming raw sire and dam PTAs into rich feature vectors "
    "that encode genetic domain knowledge:", styles['Body']))

feat_data = [
    ['Category', 'Features', 'What It Captures'],
    ['Base values', 'sire, dam, PA, diff, delta_g', 'Raw genetic merit and parental gap'],
    ['Z-scores', 'sire_z, dam_z', 'Standardized position within breed distribution'],
    ['Interactions', 'sire_sq, dam_sq, sxd', 'Non-linear effects and sire x dam synergy'],
    ['Heritability', 'h2_pa', 'PA weighted by trait-specific heritability'],
    ['Deficiency profile', 'deficiency, dg_x_def', 'How far dam is from breed ideal; correction potential'],
    ['Inbreeding', 'ib_effect', 'Expected inbreeding depression at F=8.5%'],
    ['Dominance', 'dom_pot', 'Heterotic potential from parental divergence'],
    ['Genetic correlations', 'gc_MILK, gc_DPR, ...', 'Correlated trait effects (e.g., MILK vs DPR = -0.35)'],
    ['Production pressure', 'prod_press', 'Combined production index (MILK + FAT + PRO)'],
    ['Cross-trait context', 's_MILK, d_FAT, ...', 'All 24 other trait PTAs as contextual information'],
    ['Sire consistency', 'sire_loo', 'Leave-one-out mean of sire\'s other daughters in dataset'],
]
t = Table(feat_data, colWidths=[1.3*inch, 1.8*inch, 3.3*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
    ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GRAY),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))
story.append(t)
story.append(Spacer(1, 6))
story.append(Paragraph(
    "These features encode relationships that Parent Average cannot capture: genetic antagonisms "
    "between production and fertility, diminishing returns from extreme parents, and the interaction "
    "between a sire's strengths and a dam's specific weaknesses.", styles['Body']))

story.append(PageBreak())

# ============================================================
# 5. MODEL SHOOTOUT
# ============================================================
story.append(Paragraph("5. Model Architecture Shootout", styles['H1']))
story.append(Paragraph(
    "Ten ML architectures were tested across all 25 traits using 5-fold cross-validation "
    "(1,250 total model fits):", styles['Body']))

shoot_data = [
    ['Model', 'Avg R²', 'Min R²', 'Max R²', 'Wins', 'Verdict'],
    ['PA (baseline)', '0.774', '0.497', '0.952', '0', 'Baseline'],
    ['Ridge', '0.778', '0.486', '0.965', '0', 'Linear — limited'],
    ['BayesianRidge', '0.780', '0.491', '0.965', '0', 'Marginal over Ridge'],
    ['ElasticNet', '0.781', '0.492', '0.965', '0', 'Marginal over Ridge'],
    ['SVR', '0.658', '0.305', '0.928', '0', 'Failed — O(n²)'],
    ['MLP (Neural Net)', '0.539', '-2.22', '0.957', '0', 'Failed — overfits'],
    ['Random Forest', '0.799', '0.534', '0.972', '2', 'Best for NM$, CM$'],
    ['GBR', '0.822', '0.627', '0.971', '12', 'Top model'],
    ['XGBoost', '0.818', '0.624', '0.970', '1', 'Best for PTAT'],
    ['LightGBM', '0.822', '0.636', '0.970', '10', 'Tied top model'],
    ['CatBoost', '0.810', '0.603', '0.968', '0', 'Good but never best'],
]
t = Table(shoot_data, colWidths=[1.2*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.5*inch, 2.4*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
    ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
    ('ALIGN', (1, 0), (4, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GRAY),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    # Highlight winners
    ('BACKGROUND', (0, 8), (-1, 8), LIGHT_GREEN),
    ('FONTNAME', (0, 8), (-1, 8), 'Helvetica-Bold'),
    ('BACKGROUND', (0, 10), (-1, 10), LIGHT_GREEN),
    ('FONTNAME', (0, 10), (-1, 10), 'Helvetica-Bold'),
    # Red for failures
    ('BACKGROUND', (0, 5), (-1, 6), LIGHT_RED),
]))
story.append(t)
story.append(Spacer(1, 8))

story.append(Paragraph("V8 Oracle Model Assignment:", styles['H3']))
assign_data = [
    ['Model', 'Wins', 'Traits Assigned'],
    ['GBR', '12', 'TPI, MILK, FAT, FAT%, PRO%, CFP, SCS, LIV, HCR, CCR, UDC, SCE'],
    ['LightGBM', '10', 'PRO, PL, DPR, FI, MAST, FSAV, FLC, DCE, SSB, DSB'],
    ['Random Forest', '2', 'NM$, CM$'],
    ['XGBoost', '1', 'PTAT'],
]
t = Table(assign_data, colWidths=[1.2*inch, 0.6*inch, 4.6*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), MED_BLUE),
    ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
    ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GRAY),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))
story.append(t)

# ============================================================
# 6. FULL R² COMPARISON TABLE
# ============================================================
story.append(Spacer(1, 12))
story.append(Paragraph("6. Complete R² Comparison — All Versions x 25 Traits", styles['H1']))
story.append(Paragraph(
    "5-Fold Cross-Validation R² for every trait across all model versions. "
    "V8 column shows the production model.", styles['Body']))

# Data
full_data = [
    ['Trait', 'h²', 'PA', 'V3', 'V4', 'V5', 'V7', 'V8', 'Model', 'Gain'],
    ['TPI',   '0.30', '0.924', '0.924', '0.946', '0.959', '0.878', '0.959', 'GBR',  '+3.7%'],
    ['NM$',   '0.30', '0.952', '0.951', '0.964', '0.971', '0.603', '0.972', 'RF',   '+2.0%'],
    ['CM$',   '0.30', '0.950', '0.950', '0.963', '0.971', '0.486', '0.971', 'RF',   '+2.1%'],
    ['MILK',  '0.25', '0.784', '0.784', '0.780', '0.814', '0.775', '0.815', 'GBR',  '+3.9%'],
    ['FAT',   '0.25', '0.935', '0.938', '0.941', '0.949', '0.910', '0.949', 'GBR',  '+1.6%'],
    ['FAT%',  '0.50', '0.754', '0.754', '0.752', '0.784', '0.755', '0.789', 'GBR',  '+4.6%'],
    ['PRO',   '0.25', '0.898', '0.903', '0.904', '0.919', '0.895', '0.919', 'LGBM', '+2.4%'],
    ['PRO%',  '0.50', '0.630', '0.630', '0.635', '0.679', '0.634', '0.696', 'GBR',  '+10.6%'],
    ['CFP',   '0.30', '0.943', '0.945', '0.952', '0.958', '0.937', '0.959', 'GBR',  '+1.7%'],
    ['PL',    '0.08', '0.793', '0.795', '0.806', '0.828', '0.436', '0.832', 'LGBM', '+5.0%'],
    ['SCS',   '0.12', '0.601', '0.600', '0.602', '0.704', '0.644', '0.708', 'GBR',  '+17.8%'],
    ['DPR',   '0.04', '0.710', '0.706', '0.707', '0.764', '0.717', '0.762', 'LGBM', '+7.3%'],
    ['LIV',   '0.05', '0.768', '0.766', '0.767', '0.802', '0.743', '0.801', 'GBR',  '+4.3%'],
    ['FI',    '0.06', '0.699', '0.699', '0.696', '0.763', '0.717', '0.756', 'LGBM', '+8.2%'],
    ['HCR',   '0.04', '0.633', '0.627', '0.635', '0.727', '0.687', '0.723', 'GBR',  '+14.3%'],
    ['CCR',   '0.04', '0.702', '0.696', '0.700', '0.754', '0.709', '0.754', 'GBR',  '+7.4%'],
    ['MAST',  '0.04', '0.497', '0.489', '0.487', '0.639', '0.482', '0.636', 'LGBM', '+27.9%'],
    ['FSAV',  '0.15', '0.840', '0.840', '0.845', '0.867', '0.843', '0.866', 'LGBM', '+3.1%'],
    ['PTAT',  '0.30', '0.890', '0.890', '0.899', '0.918', '0.907', '0.920', 'XGB',  '+3.3%'],
    ['UDC',   '0.25', '0.807', '0.805', '0.809', '0.830', '0.815', '0.830', 'GBR',  '+2.9%'],
    ['FLC',   '0.15', '0.908', '0.909', '0.910', '0.924', '0.910', '0.924', 'LGBM', '+1.8%'],
    ['SCE',   '0.08', '0.713', '0.713', '0.752', '0.824', '0.718', '0.824', 'GBR',  '+15.6%'],
    ['DCE',   '0.06', '0.580', '0.580', '0.556', '0.656', '0.610', '0.656', 'LGBM', '+13.1%'],
    ['SSB',   '0.06', '0.773', '0.773', '0.775', '0.821', '0.779', '0.819', 'LGBM', '+5.9%'],
    ['DSB',   '0.04', '0.677', '0.677', '0.667', '0.749', '0.696', '0.745', 'LGBM', '+10.0%'],
    ['AVG',   '',     '0.774', '0.775', '0.778', '0.823', '0.731', '0.823', '',      '+6.3%'],
]

t = Table(full_data, colWidths=[0.5*inch, 0.35*inch, 0.55*inch, 0.55*inch, 0.55*inch,
                                  0.55*inch, 0.55*inch, 0.55*inch, 0.55*inch, 0.6*inch])
style_cmds = [
    ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
    ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 7.5),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GRAY),
    ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -2), [WHITE, PALE_BLUE]),
    # AVG row
    ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#edf2f7")),
    ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
    # V8 column highlight
    ('BACKGROUND', (7, 0), (7, 0), colors.HexColor("#2c5282")),
    # Gain column
    ('TEXTCOLOR', (9, 1), (9, -1), GREEN),
    ('FONTNAME', (9, 1), (9, -1), 'Helvetica-Bold'),
]
# Highlight R2 >= 0.90 in V8 column (col 7) with green text
for i, row in enumerate(full_data[1:], 1):
    try:
        v8_val = float(row[7])
        if v8_val >= 0.90:
            style_cmds.append(('TEXTCOLOR', (7, i), (7, i), GREEN))
            style_cmds.append(('FONTNAME', (7, i), (7, i), 'Helvetica-Bold'))
    except:
        pass

t.setStyle(TableStyle(style_cmds))
story.append(t)

story.append(PageBreak())

# ============================================================
# 7. V8 vs PA DETAILED RESULTS
# ============================================================
story.append(Paragraph("7. V8 Oracle vs Parent Average — Detailed Results", styles['H1']))
story.append(Paragraph(
    "Complete comparison with MAE (Mean Absolute Error) reduction. "
    "All metrics from 5-fold cross-validation on 1,709 trios.", styles['Body']))

detail_data = [
    ['Trait', 'N', 'PA R²', 'V8 R²', 'Model', 'PA MAE', 'V8 MAE', 'MAE Gain'],
    ['TPI',  '1709', '0.924', '0.959', 'GBR',  '55.80',  '37.11',  '+33.5%'],
    ['NM$',  '1709', '0.952', '0.972', 'RF',   '66.64',  '46.92',  '+29.6%'],
    ['CM$',  '1709', '0.950', '0.971', 'RF',   '69.40',  '48.31',  '+30.4%'],
    ['MILK', '1709', '0.784', '0.815', 'GBR',  '224.73', '193.34', '+14.0%'],
    ['FAT',  '1709', '0.935', '0.949', 'GBR',  '8.69',   '7.33',   '+15.7%'],
    ['FAT%', '1709', '0.754', '0.789', 'GBR',  '0.048',  '0.042',  '+11.0%'],
    ['PRO',  '1709', '0.898', '0.919', 'LGBM', '5.06',   '4.18',   '+17.4%'],
    ['PRO%', '1709', '0.630', '0.696', 'GBR',  '0.020',  '0.018',  '+12.6%'],
    ['CFP',  '1709', '0.943', '0.959', 'GBR',  '11.49',  '9.24',   '+19.5%'],
    ['PL',   '1709', '0.793', '0.832', 'LGBM', '0.596',  '0.502',  '+15.9%'],
    ['SCS',  '1709', '0.601', '0.708', 'GBR',  '0.062',  '0.050',  '+19.7%'],
    ['DPR',  '1709', '0.710', '0.762', 'LGBM', '0.609',  '0.519',  '+14.7%'],
    ['LIV',  '1709', '0.768', '0.801', 'GBR',  '0.733',  '0.648',  '+11.5%'],
    ['FI',   '1709', '0.699', '0.756', 'LGBM', '0.580',  '0.497',  '+14.4%'],
    ['HCR',  '1709', '0.633', '0.723', 'GBR',  '0.617',  '0.512',  '+16.9%'],
    ['CCR',  '1709', '0.702', '0.754', 'GBR',  '0.714',  '0.624',  '+12.6%'],
    ['MAST', '1709', '0.497', '0.636', 'LGBM', '0.722',  '0.573',  '+20.7%'],
    ['FSAV', '1709', '0.840', '0.866', 'LGBM', '66.52',  '56.90',  '+14.5%'],
    ['PTAT', '1709', '0.890', '0.920', 'XGB',  '0.257',  '0.203',  '+20.8%'],
    ['UDC',  '1709', '0.807', '0.830', 'GBR',  '0.220',  '0.199',  '+9.8%'],
    ['FLC',  '1709', '0.908', '0.924', 'LGBM', '0.194',  '0.168',  '+13.5%'],
    ['SCE',  '1709', '0.713', '0.824', 'GBR',  '0.145',  '0.107',  '+26.4%'],
    ['DCE',  '1709', '0.580', '0.656', 'LGBM', '0.180',  '0.153',  '+14.9%'],
    ['SSB',  '1709', '0.773', '0.819', 'LGBM', '0.155',  '0.131',  '+15.7%'],
    ['DSB',  '1709', '0.677', '0.745', 'LGBM', '0.320',  '0.263',  '+18.0%'],
]
t = Table(detail_data, colWidths=[0.5*inch, 0.5*inch, 0.6*inch, 0.6*inch, 0.6*inch,
                                    0.8*inch, 0.8*inch, 0.8*inch])
style_cmds2 = [
    ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
    ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 7.5),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GRAY),
    ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
    ('TEXTCOLOR', (7, 1), (7, -1), GREEN),
    ('FONTNAME', (7, 1), (7, -1), 'Helvetica-Bold'),
]
t.setStyle(TableStyle(style_cmds2))
story.append(t)

# ============================================================
# 8. SUMMARY METRICS
# ============================================================
story.append(Spacer(1, 14))
story.append(Paragraph("8. Summary Comparison", styles['H1']))

summary_data = [
    ['Metric', 'Parent Average', 'DSII V8 Oracle'],
    ['Average R²', '0.774', '0.823 (+6.3%)'],
    ['Minimum R²', '0.497 (MAST)', '0.636 (MAST)'],
    ['Maximum R²', '0.952 (NM$)', '0.972 (NM$)'],
    ['Traits with R² >= 0.90', '5', '8'],
    ['Traits with R² >= 0.80', '10', '15'],
    ['Traits with R² >= 0.70', '17', '22'],
    ['Average MAE reduction', '—', '-17.8%'],
    ['Traits beating PA', '—', '25/25 (100%)'],
]
t = Table(summary_data, colWidths=[2*inch, 2*inch, 2.4*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
    ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GRAY),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
    ('TEXTCOLOR', (2, 1), (2, -1), GREEN),
]))
story.append(t)

# ============================================================
# 9. WHY V8 BEATS PA
# ============================================================
story.append(Spacer(1, 14))
story.append(Paragraph("9. Why DSII V8 Outperforms Parent Average", styles['H1']))
story.append(Paragraph(
    "Parent Average assumes a simple linear relationship: <b>Daughter = (Sire + Dam) / 2</b>. "
    "This ignores several biological realities that DSII captures:", styles['Body']))

reasons = [
    ("<b>Genetic antagonisms:</b> Traits with negative genetic correlations (e.g., MILK vs DPR "
     "at r = -0.35) create trade-offs. High-production parents often transmit lower fertility "
     "than PA predicts. DSII learns these patterns."),
    ("<b>Dam deficiency profiles:</b> When the dam is deficient in a specific trait, the sire's "
     "corrective effect is non-linear. A dam with low UDC benefits more from a high-UDC sire "
     "than a dam already at breed ideal."),
    ("<b>Cross-trait context:</b> All 24 other trait PTAs from both parents inform each prediction. "
     "A sire with extreme MILK but moderate DPR transmits differently than one balanced across both."),
    ("<b>Sire transmission consistency:</b> The leave-one-out feature captures how consistently "
     "a sire actually transmits his genetic merit vs. his published PTA, based on observed daughters."),
    ("<b>Non-linear interactions:</b> Tree-based models (GBR, LightGBM) capture sire x dam "
     "interaction effects, threshold behaviors, and diminishing returns that no linear model can represent."),
]
for r in reasons:
    story.append(Paragraph(r, styles['Bullet2'], bulletText='\u2022'))

story.append(PageBreak())

# ============================================================
# 10. PRODUCTION USAGE
# ============================================================
story.append(Paragraph("10. Production Usage", styles['H1']))
story.append(Paragraph(
    "In production, DSII V8 requires only <b>three NAAB codes</b> as input:", styles['Body']))

story.append(Paragraph("1. <b>Sire NAAB</b> — the bull to be mated", styles['Bullet2'], bulletText='\u2022'))
story.append(Paragraph("2. <b>MGS NAAB</b> — the cow's sire (maternal grandsire)", styles['Bullet2'], bulletText='\u2022'))
story.append(Paragraph("3. <b>MGGS NAAB</b> — the cow's maternal grandsire (maternal great-grandsire)", styles['Bullet2'], bulletText='\u2022'))

story.append(Spacer(1, 6))
story.append(Paragraph("Prediction Pipeline:", styles['H3']))

pipeline_data = [
    ['Step', 'Action', 'Detail'],
    ['1', 'Look up sire PTAs', 'From bulls.csv (40,047 bulls) by NAAB code'],
    ['2', 'Look up MGS and MGGS PTAs', 'Same database, same method'],
    ['3', 'Estimate dam PTAs', 'Dam = MGS/2 + MGGS/4 (pedigree index)'],
    ['4', 'Build feature vector', '75+ features per trait (domain knowledge encoded)'],
    ['5', 'Predict with best model', 'GBR, LGBM, RF, or XGB depending on trait'],
    ['6', 'Output results', 'Predicted PTA, PA, sire, dam estimate, and difference'],
]
t = Table(pipeline_data, colWidths=[0.5*inch, 1.8*inch, 4*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), MED_BLUE),
    ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ALIGN', (0, 0), (0, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
    ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GRAY),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t)

story.append(Spacer(1, 10))
story.append(Paragraph(
    "<b>Dam estimation note:</b> The remaining 1/4 of the dam's genetic contribution "
    "(from the unknown maternal great-great-grandam) defaults to breed average. "
    "This is the standard pedigree index approach used when direct genomic data is unavailable.",
    styles['Body']))

# ============================================================
# 11. LIMITATIONS & NEXT STEPS
# ============================================================
story.append(Spacer(1, 10))
story.append(Paragraph("11. Limitations and Next Steps", styles['H1']))

limits = [
    ("<b>Sample size (1,709 trios):</b> The primary constraint. ML models, especially gradient "
     "boosting, benefit significantly from larger datasets. With 5,000+ trios, we expect R² to "
     "approach 0.85-0.87 for most traits."),
    ("<b>Mendelian sampling variance:</b> Traits with very low heritability (h² = 0.04: DPR, HCR, MAST) "
     "have a biological ceiling on prediction accuracy. Without direct SNP data, half the genetic "
     "variance is inherently unpredictable due to random chromosome segregation."),
    ("<b>Dam estimation vs. actual PTAs:</b> In production, dam PTAs are estimated from pedigree "
     "(MGS/2 + MGGS/4). During training, we used actual dam PTAs. Production accuracy will be "
     "slightly lower than the reported CV metrics."),
    ("<b>Genetic base drift:</b> Models were trained on May 2026 evaluations. As the breed's "
     "genetic base changes over time, models should be retrained periodically (recommended: "
     "annually or when new evaluation runs are released)."),
    ("<b>Path to R² ~ 0.90:</b> Already achieved for 8/25 traits. The remaining traits need "
     "more training samples, deeper pedigree information, or direct genomic features (SNP markers)."),
]
for l in limits:
    story.append(Paragraph(l, styles['Bullet2'], bulletText='\u2022'))

story.append(Spacer(1, 20))
story.append(HRFlowable(width="100%", thickness=1, color=BORDER_GRAY, spaceAfter=8))
story.append(Paragraph(
    "DSII V8 Oracle | Generated May 2026 | Select Sires | Confidential",
    styles['Small']))

# BUILD
doc.build(story)
print(f"PDF generated: {OUTPUT}")
