"""
Gera relatorio completo de todas as femeas por cada touro.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rodar_predicao_v2 import (
    load_sires, load_dams, predict_mating, DOWNLOADS, OUTPUT_DIR,
    GENETIC_SD
)

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    sires = load_sires(os.path.join(DOWNLOADS, "Lista de touros.xlsx"))
    dams = load_dams(os.path.join(DOWNLOADS, "Lista de f\u00eameas.xlsx"))

    # Media NM$ das femeas
    all_nm = [d.nm_dollar for d in dams]
    pop_avg = sum(all_nm) / len(all_nm)
    pop_sd = max((sum((x - pop_avg)**2 for x in all_nm) / (len(all_nm)-1))**0.5, 100)

    # Roda todas as predicoes
    all_preds = {}
    for s in sires:
        all_preds[s.id] = []
        for d in dams:
            p = predict_mating(s, d, pop_avg, pop_sd)
            all_preds[s.id].append(p)
        all_preds[s.id].sort(key=lambda x: x.final_score, reverse=True)
        for i, p in enumerate(all_preds[s.id]):
            p.rank = i + 1

    # Relatorio por touro
    for s in sires:
        preds = all_preds[s.id]
        print()
        print("=" * 150)
        print(f"  TOURO: {s.name} ({s.naab})  |  TPI={s.tpi:.0f}  |  NM$={s.nm_dollar:.0f}  |  "
              f"Milk={s.ptas['MILK']:.0f}  Fat={s.ptas['FAT']:.0f}  Prot={s.ptas['PROT']:.0f}  |  "
              f"DPR={s.ptas['DPR']:+.1f}  CCR={s.ptas['CCR']:+.1f}  PL={s.ptas['PL']:.1f}  |  "
              f"UDC={s.ptas['UDC']:+.2f}  FLC={s.ptas['FLC']:+.2f}  |  "
              f"Beta={s.beta_casein}  Kappa={s.kappa_casein}  |  Pai={s.sire}  MGS={s.mgs}")
        print("=" * 150)

        hdr = (
            f"{'Rk':>3} | {'Femea':<22} | {'Reg':>5} | {'TPIf':>4} | {'NM$f':>5} | "
            f"{'NM$PA':>6} | {'NM$Cor':>6} | {'IC95-':>6} | {'IC95+':>6} | "
            f"{'F%':>5} | {'P>Med':>5} | {'PT25':>4} | {'PT10':>4} | "
            f"{'Comp':>4} | {'Bal':>4} | {'Score':>5} | "
            f"{'Milk':>6} | {'Fat':>5} | {'Prot':>4} | {'SCS':>4} | "
            f"{'DPR':>5} | {'CCR':>5} | {'PL':>4} | {'LIV':>4} | "
            f"{'UDC':>5} | {'FLC':>5} | {'BWC':>5} | "
            f"{'BetaCas':>13} | {'KapCas':>12}"
        )
        print(hdr)
        print("-" * 150)

        for p in preds:
            print(
                f"{p.rank:>3} | "
                f"{p.dam.name:<22.22} | "
                f"{p.dam.reg:>5.5} | "
                f"{p.dam.tpi:>4.0f} | "
                f"${p.dam.nm_dollar:>4.0f} | "
                f"${p.nm_pred:>5.0f} | "
                f"${p.nm_corrected:>5.0f} | "
                f"${p.nm_ic_lower:>5.0f} | "
                f"${p.nm_ic_upper:>5.0f} | "
                f"{p.expected_f*100:>4.1f}% | "
                f"{p.prob_above_avg*100:>4.0f}% | "
                f"{p.prob_top25*100:>3.0f}% | "
                f"{p.prob_top10*100:>3.0f}% | "
                f"{p.complementarity:>4.0f} | "
                f"{p.balance_score:>4.0f} | "
                f"{p.final_score:>5.1f} | "
                f"{p.trait_pred.get('MILK',0):>6.0f} | "
                f"{p.trait_pred.get('FAT',0):>5.1f} | "
                f"{p.trait_pred.get('PROT',0):>4.0f} | "
                f"{p.trait_pred.get('SCS',0):>4.2f} | "
                f"{p.trait_pred.get('DPR',0):>+5.1f} | "
                f"{p.trait_pred.get('CCR',0):>+5.1f} | "
                f"{p.trait_pred.get('PL',0):>4.1f} | "
                f"{p.trait_pred.get('LIV',0):>+4.1f} | "
                f"{p.trait_pred.get('UDC',0):>+5.2f} | "
                f"{p.trait_pred.get('FLC',0):>+5.2f} | "
                f"{p.trait_pred.get('BWC',0):>+5.2f} | "
                f"{p.beta_info[:13]:>13} | "
                f"{p.kappa_info[:12]:>12}"
            )

        # Resumo do touro
        scores = [p.final_score for p in preds]
        nms = [p.nm_corrected for p in preds]
        print("-" * 150)
        print(f"  Resumo {s.name}: Score min={min(scores):.1f} max={max(scores):.1f} med={sum(scores)/len(scores):.1f}  |  "
              f"NM$ min=${min(nms):.0f} max=${max(nms):.0f} med=${sum(nms)/len(nms):.0f}  |  "
              f"Femeas avaliadas: {len(preds)}")

    # Export CSV por touro
    csv_path = os.path.join(OUTPUT_DIR, "predicao_completa_por_touro.csv")
    fields = [
        "Touro_NAAB", "Touro_Nome", "Touro_TPI", "Touro_NM$",
        "Rank_no_Touro",
        "Femea_NAAB", "Femea_Num", "Femea_Reg", "Femea_TPI", "Femea_NM$",
        "NM$_PA", "NM$_Corrigido", "NM$_IC_Lower", "NM$_IC_Upper",
        "TPI_PA", "F_Esperada_%", "Depressao_$",
        "P_Acima_Media_%", "P_Top25_%", "P_Top10_%",
        "Complementaridade", "Balanceamento", "Score_Final",
        "PA_MILK", "PA_FAT", "PA_PROT", "PA_SCS",
        "PA_DPR", "PA_CCR", "PA_PL", "PA_LIV",
        "PA_HCR", "PA_EFC", "PA_HLIV",
        "PA_UDC", "PA_FLC", "PA_BWC",
        "IC_Low_MILK", "IC_High_MILK",
        "IC_Low_FAT", "IC_High_FAT",
        "IC_Low_PROT", "IC_High_PROT",
        "Haplo_Conflitos", "Beta_Caseina", "Kappa_Caseina",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        w.writeheader()
        for s in sires:
            for p in all_preds[s.id]:
                row = {
                    "Touro_NAAB": s.naab, "Touro_Nome": s.name,
                    "Touro_TPI": f"{s.tpi:.0f}", "Touro_NM$": f"{s.nm_dollar:.0f}",
                    "Rank_no_Touro": p.rank,
                    "Femea_NAAB": p.dam.naab, "Femea_Num": p.dam.num,
                    "Femea_Reg": p.dam.reg, "Femea_TPI": f"{p.dam.tpi:.0f}",
                    "Femea_NM$": f"{p.dam.nm_dollar:.0f}",
                    "NM$_PA": f"{p.nm_pred:.0f}",
                    "NM$_Corrigido": f"{p.nm_corrected:.0f}",
                    "NM$_IC_Lower": f"{p.nm_ic_lower:.0f}",
                    "NM$_IC_Upper": f"{p.nm_ic_upper:.0f}",
                    "TPI_PA": f"{p.tpi_pred:.0f}",
                    "F_Esperada_%": f"{p.expected_f*100:.2f}",
                    "Depressao_$": f"{p.endo_depression_nm:.0f}",
                    "P_Acima_Media_%": f"{p.prob_above_avg*100:.1f}",
                    "P_Top25_%": f"{p.prob_top25*100:.1f}",
                    "P_Top10_%": f"{p.prob_top10*100:.1f}",
                    "Complementaridade": f"{p.complementarity:.1f}",
                    "Balanceamento": f"{p.balance_score:.1f}",
                    "Score_Final": f"{p.final_score:.1f}",
                    "Haplo_Conflitos": ",".join(p.haplo_conflicts) if p.haplo_conflicts else "",
                    "Beta_Caseina": p.beta_info,
                    "Kappa_Caseina": p.kappa_info,
                }
                for t in ["MILK","FAT","PROT","SCS","DPR","CCR","PL","LIV","HCR","EFC","HLIV","UDC","FLC","BWC"]:
                    row[f"PA_{t}"] = f"{p.trait_pred.get(t,0):.1f}"
                for t in ["MILK","FAT","PROT"]:
                    row[f"IC_Low_{t}"] = f"{p.trait_ic_low.get(t,0):.1f}"
                    row[f"IC_High_{t}"] = f"{p.trait_ic_high.get(t,0):.1f}"
                w.writerow(row)

    print(f"\n[OK] CSV completo por touro: {csv_path}")


if __name__ == "__main__":
    main()
