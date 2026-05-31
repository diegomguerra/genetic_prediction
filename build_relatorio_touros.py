"""
Gera relatório HTML completo dos touros com gráficos e comentários técnicos.
"""
import csv, json, os

CSV_PATH = os.path.join(os.path.dirname(__file__), "predicao_v3_dsii.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "relatorio_touros.html")

with open(CSV_PATH, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

json_data = json.dumps(rows, ensure_ascii=False)

HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatório Técnico — Análise dos Touros | DSII v3</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/jspdf@2.5.2/dist/jspdf.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
:root {
  --green-dark: #1B5E20;
  --green: #2E7D32;
  --green-light: #43A047;
  --green-bg: #E8F5E9;
  --gold: #F9A825;
  --navy: #0D253F;
  --blue: #1565C0;
  --red: #C62828;
  --white: #FFFFFF;
  --g50: #FAFAFA; --g100: #F5F5F5; --g200: #EEEEEE; --g300: #E0E0E0;
  --g400: #BDBDBD; --g500: #9E9E9E; --g600: #757575; --g700: #616161;
  --g800: #424242; --g900: #212121;
  --shadow: 0 2px 8px rgba(0,0,0,0.10);
  --shadow-lg: 0 4px 24px rgba(0,0,0,0.12);
  --r: 10px;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Inter',sans-serif; background:#f0f2f5; color:var(--g900); line-height:1.6; }

/* Navbar */
.navbar {
  background: linear-gradient(135deg, var(--navy), var(--green-dark));
  padding: 0 40px; height: 60px;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 100;
  box-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.navbar h1 { color: var(--white); font-size: 18px; font-weight: 800; letter-spacing: -0.5px; }
.navbar-actions { display: flex; gap: 8px; }
.btn {
  padding: 8px 18px; border: none; border-radius: 6px;
  font-family: inherit; font-size: 12px; font-weight: 700;
  cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
  transition: all 0.2s;
}
.btn-gold { background: var(--gold); color: var(--navy); }
.btn-gold:hover { filter: brightness(1.1); }
.btn-green { background: var(--green); color: var(--white); }
.btn-green:hover { background: var(--green-light); }

/* Container */
.container { max-width: 1200px; margin: 0 auto; padding: 32px 24px 80px; }

/* Header do relatório */
.report-header {
  text-align: center; padding: 40px 20px; margin-bottom: 32px;
  background: linear-gradient(135deg, var(--navy) 0%, #1a3a5c 50%, var(--green-dark) 100%);
  border-radius: var(--r); color: var(--white);
  box-shadow: var(--shadow-lg);
}
.report-header h2 { font-size: 28px; font-weight: 900; letter-spacing: -1px; }
.report-header p { font-size: 14px; opacity: 0.8; margin-top: 6px; }
.report-header .meta { font-size: 11px; opacity: 0.6; margin-top: 12px; }

/* Sumário Executivo */
.summary-box {
  background: var(--white); border-radius: var(--r); padding: 28px;
  box-shadow: var(--shadow); margin-bottom: 28px;
  border-left: 5px solid var(--green);
}
.summary-box h3 {
  font-size: 16px; font-weight: 800; color: var(--navy); margin-bottom: 12px;
  padding-bottom: 8px; border-bottom: 2px solid var(--g200);
}
.summary-box p { font-size: 13px; color: var(--g800); margin-bottom: 8px; }
.summary-box ul { margin-left: 20px; margin-bottom: 8px; }
.summary-box li { font-size: 13px; color: var(--g800); margin-bottom: 4px; }
.summary-box .highlight { color: var(--green-dark); font-weight: 700; }
.summary-box .alert { color: var(--red); font-weight: 700; }

/* Section */
.section {
  background: var(--white); border-radius: var(--r);
  box-shadow: var(--shadow); margin-bottom: 28px; overflow: hidden;
}
.section-head {
  padding: 18px 24px; border-bottom: 1px solid var(--g200);
  display: flex; align-items: center; justify-content: space-between;
}
.section-head h3 { font-size: 16px; font-weight: 800; color: var(--navy); }
.section-body { padding: 24px; }

/* Charts */
.charts-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.chart-box {
  background: var(--white); border-radius: var(--r); padding: 20px;
  box-shadow: var(--shadow); border: 1px solid var(--g200);
}
.chart-box.full { grid-column: 1 / -1; }
.chart-box h4 { font-size: 13px; font-weight: 700; color: var(--navy); margin-bottom: 2px; }
.chart-box .sub { font-size: 11px; color: var(--g500); margin-bottom: 14px; }

/* Bull Report Card */
.bull-report {
  background: var(--white); border-radius: var(--r);
  box-shadow: var(--shadow); margin-bottom: 32px; overflow: hidden;
  border: 1px solid var(--g200);
  page-break-inside: avoid;
}
.bull-report-header {
  padding: 20px 28px; color: var(--white);
  display: flex; align-items: center; justify-content: space-between;
}
.bull-report-header h3 { font-size: 22px; font-weight: 900; }
.bull-report-header .rank-badge {
  background: rgba(255,255,255,0.2); padding: 6px 16px;
  border-radius: 20px; font-weight: 700; font-size: 13px;
}

/* KPI row inside bull */
.bull-kpis {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 1px; background: var(--g200);
}
.bull-kpi {
  background: var(--g50); padding: 14px 16px; text-align: center;
}
.bull-kpi .val { font-size: 20px; font-weight: 800; color: var(--navy); letter-spacing: -0.5px; }
.bull-kpi .lbl { font-size: 9px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--g500); font-weight: 700; }

/* Análise técnica */
.tech-analysis { padding: 24px 28px; }
.tech-analysis h4 {
  font-size: 14px; font-weight: 700; color: var(--navy); margin-bottom: 10px;
  display: flex; align-items: center; gap: 8px;
}
.tech-analysis h4 .badge {
  font-size: 10px; padding: 2px 8px; border-radius: 10px; font-weight: 700;
}
.tech-analysis h4 .badge-green { background: #C8E6C9; color: #1B5E20; }
.tech-analysis h4 .badge-red { background: #FFCDD2; color: #B71C1C; }
.tech-analysis h4 .badge-gold { background: #FFF9C4; color: #F57F17; }

.tech-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 16px; }
.tech-col { }
.tech-item {
  display: flex; gap: 8px; padding: 5px 0; font-size: 12.5px;
  color: var(--g800); line-height: 1.5;
}
.tech-icon {
  flex-shrink: 0; width: 18px; height: 18px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 800; margin-top: 2px;
}
.ti-green { background: #C8E6C9; color: #1B5E20; }
.ti-red { background: #FFCDD2; color: #B71C1C; }
.ti-yellow { background: #FFF9C4; color: #F57F17; }
.ti-blue { background: #BBDEFB; color: #0D47A1; }

.verdict-box {
  background: var(--g50); border: 1px solid var(--g200);
  border-radius: 8px; padding: 16px 20px; margin-top: 8px;
  font-size: 13px; color: var(--g800); line-height: 1.6;
}
.verdict-box strong { color: var(--navy); }

/* Tabela comparativa */
.comp-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.comp-table th {
  background: var(--navy); color: var(--white); padding: 8px 10px;
  text-align: center; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
}
.comp-table td {
  padding: 8px 10px; text-align: center; border-bottom: 1px solid var(--g200);
}
.comp-table tr:hover { background: var(--green-bg); }
.comp-table .bull-name { text-align: left; font-weight: 700; }
.best-val { background: #C8E6C9; font-weight: 700; color: #1B5E20; border-radius: 4px; }
.worst-val { background: #FFCDD2; font-weight: 700; color: #B71C1C; border-radius: 4px; }

/* Print */
@media print {
  .navbar, .btn, .navbar-actions { display: none !important; }
  body { background: white; }
  .bull-report, .section { break-inside: avoid; box-shadow: none; border: 1px solid #ccc; }
  .container { max-width: 100%; padding: 10px; }
}
</style>
</head>
<body>

<nav class="navbar">
  <h1>Relatório Técnico — Análise dos Touros</h1>
  <div class="navbar-actions">
    <button class="btn btn-gold" onclick="exportExcel()">Exportar Excel</button>
    <button class="btn btn-green" onclick="exportPDF()">Exportar PDF</button>
    <button class="btn btn-gold" onclick="window.print()">Imprimir</button>
  </div>
</nav>

<div class="container" id="report-content">

  <div class="report-header">
    <h2>Relatório de Análise Genética dos Touros</h2>
    <p>Predição DSII v3 — Dam-Specific Improvement Index</p>
    <div class="meta">276 combinações avaliadas (6 touros × 46 fêmeas) | Gerado em REPORT_DATE</div>
  </div>

  <div id="executive-summary"></div>

  <!-- Gráficos Comparativos -->
  <div class="section">
    <div class="section-head"><h3>Comparativo Geral entre Touros</h3></div>
    <div class="section-body">
      <div class="charts-2col">
        <div class="chart-box">
          <h4>Score Final Médio</h4>
          <div class="sub">Pontuação composta DSII v3 — considera mérito econômico, correção de deficiências, penalidades e bônus</div>
          <canvas id="c1"></canvas>
        </div>
        <div class="chart-box">
          <h4>NM$ Corrigido Médio</h4>
          <div class="sub">Valor econômico da progênie ajustado por depressão endogâmica</div>
          <canvas id="c2"></canvas>
        </div>
        <div class="chart-box">
          <h4>TPI Corrigido Médio</h4>
          <div class="sub">Índice Total de Performance — combina produção, saúde, fertilidade e tipo</div>
          <canvas id="c3"></canvas>
        </div>
        <div class="chart-box">
          <h4>DSII Médio (Capacidade de Correção)</h4>
          <div class="sub">Quanto maior, mais o touro corrige deficiências específicas de cada fêmea</div>
          <canvas id="c4"></canvas>
        </div>
        <div class="chart-box">
          <h4>Penalidade de Antagonismo Média</h4>
          <div class="sub">Penalidade por conflito produção × funcionalidade (correlações genéticas negativas)</div>
          <canvas id="c6"></canvas>
        </div>
        <div class="chart-box">
          <h4>Nº de Fêmeas Onde é o Melhor Touro</h4>
          <div class="sub">Quantidade de fêmeas para as quais este touro tem o maior Score Final</div>
          <canvas id="c7"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- Tabela Comparativa -->
  <div class="section">
    <div class="section-head"><h3>Tabela Comparativa Resumida</h3></div>
    <div class="section-body" style="padding:0; overflow-x:auto;">
      <table class="comp-table" id="comp-table"></table>
    </div>
  </div>

  <!-- Reports individuais por touro -->
  <div id="bull-reports"></div>

</div>

<script>
const DATA = %%DATA%%;

const BULLS = ['Commander','Betzold','POwered','Rosemary','Shimmy','Hendricks'];
const COLORS = {
  Commander:'#1565C0', Betzold:'#2E7D32', POwered:'#E65100',
  Rosemary:'#6A1B9A', Shimmy:'#C62828', Hendricks:'#4E342E'
};
const COLORS_L = {
  Commander:'rgba(21,101,192,0.15)', Betzold:'rgba(46,125,50,0.15)', POwered:'rgba(230,81,0,0.15)',
  Rosemary:'rgba(106,27,154,0.15)', Shimmy:'rgba(198,40,40,0.15)', Hendricks:'rgba(78,52,46,0.15)'
};

const N = v => parseFloat(v)||0;
const avg = a => a.length ? a.reduce((s,v)=>s+v,0)/a.length : 0;

// Group by bull
const byBull = {};
DATA.forEach(r => { if(!byBull[r.Touro_Nome]) byBull[r.Touro_Nome]=[]; byBull[r.Touro_Nome].push(r); });
const byFemea = {};
DATA.forEach(r => { if(!byFemea[r.Femea_Num]) byFemea[r.Femea_Num]=[]; byFemea[r.Femea_Num].push(r); });

// Stats per bull
const stats = {};
BULLS.forEach(b => {
  const rows = byBull[b]||[];
  const sc = rows.map(r=>N(r.Score_Final_v3)).sort((a,b)=>a-b);
  const nm = rows.map(r=>N(r['NM$_Corrigido']));
  const tp = rows.map(r=>N(r.TPI_Corrigido));
  const ds = rows.map(r=>N(r.DSII));
  let best1s=0, best1n=0;
  Object.values(byFemea).forEach(fr => {
    if(fr.reduce((a,c)=>N(a.Score_Final_v3)>N(c.Score_Final_v3)?a:c).Touro_Nome===b) best1s++;
    if(fr.reduce((a,c)=>N(a['NM$_Corrigido'])>N(c['NM$_Corrigido'])?a:c).Touro_Nome===b) best1n++;
  });
  stats[b] = {
    tpi: rows[0]?.Touro_TPI, nm_bull: rows[0]?.['Touro_NM$'], naab: rows[0]?.Touro_NAAB,
    sc_avg:avg(sc), sc_min:sc[0], sc_max:sc[sc.length-1],
    sc_q1:sc[Math.floor(sc.length*0.25)], sc_med:sc[Math.floor(sc.length*0.5)], sc_q3:sc[Math.floor(sc.length*0.75)],
    nm_avg:avg(nm), nm_min:Math.min(...nm), nm_max:Math.max(...nm),
    tp_avg:avg(tp), tp_min:Math.min(...tp), tp_max:Math.max(...tp),
    ds_avg:avg(ds), ds_min:Math.min(...ds), ds_max:Math.max(...ds),
    pCrit:avg(rows.map(r=>N(r.Penalidade_Critica))),
    pAnt:avg(rows.map(r=>N(r.Penalidade_Antagonismo))),
    dist:avg(rows.map(r=>N(r.Distancia_Ideal))),
    f:avg(rows.map(r=>N(r['F_Esperada_%']))),
    dep:avg(rows.map(r=>N(r['Depressao_$']))),
    pAbove:avg(rows.map(r=>N(r['P_Acima_Media_%']))),
    pT25:avg(rows.map(r=>N(r['P_Top25_%']))),
    pT10:avg(rows.map(r=>N(r['P_Top10_%']))),
    deltaNM:avg(rows.map(r=>N(r['Delta_NM$_sobre_Mae']))),
    deltaTPI:avg(rows.map(r=>N(r.Delta_TPI_sobre_Mae))),
    best1s, best1n,
    bestF: rows.reduce((a,c)=>N(a.Score_Final_v3)>N(c.Score_Final_v3)?a:c),
    worstF: rows.reduce((a,c)=>N(a.Score_Final_v3)<N(c.Score_Final_v3)?a:c),
  };
  // DeltaG per trait
  const traits = ['MILK','FAT','PROT','SCS','DPR','CCR','PL','LIV','UDC','FLC','BWC','HCR','EFC','SCE'];
  stats[b].dg = {};
  traits.forEach(t => stats[b].dg[t] = avg(rows.map(r=>N(r['DeltaG_'+t]))));
});

// Sort bulls by score
const bullsSorted = BULLS.slice().sort((a,b) => stats[b].sc_avg - stats[a].sc_avg);

// ===== EXECUTIVE SUMMARY =====
function renderSummary() {
  const best = bullsSorted[0], second = bullsSorted[1], worst = bullsSorted[bullsSorted.length-1];
  const s = stats;
  document.getElementById('executive-summary').innerHTML = `
  <div class="summary-box">
    <h3>Sumário Executivo</h3>
    <p>Este relatório analisa <strong>6 touros Holstein</strong> avaliados contra <strong>46 fêmeas</strong> do rebanho,
    totalizando <strong>276 combinações</strong>. A metodologia DSII v3 (Dam-Specific Improvement Index) vai além da
    média parental simples, incorporando correção de deficiências específicas de cada fêmea, depressão endogâmica,
    penalidades de antagonismo produção × funcionalidade e distância ao perfil ideal da raça.</p>

    <p><strong>Principais conclusões:</strong></p>
    <ul>
      <li><span class="highlight">${best}</span> é o touro mais versátil do grupo, com Score médio de
      <strong>${s[best].sc_avg.toFixed(1)}</strong> e é o melhor para <strong>${s[best].best1s} de 46 fêmeas</strong>.
      TPI ${s[best].tpi}, NM$ ${s[best].nm_bull}.</li>

      <li><span class="highlight">${second}</span> empata virtualmente com o líder (Score ${s[second].sc_avg.toFixed(1)})
      e é o melhor para <strong>${s[second].best1s} fêmeas</strong>. Supera o líder em fêmeas com deficiências
      específicas que ele corrige melhor — é essencial para diversificação.</li>

      <li><span class="highlight">Rosemary</span> se destaca pelo <strong>melhor equilíbrio produção × funcionalidade</strong>:
      menor penalidade de antagonismo (${s.Rosemary.pAnt.toFixed(1)}) e menor penalidade crítica (${s.Rosemary.pCrit.toFixed(1)}).
      Melhor em PL (+${s.Rosemary.dg.PL.toFixed(2)} meses) e LIV (+${s.Rosemary.dg.LIV.toFixed(2)}).</li>

      <li><span class="alert">${worst}</span> fica em <strong>último lugar</strong> com Score médio de
      ${s[worst].sc_avg.toFixed(1)} — NM$ base de apenas $${s[worst].nm_bull} (vs $${s[best].nm_bull} do líder).
      Comprometido em fertilidade (DPR ${s[worst].dg.DPR.toFixed(2)}) e vivabilidade (LIV ${s[worst].dg.LIV.toFixed(2)}).
      Único destaque: UDC +${s[worst].dg.UDC.toFixed(2)} (melhor do grupo).</li>
    </ul>

    <p><strong>Recomendação de uso:</strong> Priorizar <span class="highlight">${best}</span> e
    <span class="highlight">${second}</span> como touros principais, usando <span class="highlight">Rosemary</span>
    para fêmeas onde o equilíbrio funcional é prioritário. Evitar concentração excessiva de um único touro —
    diversificar para prevenir endogamia futura. ${worst} apenas para casos específicos de correção de tipo (úbere).</p>
  </div>`;
}

// ===== CHARTS =====
function bar(id, data, opts={}) {
  new Chart(document.getElementById(id), {
    type: 'bar',
    data: { labels: bullsSorted, datasets: [{ data: bullsSorted.map(b=>data(b)), backgroundColor: bullsSorted.map(b=>COLORS[b]), borderRadius:6 }] },
    options: { plugins:{legend:{display:false}}, scales:{y:{beginAtZero: opts.zero||false}}, ...opts.extra }
  });
}

function renderCharts() {
  bar('c1', b=>stats[b].sc_avg);
  bar('c2', b=>stats[b].nm_avg);
  bar('c3', b=>stats[b].tp_avg);
  bar('c4', b=>stats[b].ds_avg);

  bar('c6', b=>stats[b].pAnt, {zero:true});

  // Doughnut best count
  new Chart(document.getElementById('c7'), {
    type:'doughnut',
    data:{ labels:bullsSorted, datasets:[{ data:bullsSorted.map(b=>stats[b].best1s), backgroundColor:bullsSorted.map(b=>COLORS[b]) }] },
    options:{ plugins:{legend:{position:'bottom'}} }
  });

}

// ===== COMPARISON TABLE =====
function renderTable() {
  const metrics = [
    {k:'tpi', l:'TPI Touro', hi:true},
    {k:'nm_bull', l:'NM$ Touro', hi:true, dollar:true},
    {k:'sc_avg', l:'Score Médio', hi:true, dec:1},
    {k:'pT25', l:'P(Top 25%)', hi:true, dec:1, pct:true},
    {k:'pT10', l:'P(Top 10%)', hi:true, dec:1, pct:true},
  ];

  let html = '<thead><tr><th></th>';
  bullsSorted.forEach(b => html += `<th style="background:${COLORS[b]}">${b}</th>`);
  html += '</tr></thead><tbody>';

  metrics.forEach(m => {
    const vals = bullsSorted.map(b => {
      let v = stats[b][m.k];
      if (typeof v === 'string') v = parseFloat(v);
      return v;
    });
    const bestVal = m.hi ? Math.max(...vals) : Math.min(...vals);
    const worstVal = m.hi ? Math.min(...vals) : Math.max(...vals);

    html += `<tr><td class="bull-name">${m.l}</td>`;
    bullsSorted.forEach((b,i) => {
      let v = vals[i];
      let display = m.dec !== undefined ? v.toFixed(m.dec) : v;
      if (m.dollar) display = '$' + display;
      if (m.pct) display = display + '%';
      const cls = v === bestVal ? 'best-val' : v === worstVal ? 'worst-val' : '';
      html += `<td class="${cls}">${display}</td>`;
    });
    html += '</tr>';
  });

  // DeltaG traits
  const traits = ['MILK','FAT','PROT','DPR','CCR','PL','LIV','UDC','FLC','SCS','HCR','EFC'];
  const lowerBetter = new Set(['SCS']);
  traits.forEach(t => {
    const vals = bullsSorted.map(b => stats[b].dg[t]);
    const hi = !lowerBetter.has(t);
    const bestVal = hi ? Math.max(...vals) : Math.min(...vals);
    const worstVal = hi ? Math.min(...vals) : Math.max(...vals);
    html += `<tr><td class="bull-name">DeltaG ${t}</td>`;
    bullsSorted.forEach((b,i) => {
      const cls = vals[i]===bestVal?'best-val':vals[i]===worstVal?'worst-val':'';
      html += `<td class="${cls}">${vals[i].toFixed(2)}</td>`;
    });
    html += '</tr>';
  });

  html += '</tbody>';
  document.getElementById('comp-table').innerHTML = html;
}

// ===== INDIVIDUAL BULL REPORTS =====
function renderBullReports() {
  const lowerBetter = new Set(['SCS','BWC','SCE']);
  let html = '';

  bullsSorted.forEach((bName, idx) => {
    const s = stats[bName];
    const rank = idx + 1;
    const rankLabel = rank<=3 ? ['1º','2º','3º'][rank-1]+' LUGAR' : rank===6 ? '6º (ÚLTIMO)' : rank+'º LUGAR';

    // Build strengths/weaknesses
    const strengths = [], weaknesses = [];
    const dg = s.dg;

    // Production
    if(dg.MILK>900) strengths.push({i:'green',t:`<b>Produção de leite excepcional:</b> DeltaG MILK = +${dg.MILK.toFixed(0)} lbs — forte ganho genético sobre as mães`});
    else if(dg.MILK>600) strengths.push({i:'green',t:`<b>Boa produção de leite:</b> DeltaG MILK = +${dg.MILK.toFixed(0)} lbs`});
    else strengths.push({i:'blue',t:`<b>Produção de leite moderada:</b> DeltaG MILK = +${dg.MILK.toFixed(0)} lbs — não é destaque em volume`});

    if(dg.FAT>55) strengths.push({i:'green',t:`<b>Gordura excepcional:</b> DeltaG FAT = +${dg.FAT.toFixed(1)} lbs — altíssimo valor para pagamento do leite`});
    else if(dg.FAT>40) strengths.push({i:'green',t:`<b>Boa gordura:</b> DeltaG FAT = +${dg.FAT.toFixed(1)} lbs`});
    else weaknesses.push({i:'yellow',t:`<b>Gordura modesta:</b> DeltaG FAT = +${dg.FAT.toFixed(1)} lbs — inferior aos melhores touros do grupo`});

    if(dg.PROT>35) strengths.push({i:'green',t:`<b>Proteína alta:</b> DeltaG PROT = +${dg.PROT.toFixed(1)} lbs`});

    // Fertility
    if(dg.DPR<-1.0) weaknesses.push({i:'red',t:`<b>Fertilidade comprometida:</b> DeltaG DPR = ${dg.DPR.toFixed(2)} — piora significativa na taxa de prenhez. Impacta intervalo entre partos e custos reprodutivos. Este é um trade-off clássico de touros de alta produção.`});
    else if(dg.DPR<-0.5) weaknesses.push({i:'yellow',t:`<b>Fertilidade moderadamente reduzida:</b> DeltaG DPR = ${dg.DPR.toFixed(2)} — leve piora na taxa de prenhez`});
    else if(dg.DPR>0) strengths.push({i:'green',t:`<b>Melhora a fertilidade:</b> DeltaG DPR = +${dg.DPR.toFixed(2)}`});

    if(dg.CCR<-0.3) weaknesses.push({i:'red',t:`<b>Concepção de vacas reduzida:</b> DeltaG CCR = ${dg.CCR.toFixed(2)}`});
    else if(dg.CCR>0.3) strengths.push({i:'green',t:`<b>Boa taxa de concepção:</b> DeltaG CCR = +${dg.CCR.toFixed(2)}`});

    // Longevity
    if(dg.PL>1.2) strengths.push({i:'green',t:`<b>Vida produtiva longa:</b> DeltaG PL = +${dg.PL.toFixed(2)} meses — filhas duram mais no rebanho`});
    else if(dg.PL<0.7) weaknesses.push({i:'yellow',t:`<b>Vida produtiva curta:</b> DeltaG PL = +${dg.PL.toFixed(2)} — ganho limitado em longevidade`});

    if(dg.LIV<-1.0) weaknesses.push({i:'red',t:`<b>Vivabilidade comprometida:</b> DeltaG LIV = ${dg.LIV.toFixed(2)} — risco elevado de perdas. É um dos custos ocultos mais relevantes na pecuária leiteira.`});
    else if(dg.LIV<-0.3) weaknesses.push({i:'yellow',t:`<b>Vivabilidade reduzida:</b> DeltaG LIV = ${dg.LIV.toFixed(2)}`});
    else if(dg.LIV>0.3) strengths.push({i:'green',t:`<b>Boa vivabilidade:</b> DeltaG LIV = +${dg.LIV.toFixed(2)}`});

    // Type
    if(dg.UDC>1.3) strengths.push({i:'green',t:`<b>Úbere excelente:</b> DeltaG UDC = +${dg.UDC.toFixed(2)} — forte melhoria no composto de úbere`});
    else if(dg.UDC>0.8) strengths.push({i:'green',t:`<b>Bom composto de úbere:</b> DeltaG UDC = +${dg.UDC.toFixed(2)}`});

    if(dg.FLC>0.5) strengths.push({i:'green',t:`<b>Boas pernas e pés:</b> DeltaG FLC = +${dg.FLC.toFixed(2)}`});
    else if(dg.FLC<0.35) weaknesses.push({i:'yellow',t:`<b>Pernas e pés limitado:</b> DeltaG FLC = +${dg.FLC.toFixed(2)}`});

    // Penalties
    if(s.pAnt>12) weaknesses.push({i:'red',t:`<b>Forte antagonismo produção × funcionalidade:</b> Pen. = ${s.pAnt.toFixed(1)} — a alta produção puxa fertilidade e saúde para baixo via correlações genéticas (MILK×DPR = -0.35, MILK×CCR = -0.30).`});
    else if(s.pAnt>8) weaknesses.push({i:'yellow',t:`<b>Antagonismo moderado:</b> Pen. = ${s.pAnt.toFixed(1)}`});
    else strengths.push({i:'green',t:`<b>Bom equilíbrio produção/funcionalidade:</b> Pen. antagonismo = ${s.pAnt.toFixed(1)} — consegue produzir sem comprometer saúde`});

    if(s.pCrit>4) weaknesses.push({i:'red',t:`<b>Penalidade crítica alta:</b> ${s.pCrit.toFixed(1)} — traits da progênie abaixo dos limiares críticos em várias combinações`});
    else if(s.pCrit<1.5) strengths.push({i:'green',t:`<b>Poucos riscos de extremos:</b> Pen. crítica = ${s.pCrit.toFixed(1)}`});

    if(s.nm_avg<100) weaknesses.push({i:'red',t:`<b>Valor econômico baixo:</b> NM$ Corr. médio = $${s.nm_avg.toFixed(0)} — progênie com mérito econômico muito limitado comparado ao grupo`});

    if(dg.HCR>1.5) strengths.push({i:'green',t:`<b>Boa concepção de novilhas:</b> DeltaG HCR = +${dg.HCR.toFixed(2)}`});
    if(dg.EFC>3) strengths.push({i:'green',t:`<b>Eficiência alimentar excelente:</b> DeltaG EFC = +${dg.EFC.toFixed(2)}`});
    else if(dg.EFC>2) strengths.push({i:'green',t:`<b>Boa eficiência alimentar:</b> DeltaG EFC = +${dg.EFC.toFixed(2)}`});

    // Verdict
    let verdict = '';
    if(rank===1) {
      verdict = `<strong>${bName}</strong> é o touro mais completo e versátil do grupo, ocupando o <strong>1º lugar</strong> no ranking geral. `
        + `Com NM$ $${s.nm_bull} e TPI ${s.tpi}, apresenta o maior mérito genético base. `
        + `É o melhor para <strong>${s.best1s} de 46 fêmeas</strong>, demonstrando ampla adaptação ao rebanho. `
        + (s.pAnt>12 ? `Sua principal limitação é o antagonismo produção × funcionalidade — sua altíssima produção (MILK, FAT) penaliza DPR e LIV via correlações genéticas negativas. ` : '')
        + `<strong>Recomendação:</strong> Touro principal, mas diversificar com o 2º colocado para evitar concentração genética.`;
    } else if(rank===2) {
      verdict = `<strong>${bName}</strong> empata virtualmente com o líder (diferença de apenas ${(stats[bullsSorted[0]].sc_avg - s.sc_avg).toFixed(1)} pontos) e é o melhor para <strong>${s.best1s} fêmeas</strong>. `
        + `Há combinações específicas onde supera o líder — especialmente fêmeas cujas deficiências são melhor corrigidas pelo seu perfil genético. `
        + `<strong>Recomendação:</strong> Usar como touro complementar ao líder, priorizando nas fêmeas onde é #1.`;
    } else if(rank<=4) {
      verdict = `<strong>${bName}</strong> ocupa a <strong>${rank}ª posição</strong>. `
        + (s.best1s>0 ? `É o melhor para <strong>${s.best1s} fêmeas</strong> específicas. ` : `Não é o #1 para nenhuma fêmea pelo Score, mas apresenta diferenciais em traits específicos. `)
        + `<strong>Recomendação:</strong> Usar seletivamente para fêmeas onde seus pontos fortes são mais necessários, ou para diversificação genética.`;
    } else {
      verdict = `<strong>${bName}</strong> está na <strong>${rank}ª posição</strong>. `
        + `Seu NM$ base ($${s.nm_bull}) é substancialmente inferior ao do líder ($${stats[bullsSorted[0]].nm_bull}), o que limita o valor econômico da progênie. `
        + `DSII médio de ${s.ds_avg.toFixed(0)} (vs ${stats[bullsSorted[0]].ds_avg.toFixed(0)} do líder) indica menor capacidade de melhoria ponderada. `
        + `<strong>Recomendação:</strong> Uso restrito — apenas para fêmeas muito específicas onde seus pontos fortes (${dg.UDC>1.3?'úbere':'tipo'}) compensem o menor mérito econômico, ou para diversificação de pedigree quando necessário.`;
    }

    html += `
    <div class="bull-report">
      <div class="bull-report-header" style="background:linear-gradient(135deg,${COLORS[bName]},${COLORS[bName]}dd)">
        <h3>${bName} <span style="font-weight:400;font-size:14px;opacity:0.8">(${s.naab})</span></h3>
        <span class="rank-badge">${rankLabel} — Score Médio: ${s.sc_avg.toFixed(1)}</span>
      </div>
      <div class="bull-kpis">
      </div>
      <div class="tech-analysis">
        <div class="tech-grid">
          <div class="tech-col">
            <h4>Pontos Fortes <span class="badge badge-green">${strengths.length}</span></h4>
            ${strengths.map(x=>`<div class="tech-item"><div class="tech-icon ti-${x.i}">+</div><div>${x.t}</div></div>`).join('')}
          </div>
          <div class="tech-col">
            <h4>Pontos Fracos <span class="badge badge-red">${weaknesses.length}</span></h4>
            ${weaknesses.length ? weaknesses.map(x=>`<div class="tech-item"><div class="tech-icon ti-${x.i}">!</div><div>${x.t}</div></div>`).join('') : '<div class="tech-item"><div class="tech-icon ti-green">✓</div><div>Nenhum ponto fraco significativo identificado</div></div>'}
          </div>
        </div>
        <div class="verdict-box">${verdict}</div>
      </div>
    </div>`;
  });

  document.getElementById('bull-reports').innerHTML = html;
}

// ===== EXPORTS =====
function exportExcel() {
  // Summary sheet
  const summary = bullsSorted.map(b => {
    const s = stats[b];
    return {
      Rank: bullsSorted.indexOf(b)+1, Touro: b, NAAB: s.naab, TPI: s.tpi, 'NM$': s.nm_bull,
      'Score Médio': s.sc_avg.toFixed(1), 'Score Mín': s.sc_min.toFixed(1), 'Score Máx': s.sc_max.toFixed(1),
      'NM$ Corr. Médio': s.nm_avg.toFixed(0), 'TPI Corr. Médio': s.tp_avg.toFixed(0),
      'DSII Médio': s.ds_avg.toFixed(0), '#1 Score': s.best1s, '#1 NM$': s.best1n,
      'Delta NM$': s.deltaNM.toFixed(0), 'Delta TPI': s.deltaTPI.toFixed(0),
      'Pen. Crítica': s.pCrit.toFixed(1), 'Pen. Antagonismo': s.pAnt.toFixed(1),
      'Dist. Ideal': s.dist.toFixed(2), 'P(Top25%)': s.pT25.toFixed(1), 'P(Top10%)': s.pT10.toFixed(1),
    };
  });
  const ws1 = XLSX.utils.json_to_sheet(summary);

  // DeltaG sheet
  const traits = ['MILK','FAT','PROT','SCS','DPR','CCR','PL','LIV','UDC','FLC','BWC','HCR','EFC','SCE'];
  const dgRows = traits.map(t => {
    const row = { Trait: t };
    bullsSorted.forEach(b => row[b] = stats[b].dg[t].toFixed(2));
    return row;
  });
  const ws2 = XLSX.utils.json_to_sheet(dgRows);

  // All data
  const ws3 = XLSX.utils.json_to_sheet(DATA);

  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws1, 'Resumo Touros');
  XLSX.utils.book_append_sheet(wb, ws2, 'DeltaG por Trait');
  XLSX.utils.book_append_sheet(wb, ws3, 'Dados Completos');
  XLSX.writeFile(wb, 'relatorio_touros_dsii_v3.xlsx');
}

async function exportPDF() {
  const btn = event.target; btn.textContent='Gerando PDF...'; btn.disabled=true;

  // Overlay de progresso
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:9999;display:flex;align-items:center;justify-content:center;';
  overlay.innerHTML = '<div style="background:#fff;padding:32px 48px;border-radius:12px;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.3);">'
    + '<div style="font-size:16px;font-weight:700;color:#0D253F;margin-bottom:8px;" id="pdf-status">Preparando PDF...</div>'
    + '<div style="font-size:12px;color:#757575;" id="pdf-detail">Capturando seções...</div>'
    + '<div style="margin-top:12px;height:4px;background:#E0E0E0;border-radius:2px;width:300px;"><div id="pdf-bar" style="height:100%;background:#2E7D32;border-radius:2px;width:0%;transition:width 0.3s;"></div></div></div>';
  document.body.appendChild(overlay);

  const setProgress = (pct, text) => {
    document.getElementById('pdf-bar').style.width = pct+'%';
    document.getElementById('pdf-detail').textContent = text;
  };

  try {
    const {jsPDF} = window.jspdf;
    const pdf = new jsPDF('l','mm','a4');
    const pageW = pdf.internal.pageSize.getWidth();
    const pageH = pdf.internal.pageSize.getHeight();
    const margin = 12;
    const usableW = pageW - margin*2;
    const usableH = pageH - margin*2;

    // Collect all sections to capture
    const sections = [];

    // 1. Header + Summary
    const header = document.querySelector('.report-header');
    const summary = document.querySelector('.summary-box');
    if(header) sections.push({el:header, label:'Cabeçalho'});
    if(summary) sections.push({el:summary, label:'Sumário Executivo'});

    // 2. Chart boxes (each individually)
    document.querySelectorAll('.chart-box').forEach((el,i) => {
      const title = el.querySelector('h4')?.textContent || 'Gráfico '+(i+1);
      sections.push({el, label:title});
    });

    // 3. Comparison table
    const compTable = document.querySelector('.comp-table');
    if(compTable) sections.push({el:compTable.closest('.section'), label:'Tabela Comparativa'});

    // 4. Bull reports (each one)
    document.querySelectorAll('.bull-report').forEach(el => {
      const name = el.querySelector('h3')?.textContent?.split('(')[0]?.trim() || 'Touro';
      sections.push({el, label:'Ficha: '+name});
    });

    // Capture each section as canvas, then place on PDF pages
    let curY = margin;
    let isFirstPage = true;

    // Title page header
    pdf.setFillColor(13,37,63);
    pdf.rect(0, 0, pageW, 28, 'F');
    pdf.setFillColor(27,94,32);
    pdf.rect(0, 28, pageW, 3, 'F');
    pdf.setTextColor(255,255,255);
    pdf.setFontSize(18);
    pdf.setFont(undefined, 'bold');
    pdf.text('Relatório Técnico — Análise dos Touros | DSII v3', pageW/2, 17, {align:'center'});
    pdf.setFontSize(9);
    pdf.setFont(undefined, 'normal');
    pdf.text('Gerado em ' + new Date().toLocaleString('pt-BR'), pageW/2, 25, {align:'center'});
    curY = 38;

    for(let i = 0; i < sections.length; i++) {
      const sec = sections[i];
      setProgress(Math.round((i/sections.length)*100), `Capturando: ${sec.label} (${i+1}/${sections.length})`);

      // Small delay to let browser breathe
      await new Promise(r => setTimeout(r, 50));

      const canvas = await html2canvas(sec.el, {
        scale: 1.8,
        useCORS: true,
        backgroundColor: '#FFFFFF',
        logging: false,
        windowWidth: 1200,
      });

      const imgData = canvas.toDataURL('image/jpeg', 0.92);
      const imgW = usableW;
      const imgH = (canvas.height * imgW) / canvas.width;

      // If image is too tall for a single page, scale it down
      let finalW = imgW;
      let finalH = imgH;
      if(finalH > usableH) {
        const scale = usableH / finalH;
        finalW *= scale;
        finalH = usableH;
      }

      // Check if fits on current page
      if(curY + finalH > pageH - margin) {
        // New page
        pdf.addPage();
        curY = margin;
        // Page header bar
        pdf.setFillColor(13,37,63);
        pdf.rect(0, 0, pageW, 8, 'F');
        pdf.setFillColor(27,94,32);
        pdf.rect(0, 8, pageW, 1.5, 'F');
        curY = 14;
      }

      // Center horizontally
      const x = margin + (usableW - finalW)/2;
      pdf.addImage(imgData, 'JPEG', x, curY, finalW, finalH);
      curY += finalH + 6;
    }

    // Footer on all pages
    const totalPages = pdf.internal.getNumberOfPages();
    for(let p = 1; p <= totalPages; p++) {
      pdf.setPage(p);
      pdf.setFontSize(8);
      pdf.setTextColor(150,150,150);
      pdf.setFont(undefined, 'normal');
      pdf.text(`Predição Genética DSII v3 — Página ${p} de ${totalPages}`, pageW/2, pageH - 5, {align:'center'});
    }

    setProgress(100, 'Salvando arquivo...');
    await new Promise(r => setTimeout(r, 200));

    pdf.save('relatorio_touros_dsii_v3.pdf');
  } catch(e) {
    console.error(e);
    alert('Erro ao gerar PDF: ' + e.message);
  }

  document.body.removeChild(overlay);
  btn.textContent='Exportar PDF'; btn.disabled=false;
}

// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => {
  renderSummary();
  renderCharts();
  renderTable();
  renderBullReports();
});
</script>
</body>
</html>"""

# Inject data
import datetime
html = HTML.replace('%%DATA%%', json_data)
html = html.replace('REPORT_DATE', datetime.datetime.now().strftime('%d/%m/%Y %H:%M'))

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'[OK] Relatório gerado: {OUTPUT_PATH} ({len(html)//1024}KB)')
