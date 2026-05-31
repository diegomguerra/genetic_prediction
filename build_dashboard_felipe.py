"""Build HTML dashboard for Felipe Santana prediction results."""
import csv, json, os

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, 'predicao_felipe_IA.csv'), encoding='utf-8-sig') as f:
    rows_ia = list(csv.DictReader(f, delimiter=';'))
with open(os.path.join(BASE, 'predicao_felipe_embriao.csv'), encoding='utf-8-sig') as f:
    rows_emb = list(csv.DictReader(f, delimiter=';'))

for r in rows_ia:
    r['_tipo'] = 'IA'
for r in rows_emb:
    r['_tipo'] = 'Embrião'

all_data = rows_ia + rows_emb
json_str = json.dumps(all_data, ensure_ascii=False)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predição Felipe Santana — DSII v3</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<style>
:root{--navy:#0D253F;--green:#2E7D32;--green-bg:#E8F5E9;--gold:#F9A825;--red:#C62828;--g100:#F5F5F5;--g200:#EEE;--g300:#E0E0E0;--g500:#9E9E9E;--g800:#424242;--g900:#212121;--white:#FFF;--shadow:0 2px 8px rgba(0,0,0,.1);--r:10px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Inter,sans-serif;background:var(--g100);color:var(--g900);line-height:1.5}
.navbar{background:linear-gradient(135deg,var(--navy),var(--green));padding:0 32px;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;box-shadow:0 2px 12px rgba(0,0,0,.2)}
.navbar h1{color:var(--white);font-size:17px;font-weight:800}
.navbar-actions{display:flex;gap:8px}
.btn{padding:8px 16px;border:none;border-radius:6px;font-family:inherit;font-size:12px;font-weight:700;cursor:pointer;transition:.2s}
.btn-gold{background:var(--gold);color:var(--navy)}.btn-gold:hover{filter:brightness(1.1)}
.container{max-width:1500px;margin:0 auto;padding:24px}
.tabs{display:flex;gap:0;background:var(--white);border-bottom:2px solid var(--g200);margin-bottom:20px;border-radius:var(--r) var(--r) 0 0;overflow:hidden}
.tab{padding:12px 24px;font-size:13px;font-weight:700;cursor:pointer;border-bottom:3px solid transparent;color:var(--g500)}
.tab.active{color:var(--green);border-bottom-color:var(--green)}.tab:hover{color:var(--green)}
.panel{display:none}.panel.active{display:block}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px;margin-bottom:20px}
.kpi{background:var(--white);border-radius:var(--r);padding:16px;box-shadow:var(--shadow);border-left:4px solid var(--green)}
.kpi.gold{border-left-color:var(--gold)}.kpi.red{border-left-color:var(--red)}
.kpi .lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--g500)}
.kpi .val{font-size:24px;font-weight:900;color:var(--navy);letter-spacing:-1px}
.kpi .sub{font-size:11px;color:var(--g500)}
.charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
.chart-box{background:var(--white);border:1px solid var(--g200);border-radius:var(--r);padding:16px}
.chart-box h4{font-size:13px;font-weight:700;color:var(--navy);margin-bottom:12px}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:var(--navy);color:var(--white);padding:8px 10px;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.5px;white-space:nowrap;position:sticky;top:0;cursor:pointer}
thead th:hover{background:#1a3a5c}
tbody td{padding:7px 10px;border-bottom:1px solid var(--g200);white-space:nowrap}
tbody tr:hover{background:var(--green-bg)}
.bull-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:700;font-size:11px;color:var(--white);background:var(--green)}
.score-pill{display:inline-block;padding:2px 10px;border-radius:20px;font-weight:700;font-size:12px}
.sc-hi{background:#C8E6C9;color:#1B5E20}.sc-md{background:#FFF9C4;color:#F57F17}.sc-lo{background:#FFCDD2;color:#B71C1C}
.delta-pos{color:var(--green);font-weight:600}.delta-neg{color:var(--red);font-weight:600}
.filter-bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
.filter-bar select,.filter-bar input{padding:7px 12px;border:1.5px solid var(--g300);border-radius:6px;font-family:inherit;font-size:13px}
</style>
</head>
<body>
<nav class="navbar">
  <h1>Predição Genética — Felipe Santana | DSII v3</h1>
  <div class="navbar-actions">
    <button class="btn btn-gold" onclick="exportExcel()">Exportar Excel</button>
  </div>
</nav>
<div class="container">
  <div class="tabs">
    <div class="tab active" onclick="switchTab(0)">Visão Geral</div>
    <div class="tab" onclick="switchTab(1)">IAs (Sêmen)</div>
    <div class="tab" onclick="switchTab(2)">Embriões</div>
  </div>
  <div class="panel active" id="p0">
    <div class="kpi-row" id="kpis"></div>
    <div class="charts-grid" id="charts"></div>
  </div>
  <div class="panel" id="p1">
    <div class="filter-bar">
      <select id="fBull" onchange="renderTable()"><option value="">Todos os Touros</option></select>
      <input id="fSearch" placeholder="Buscar fêmea..." oninput="renderTable()">
    </div>
    <div style="max-height:75vh;overflow:auto"><table id="tblIA"><thead></thead><tbody></tbody></table></div>
  </div>
  <div class="panel" id="p2">
    <div style="max-height:75vh;overflow:auto"><table id="tblEmb"><thead></thead><tbody></tbody></table></div>
  </div>
</div>
<script>
const DATA = %%JSON%%;
const IA = DATA.filter(r=>r._tipo==='IA');
const EMB = DATA.filter(r=>r._tipo==='Embrião');
const N = v => parseFloat(v)||0;
const avg = a => a.length ? a.reduce((s,v)=>s+v,0)/a.length : 0;

const BULLS = [...new Set(IA.map(r=>r.Touro_Nome))].sort();
const BULL_COLORS = {};
const palette = ['#1565C0','#2E7D32','#E65100','#6A1B9A','#C62828','#4E342E','#00838F','#AD1457','#F57F17','#1B5E20','#283593','#4A148C'];
BULLS.forEach((b,i) => BULL_COLORS[b] = palette[i % palette.length]);

function switchTab(n) {
  document.querySelectorAll('.panel').forEach((p,i) => p.classList.toggle('active', i===n));
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', i===n));
}

function renderKPIs() {
  const scores = IA.map(r=>N(r.Score_Final_v3));
  const nms = IA.map(r=>N(r['NM$_Corrigido']));
  const best = IA.reduce((a,b)=>N(a.Score_Final_v3)>N(b.Score_Final_v3)?a:b);
  document.getElementById('kpis').innerHTML = `
    <div class="kpi"><div class="lbl">Total IAs</div><div class="val">${IA.length}</div><div class="sub">12 touros</div></div>
    <div class="kpi"><div class="lbl">Total Embriões</div><div class="val">${EMB.length}</div><div class="sub">2 touros</div></div>
    <div class="kpi gold"><div class="lbl">Melhor IA (Score)</div><div class="val">${N(best.Score_Final_v3).toFixed(1)}</div><div class="sub">${best.Touro_Nome} x #${best.Femea_Num}</div></div>
    <div class="kpi"><div class="lbl">NM$ Corr. Médio IAs</div><div class="val">$${avg(nms).toFixed(0)}</div><div class="sub">$${Math.min(...nms).toFixed(0)} a $${Math.max(...nms).toFixed(0)}</div></div>
    <div class="kpi"><div class="lbl">Score Médio IAs</div><div class="val">${avg(scores).toFixed(1)}</div><div class="sub">${Math.min(...scores).toFixed(1)} a ${Math.max(...scores).toFixed(1)}</div></div>
    <div class="kpi red"><div class="lbl">Melhor Embrião</div><div class="val">${EMB.length?N(EMB[0].Score_Final_v3).toFixed(1):'-'}</div><div class="sub">${EMB.length?EMB[0].Touro_Nome+' x Mãe #'+EMB[0].Femea_Num:''}</div></div>
  `;
}

function renderCharts() {
  const byBull = {};
  IA.forEach(r => { if(!byBull[r.Touro_Nome]) byBull[r.Touro_Nome]=[]; byBull[r.Touro_Nome].push(r); });
  const sorted = BULLS.slice().sort((a,b) => avg((byBull[b]||[]).map(r=>N(r.Score_Final_v3))) - avg((byBull[a]||[]).map(r=>N(r.Score_Final_v3))));

  document.getElementById('charts').innerHTML = `
    <div class="chart-box"><h4>Score Médio por Touro (IAs)</h4><canvas id="c1"></canvas></div>
    <div class="chart-box"><h4>NM$ Corrigido Médio por Touro</h4><canvas id="c2"></canvas></div>
    <div class="chart-box"><h4>TPI Corrigido Médio por Touro</h4><canvas id="c3"></canvas></div>
    <div class="chart-box"><h4>Nº de IAs por Touro</h4><canvas id="c4"></canvas></div>
  `;

  const mkBar = (id, data) => new Chart(document.getElementById(id), {
    type:'bar', data:{labels:sorted, datasets:[{data:sorted.map(b=>data(byBull[b]||[])), backgroundColor:sorted.map(b=>BULL_COLORS[b]), borderRadius:6}]},
    options:{plugins:{legend:{display:false}}, scales:{y:{beginAtZero:false}}}
  });
  mkBar('c1', rows=>avg(rows.map(r=>N(r.Score_Final_v3))));
  mkBar('c2', rows=>avg(rows.map(r=>N(r['NM$_Corrigido']))));
  mkBar('c3', rows=>avg(rows.map(r=>N(r.TPI_Corrigido))));
  mkBar('c4', rows=>rows.length);
}

function renderTable() {
  const bull = document.getElementById('fBull').value;
  const search = document.getElementById('fSearch').value.toLowerCase();
  let data = IA;
  if(bull) data = data.filter(r=>r.Touro_Nome===bull);
  if(search) data = data.filter(r=>r.Femea_Num.includes(search)||r.Pai_da_Femea.toLowerCase().includes(search));
  data = [...data].sort((a,b)=>N(a.Rank)-N(b.Rank));

  const cols = ['Rank','Touro_Nome','Femea_Num','Pai_da_Femea','Score_Final_v3','NM$_Corrigido','NM$_IC_Lower','NM$_IC_Upper','Delta_NM$_sobre_Mae','TPI_Corrigido','Delta_TPI_sobre_Mae','DSII','F_Esperada_%','Penalidade_Cr\u00edtica','Penalidade_Antagonismo','P_Top25_%','P_Top10_%'];
  const labels = ['Rk','Touro','Fêmea','Pai Fêmea','Score','NM$ Corr','IC-','IC+','Delta NM$','TPI Corr','Delta TPI','DSII','F%','Pen.Crít','Pen.Ant','P T25%','P T10%'];

  const thead = document.querySelector('#tblIA thead');
  const tbody = document.querySelector('#tblIA tbody');
  thead.innerHTML = '<tr>'+labels.map(l=>`<th>${l}</th>`).join('')+'</tr>';
  tbody.innerHTML = data.map(r=>'<tr>'+cols.map(c=>{
    const v = r[c]||'';
    if(c==='Score_Final_v3'){const n=N(v);return `<td><span class="score-pill ${n>=60?'sc-hi':n>=50?'sc-md':'sc-lo'}">${n.toFixed(1)}</span></td>`}
    if(c.startsWith('Delta')){const n=N(v);return `<td class="${n>=0?'delta-pos':'delta-neg'}">${n>=0?'\u25B2':'\u25BC'} ${n.toFixed(0)}</td>`}
    if(c==='Touro_Nome') return `<td><span class="bull-badge" style="background:${BULL_COLORS[v]||'#666'}">${v}</span></td>`;
    return `<td>${v}</td>`;
  }).join('')+'</tr>').join('');
}

function renderEmbTable() {
  const cols = ['Rank','Touro_Nome','Femea_Num','Produto_Num','Pai_da_Femea','Score_Final_v3','NM$_Corrigido','Delta_NM$_sobre_Mae','TPI_Corrigido','Delta_TPI_sobre_Mae','DSII','P_Top25_%'];
  const labels = ['Rk','Touro','Mãe','Produto','Pai Mãe','Score','NM$ Corr','Delta NM$','TPI Corr','Delta TPI','DSII','P T25%'];

  const thead = document.querySelector('#tblEmb thead');
  const tbody = document.querySelector('#tblEmb tbody');
  thead.innerHTML = '<tr>'+labels.map(l=>`<th>${l}</th>`).join('')+'</tr>';
  const data = [...EMB].sort((a,b)=>N(a.Rank)-N(b.Rank));
  tbody.innerHTML = data.map(r=>'<tr>'+cols.map(c=>{
    const v = r[c]||'';
    if(c==='Score_Final_v3'){const n=N(v);return `<td><span class="score-pill ${n>=60?'sc-hi':n>=50?'sc-md':'sc-lo'}">${n.toFixed(1)}</span></td>`}
    if(c.startsWith('Delta')){const n=N(v);return `<td class="${n>=0?'delta-pos':'delta-neg'}">${n>=0?'\u25B2':'\u25BC'} ${n.toFixed(0)}</td>`}
    if(c==='Touro_Nome') return `<td><span class="bull-badge" style="background:${BULL_COLORS[v]||'#666'}">${v}</span></td>`;
    return `<td>${v}</td>`;
  }).join('')+'</tr>').join('');
}

function exportExcel() {
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(IA), 'IAs');
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(EMB), 'Embrioes');
  XLSX.writeFile(wb, 'predicao_felipe_santana_dsii_v3.xlsx');
}

document.addEventListener('DOMContentLoaded', () => {
  BULLS.forEach(b => { const o=document.createElement('option'); o.value=b; o.textContent=b; document.getElementById('fBull').appendChild(o); });
  renderKPIs();
  renderCharts();
  renderTable();
  renderEmbTable();
});
</script>
</body>
</html>"""

html = HTML_TEMPLATE.replace('%%JSON%%', json_str)
output = os.path.join(BASE, 'dashboard_felipe.html')
with open(output, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'OK: {output} ({len(html)//1024}KB)')
