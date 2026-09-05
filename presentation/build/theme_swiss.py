"""SWISS MODERN — Bauhaus-derived international style.

Pure white, pure black, one red. Archivo at 800/900 for uppercase display, Nunito for
substance, JetBrains Mono for data and chrome. Structure is a visible modular grid and
heavy black rules; the red is spent only on the single most important figure per slide.
No gradients, no shadows, no curves — weight and alignment carry everything.
"""

NAME = "Swiss Modern"
FONTS = ("https://fonts.googleapis.com/css2?"
         "family=Archivo:wght@500;600;700;800;900&family=Nunito:wght@300;400;600;700&"
         "family=JetBrains+Mono:wght@400;500;700&display=swap")

CSS = r"""
:root{
  --paper:#FFFFFF; --paper-2:#F2F2F0; --ink:#000000;
  --red:#FF3300; --grey:#6E6E6E; --grey-2:#9A9A9A;
  --line:#000000; --line-soft:#D8D8D6;
  --stage-bg:#111111; --slide-bg:#FFFFFF;
  --font-display:'Archivo',system-ui,sans-serif;
  --font-sans:'Nunito',system-ui,sans-serif;
  --font-mono:'JetBrains Mono',monospace;
  --chrome-accent:#FF3300; --chrome-fg:#9A9A9A;
}
*{margin:0;padding:0;box-sizing:border-box}

.slide{background:var(--paper);color:var(--ink)}
.slide.alt{background:var(--paper-2)}

/* the modular grid is visible — Swiss structure is never hidden */
.slide::before{content:"";position:absolute;inset:0;pointer-events:none;
  background-image:linear-gradient(to right,rgba(0,0,0,.055) 1px,transparent 1px);
  background-size:160px 100%}

.s-pad{position:absolute;inset:0;padding:56px 96px 50px;display:flex;flex-direction:column}

/* --- chrome: heavy rules --- */
.s-chrome{display:flex;justify-content:space-between;align-items:baseline;
  font-family:var(--font-mono);font-size:15px;font-weight:700;letter-spacing:.2em;
  text-transform:uppercase;color:var(--ink);border-bottom:4px solid var(--line);
  padding-bottom:13px;flex:0 0 auto}
.s-num{color:var(--red)}
.s-foot{display:flex;justify-content:space-between;align-items:baseline;
  border-top:2px solid var(--line);padding-top:12px;margin-top:auto;flex:0 0 auto;
  font-family:var(--font-mono);font-size:14px;letter-spacing:.18em;text-transform:uppercase;color:var(--grey)}
.s-brand{color:var(--red);font-weight:700}

.s-body{flex:1 1 auto;min-height:0;padding:24px 0 16px;display:flex;flex-direction:column;
  gap:22px;justify-content:space-between}

/* --- headings: Archivo Black, uppercase, tight --- */
.s-kicker{font-family:var(--font-mono);font-size:14px;font-weight:700;letter-spacing:.24em;
  text-transform:uppercase;color:var(--red);margin-bottom:12px}
.s-kicker.warn-kicker::before{content:"\25A0  "}
.s-head{font-family:var(--font-display);font-weight:900;font-size:60px;line-height:.94;
  letter-spacing:-.032em;text-transform:uppercase;color:var(--ink)}
.s-head em{font-style:normal;color:var(--red)}
.s-lead{font-family:var(--font-sans);font-weight:300;font-size:25px;line-height:1.46;
  color:#333;max-width:1480px;margin-top:16px}
.s-lead strong{color:var(--ink);font-weight:700}
.s-lead em,.panel p em,.note em,.act em{font-style:normal;color:var(--red);font-weight:700}
.s-head-block.tight .s-lead{margin-top:12px;font-size:23px}
.head-flex{display:flex;justify-content:space-between;align-items:flex-start;gap:56px}
.head-flex .s-head-block{flex:1}

/* --- panels: black top rule --- */
.panel{border-top:3px solid var(--line);padding:16px 0 0}
.panel.tight{padding-top:14px}
.panel-title{font-family:var(--font-display);font-weight:800;font-size:24px;line-height:1.16;
  letter-spacing:-.012em;text-transform:uppercase;color:var(--ink);margin-bottom:10px}
.panel p{font-family:var(--font-sans);font-size:20px;line-height:1.55;color:#3A3A3A}
.act{font-family:var(--font-sans);font-size:19px;line-height:1.48;color:#3A3A3A;
  border-left:5px solid var(--red);padding-left:15px;margin-top:13px}
.note{font-family:var(--font-sans);font-size:20px;line-height:1.5;color:#3A3A3A;
  border-left:5px solid var(--red);padding-left:18px}
.note.tight{font-size:18px;margin:10px 0}
.note strong,.act strong,.panel p strong{color:var(--ink);font-weight:700}

.col2{display:grid;grid-template-columns:1fr 1fr;gap:50px}
.split-note{display:grid;grid-template-columns:1fr 1fr;gap:50px;align-items:start}

/* --- index rows --- */
.xp-cap{font-family:var(--font-mono);font-size:14px;font-weight:700;letter-spacing:.2em;
  text-transform:uppercase;color:var(--ink);margin-bottom:12px;padding-bottom:9px;
  border-bottom:3px solid var(--line)}
.xp-row{display:flex;justify-content:space-between;gap:24px;padding:11px 0;border-bottom:1px solid var(--line-soft)}
.xp-k{font-family:var(--font-display);font-weight:800;font-size:16px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--ink)}
.xp-v{font-family:var(--font-sans);font-size:19px;color:#3A3A3A;text-align:right}

/* --- tables --- */
.s-table{width:100%;border-collapse:collapse;font-family:var(--font-sans)}
.s-table th{font-family:var(--font-mono);font-size:14px;font-weight:700;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink);text-align:left;padding:0 14px 10px 0;
  border-bottom:3px solid var(--line)}
.s-table td{font-size:19px;color:#3A3A3A;padding:12px 14px 12px 0;border-bottom:1px solid var(--line-soft);
  vertical-align:top;line-height:1.38}
.s-table td:first-child{color:var(--ink);font-weight:600}
.s-table .num,.s-table th.num{text-align:right;font-family:var(--font-mono);font-variant-numeric:tabular-nums}
.s-table .hl{color:var(--red);font-weight:700}
.s-table.wide td{font-size:20px;padding:14px 16px 14px 0}
.s-table.res td{font-size:20px;padding:11px 14px 11px 0}
.s-table.res .base-row td{color:var(--grey-2);font-style:italic}
.s-table.gap td{font-size:19px;padding:12px 12px 12px 0}
.s-table.gap .tot td{border-bottom:0;border-top:3px solid var(--line);color:var(--ink);font-weight:700}
.s-table.bands td{font-size:19px;padding:10px 12px 10px 0}
.s-table.work td{font-size:18px;padding:10px 12px 10px 0}
.s-table.work .role{color:#3A3A3A;text-align:right;font-weight:400}
.mono{font-family:var(--font-mono);font-size:.92em}
.tbd{font-family:var(--font-sans);text-transform:none;letter-spacing:0;color:var(--grey);
  font-size:15px;font-weight:400}

/* --- stats: huge Archivo numerals --- */
.stat-row{display:flex;gap:40px}
.stat-row.vert{gap:28px;flex:0 0 auto}
.stat{border-top:3px solid var(--line);padding-top:10px;min-width:96px}
.stat-n{font-family:var(--font-display);font-weight:900;font-size:56px;line-height:.92;
  letter-spacing:-.045em;color:var(--ink)}
.stat:last-child .stat-n{color:var(--red)}
.stat-l{font-family:var(--font-mono);font-size:14px;font-weight:500;letter-spacing:.16em;
  text-transform:uppercase;color:var(--grey);margin-top:7px}

/* --- bullets: red square --- */
.bul{list-style:none}
.bul li{font-family:var(--font-sans);font-size:19px;line-height:1.48;color:#3A3A3A;
  padding-left:28px;position:relative;margin-bottom:12px}
.bul li::before{content:"";position:absolute;left:0;top:.5em;width:10px;height:10px;background:var(--red)}
.bul li em{font-style:normal;color:var(--red);font-weight:700}

/* --- figures --- */
.s-fig{display:flex;flex-direction:column;gap:10px;min-height:0}
.s-fig img{width:100%;height:auto;object-fit:contain;border:2px solid var(--line);background:#fff}
.s-fig figcaption{font-family:var(--font-sans);font-size:16px;line-height:1.4;color:var(--grey)}
.fig-grid{display:grid;grid-template-columns:1fr 1fr;gap:50px;align-items:start;min-height:0;flex:1}
.fig-grid .s-fig img{max-height:500px}
.wide-fig{grid-column:1 / -1}
.wide-fig img{max-height:300px}
.s-fig.sm img{max-height:230px}

/* --- specific blocks --- */
.rel-grid{display:grid;grid-template-columns:1.05fr 1fr;gap:54px;align-items:start}
.cb-wrap{display:flex;flex-direction:column;gap:14px}
.cb{border-left:5px solid var(--ink);padding:2px 0 2px 16px}
.cb-t{font-family:var(--font-display);font-weight:800;font-size:20px;letter-spacing:-.01em;
  text-transform:uppercase;color:var(--ink);margin-bottom:4px}
.cb-b{font-family:var(--font-sans);font-size:17px;line-height:1.45;color:#3A3A3A}

.chip-row{display:flex;gap:16px;margin-bottom:20px}
.chip{border:3px solid var(--ink);padding:7px 15px;display:flex;gap:10px;align-items:baseline}
.chip-k{font-family:var(--font-mono);font-size:14px;font-weight:700;letter-spacing:.12em;color:var(--ink)}
.chip-v{font-family:var(--font-display);font-size:24px;font-weight:900;color:var(--red)}

.verdict-row{margin-top:4px}
.verdict{border-left:6px solid var(--red);padding-left:20px}
.verdict.big{margin:14px 0}
.v-tag{font-family:var(--font-mono);font-size:14px;font-weight:700;letter-spacing:.2em;
  text-transform:uppercase;color:var(--red);margin-bottom:8px}
.verdict p{font-family:var(--font-sans);font-size:20px;line-height:1.48;color:#3A3A3A}
.verdict p strong{color:var(--ink);font-weight:700}

.step-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px 42px;flex:1;align-content:space-between}
.step{border-top:2px solid var(--line);padding-top:11px}
.step-n{font-family:var(--font-mono);font-size:14px;font-weight:700;letter-spacing:.16em;
  color:var(--red);margin-bottom:5px}
.step-t{font-family:var(--font-display);font-weight:800;font-size:20px;line-height:1.16;
  letter-spacing:-.012em;text-transform:uppercase;color:var(--ink);margin-bottom:5px}
.step-w{font-family:var(--font-sans);font-size:16px;line-height:1.42;color:#4A4A4A}
.step-t .mono{text-transform:none;font-size:.95em}

.ev-grid{display:grid;grid-template-columns:1fr 1fr;gap:32px 50px;flex:1;align-content:space-between}

.mth-wrap{display:flex;flex-direction:column;margin-bottom:6px}
.mth{display:flex;justify-content:space-between;gap:20px;padding:9px 0;border-bottom:1px solid var(--line-soft)}
.mth-k{font-family:var(--font-display);font-weight:800;font-size:19px;text-transform:uppercase;
  letter-spacing:-.008em;color:var(--ink)}
.mth-v{font-family:var(--font-sans);font-size:16px;color:var(--grey);text-align:right}

.mdl-grid{display:grid;grid-template-columns:.85fr 1.15fr;gap:54px;flex:1;align-content:start}
.mdl-grid>div{display:flex;flex-direction:column}
.mdl-wrap,.pr-wrap{flex:1;justify-content:space-between}
.mdl-wrap{display:flex;flex-direction:column}
.mdl{display:flex;justify-content:space-between;gap:18px;padding:10px 0;border-bottom:1px solid var(--line-soft)}
.mdl-k{font-family:var(--font-mono);font-size:17px;font-weight:500;color:var(--ink)}
.mdl-v{font-family:var(--font-sans);font-size:16px;color:var(--grey);text-align:right}
.pr-wrap{display:flex;flex-direction:column;gap:12px}
.pr{border-top:2px solid var(--line);padding-top:9px}
.pr-k{font-family:var(--font-display);font-weight:800;font-size:19px;text-transform:uppercase;
  letter-spacing:-.008em;color:var(--ink);margin-bottom:3px}
.pr-v{font-family:var(--font-sans);font-size:17px;line-height:1.4;color:#3A3A3A}
.warn-box{border:3px solid var(--red);padding:15px 20px;font-family:var(--font-sans);
  font-size:18px;line-height:1.46;color:#3A3A3A}

.res-grid{display:grid;grid-template-columns:1fr 1fr;gap:80px;flex:1;align-items:start}
.punch{text-align:right;border-top:4px solid var(--red);padding-top:10px;flex:0 0 auto}
.punch-n{font-family:var(--font-display);font-weight:900;font-size:82px;line-height:.9;
  letter-spacing:-.05em;color:var(--red)}
.punch-l{font-family:var(--font-mono);font-size:14px;font-weight:500;letter-spacing:.14em;
  text-transform:uppercase;color:var(--grey);margin-top:9px;max-width:240px;margin-left:auto}

.err-grid{display:grid;grid-template-columns:1.15fr .95fr .9fr;grid-template-rows:auto auto;
  gap:28px 40px;flex:1;align-content:space-between}
.err-grid .s-fig img{max-height:330px}

.fnd-row{display:grid;grid-template-columns:repeat(3,1fr);gap:42px}
.fnd{border-top:6px solid var(--red);padding-top:12px}
.fnd-t{font-family:var(--font-display);font-weight:900;font-size:23px;line-height:1.1;
  letter-spacing:-.022em;text-transform:uppercase;color:var(--ink);margin-bottom:8px}
.fnd-b{font-family:var(--font-sans);font-size:18px;line-height:1.45;color:#3A3A3A}
.close-grid{display:grid;grid-template-columns:1fr 1fr;gap:54px;margin-top:4px}
.repro{font-family:var(--font-sans);font-size:18px;color:var(--grey);border-top:2px solid var(--line);padding-top:12px}
.repro .mono{color:var(--ink);font-weight:600}

/* --- cover: red block as the signature mark --- */
.slide.cover{background:var(--paper)}
.cover .deco{position:absolute;right:0;top:0;width:96px;height:340px;background:var(--red);pointer-events:none}
.cover .s-pad{padding:84px 96px 76px;justify-content:space-between}
.cover-top{display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:16px;
  font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:var(--ink);
  border-bottom:5px solid var(--line);padding-bottom:20px}
.cover-mid{display:grid;grid-template-columns:1.36fr .74fr;gap:80px;align-items:center;flex:1}
.cover h1{font-family:var(--font-display);font-weight:900;font-size:130px;line-height:.86;
  letter-spacing:-.048em;text-transform:uppercase;color:var(--ink)}
.cover h1 em{font-style:normal;color:var(--red)}
.cover-rule{width:180px;height:8px;background:var(--red);margin:36px 0 30px}
.cover-sub{font-family:var(--font-sans);font-weight:300;font-size:26px;line-height:1.5;
  color:#3A3A3A;max-width:790px}
.cover-sub strong{color:var(--ink);font-weight:700}
.cover-panel{border-left:4px solid var(--ink);padding-left:32px}
.g-cap{font-family:var(--font-mono);font-size:14px;font-weight:700;letter-spacing:.2em;
  text-transform:uppercase;color:var(--red);padding-bottom:14px;margin-bottom:4px}
.g-row{display:flex;justify-content:space-between;align-items:baseline;padding:14px 0;
  border-bottom:1px solid var(--line-soft)}
.g-row:last-child{border-bottom:0}
.g-k{font-family:var(--font-sans);font-size:21px;color:#3A3A3A}
.g-v{font-family:var(--font-display);font-size:30px;font-weight:900;letter-spacing:-.03em;color:var(--ink)}
.cover-foot{display:flex;justify-content:space-between;align-items:flex-end;
  border-top:2px solid var(--line);padding-top:20px}
.cover-names{font-family:var(--font-sans);font-size:19px;color:#3A3A3A}
.cover-foot .s-brand{font-family:var(--font-mono);font-size:16px;font-weight:700;letter-spacing:.16em}

/* --- entrance --- */
.reveal{opacity:0;transform:translateY(26px);
  transition:opacity .7s cubic-bezier(.16,1,.3,1),transform .7s cubic-bezier(.16,1,.3,1)}
.slide.visible .reveal{opacity:1;transform:none}
.slide.visible .reveal:nth-child(1){transition-delay:.06s}
.slide.visible .reveal:nth-child(2){transition-delay:.16s}
.slide.visible .reveal:nth-child(3){transition-delay:.26s}
.slide.visible .reveal:nth-child(4){transition-delay:.36s}
.s-body>*{opacity:0;transform:translateY(18px);
  transition:opacity .65s cubic-bezier(.16,1,.3,1),transform .65s cubic-bezier(.16,1,.3,1)}
.slide.visible .s-body>*{opacity:1;transform:none}
.slide.visible .s-body>*:nth-child(1){transition-delay:.08s}
.slide.visible .s-body>*:nth-child(2){transition-delay:.18s}
.slide.visible .s-body>*:nth-child(3){transition-delay:.28s}
.slide.visible .s-body>*:nth-child(4){transition-delay:.38s}
.s-chrome,.s-foot{opacity:0;transition:opacity .55s ease .04s}
.slide.visible .s-chrome,.slide.visible .s-foot{opacity:1}
.cover .deco{transform:scaleY(0);transform-origin:top;
  transition:transform .8s cubic-bezier(.76,0,.24,1) .25s}
.slide.visible .deco{transform:scaleY(1)}
"""
