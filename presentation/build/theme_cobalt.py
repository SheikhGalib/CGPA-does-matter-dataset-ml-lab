"""COBALT GRID — two-colour risograph trend-report system.

Warm cream paper, one electric cobalt ink, and a graph-paper grid that sits permanently
behind every slide. Newsreader serif headlines, Hanken Grotesk for substance, DM Mono for
chrome. Decoration is a stair-stepped pixel-glitch column and QR-style 8x8 patches.
Structure comes from 1.5px cobalt rules and 1px faint-cobalt row dividers — never cards.
"""

NAME = "Cobalt Grid"
FONTS = ("https://fonts.googleapis.com/css2?"
         "family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,500;"
         "1,6..72,300;1,6..72,400&family=Hanken+Grotesk:wght@300;400;500;700&"
         "family=DM+Mono:wght@400;500&display=swap")

CSS = r"""
:root{
  --paper:#F0EBDE; --paper-2:#E6E0CE; --ink:#1F2BE0; --ink-soft:#5560E5;
  --text:#22283A; --text-2:#4A5163; --text-3:#7C8398;
  --line:rgba(31,43,224,.22); --line-soft:rgba(31,43,224,.12);
  --stage-bg:#2A2A26; --slide-bg:var(--paper);
  --font-serif:'Newsreader',Georgia,serif;
  --font-sans:'Hanken Grotesk',system-ui,sans-serif;
  --font-mono:'DM Mono',monospace;
  --chrome-accent:#1F2BE0; --chrome-fg:#7C8398;
}
*{margin:0;padding:0;box-sizing:border-box}

.slide{background:var(--paper);color:var(--text)}
.slide.alt{background:var(--paper-2)}

/* the graph paper is permanent — it is the system, not decoration */
.slide::before{content:"";position:absolute;inset:0;pointer-events:none;opacity:.10;
  background-image:linear-gradient(to right,var(--ink) 1px,transparent 1px),
                   linear-gradient(to bottom,var(--ink) 1px,transparent 1px);
  background-size:40px 40px}

/* 1.5px cobalt hairline frame, top and bottom of every slide */
.slide::after{content:"";position:absolute;left:80px;right:80px;top:46px;height:1.5px;
  background:var(--ink);pointer-events:none}
.s-pad{position:absolute;inset:0;padding:74px 80px 70px;display:flex;flex-direction:column}
.s-pad::after{content:"";position:absolute;left:0;right:0;bottom:46px;height:1.5px;background:var(--ink)}

/* --- chrome --- */
.s-chrome{display:flex;justify-content:space-between;align-items:baseline;
  font-family:var(--font-mono);font-size:15px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--ink);padding-bottom:14px;flex:0 0 auto}
.s-num{color:var(--ink)}
.s-foot{display:flex;justify-content:space-between;align-items:baseline;
  padding-top:13px;margin-top:auto;flex:0 0 auto;font-family:var(--font-mono);font-size:14px;
  letter-spacing:.15em;text-transform:uppercase;color:var(--text-3)}
.s-brand{color:var(--ink)}

.s-body{flex:1 1 auto;min-height:0;padding:22px 0 16px;display:flex;flex-direction:column;
  gap:22px;justify-content:space-between}

/* --- headings: Newsreader, light weight, cobalt --- */
.s-kicker{font-family:var(--font-mono);font-size:14px;letter-spacing:.2em;text-transform:uppercase;
  color:var(--ink-soft);margin-bottom:12px}
.s-kicker.warn-kicker::before{content:"\25A0  "}
.s-head{font-family:var(--font-serif);font-weight:300;font-size:64px;line-height:1.02;
  letter-spacing:-.026em;color:var(--ink);border-bottom:1.5px solid var(--ink);padding-bottom:16px}
.s-head em{font-style:italic;font-weight:400}
.s-lead{font-family:var(--font-sans);font-weight:300;font-size:25px;line-height:1.5;
  color:var(--text-2);max-width:1500px;margin-top:16px}
.s-lead strong{color:var(--text);font-weight:500}
.s-lead em,.panel p em,.note em,.act em{font-family:var(--font-serif);font-style:italic;color:var(--ink);font-weight:400}
.s-head-block.tight .s-lead{margin-top:13px;font-size:23px}
.head-flex{display:flex;justify-content:space-between;align-items:flex-start;gap:60px}
.head-flex .s-head-block{flex:1}

/* --- panels: rules only, never chrome --- */
.panel{border-top:1.5px solid var(--ink);padding:18px 0 0}
.panel.tight{padding-top:15px}
.panel-title{font-family:var(--font-serif);font-weight:400;font-size:28px;line-height:1.2;
  color:var(--ink);margin-bottom:10px}
.panel p{font-family:var(--font-sans);font-size:20px;line-height:1.58;color:var(--text-2)}
.act{font-family:var(--font-sans);font-size:19px;line-height:1.5;color:var(--text-2);
  background:rgba(31,43,224,.055);padding:12px 16px;margin-top:13px}
.note{font-family:var(--font-sans);font-size:20px;line-height:1.55;color:var(--text-2);
  border-left:3px solid var(--ink);padding-left:18px}
.note.tight{font-size:18px;margin:10px 0}
.note strong,.act strong,.panel p strong{color:var(--ink);font-weight:500}

.col2{display:grid;grid-template-columns:1fr 1fr;gap:52px}
.split-note{display:grid;grid-template-columns:1fr 1fr;gap:52px;align-items:start}

/* --- index rows: 1px faint cobalt dividers --- */
.xp-cap{font-family:var(--font-mono);font-size:14px;letter-spacing:.19em;text-transform:uppercase;
  color:var(--ink-soft);margin-bottom:12px;padding-bottom:9px;border-bottom:1.5px solid var(--ink)}
.xp-row{display:flex;justify-content:space-between;gap:24px;padding:11px 0;border-bottom:1px solid var(--line-soft)}
.xp-k{font-family:var(--font-mono);font-size:16px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink)}
.xp-v{font-family:var(--font-sans);font-size:19px;color:var(--text-2);text-align:right}

/* --- tables --- */
.s-table{width:100%;border-collapse:collapse;font-family:var(--font-sans)}
.s-table th{font-family:var(--font-mono);font-size:14px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--ink);text-align:left;padding:0 14px 10px 0;border-bottom:1.5px solid var(--ink);font-weight:500}
.s-table td{font-size:19px;color:var(--text-2);padding:12px 14px 12px 0;
  border-bottom:1px solid var(--line-soft);vertical-align:top;line-height:1.4}
.s-table td:first-child{color:var(--text);font-weight:500}
.s-table .num,.s-table th.num{text-align:right;font-family:var(--font-mono);font-variant-numeric:tabular-nums}
.s-table .hl{color:var(--ink);font-weight:500}
.s-table.wide td{font-size:20px;padding:14px 16px 14px 0}
.s-table.res td{font-size:20px;padding:11px 14px 11px 0}
.s-table.res .base-row td{color:var(--text-3);font-style:italic}
.s-table.gap td{font-size:19px;padding:12px 12px 12px 0}
.s-table.gap .tot td{border-bottom:0;border-top:1.5px solid var(--ink);color:var(--ink);font-weight:500}
.s-table.bands td{font-size:19px;padding:10px 12px 10px 0}
.s-table.work td{font-size:18px;padding:10px 12px 10px 0}
.s-table.work .role{color:var(--text-2);text-align:right;font-weight:400}
.mono{font-family:var(--font-mono);font-size:.92em}
.tbd{font-family:var(--font-sans);text-transform:none;letter-spacing:0;color:var(--text-3);font-size:15px}

/* --- stats --- */
.stat-row{display:flex;gap:44px}
.stat-row.vert{gap:30px;flex:0 0 auto}
.stat{border-top:1.5px solid var(--ink);padding-top:11px;min-width:96px}
.stat-n{font-family:var(--font-serif);font-weight:400;font-size:56px;line-height:1;
  letter-spacing:-.03em;color:var(--ink)}
.stat-l{font-family:var(--font-mono);font-size:14px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--text-3);margin-top:6px}

/* --- bullets: cobalt square marker --- */
.bul{list-style:none}
.bul li{font-family:var(--font-sans);font-size:19px;line-height:1.5;color:var(--text-2);
  padding-left:28px;position:relative;margin-bottom:12px}
.bul li::before{content:"";position:absolute;left:0;top:.55em;width:9px;height:9px;background:var(--ink)}
.bul li em{font-family:var(--font-serif);font-style:italic;color:var(--ink)}

/* --- figures --- */
.s-fig{display:flex;flex-direction:column;gap:10px;min-height:0}
.s-fig img{width:100%;height:auto;object-fit:contain;border:1.5px solid var(--ink);background:#fff}
.s-fig figcaption{font-family:var(--font-sans);font-size:16px;line-height:1.42;color:var(--text-3)}
.fig-grid{display:grid;grid-template-columns:1fr 1fr;gap:52px;align-items:start;min-height:0;flex:1}
.fig-grid .s-fig img{max-height:500px}
.wide-fig{grid-column:1 / -1}
.wide-fig img{max-height:300px}
.s-fig.sm img{max-height:230px}

/* --- specific blocks --- */
.rel-grid{display:grid;grid-template-columns:1.05fr 1fr;gap:56px;align-items:start}
.cb-wrap{display:flex;flex-direction:column;gap:14px}
.cb{border-left:3px solid var(--ink);padding:2px 0 2px 16px}
.cb-t{font-family:var(--font-serif);font-weight:400;font-size:23px;color:var(--ink);margin-bottom:4px}
.cb-b{font-family:var(--font-sans);font-size:17px;line-height:1.48;color:var(--text-2)}

.chip-row{display:flex;gap:16px;margin-bottom:20px}
.chip{border:1.5px solid var(--ink);padding:8px 16px;display:flex;gap:10px;align-items:baseline}
.chip-k{font-family:var(--font-mono);font-size:14px;letter-spacing:.12em;color:var(--ink-soft)}
.chip-v{font-family:var(--font-serif);font-size:25px;font-weight:400;color:var(--ink)}

.verdict-row{margin-top:4px}
.verdict{background:rgba(31,43,224,.06);padding:16px 20px}
.verdict.big{margin:14px 0}
.v-tag{font-family:var(--font-mono);font-size:14px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--ink);margin-bottom:8px}
.verdict p{font-family:var(--font-sans);font-size:20px;line-height:1.52;color:var(--text-2)}
.verdict p strong{color:var(--ink);font-weight:500}

.step-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:26px 44px;flex:1;align-content:space-between}
.step{border-top:1px solid var(--line);padding-top:11px;position:relative}
.step-n{font-family:var(--font-mono);font-size:14px;letter-spacing:.14em;color:var(--ink-soft);margin-bottom:5px}
.step-t{font-family:var(--font-serif);font-weight:400;font-size:23px;line-height:1.2;color:var(--ink);margin-bottom:5px}
.step-w{font-family:var(--font-sans);font-size:16px;line-height:1.44;color:var(--text-2)}

.ev-grid{display:grid;grid-template-columns:1fr 1fr;gap:34px 52px;flex:1;align-content:space-between}

.mth-wrap{display:flex;flex-direction:column;margin-bottom:6px}
.mth{display:flex;justify-content:space-between;gap:20px;padding:9px 0;border-bottom:1px solid var(--line-soft)}
.mth-k{font-family:var(--font-serif);font-size:21px;font-weight:400;color:var(--ink)}
.mth-v{font-family:var(--font-sans);font-size:16px;color:var(--text-3);text-align:right}

.mdl-grid{display:grid;grid-template-columns:.85fr 1.15fr;gap:56px;flex:1;align-content:start}
.mdl-grid>div{display:flex;flex-direction:column}
.mdl-wrap,.pr-wrap{flex:1;justify-content:space-between}
.mdl-wrap{display:flex;flex-direction:column}
.mdl{display:flex;justify-content:space-between;gap:18px;padding:10px 0;border-bottom:1px solid var(--line-soft)}
.mdl-k{font-family:var(--font-mono);font-size:17px;letter-spacing:.04em;color:var(--ink)}
.mdl-v{font-family:var(--font-sans);font-size:16px;color:var(--text-3);text-align:right}
.pr-wrap{display:flex;flex-direction:column;gap:12px}
.pr{border-top:1px solid var(--line);padding-top:10px}
.pr-k{font-family:var(--font-serif);font-weight:400;font-size:22px;color:var(--ink);margin-bottom:3px}
.pr-v{font-family:var(--font-sans);font-size:17px;line-height:1.42;color:var(--text-2)}
.warn-box{border:1.5px solid var(--ink);padding:15px 20px;font-family:var(--font-sans);
  font-size:18px;line-height:1.48;color:var(--text-2)}

.res-grid{display:grid;grid-template-columns:1fr 1fr;gap:84px;flex:1;align-items:start}
.punch{text-align:right;border-top:1.5px solid var(--ink);padding-top:11px;flex:0 0 auto}
.punch-n{font-family:var(--font-serif);font-weight:400;font-size:78px;line-height:1;
  letter-spacing:-.035em;color:var(--ink)}
.punch-l{font-family:var(--font-mono);font-size:14px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--text-3);margin-top:7px;max-width:240px;margin-left:auto}

.err-grid{display:grid;grid-template-columns:1.15fr .95fr .9fr;grid-template-rows:auto auto;
  gap:30px 42px;flex:1;align-content:space-between}
.err-grid .s-fig img{max-height:330px}

.fnd-row{display:grid;grid-template-columns:repeat(3,1fr);gap:44px}
.fnd{border-top:3px solid var(--ink);padding-top:12px}
.fnd-t{font-family:var(--font-serif);font-weight:400;font-size:26px;line-height:1.16;color:var(--ink);margin-bottom:7px}
.fnd-b{font-family:var(--font-sans);font-size:18px;line-height:1.48;color:var(--text-2)}
.close-grid{display:grid;grid-template-columns:1fr 1fr;gap:56px;margin-top:4px}
.repro{font-family:var(--font-sans);font-size:18px;color:var(--text-3);border-top:1px solid var(--line);padding-top:13px}
.repro .mono{color:var(--ink)}

/* --- cover: pixel-glitch column + QR patch --- */
.slide.cover{background:var(--paper)}
.cover .s-pad{padding:96px 80px 92px;justify-content:space-between}
.cover .deco{position:absolute;right:80px;top:200px;bottom:200px;width:150px;pointer-events:none;
  display:flex;flex-direction:column;justify-content:center;gap:7px}
.cover .deco i{display:block;height:7px;background:var(--ink);margin-left:auto}
.cover-top{display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:16px;
  letter-spacing:.16em;text-transform:uppercase;color:var(--ink)}
.cover-mid{display:grid;grid-template-columns:1.4fr .78fr;gap:70px;align-items:center;flex:1;padding-right:200px}
.cover h1{font-family:var(--font-serif);font-weight:300;font-size:132px;line-height:.94;
  letter-spacing:-.032em;color:var(--ink)}
.cover h1 em{font-style:italic;font-weight:400}
.cover-rule{width:300px;height:1.5px;background:var(--ink);margin:38px 0 30px}
.cover-sub{font-family:var(--font-sans);font-weight:300;font-size:27px;line-height:1.52;
  color:var(--text-2);max-width:800px}
.cover-sub strong{color:var(--ink);font-weight:500}
.cover-panel{padding:0}
.g-cap{font-family:var(--font-mono);font-size:14px;letter-spacing:.19em;text-transform:uppercase;
  color:var(--ink-soft);padding-bottom:14px;border-bottom:1.5px solid var(--ink);margin-bottom:4px}
.g-row{display:flex;justify-content:space-between;align-items:baseline;padding:15px 0;
  border-bottom:1px solid var(--line-soft)}
.g-row:last-child{border-bottom:0}
.g-k{font-family:var(--font-sans);font-size:21px;color:var(--text)}
.g-v{font-family:var(--font-mono);font-size:26px;color:var(--ink)}
.cover-foot{display:flex;justify-content:space-between;align-items:flex-end}
.cover-names{font-family:var(--font-sans);font-size:19px;color:var(--text-2)}
.cover-foot .s-brand{font-family:var(--font-mono);font-size:16px;letter-spacing:.13em}

/* --- entrance: rules extend, content rises --- */
.reveal{opacity:0;transform:translateY(22px);
  transition:opacity .8s cubic-bezier(.16,1,.3,1),transform .8s cubic-bezier(.16,1,.3,1)}
.slide.visible .reveal{opacity:1;transform:none}
.slide.visible .reveal:nth-child(1){transition-delay:.06s}
.slide.visible .reveal:nth-child(2){transition-delay:.16s}
.slide.visible .reveal:nth-child(3){transition-delay:.26s}
.slide.visible .reveal:nth-child(4){transition-delay:.36s}
.s-body>*{opacity:0;transform:translateY(18px);
  transition:opacity .7s cubic-bezier(.16,1,.3,1),transform .7s cubic-bezier(.16,1,.3,1)}
.slide.visible .s-body>*{opacity:1;transform:none}
.slide.visible .s-body>*:nth-child(1){transition-delay:.08s}
.slide.visible .s-body>*:nth-child(2){transition-delay:.18s}
.slide.visible .s-body>*:nth-child(3){transition-delay:.28s}
.slide.visible .s-body>*:nth-child(4){transition-delay:.38s}
.s-head{transform-origin:left;transition:border-color .5s ease .3s}
.s-chrome,.s-foot{opacity:0;transition:opacity .6s ease .04s}
.slide.visible .s-chrome,.slide.visible .s-foot{opacity:1}
.cover .deco i{transform:scaleX(0);transform-origin:right}
.slide.visible .deco i{animation:gx .5s cubic-bezier(.22,1,.36,1) forwards}
@keyframes gx{to{transform:scaleX(1)}}
"""

# Stair-stepped glitch bars injected into the cover's .deco element.
EXTRA_JS = """
/* Cobalt Grid: build the stair-stepped pixel-glitch column on the cover */
(function(){
  const d = document.querySelector('.cover .deco');
  if(!d) return;
  [132,96,150,64,112,40,86,120,54,100,72,140,48,92,66,118].forEach(function(w,i){
    const el = document.createElement('i');
    el.style.width = w + 'px';
    el.style.animationDelay = (0.45 + i*0.035) + 's';
    d.appendChild(el);
  });
})();
"""
