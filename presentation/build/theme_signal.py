"""SIGNAL — literary editorial system.

Dual surfaces (navy / cream), Source Serif 4 headlines with gold italics, DM Sans for
substance, IBM Plex Mono for all chrome. One accent: antique gold, used only on rules,
italic emphasis and numerals. Flat plus hairline — no shadows, no rounded cards.
An 80px grid texture at 3% white is the fingerprint on every dark slide.
"""

NAME = "Signal"
FONTS = ("https://fonts.googleapis.com/css2?"
         "family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;"
         "1,8..60,400;1,8..60,600&family=DM+Sans:wght@300;400;500&"
         "family=IBM+Plex+Mono:wght@400;500&display=swap")

CSS = r"""
:root{
  --navy:#1C2644; --navy-alt:#232F55; --cream:#F0ECE3; --cream-alt:#E6E0D4;
  --warm:#E2DCD0; --muted:#8A96A8; --hint:#4E5A6E;
  --ink:#1A2030; --muted-l:#5A6270; --hint-l:#9AA0A8;
  --gold:#C8A870; --bd:#2E3D5C; --bd-l:#CAC4B4;
  --stage-bg:#0D1222; --slide-bg:var(--navy);
  --font-serif:'Source Serif 4',Georgia,serif;
  --font-sans:'DM Sans',system-ui,sans-serif;
  --font-mono:'IBM Plex Mono',monospace;
  --chrome-accent:#C8A870; --chrome-fg:#8A96A8;
}
*{margin:0;padding:0;box-sizing:border-box}

/* dual surface: default navy, .alt flips to cream */
.slide{background:var(--navy);color:var(--warm)}
.slide.alt{background:var(--cream);color:var(--ink)}

/* 80px grid fingerprint — dark slides only */
.slide::before{content:"";position:absolute;inset:0;pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.03) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(255,255,255,.03) 1px,transparent 1px);
  background-size:80px 80px}
.slide.alt::before{opacity:0}

.s-pad{position:absolute;inset:0;padding:58px 104px 52px;display:flex;flex-direction:column}

/* --- chrome --- */
.s-chrome{display:flex;justify-content:space-between;align-items:baseline;
  font-family:var(--font-mono);font-size:15px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--hint);border-bottom:1px solid var(--bd);padding-bottom:14px;flex:0 0 auto}
.slide.alt .s-chrome{color:var(--hint-l);border-color:var(--bd-l)}
.s-num{color:var(--gold)}
.s-foot{display:flex;justify-content:space-between;align-items:baseline;
  border-top:1px solid var(--bd);padding-top:13px;margin-top:auto;flex:0 0 auto;
  font-family:var(--font-mono);font-size:14px;letter-spacing:.16em;text-transform:uppercase;color:var(--hint)}
.slide.alt .s-foot{color:var(--hint-l);border-color:var(--bd-l)}
.s-brand{color:var(--gold)}

.s-body{flex:1 1 auto;min-height:0;padding:26px 0 18px;display:flex;flex-direction:column;gap:22px;justify-content:space-between}

/* --- headings --- */
.s-kicker{font-family:var(--font-mono);font-size:15px;letter-spacing:.2em;text-transform:uppercase;
  color:var(--gold);margin-bottom:14px}
.s-kicker.warn-kicker::before{content:"\25C6  "}
.s-head{font-family:var(--font-serif);font-weight:600;font-size:58px;line-height:1.1;
  letter-spacing:-.012em;color:var(--warm)}
.slide.alt .s-head{color:var(--ink)}
.s-head em{font-style:italic;color:var(--gold);font-weight:600}
.s-lead{font-family:var(--font-sans);font-weight:300;font-size:25px;line-height:1.5;
  color:var(--muted);max-width:1500px;margin-top:16px}
.slide.alt .s-lead{color:var(--muted-l)}
.s-lead strong{color:var(--warm);font-weight:500}
.slide.alt .s-lead strong{color:var(--ink)}
.s-lead em,.panel p em,.note em,.act em{font-family:var(--font-serif);font-style:italic;color:var(--gold);font-weight:600}
.s-head-block.tight .s-lead{margin-top:12px;font-size:23px}
.head-flex{display:flex;justify-content:space-between;align-items:flex-start;gap:60px}

/* --- panels: hairline top rule, never a card --- */
.panel{border-top:1px solid var(--bd);padding:20px 0 0}
.slide.alt .panel{border-color:var(--bd-l)}
.panel.tight{padding-top:16px}
.panel-title{font-family:var(--font-serif);font-weight:600;font-size:27px;line-height:1.25;
  color:var(--warm);margin-bottom:11px}
.slide.alt .panel-title{color:var(--ink)}
.panel p{font-family:var(--font-sans);font-size:20px;line-height:1.6;color:var(--muted)}
.slide.alt .panel p{color:var(--muted-l)}
.act{font-family:var(--font-sans);font-size:19px;line-height:1.55;color:var(--hint);
  border-left:2px solid var(--gold);padding-left:16px;margin-top:14px}
.slide.alt .act{color:var(--muted-l)}
.note{font-family:var(--font-sans);font-size:20px;line-height:1.58;color:var(--muted);
  border-left:2px solid var(--gold);padding-left:18px}
.slide.alt .note{color:var(--muted-l)}
.note.tight{font-size:18px;margin:12px 0}
.note strong,.act strong,.panel p strong{color:var(--warm);font-weight:500}
.slide.alt .note strong,.slide.alt .act strong,.slide.alt .panel p strong{color:var(--ink)}

.col2{display:grid;grid-template-columns:1fr 1fr;gap:52px}
.split-note{display:grid;grid-template-columns:1fr 1fr;gap:52px;align-items:start}

/* --- mono index rows --- */
.xp-cap{font-family:var(--font-mono);font-size:14px;letter-spacing:.2em;text-transform:uppercase;
  color:var(--gold);margin-bottom:13px}
.xp-row{display:flex;justify-content:space-between;gap:24px;padding:11px 0;border-bottom:1px solid var(--bd)}
.slide.alt .xp-row{border-color:var(--bd-l)}
.xp-k{font-family:var(--font-mono);font-size:16px;letter-spacing:.1em;text-transform:uppercase;color:var(--warm)}
.slide.alt .xp-k{color:var(--ink)}
.xp-v{font-family:var(--font-sans);font-size:19px;color:var(--muted);text-align:right}
.slide.alt .xp-v{color:var(--muted-l)}

/* --- tables --- */
.s-table{width:100%;border-collapse:collapse;font-family:var(--font-sans)}
.s-table th{font-family:var(--font-mono);font-size:14px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--gold);text-align:left;padding:0 14px 11px 0;border-bottom:1px solid var(--gold);font-weight:500}
.s-table td{font-size:19px;color:var(--muted);padding:12px 14px 12px 0;border-bottom:1px solid var(--bd);
  vertical-align:top;line-height:1.42}
.slide.alt .s-table td{color:var(--muted-l);border-color:var(--bd-l)}
.s-table td:first-child{color:var(--warm)}
.slide.alt .s-table td:first-child{color:var(--ink)}
.s-table .num,.s-table th.num{text-align:right;font-family:var(--font-mono);font-variant-numeric:tabular-nums}
.s-table .hl{color:var(--gold)}
.s-table.wide td{font-size:20px;padding:14px 16px 14px 0}
.s-table.res td{font-size:20px;padding:11px 14px 11px 0}
.s-table.res .base-row td{color:var(--hint);font-style:italic}
.slide.alt .s-table.res .base-row td{color:var(--hint-l)}
.s-table.gap td{font-size:19px;padding:12px 12px 12px 0}
.s-table.gap .tot td{border-bottom:0;border-top:1px solid var(--gold);color:var(--warm);font-weight:500}
.slide.alt .s-table.gap .tot td{color:var(--ink)}
.s-table.bands td{font-size:19px;padding:10px 12px 10px 0}
.s-table.work td{font-size:18px;padding:10px 12px 10px 0}
.s-table.work .role{color:var(--muted);text-align:right}
.slide.alt .s-table.work .role{color:var(--muted-l)}
.mono{font-family:var(--font-mono)}
.tbd{font-family:var(--font-sans);text-transform:none;letter-spacing:0;color:var(--hint);font-size:15px}

/* --- stats: gold serif numerals --- */
.stat-row{display:flex;gap:44px}
.stat-row.vert{gap:34px;flex:0 0 auto}
.stat{border-top:1px solid var(--bd);padding-top:12px;min-width:96px}
.slide.alt .stat{border-color:var(--bd-l)}
.stat-n{font-family:var(--font-serif);font-weight:600;font-size:54px;line-height:1;
  letter-spacing:-.02em;color:var(--gold)}
.stat-l{font-family:var(--font-mono);font-size:14px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--hint);margin-top:7px}
.slide.alt .stat-l{color:var(--hint-l)}

/* --- bullets: em-dash in gold mono --- */
.bul{list-style:none}
.bul li{font-family:var(--font-sans);font-size:19px;line-height:1.5;color:var(--muted);
  padding-left:30px;position:relative;margin-bottom:12px}
.slide.alt .bul li{color:var(--muted-l)}
.bul li::before{content:"\2014";position:absolute;left:0;color:var(--gold);font-family:var(--font-mono)}
.bul li em{font-family:var(--font-serif);font-style:italic;color:var(--gold)}

/* --- figures --- */
.s-fig{display:flex;flex-direction:column;gap:10px;min-height:0}
.s-fig img{width:100%;height:auto;object-fit:contain;border:1px solid var(--bd);background:var(--cream)}
.slide.alt .s-fig img{border-color:var(--bd-l)}
.s-fig figcaption{font-family:var(--font-sans);font-size:16px;line-height:1.45;color:var(--hint)}
.slide.alt .s-fig figcaption{color:var(--hint-l)}
.fig-grid{display:grid;grid-template-columns:1fr 1fr;gap:52px;align-items:start;min-height:0;flex:1}
.fig-grid.rev{grid-template-columns:1fr 1fr}
.fig-grid .s-fig img{max-height:500px}
.wide-fig{grid-column:1 / -1}
.wide-fig img{max-height:300px}
.s-fig.sm img{max-height:230px}

/* --- specific blocks --- */
.rel-grid{display:grid;grid-template-columns:1.05fr 1fr;gap:56px;align-items:start}
.cb-wrap{display:flex;flex-direction:column;gap:15px}
.cb{border-top:1px solid var(--bd);padding-top:12px}
.slide.alt .cb{border-color:var(--bd-l)}
.cb-t{font-family:var(--font-serif);font-weight:600;font-size:22px;color:var(--warm);margin-bottom:5px}
.slide.alt .cb-t{color:var(--ink)}
.cb-b{font-family:var(--font-sans);font-size:17px;line-height:1.5;color:var(--muted)}
.slide.alt .cb-b{color:var(--muted-l)}

.chip-row{display:flex;gap:18px;margin-bottom:22px}
.chip{border:1px solid var(--gold);padding:9px 17px;display:flex;gap:11px;align-items:baseline}
.chip-k{font-family:var(--font-mono);font-size:14px;letter-spacing:.14em;color:var(--gold)}
.chip-v{font-family:var(--font-serif);font-size:24px;font-weight:600;color:var(--warm)}
.slide.alt .chip-v{color:var(--ink)}

.verdict-row{margin-top:6px}
.verdict{border-left:3px solid var(--gold);padding-left:20px}
.verdict.big{margin:16px 0}
.v-tag{font-family:var(--font-mono);font-size:14px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--gold);margin-bottom:9px}
.verdict p{font-family:var(--font-sans);font-size:20px;line-height:1.55;color:var(--muted)}
.slide.alt .verdict p{color:var(--muted-l)}
.verdict p strong{color:var(--warm);font-weight:500}
.slide.alt .verdict p strong{color:var(--ink)}

.step-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:26px 44px;flex:1;align-content:space-between}
.step{border-top:1px solid var(--bd);padding-top:12px}
.slide.alt .step{border-color:var(--bd-l)}
.step-n{font-family:var(--font-mono);font-size:14px;letter-spacing:.16em;color:var(--gold);margin-bottom:6px}
.step-t{font-family:var(--font-serif);font-weight:600;font-size:21px;line-height:1.25;color:var(--warm);margin-bottom:5px}
.slide.alt .step-t{color:var(--ink)}
.step-w{font-family:var(--font-sans);font-size:16px;line-height:1.45;color:var(--muted)}
.slide.alt .step-w{color:var(--muted-l)}
.step-t .mono,.step-w .mono{font-family:var(--font-mono);font-size:.9em}

.ev-grid{display:grid;grid-template-columns:1fr 1fr;gap:34px 52px;flex:1;align-content:space-between}

.mth-wrap{display:flex;flex-direction:column;gap:0;margin-bottom:6px}
.mth{display:flex;justify-content:space-between;gap:20px;padding:9px 0;border-bottom:1px solid var(--bd)}
.slide.alt .mth{border-color:var(--bd-l)}
.mth-k{font-family:var(--font-serif);font-size:20px;font-weight:600;color:var(--warm)}
.slide.alt .mth-k{color:var(--ink)}
.mth-v{font-family:var(--font-sans);font-size:16px;color:var(--hint);text-align:right}
.slide.alt .mth-v{color:var(--hint-l)}

.mdl-grid{display:grid;grid-template-columns:.85fr 1.15fr;gap:56px;flex:1;align-content:start}
.mdl-grid>div{display:flex;flex-direction:column}
.mdl-wrap,.pr-wrap{flex:1;justify-content:space-between}
.mdl-wrap{display:flex;flex-direction:column}
.mdl{display:flex;justify-content:space-between;gap:18px;padding:10px 0;border-bottom:1px solid var(--bd)}
.slide.alt .mdl{border-color:var(--bd-l)}
.mdl-k{font-family:var(--font-mono);font-size:17px;letter-spacing:.06em;color:var(--warm)}
.slide.alt .mdl-k{color:var(--ink)}
.mdl-v{font-family:var(--font-sans);font-size:16px;color:var(--hint);text-align:right}
.slide.alt .mdl-v{color:var(--hint-l)}
.pr-wrap{display:flex;flex-direction:column;gap:13px}
.pr{border-top:1px solid var(--bd);padding-top:10px}
.slide.alt .pr{border-color:var(--bd-l)}
.pr-k{font-family:var(--font-serif);font-weight:600;font-size:21px;color:var(--warm);margin-bottom:4px}
.slide.alt .pr-k{color:var(--ink)}
.pr-v{font-family:var(--font-sans);font-size:17px;line-height:1.45;color:var(--muted)}
.slide.alt .pr-v{color:var(--muted-l)}
.warn-box{border:1px solid var(--gold);padding:16px 22px;font-family:var(--font-sans);
  font-size:18px;line-height:1.5;color:var(--muted)}
.slide.alt .warn-box{color:var(--muted-l)}

.res-grid{display:grid;grid-template-columns:1fr 1fr;gap:84px;flex:1;align-items:start}
.punch{text-align:right;border-top:1px solid var(--gold);padding-top:12px;flex:0 0 auto}
.punch-n{font-family:var(--font-serif);font-weight:600;font-size:76px;line-height:1;
  letter-spacing:-.02em;color:var(--gold)}
.punch-l{font-family:var(--font-mono);font-size:14px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--hint);margin-top:8px;max-width:230px;margin-left:auto}
.slide.alt .punch-l{color:var(--hint-l)}

.err-grid{display:grid;grid-template-columns:1.15fr .95fr .9fr;grid-template-rows:auto auto;
  gap:30px 42px;flex:1;align-content:space-between}
.err-grid .s-fig img{max-height:330px}

.fnd-row{display:grid;grid-template-columns:repeat(3,1fr);gap:44px}
.fnd{border-top:2px solid var(--gold);padding-top:13px}
.fnd-t{font-family:var(--font-serif);font-weight:600;font-size:25px;line-height:1.2;color:var(--warm);margin-bottom:8px}
.slide.alt .fnd-t{color:var(--ink)}
.fnd-b{font-family:var(--font-sans);font-size:18px;line-height:1.5;color:var(--muted)}
.slide.alt .fnd-b{color:var(--muted-l)}
.close-grid{display:grid;grid-template-columns:1fr 1fr;gap:56px;margin-top:6px}
.repro{font-family:var(--font-sans);font-size:18px;color:var(--hint);border-top:1px solid var(--bd);padding-top:14px}
.slide.alt .repro{color:var(--hint-l);border-color:var(--bd-l)}
.repro .mono{font-family:var(--font-mono);color:var(--gold)}

/* --- cover --- */
.slide.cover{background:var(--navy)}
.cover .deco{position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(circle 620px at 88% -6%,rgba(200,168,112,.13) 0%,rgba(200,168,112,0) 70%)}
.cover .s-pad{padding:72px 104px 62px;justify-content:space-between}
.cover-top{display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:17px;
  letter-spacing:.18em;text-transform:uppercase;color:var(--muted);
  border-bottom:1px solid var(--bd);padding-bottom:20px}
.cover-mid{display:grid;grid-template-columns:1.32fr 1fr;gap:92px;align-items:center;flex:1}
.cover h1{font-family:var(--font-serif);font-weight:400;font-size:112px;line-height:1.04;
  letter-spacing:-.022em;color:var(--warm)}
.cover h1 em{font-style:italic;color:var(--gold);font-weight:600}
.cover-rule{width:118px;height:2px;background:var(--gold);margin:40px 0 32px}
.cover-sub{font-family:var(--font-sans);font-weight:300;font-size:27px;line-height:1.55;
  color:#B9C1CE;max-width:820px}
.cover-sub strong{color:var(--warm);font-weight:500}
.cover-panel{background:var(--cream);padding:44px 44px 38px}
.g-cap{font-family:var(--font-mono);font-size:14px;letter-spacing:.2em;text-transform:uppercase;
  color:#6E6B62;padding-bottom:16px;border-bottom:1px solid rgba(28,38,68,.16);margin-bottom:8px}
.g-row{display:flex;justify-content:space-between;align-items:baseline;padding:13px 0;
  border-bottom:1px solid rgba(28,38,68,.09)}
.g-row:last-child{border-bottom:0}
.g-k{font-family:var(--font-sans);font-size:21px;color:#2A3350}
.g-v{font-family:var(--font-serif);font-size:29px;font-weight:600;color:#9C7F45}
.cover-foot{display:flex;justify-content:space-between;align-items:flex-end;
  border-top:1px solid var(--bd);padding-top:20px}
.cover-names{font-family:var(--font-sans);font-size:19px;color:var(--muted)}
.cover-foot .s-brand{font-family:var(--font-mono);font-size:16px;letter-spacing:.14em}

/* --- entrance --- */
.reveal{opacity:0;transform:translateY(24px);
  transition:opacity .75s cubic-bezier(.16,1,.3,1),transform .75s cubic-bezier(.16,1,.3,1)}
.slide.visible .reveal{opacity:1;transform:none}
.slide.visible .reveal:nth-child(1){transition-delay:.06s}
.slide.visible .reveal:nth-child(2){transition-delay:.16s}
.slide.visible .reveal:nth-child(3){transition-delay:.26s}
.slide.visible .reveal:nth-child(4){transition-delay:.36s}
.s-body>*{opacity:0;transform:translateY(20px);
  transition:opacity .7s cubic-bezier(.16,1,.3,1),transform .7s cubic-bezier(.16,1,.3,1)}
.slide.visible .s-body>*{opacity:1;transform:none}
.slide.visible .s-body>*:nth-child(1){transition-delay:.08s}
.slide.visible .s-body>*:nth-child(2){transition-delay:.18s}
.slide.visible .s-body>*:nth-child(3){transition-delay:.28s}
.slide.visible .s-body>*:nth-child(4){transition-delay:.38s}
.s-chrome,.s-foot{opacity:0;transition:opacity .6s ease .04s}
.slide.visible .s-chrome,.slide.visible .s-foot{opacity:1}
"""
