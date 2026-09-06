"""Semantic slide markup shared by all three decks.

The markup is intentionally theme-agnostic: each deck supplies its own stylesheet that
gives these classes a completely different typographic and decorative treatment.
"""
from pres_content import META, SLIDES


def _chrome(s, total):
    return (f'<div class="s-chrome"><span class="s-sec">{s["section"]}</span>'
            f'<span class="s-num">{s["n"]} / {total:02d}</span></div>')


def _foot(text=""):
    return f'<div class="s-foot"><span>{text}</span><span class="s-brand">CSE&#8209;4112</span></div>'


def _table(cols, rows, cls=""):
    th = "".join(f"<th>{c}</th>" for c in cols)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="s-table {cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def _bullets(items):
    return '<ul class="bul">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


# ------------------------------------------------------------------ layouts

def cover(s, total, figs):
    names = " &middot; ".join(n for _, n in META["team"])
    glance = "".join(
        f'<div class="g-row"><span class="g-k">{k}</span><span class="g-v">{v}</span></div>'
        for k, v in META["glance"])
    return f'''
<section class="slide cover active">
  <div class="deco"></div>
  <div class="s-pad">
    <div class="cover-top reveal">
      <span>{META["course"]} &mdash; {META["course_full"]}</span><span>{META["date"]}</span>
    </div>
    <div class="cover-mid">
      <div>
        <h1 class="reveal">{META["title_a"]}<br>{META["title_b"]} <em>{META["title_em"]}</em>?</h1>
        <div class="cover-rule reveal"></div>
        <p class="cover-sub reveal">{META["subtitle"]}</p>
      </div>
      <div class="cover-panel reveal">
        <div class="g-cap">Dataset at a glance</div>
        {glance}
      </div>
    </div>
    <div class="cover-foot reveal"><span class="cover-names">{names}</span><span class="s-brand">01</span></div>
  </div>
</section>'''


def two_col(s, total, figs):
    rows = "".join(f'<div class="xp-row"><span class="xp-k">{a}</span><span class="xp-v">{b}</span></div>'
                   for a, b in s["rows"])
    return f'''
<section class="slide">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="s-head-block">
        <div class="s-kicker">{s["kicker"]}</div>
        <h2 class="s-head">{s["head"]}</h2>
        <p class="s-lead">{s["lead"]}</p>
      </div>
      <div class="col2">
        <div class="panel">
          <h3 class="panel-title">{s["left_title"]}</h3>
          <p>{s["left_body"]}</p>
        </div>
        <div class="panel">
          <h3 class="panel-title">{s["right_title"]}</h3>
          <p>{s["right_body"]}</p>
        </div>
      </div>
      <div class="split-note">
        <div class="xp">
          <div class="xp-cap">{s["table_title"]}</div>
          {rows}
        </div>
        <div class="note">{s["note"]}</div>
      </div>
    </div>
    {_foot(s.get("foot", "Scope"))}
  </div>
</section>'''


def related(s, total, figs):
    stats = "".join(f'<div class="stat"><div class="stat-n">{n}</div><div class="stat-l">{l}</div></div>'
                    for n, l in s["base_stats"])
    contribs = "".join(
        f'<div class="cb"><div class="cb-t">{t}</div><div class="cb-b">{b}</div></div>'
        for t, b in s["contribs"])
    return f'''
<section class="slide">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="s-head-block tight">
        <div class="s-kicker">{s["kicker"]}</div>
        <h2 class="s-head">{s["head"]}</h2>
      </div>
      <div class="rel-grid">
        <div class="panel base">
          <h3 class="panel-title">{s["base_title"]}</h3>
          <p>{s["base_body"]}</p>
          <div class="stat-row">{stats}</div>
          <div class="note tight">{s["finding"]}</div>
        </div>
        <div>
          <div class="xp-cap">{s["contrib_title"]}</div>
          <div class="cb-wrap">{contribs}</div>
        </div>
      </div>
    </div>
    {_foot("Related works")}
  </div>
</section>'''


def dataset(s, total, figs):
    unis = "".join(f'<div class="chip"><span class="chip-k">{k}</span><span class="chip-v">{v}</span></div>'
                   for k, v in s["unis"])
    bands = "".join(f'<tr><td class="mono">{c}</td><td>{r}</td><td class="num">{n}</td></tr>'
                    for c, r, n in s["bands"])
    return f'''
<section class="slide">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="s-head-block tight">
        <div class="s-kicker">{s["kicker"]}</div>
        <h2 class="s-head">{s["head"]}</h2>
        <p class="s-lead">{s["lead"]}</p>
      </div>
      <div class="fig-grid">
        <figure class="s-fig"><img src="{figs[s["fig"]]}" alt="Target distribution">
          <figcaption>{s["cap"]}</figcaption></figure>
        <div>
          <div class="chip-row">{unis}</div>
          <table class="s-table bands"><thead><tr><th>Class</th><th>Range</th><th class="num">n</th></tr></thead>
            <tbody>{bands}</tbody></table>
          <div class="note">{s["note"]}</div>
        </div>
      </div>
    </div>
    {_foot("Dataset information")}
  </div>
</section>'''


def integrity(s, total, figs):
    return f'''
<section class="slide alt">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="s-head-block tight">
        <div class="s-kicker {s.get("kicker_cls", "warn-kicker")}">{s["kicker"]}</div>
        <h2 class="s-head">{s["head"]}</h2>
        <p class="s-lead">{s["lead"]}</p>
      </div>
      {_table(s["cols"], s["rows"], "wide")}
      <div class="col2 verdict-row">
        <div class="verdict"><div class="v-tag">Verdict</div><p>{s["verdict"]}</p></div>
        <div class="verdict"><div class="v-tag">What we did</div><p>{s["action"]}</p></div>
      </div>
    </div>
    {_foot(s.get("foot", "Data integrity"))}
  </div>
</section>'''


def preprocess(s, total, figs):
    steps = "".join(
        f'<div class="step"><div class="step-n">{i+1:02d}</div>'
        f'<div class="step-t">{t}</div><div class="step-w">{w}</div></div>'
        for i, (t, w) in enumerate(s["steps"]))
    stats = "".join(f'<div class="stat"><div class="stat-n">{n}</div><div class="stat-l">{l}</div></div>'
                    for n, l in s["stats"])
    return f'''
<section class="slide">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="head-flex">
        <div class="s-head-block tight">
          <div class="s-kicker">{s["kicker"]}</div>
          <h2 class="s-head">{s["head"]}</h2>
          <p class="s-lead">{s["lead"]}</p>
        </div>
        <div class="stat-row vert">{stats}</div>
      </div>
      <div class="step-grid">{steps}</div>
    </div>
    {_foot("Preprocessing")}
  </div>
</section>'''


def evidence(s, total, figs):
    return f'''
<section class="slide alt">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="s-head-block tight">
        <div class="s-kicker">{s["kicker"]}</div>
        <h2 class="s-head">{s["head"]}</h2>
      </div>
      <div class="ev-grid">
        <div class="panel">
          <h3 class="panel-title">{s["a_title"]}</h3>
          <p>{s["a_body"]}</p>
          <div class="act">{s["a_action"]}</div>
        </div>
        <div class="panel">
          <h3 class="panel-title">{s["b_title"]}</h3>
          <p>{s["b_body"]}</p>
          <div class="act">{s["b_action"]}</div>
        </div>
        <figure class="s-fig wide-fig"><img src="{figs[s["fig"]]}" alt="Semester by university">
          <figcaption>{s["cap"]}</figcaption></figure>
      </div>
    </div>
    {_foot(s.get("foot", "Preprocessing evidence"))}
  </div>
</section>'''


def importance(s, total, figs):
    ms = "".join(f'<div class="mth"><span class="mth-k">{a}</span><span class="mth-v">{b}</span></div>'
                 for a, b in s["methods"])
    return f'''
<section class="slide">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="s-head-block tight">
        <div class="s-kicker">{s["kicker"]}</div>
        <h2 class="s-head">{s["head"]}</h2>
        <p class="s-lead">{s["lead"]}</p>
      </div>
      <div class="fig-grid">
        <figure class="s-fig"><img src="{figs[s["fig"]]}" alt="Consensus feature importance"></figure>
        <div>
          <div class="mth-wrap">{ms}</div>
          <div class="note tight">{s["why"]}</div>
          <div class="verdict big">
            <div class="v-tag">{s["verdict_head"]}</div>
            <p>{s["verdict"]}</p>
          </div>
          <div class="act">{s["honest"]}</div>
        </div>
      </div>
    </div>
    {_foot("Pipeline A &mdash; feature influence")}
  </div>
</section>'''


def counter(s, total, figs):
    return f'''
<section class="slide alt">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="s-head-block tight">
        <div class="s-kicker">{s["kicker"]}</div>
        <h2 class="s-head">{s["head"]}</h2>
        <p class="s-lead">{s["lead"]}</p>
      </div>
      <div class="fig-grid rev">
        <div>
          <div class="panel">
            <h3 class="panel-title">{s["a_title"]}</h3>
            <p>{s["a_body"]}</p><div class="act">{s["a_read"]}</div>
          </div>
          <div class="panel">
            <h3 class="panel-title">{s["b_title"]}</h3>
            <p>{s["b_body"]}</p><div class="act">{s["b_read"]}</div>
          </div>
        </div>
        <figure class="s-fig"><img src="{figs[s["fig"]]}" alt="Counterintuitive effects">
          <figcaption>{s["caveat"]}</figcaption></figure>
      </div>
    </div>
    {_foot(s.get("foot", "Pipeline A &mdash; sanity checks"))}
  </div>
</section>'''


def models(s, total, figs):
    ms = "".join(f'<div class="mdl"><span class="mdl-k">{a}</span><span class="mdl-v">{b}</span></div>'
                 for a, b in s["models"])
    pr = "".join(f'<div class="pr"><div class="pr-k">{a}</div><div class="pr-v">{b}</div></div>'
                 for a, b in s["protocol"])
    return f'''
<section class="slide">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="s-head-block tight">
        <div class="s-kicker">{s["kicker"]}</div>
        <h2 class="s-head">{s["head"]}</h2>
      </div>
      <div class="mdl-grid">
        <div>
          <div class="xp-cap">Classifiers compared</div>
          <div class="mdl-wrap">{ms}</div>
        </div>
        <div>
          <div class="xp-cap">{s["protocol_title"]}</div>
          <div class="pr-wrap">{pr}</div>
        </div>
      </div>
      <div class="warn-box">{s["warn"]}</div>
    </div>
    {_foot("Models &amp; protocol")}
  </div>
</section>'''


def results(s, total, figs):
    def tbl(rows):
        body = "".join(
            '<tr class="' + ("base-row" if "ZeroR" in r[0] else "") + '">'
            + f'<td>{r[0]}</td><td class="num">{r[1]}</td><td class="num">{r[2]}</td><td class="num hl">{r[3]}</td></tr>'
            for r in rows)
        th = "".join(f'<th{" class=num" if i else ""}>{c}</th>' for i, c in enumerate(s["cols"]))
        return f'<table class="s-table res"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'
    return f'''
<section class="slide">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="head-flex">
        <div class="s-head-block tight">
          <div class="s-kicker">{s["kicker"]}</div>
          <h2 class="s-head">{s["head"]}</h2>
        </div>
        <div class="punch"><div class="punch-n">{s["punch_n"]}</div><div class="punch-l">{s["punch_l"]}</div></div>
      </div>
      <div class="res-grid">
        <div><div class="xp-cap">{s["t1_title"]}</div>{tbl(s["t1"])}</div>
        <div><div class="xp-cap">{s["t2_title"]}</div>{tbl(s["t2"])}</div>
      </div>
      <div class="note">{s["read"]}</div>
    </div>
    {_foot("Comparison between models")}
  </div>
</section>'''


def errors(s, total, figs):
    gaps = "".join(
        f'<tr><td class="mono num">{g}</td><td>{m}</td><td class="num">{a}</td><td class="num">{b}</td></tr>'
        for g, m, a, b in s["gaps"])
    th = "".join(f'<th{" class=num" if i in (0,2,3) else ""}>{c}</th>' for i, c in enumerate(s["gap_cols"]))
    return f'''
<section class="slide alt">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="s-head-block tight">
        <div class="s-kicker">{s["kicker"]}</div>
        <h2 class="s-head">{s["head"]}</h2>
      </div>
      <div class="err-grid">
        <figure class="s-fig"><img src="{figs[s["fig"]]}" alt="Confusion matrices"></figure>
        <div>
          <div class="xp-cap">{s["gap_title"]}</div>
          <table class="s-table gap"><thead><tr>{th}</tr></thead><tbody>{gaps}
            <tr class="tot"><td colspan="2">within one band</td>
            <td class="num hl">{s["within"][0]}</td><td class="num hl">{s["within"][1]}</td></tr></tbody></table>
        </div>
        <div>
          <figure class="s-fig sm"><img src="{figs[s["fig2"]]}" alt="Learning curve"></figure>
          <div class="panel tight">
            <h3 class="panel-title">{s["lc_title"]}</h3>
            <p>{s["lc_body"]}</p>
          </div>
        </div>
        <div class="panel tight">
          <h3 class="panel-title">{s["improve_title"]}</h3>
          {_bullets(s["improve"])}
        </div>
      </div>
    </div>
    {_foot("Error analysis")}
  </div>
</section>'''


def closing(s, total, figs):
    fs = "".join(f'<div class="fnd"><div class="fnd-t">{t}</div><div class="fnd-b">{b}</div></div>'
                 for t, b in s["findings"])
    wk = "".join(f'<tr><td>{n}</td><td class="role">{r}</td></tr>' for n, r in s["work"])
    return f'''
<section class="slide">
  <div class="s-pad">
    {_chrome(s, total)}
    <div class="s-body">
      <div class="s-head-block tight">
        <div class="s-kicker">{s["kicker"]}</div>
        <h2 class="s-head">{s["head"]}</h2>
      </div>
      <div class="fnd-row">{fs}</div>
      <div class="close-grid">
        <div>
          <div class="xp-cap">{s["lim_title"]}</div>
          {_bullets(s["lims"])}
        </div>
        <div>
          <div class="xp-cap">{s["work_title"]} <span class="tbd">&mdash; {s["work_note"]}</span></div>
          <table class="s-table work"><tbody>{wk}</tbody></table>
        </div>
      </div>
      <div class="repro">{s["repro"]}</div>
    </div>
    {_foot("Conclusions")}
  </div>
</section>'''


LAYOUTS = {
    "cover": cover, "two-col": two_col, "related": related, "dataset": dataset,
    "integrity": integrity, "preprocess": preprocess, "evidence": evidence,
    "importance": importance, "counter": counter, "models": models,
    "results": results, "errors": errors, "closing": closing,
}


def render_slides(figs):
    total = len(SLIDES)
    return "\n".join(LAYOUTS[s["kind"]](s, total, figs) for s in SLIDES)
