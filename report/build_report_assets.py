r"""Build every number, table and figure used in report/CSE_4112_Dataset_Report.tex.

Reads the same inputs as notebooks/final_weka_results.py, so the report matches the presentation:
    cleaned-dataset/ours/merged/merged_full_clean.csv, raw-data/*.csv, docs/final_results.json
Writes:
    report/tables/*.tex   LaTeX table bodies (\input{} from the report)
    report/figures/*.png  charts drawn for the report
    report/figures/deck/  charts copied from docs/figures/final (the ones shown in the slides)
    report/data/report_stats.json
Run:  .\venv\Scripts\python.exe report\build_report_assets.py
"""
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"
FIG, TAB, DATA = REP / "figures", REP / "tables", REP / "data"
for d in (FIG, TAB, DATA, FIG / "deck"):
    d.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "docs" / "final_results.json").read_text(encoding="utf-8"))

ITEMS = ["desired_department_match", "attendance", "weekly_study_time", "study_style", "topic_clarity",
         "sleep_duration", "stress_frequency", "weekly_responsibilities", "distraction_frequency",
         "study_environment", "support_level", "routine_manageability"]
EXTRA = ["admission_satisfaction", "career_expectation", "result_satisfaction"]
BANDS = ["C0", "C1", "C2", "C3"]
BAND_COLORS = ["#A83A3A", "#DC713D", "#4C91D8", "#285B9F"]     # same colours as the slides
BLUE, ORANGE, INK, MUTED, GRID = "#2A78D6", "#DC713D", "#1F2328", "#5B6470", "#D9DDE2"

QUESTION = {
    "desired_department_match": "Did you get into your desired subject or department?",
    "attendance": "How often were you actually present in class?",
    "weekly_study_time": "Outside class, how much did you usually study per week?",
    "study_style": "Which study style sounds most like you?",
    "topic_clarity": "When a class ended, how much of the topic was usually clear to you?",
    "sleep_duration": "On a normal night, how much sleep did you get?",
    "stress_frequency": "How often did stress make it difficult for you to study?",
    "weekly_responsibilities": "How much time did tuition, a job, club activities or other responsibilities take each week?",
    "distraction_frequency": "How often did your phone, gaming, social media or entertainment steal your planned study time?",
    "study_environment": "How study-friendly was the place where you lived?",
    "support_level": "When academic life became difficult, how much support did you receive from family or close friends?",
    "routine_manageability": "Overall, how healthy and manageable was your semester routine?",
    "admission_satisfaction": "How satisfied are you with getting admitted into your university (KUET form: into KUET)?",
    "career_expectation": "Do you think this course of study will take you to your dream career?",
    "result_satisfaction": "How satisfied were you with your latest semester result?",
}
SHORT = {
    "desired_department_match": "Got desired department", "attendance": "Class attendance",
    "weekly_study_time": "Self-study per week", "study_style": "Study style",
    "topic_clarity": "Topic clear after class", "sleep_duration": "Sleep per night",
    "stress_frequency": "Stress blocked studying", "weekly_responsibilities": "Outside commitments",
    "distraction_frequency": "Phone / social media distraction", "study_environment": "Study-friendly living place",
    "support_level": "Family / friend support", "routine_manageability": "Semester routine manageable",
    "admission_satisfaction": "Satisfied with admission", "career_expectation": "Course leads to dream career",
    "result_satisfaction": "Satisfied with last result",
}
# raw column index of each question in Source A (multi-university form) and Source B (KUET form)
RAW_A = {"admission_satisfaction": 2, "desired_department_match": 5, "result_satisfaction": 9, "attendance": 10,
         "weekly_study_time": 11, "study_style": 12, "topic_clarity": 13, "sleep_duration": 14,
         "stress_frequency": 15, "weekly_responsibilities": 16, "distraction_frequency": 17,
         "study_environment": 18, "support_level": 19, "routine_manageability": 20, "career_expectation": 21,
         "department": 3, "semester": 6, "recent_sgpa": 7, "cgpa": 8}
RAW_B = {"attendance": 5, "weekly_study_time": 6, "study_style": 7, "topic_clarity": 8, "sleep_duration": 9,
         "stress_frequency": 10, "weekly_responsibilities": 11, "desired_department_match": 12,
         "distraction_frequency": 13, "study_environment": 14, "support_level": 15, "admission_satisfaction": 16,
         "career_expectation": 17, "result_satisfaction": 18, "routine_manageability": 19,
         "department": 1, "semester": 2, "recent_sgpa": 3, "cgpa": 4}

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.titleweight": "bold",
                     "axes.titlesize": 8.5, "axes.titlecolor": INK, "savefig.dpi": 300})


def esc(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(ch, ch) for ch in str(s))


def write(name, text):
    (TAB / name).write_text(text, encoding="utf-8")


def n(x):
    return f"{int(x):,}"


# ================================================================ rebuild 1,108 -> 1,102 -> 920 (as the deck)
clean = pd.read_csv(ROOT / "cleaned-dataset" / "ours" / "merged" / "merged_full_clean.csv",
                    keep_default_na=False, na_values=[""])
raw_a = pd.read_csv(ROOT / "raw-data" / "University Student Performance Analysis Form (Responses) - Form Responses 1.csv",
                    keep_default_na=False, na_values=[""])
raw_b = pd.read_csv(ROOT / "raw-data" / "Student Performance Analysis Form (Responses) - Form Responses 1.csv",
                    keep_default_na=False, na_values=[""])
BAND_CODE = {"Below": 0, "3.0+": 0, "3.2+": 1, "3.5+": 2, "3.75+": 3, "3.9+": 3,
             "3.20 - 3.49": 1, "3.2 - 3.49": 1, "3.20-3.49": 1, "3.2-3.49": 1,
             "3.50 - 3.74": 2, "3.50-3.74": 2, "3.5 - 3.74": 2}


def band_code(v):
    if not isinstance(v, str):
        return np.nan
    s = " ".join(v.split(" (")[0].split())
    if s in BAND_CODE:
        return BAND_CODE[s]
    try:
        x = float(s)
        if 0 <= x <= 4:
            return 0 if x < 3.2 else 1 if x < 3.5 else 2 if x < 3.75 else 3
    except ValueError:
        pass
    return np.nan


stamp = pd.concat([
    pd.DataFrame({"ts": raw_a.iloc[:, 0], "cg": raw_a.iloc[:, 8].map(band_code), "sg": raw_a.iloc[:, 7].map(band_code)}),
    pd.DataFrame({"ts": raw_b.iloc[:, 0], "cg": raw_b.iloc[:, 4].map(band_code), "sg": raw_b.iloc[:, 3].map(band_code)}),
], ignore_index=True)
stamp = stamp[stamp.cg.notna()].reset_index(drop=True)
assert len(stamp) == len(clean) and (stamp.cg.values == clean.target_code.values).all()
clean["submitted"] = pd.to_datetime(stamp.ts, format="mixed")
has_sgpa = ~clean.recent_sgpa_imputed.astype(str).str.lower().eq("true")
base = clean[has_sgpa.values].reset_index(drop=True)
base["date"] = base.submitted.dt.strftime("%Y-%m-%d")
base["gap"] = (base.recent_sgpa_code - base.target_code).abs().astype(int)
base["aug67_group"] = (base.source_form == "kuet") & (base.semester_raw == 7) & base.date.isin(["2026-08-06", "2026-08-07"])
base["dropped"] = base.aug67_group & (base.gap >= 2)
final = base[~base.dropped].reset_index(drop=True)
assert len(clean) == 1108 and len(base) == 1102 and len(final) == 920
F = len(final)
S = {}

# ================================================================ 1. sources and funnel
raw_ts_a = pd.to_datetime(raw_a.iloc[:, 0], format="mixed")
raw_ts_b = pd.to_datetime(raw_b.iloc[:, 0], format="mixed")
src_rows = []
for key, label, raw, ts, unis in [("multi_uni", "A: University Student Performance Analysis Form", raw_a, raw_ts_a, "BUET, BRAC, CUET"),
                                  ("kuet", "B: Student Performance Analysis Form (KUET)", raw_b, raw_ts_b, "KUET")]:
    src_rows.append({
        "key": key, "label": label, "universities": unis, "raw": len(raw), "cols": raw.shape[1],
        "start": ts.min().strftime("%d %b %Y"), "end": ts.max().strftime("%d %b %Y"),
        "clean": int((clean.source_form == key).sum()), "base": int((base.source_form == key).sum()),
        "final": int((final.source_form == key).sum()),
    })
S["sources"] = src_rows
write("sources.tex", "".join(
    f"{esc(r['label'])} & {esc(r['universities'])} & {r['start']} -- {r['end']} & {r['cols']} & {n(r['raw'])} & "
    f"{n(r['clean'])} & {n(r['final'])} \\\\ \\hline\n" for r in src_rows)
    + f"\\textbf{{Total}} & 4 universities & 31 Jul -- 03 Sep 2026 & -- & \\textbf{{{n(len(raw_a) + len(raw_b))}}} & "
      f"\\textbf{{{n(len(clean))}}} & \\textbf{{{F}}} \\\\ \\hline\n")

G = R["gap_analysis"]
S["funnel"] = [
    ("Raw responses from the two Google Forms", len(raw_a) + len(raw_b), "--"),
    ("Remove responses with no usable CGPA answer (blank or ``I don't know'')", len(clean), len(raw_a) + len(raw_b) - len(clean)),
    ("Remove responses with no real recent-SGPA answer", len(base), len(clean) - len(base)),
    ("Remove 6--7 Aug KUET semester-7 answers whose SGPA and CGPA bands are 2--3 apart", F, G["dropped_rows"]),
]

# ================================================================ 2. raw missingness per question
miss_rows = []
for key in ["department", "semester", "recent_sgpa", "cgpa"] + ITEMS + EXTRA:
    a = int(raw_a.iloc[:, RAW_A[key]].isna().sum())
    b = int(raw_b.iloc[:, RAW_B[key]].isna().sum())
    miss_rows.append({"field": key, "a": a, "b": b, "total": a + b, "pct": 100 * (a + b) / 1126})
S["raw_missing"] = miss_rows
S["raw_a_weekly_responsibilities_values"] = raw_a.iloc[:, 16].value_counts(dropna=False).to_dict()
S["raw_a_department_other_blank"] = int(raw_a.iloc[:, 4].isna().sum())
S["raw_b_email_blank"] = int(raw_b.iloc[:, 20].isna().sum())

# ================================================================ 3. composition of the 920
uni_order = ["KUET", "BUET", "BRAC", "CUET"]
ct = pd.crosstab(final.university, final.target_code).reindex(uni_order).reindex(columns=range(4), fill_value=0)
rows = ""
for u in uni_order:
    sub = final[final.university == u]
    sem = sub.semester_raw.dropna()
    rows += (f"{u} & {n(len(sub))} & {100 * len(sub) / F:.1f} & " + " & ".join(str(int(v)) for v in ct.loc[u])
             + f" & {int(sem.min())}--{int(sem.max())} & {sub.submitted.min():%d %b} -- {sub.submitted.max():%d %b} \\\\ \\hline\n")
tot = ct.sum()
rows += (f"\\textbf{{Total}} & \\textbf{{{F}}} & 100.0 & " + " & ".join(f"\\textbf{{{int(v)}}}" for v in tot)
         + " & 1--8 & 31 Jul -- 03 Sep \\\\ \\hline\n")
rows += ("\\% of 920 & & & " + " & ".join(f"{100 * v / F:.1f}" for v in tot) + " & & \\\\ \\hline\n")
write("university_cgpa.tex", rows)
S["university_cgpa"] = {u: [int(v) for v in ct.loc[u]] for u in uni_order}

sem_ct = pd.crosstab(final.semester_raw.astype(int), final.university).reindex(columns=uni_order, fill_value=0)
rows = ""
for s_, r in sem_ct.iterrows():
    rows += f"{s_} & " + " & ".join(str(int(v)) for v in r) + f" & {int(r.sum())} & {100 * r.sum() / F:.1f} \\\\ \\hline\n"
rows += "\\textbf{Total} & " + " & ".join(f"\\textbf{{{int(v)}}}" for v in sem_ct.sum()) + f" & \\textbf{{{F}}} & 100.0 \\\\ \\hline\n"
write("semester_university.tex", rows)
S["semester_university"] = {int(k): [int(x) for x in v] for k, v in sem_ct.iterrows()}

dep = final.department.value_counts()
dep_items = list(dep.items())
half = (len(dep_items) + 1) // 2
rows = ""
for i in range(half):
    left = dep_items[i]
    right = dep_items[i + half] if i + half < len(dep_items) else ("", "")
    rows += (f"{esc(left[0])} & {left[1]} & {100 * left[1] / F:.1f} & "
             + (f"{esc(right[0])} & {right[1]} & {100 * right[1] / F:.1f}" if right[0] else " & & ")
             + " \\\\ \\hline\n")
write("departments.tex", rows)
S["departments"] = {k: int(v) for k, v in dep.items()}

sg_cg = pd.crosstab(final.recent_sgpa_code.astype(int), final.target_code.astype(int)).reindex(index=range(4), columns=range(4), fill_value=0)
rows = ""
for i in range(4):
    cells = [f"\\cellcolor{{diag}}\\textbf{{{int(v)}}}" if i == j else str(int(v)) for j, v in enumerate(sg_cg.loc[i])]
    rows += f"Recent SGPA {BANDS[i]} & " + " & ".join(cells) + f" & {int(sg_cg.loc[i].sum())} \\\\ \\hline\n"
rows += "Total & " + " & ".join(str(int(v)) for v in sg_cg.sum()) + f" & {F} \\\\ \\hline\n"
write("sgpa_cgpa_final.tex", rows)
S["final_gap"] = {int(k): int(v) for k, v in final.gap.value_counts().sort_index().items()}
S["sgpa_bands"] = [int(v) for v in final.recent_sgpa_code.value_counts().sort_index()]

# ================================================================ 4. the 182 removed answers
drop = base[base.dropped]
S["dropped"] = {
    "n": len(drop), "by_gap": {int(k): int(v) for k, v in drop.gap.value_counts().sort_index().items()},
    "by_date": {k: int(v) for k, v in drop.date.value_counts().sort_index().items()},
    "sgpa_lower": int((drop.recent_sgpa_code < drop.target_code).sum()),
    "sgpa_higher": int((drop.recent_sgpa_code > drop.target_code).sum()),
    "cgpa_band": [int(v) for v in drop.target_code.value_counts().reindex(range(4), fill_value=0)],
    "sgpa_band": [int(v) for v in drop.recent_sgpa_code.value_counts().reindex(range(4), fill_value=0)],
}
group = base[base.aug67_group]
rest = base[~base.aug67_group]
rows = ""
for label, fr in [("6--7 Aug KUET semester 7 (checked group)", group), ("All other students", rest),
                  ("Final dataset", final)]:
    g = fr.gap.value_counts().reindex(range(4), fill_value=0)
    rho = fr[["recent_sgpa_code", "target_code"]].corr("spearman").iloc[0, 1]
    rows += (f"{label} & {n(len(fr))} & " + " & ".join(str(int(v)) for v in g)
             + f" & {100 * (fr.gap >= 2).mean():.1f} & {rho:.2f} \\\\ \\hline\n")
write("gap_groups.tex", rows)

uni_rows = ""
for r in G["far_gap_by_university"]:
    uni_rows += f"{r['university']} & {r['n']} & {r['far']} & {100 * r['far'] / r['n']:.1f} \\\\ \\hline\n"
write("gap_university.tex", uni_rows)
sem_rows = ""
for r in G["far_gap_by_kuet_semester"]:
    sem_rows += f"{int(r['semester'])} & {r['n']} & {r['far']} & {100 * r['far'] / r['n']:.1f} \\\\ \\hline\n"
write("gap_kuet_semester.tex", sem_rows)
date_rows = ""
for r in G["kuet_sem7_by_date"]:
    d = pd.Timestamp(r["date"])
    date_rows += (f"{d:%d %b} & {r['n']} & {r['far']} & {100 * r['far'] / r['n']:.1f} & "
                  f"{S['dropped']['by_date'].get(r['date'], 0)} \\\\ \\hline\n")
write("gap_kuet_dates.tex", date_rows)

# ================================================================ 5. per-question statistics
assoc_c = R["association_cgpa"]
assoc_s = R["association_sgpa"]


def dist(frame, item):
    out = []
    for code, g in frame.groupby(item + "_code"):
        mix = g.target_code.value_counts(normalize=True).reindex(range(4), fill_value=0) * 100
        out.append({"code": int(code), "answer": str(g[item].iloc[0]), "n": len(g), "pct": 100 * len(g) / len(frame),
                    "mix": [float(round(x, 1)) for x in mix], "mean_cgpa_code": float(g.target_code.mean())})
    return out


qstats = {}
detail = ""
summary = ""
for item in ITEMS + EXTRA:
    d = dist(final, item)
    d_base = {r["code"]: r["pct"] for r in dist(base, item)}
    codes = final[item + "_code"]
    mode = max(d, key=lambda r: r["n"])
    med_code = int(np.median(codes))
    med_label = next(r["answer"] for r in d if r["code"] == med_code)
    shift = max(abs(r["pct"] - d_base.get(r["code"], 0)) for r in d)
    if item in assoc_c:
        rho, q = assoc_c[item]["rho"], assoc_c[item]["q"]
        rel = "yes" if assoc_c[item]["reliable"] else "no"
        rho_s = assoc_s[item]["rho"]
    else:
        rho, p = spearmanr(codes, final.target_code)
        rho_s, _ = spearmanr(codes, final.recent_sgpa_code)
        q, rel = p, "not used"
    miss = next(m for m in miss_rows if m["field"] == item)
    qstats[item] = {"answers": d, "mode": mode["answer"], "median": med_label, "mean": float(codes.mean()),
                    "sd": float(codes.std()), "rho_cgpa": float(rho), "rho_sgpa": float(rho_s), "q": float(q),
                    "reliable": rel, "max_shift_pp": float(shift), "raw_blank_a": miss["a"], "raw_blank_b": miss["b"]}
    used = "Model feature" if item in ITEMS else "Collected, not a model feature"
    detail += (f"\\rowcolor{{qhead}}\\multicolumn{{8}}{{|p{{\\dimexpr\\linewidth-2\\tabcolsep\\relax}}|}}{{\\textbf{{{esc(SHORT[item])}}} "
               f"(\\texttt{{{esc(item)}}}, {used}) --- \\emph{{{esc(QUESTION[item])}}}}} \\\\ \\hline\n")
    for r in d:
        detail += (f"{r['code']} & {esc(r['answer'])} & {r['n']} & {r['pct']:.1f} & "
                   + " & ".join(f"{x:.0f}" for x in r["mix"]) + " \\\\ \\hline\n")
    detail += (f"\\multicolumn{{8}}{{|p{{\\dimexpr\\linewidth-2\\tabcolsep\\relax}}|}}{{\\footnotesize Most common: {esc(mode['answer'])}; median: {esc(med_label)}; "
               f"mean code {codes.mean():.2f} (SD {codes.std():.2f}); Spearman $\\rho$ with CGPA band = {rho:+.3f}; "
               f"raw blanks A/B = {miss['a']}/{miss['b']}}} \\\\ \\hline\n")
    summary += (f"{esc(SHORT[item])} & {esc(mode['answer'])} & {esc(med_label)} & {codes.mean():.2f} & {codes.std():.2f} & "
                f"{rho:+.3f} & {rho_s:+.3f} & {esc(rel)} & {shift:.1f} \\\\ \\hline\n")
write("questions_detail.tex", detail)
write("questions_summary.tex", summary)
S["questions"] = qstats

# ================================================================ 6. missingness table (raw) and dictionary answers
rows = ""
for m in miss_rows:
    label = SHORT.get(m["field"], {"department": "Department", "semester": "Semester", "recent_sgpa": "Recent SGPA",
                                   "cgpa": "CGPA (target)"}.get(m["field"]))
    rows += f"{esc(label)} & {m['a']} & {m['b']} & {m['total']} & {m['pct']:.1f} \\\\ \\hline\n"
write("raw_missing.tex", rows)

# ================================================================ 7. WEKA results
W = R["weka"]
MODELS = ["ZeroR", "OneR", "J48", "RandomForest", "SMO", "Logistic", "NaiveBayes"]
NAMES = {"RandomForest": "Random Forest", "NaiveBayes": "Naive Bayes", "J48": "J48 (C4.5 tree)"}
rows = ""
for m in MODELS:
    r = W["main_cgpa"]["cv5"][m]
    rows += (f"{NAMES.get(m, m)} & {100 * r['accuracy']:.2f} & {r['kappa']:.3f} & {r['macro_f1']:.3f} & "
             f"{r['weighted_f1']:.3f} & {100 * r['within_1']:.1f} & {100 * W['questionnaire_cgpa']['cv5'][m]['accuracy']:.2f} & "
             f"{100 * W['sgpa']['cv5'][m]['accuracy']:.2f} \\\\ \\hline\n")
write("weka_cv5.tex", rows)
rows = ""
for m in MODELS:
    rows += (f"{NAMES.get(m, m)} & {100 * W['main_cgpa']['split80'][m]['accuracy']:.2f} & "
             f"{W['main_cgpa']['split80'][m]['kappa']:.3f} & "
             f"{100 * W['questionnaire_cgpa']['split80'][m]['accuracy']:.2f} & "
             f"{100 * W['sgpa']['split80'][m]['accuracy']:.2f} \\\\ \\hline\n")
write("weka_split80.tex", rows)


def cm_tex(cm, name):
    rows = ""
    for i, row in enumerate(cm):
        cells = [f"\\cellcolor{{diag}}\\textbf{{{v}}}" if i == j else str(v) for j, v in enumerate(row)]
        rows += f"Actual {BANDS[i]} & " + " & ".join(cells) + f" & {sum(row)} \\\\ \\hline\n"
    write(name, rows)


cm_tex(W["main_cgpa"]["cv5"]["SMO"]["confusion_matrix"], "cm_smo_cv5.tex")
cm_tex(W["main_cgpa"]["cv5"]["J48"]["confusion_matrix"], "cm_j48_cv5.tex")
S["before_check_1102_cv5"] = R.get("before_check_1102_cv5")
S["j48_sizes"] = R.get("j48_sizes")
S["weka_main_cv5"] = {m: W["main_cgpa"]["cv5"][m] for m in MODELS}

# ================================================================ 8. sample rows (processed) and ARFF header
sample_idx = [0, 150, 330, 520, 700, 880]
rows = ""
for i in sample_idx:
    r = final.iloc[i]
    rows += (f"{r.university} & {esc(r.department)} & {int(r.semester_raw)} & {esc(r.attendance)} & "
             f"{esc(r.weekly_study_time)} & {esc(r.sleep_duration)} & {esc(r.stress_frequency)} & "
             f"C{int(r.recent_sgpa_code)} & C{int(r.target_code)} \\\\ \\hline\n")
write("sample_rows.tex", rows)
arff = (ROOT / "cleaned-dataset" / "ours" / "final" / "final_920.arff").read_text(encoding="utf-8").splitlines()
start = next(i for i, line in enumerate(arff) if line.strip().lower() == "@data")
(TAB / "arff_head.txt").write_text("\n".join(arff[:start + 5]) + "\n...\n", encoding="utf-8")
csv_head = (ROOT / "cleaned-dataset" / "ours" / "final" / "final_920.csv").read_text(encoding="utf-8").splitlines()[:4]
(TAB / "csv_head.txt").write_text("\n".join(csv_head) + "\n...\n", encoding="utf-8")

# ================================================================ figures
def clean_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, labelsize=7)


def wrap(s, width=22):
    words, lines, cur = str(s).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return "\n".join(lines)


def answers_grid(items, name, ncols=3):
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, 2.05 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, item in zip(axes, items):
        d = qstats[item]["answers"]
        labels = [wrap(r["answer"]) for r in d]
        vals = [r["n"] for r in d]
        y = np.arange(len(d))[::-1]
        ax.barh(y, vals, color=BLUE, height=0.62)
        top = max(vals)
        for yy, r in zip(y, d):
            ax.text(r["n"] + top * 0.03, yy, f"{r['n']} ({r['pct']:.0f}%)", va="center", fontsize=6.5, color=INK)
        ax.set_yticks(y, labels, fontsize=6.3)
        ax.set_xlim(0, top * 1.5)
        ax.set_xticks([])
        clean_axes(ax)
        ax.spines["bottom"].set_visible(False)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_title(SHORT[item], loc="left", pad=4)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout(h_pad=1.2, w_pad=0.6)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def mix_grid(items, name, ncols=3):
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, 1.85 * nrows + 0.4))
    axes = np.atleast_1d(axes).ravel()
    for ax, item in zip(axes, items):
        d = qstats[item]["answers"]
        y = np.arange(len(d))[::-1]
        left = np.zeros(len(d))
        for k in range(4):
            vals = np.array([r["mix"][k] for r in d])
            ax.barh(y, vals, left=left, color=BAND_COLORS[k], height=0.66, edgecolor="white", linewidth=1.2)
            for yy, v, l0 in zip(y, vals, left):
                if v >= 10:
                    ax.text(l0 + v / 2, yy, f"{v:.0f}", ha="center", va="center", fontsize=5.8, color="white")
            left += vals
        ax.set_yticks(y, [wrap(r["answer"], 18) + f"  (n={r['n']})" for r in d], fontsize=5.8)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100], ["0%", "50%", "100%"])
        clean_axes(ax)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(SHORT[item], loc="left", pad=4)
    for ax in axes[len(items):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BAND_COLORS]
    fig.legend(handles, ["C0  below 3.20", "C1  3.20-3.49", "C2  3.50-3.74", "C3  3.75 and above"],
               loc="upper center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.1, w_pad=0.4)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


answers_grid(ITEMS, "answers_12_features.png")
answers_grid(EXTRA, "answers_3_extra.png")
mix_grid(ITEMS[:6], "cgpa_mix_part1.png")
mix_grid(ITEMS[6:], "cgpa_mix_part2.png")

# collection timeline: raw responses per day, by form
days = pd.date_range("2026-07-31", "2026-09-03", freq="D")
ca = raw_ts_a.dt.normalize().value_counts().reindex(days, fill_value=0)
cb = raw_ts_b.dt.normalize().value_counts().reindex(days, fill_value=0)
fig, ax = plt.subplots(figsize=(6.9, 2.1))
x = np.arange(len(days))
ax.bar(x, cb.values, color=BLUE, width=0.8, label=f"Form B, KUET ({len(raw_b)} responses)")
ax.bar(x, ca.values, bottom=cb.values, color=ORANGE, width=0.8, label=f"Form A, BUET/BRAC/CUET ({len(raw_a)} responses)")
for xi, v in zip(x, (ca + cb).values):
    if v >= 40:
        ax.text(xi, v + 6, str(v), ha="center", fontsize=6.5, color=INK)
ticks = [i for i, d in enumerate(days) if d.day in (1, 5, 10, 15, 20, 25)]
ax.set_xticks(ticks, [days[i].strftime("%d %b") for i in ticks])
ax.set_ylabel("Responses per day")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(length=0)
ax.legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "collection_timeline.png", bbox_inches="tight")
plt.close(fig)

# copy the slide figures the report re-uses
for f in ["rc_university", "rc_cgpa", "gap_distribution_1102", "far_gap_by_university", "far_gap_by_kuet_semester",
          "kuet_sem7_by_date", "sgpa_x_cgpa_aug67", "sgpa_x_cgpa_rest", "sgpa_x_cgpa_final", "association_strength", "sgpa_vs_cgpa_pattern",
          "tree_main_cgpa", "tree_questionnaire_cgpa", "tree_sgpa", "weka_output_main_smo_cv5"]:
    shutil.copy2(ROOT / "docs" / "figures" / "final" / f"{f}.png", FIG / "deck" / f"{f}.png")

(DATA / "report_stats.json").write_text(json.dumps(S, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
print("raw A weekly_responsibilities values:", S["raw_a_weekly_responsibilities_values"])
print("before_check:", json.dumps(S["before_check_1102_cv5"])[:900])
print("j48_sizes:", json.dumps(S["j48_sizes"])[:600])
print("dropped:", S["dropped"])
print("final gap:", S["final_gap"], "sgpa bands:", S["sgpa_bands"])
print("sources:", src_rows)
print("done")
