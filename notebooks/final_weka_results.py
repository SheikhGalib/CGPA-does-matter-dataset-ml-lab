"""Build the FINAL dataset (920 students) and every WEKA result used in the presentation.

Decisions (agreed by the team):
  * start from the 1,102 students who reported a real recent SGPA
  * the 455 KUET semester-7 responses submitted on 6-7 Aug 2026 are checked: keep those whose recent
    SGPA band and CGPA band are at most 1 band apart, drop the other 182  ->  920 students
  * features: 12 questionnaire answers (codes 0-4) + recent SGPA band; result satisfaction NOT used
  * evaluation: 5-fold cross-validation (seed 1) for the main slides, 80/20 split (seed 1) for backup
  * models: ZeroR, J48 (minNumObj 40), RandomForest, SMO, Logistic, NaiveBayes (+ OneR as a check)

Run:  venv\\Scripts\\python.exe notebooks\\final_weka_results.py
Every number comes from a WEKA 3.8.7 log in docs/weka_runs/final/.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import weka_runner as W  # noqa: E402

W.RUN_DIR = ROOT / "docs" / "weka_runs" / "final"
OUT = ROOT / "cleaned-dataset" / "ours" / "final"
OUT.mkdir(parents=True, exist_ok=True)
DOCS = ROOT / "docs"

ITEMS = ["desired_department_match", "attendance", "weekly_study_time", "study_style", "topic_clarity",
         "sleep_duration", "stress_frequency", "weekly_responsibilities", "distraction_frequency",
         "study_environment", "support_level", "routine_manageability"]
BANDS = ["C0", "C1", "C2", "C3"]

# --------------------------------------------------------------------------------------------- data
clean = pd.read_csv(ROOT / "cleaned-dataset" / "ours" / "merged" / "merged_full_clean.csv",
                    keep_default_na=False, na_values=[""])          # keep the answer "None" as text
raw_multi = pd.read_csv(ROOT / "raw-data" / "University Student Performance Analysis Form (Responses) - Form Responses 1.csv")
raw_kuet = pd.read_csv(ROOT / "raw-data" / "Student Performance Analysis Form (Responses) - Form Responses 1.csv")

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
    pd.DataFrame({"ts": raw_multi.iloc[:, 0], "cg": raw_multi.iloc[:, 8].map(band_code), "sg": raw_multi.iloc[:, 7].map(band_code)}),
    pd.DataFrame({"ts": raw_kuet.iloc[:, 0], "cg": raw_kuet.iloc[:, 4].map(band_code), "sg": raw_kuet.iloc[:, 3].map(band_code)}),
], ignore_index=True)
stamp = stamp[stamp.cg.notna()].reset_index(drop=True)
assert len(stamp) == len(clean) and (stamp.cg.values == clean.target_code.values).all()
has_sgpa = ~clean.recent_sgpa_imputed.astype(str).str.lower().eq("true")
base = clean[has_sgpa.values].reset_index(drop=True)
stamp = stamp[has_sgpa.values].reset_index(drop=True)
assert (stamp.sg.values == base.recent_sgpa_code.values).all()
base["submitted"] = pd.to_datetime(stamp.ts, format="mixed")
base["date"] = base.submitted.dt.strftime("%Y-%m-%d")
base["gap"] = (base.recent_sgpa_code - base.target_code).abs().astype(int)
base["aug67_group"] = (base.source_form == "kuet") & (base.semester_raw == 7) & base.date.isin(["2026-08-06", "2026-08-07"])
base["dropped"] = base.aug67_group & (base.gap >= 2)
final = base[~base.dropped].reset_index(drop=True)
print(f"1,102 base -> dropped {int(base.dropped.sum())} -> final {len(final)}")
assert len(base) == 1102 and len(final) == 920

# --------------------------------------------------------------------------------------------- exports
weka_table = pd.DataFrame({i: final[i + "_code"].astype(int) for i in ITEMS})
weka_table["recent_sgpa_band"] = ["C%d" % v for v in final.recent_sgpa_code.astype(int)]
weka_table["cgpa_band"] = ["C%d" % v for v in final.target_code.astype(int)]
FINAL_CSV = OUT / "final_920.csv"
FINAL_ARFF = OUT / "final_920.arff"
weka_table.to_csv(FINAL_CSV, index=False)
W.write_arff(weka_table, FINAL_ARFF, "student_cgpa_final_920",
             {"recent_sgpa_band": BANDS, "cgpa_band": BANDS}, class_col="cgpa_band")

detail_cols = (["university", "department", "semester_raw", "source_form", "date", "gap", "aug67_group"]
               + ITEMS + [i + "_code" for i in ITEMS] + ["recent_sgpa_band", "cgpa_band"])
details = final.copy()
details["recent_sgpa_band"] = weka_table.recent_sgpa_band.values
details["cgpa_band"] = weka_table.cgpa_band.values
details[detail_cols].rename(columns={"semester_raw": "semester"}).to_csv(OUT / "final_920_with_details.csv", index=False)

codes = []
for i in ITEMS:
    for code, grp in final.groupby(i + "_code"):
        codes.append({"question": i, "code": int(code), "answer": grp[i].iloc[0]})
pd.DataFrame(codes).to_csv(OUT / "answer_codes.csv", index=False)
base.loc[base.dropped, ["university", "semester_raw", "date", "recent_sgpa_band", "cgpa_band", "gap"]] \
    .rename(columns={"semester_raw": "semester"}).to_csv(OUT / "dropped_182_rows.csv", index=False)

# three experiment files (same rows) so every run is one WEKA command
table_sgpa = weka_table.drop(columns=["cgpa_band"]).rename(columns={"recent_sgpa_band": "sgpa_band"})
table_sgpa["sgpa_band"] = weka_table.recent_sgpa_band
P_MAIN = FINAL_ARFF
P_QUEST = W.write_arff(weka_table.drop(columns=["recent_sgpa_band"]), OUT / "_work_questionnaire_cgpa.arff",
                       "questionnaire_cgpa", {"cgpa_band": BANDS}, class_col="cgpa_band")
P_SGPA = W.write_arff(table_sgpa, OUT / "_work_questionnaire_sgpa.arff", "questionnaire_sgpa",
                      {"sgpa_band": BANDS}, class_col="sgpa_band")
(OUT / ".gitignore").write_text("_work_*\n")

# --------------------------------------------------------------------------------------------- WEKA runs
MODELS = {"ZeroR": W.ZEROR, "J48": W.J48, "RandomForest": W.RF, "SMO": W.SMO,
          "Logistic": W.LOGISTIC, "NaiveBayes": W.NAIVE_BAYES, "OneR": ["weka.classifiers.rules.OneR", "-B", "6"]}
TARGETS = {"sgpa": P_SGPA, "questionnaire_cgpa": P_QUEST, "main_cgpa": P_MAIN}

def weighted_f1(cm):
    cm = cm.astype(float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float((f1 * cm.sum(1)).sum() / cm.sum())

jobs = []
for t, path in TARGETS.items():
    for m, scheme in MODELS.items():
        jobs.append(dict(name=f"{t}_{m}_cv5", scheme=scheme, train=path, folds=5, seed=1, target=t, model=m, eval="cv5"))
        jobs.append(dict(name=f"{t}_{m}_split80", scheme=scheme, train=path, split=80, seed=1, target=t, model=m, eval="split80"))
results = {}
for r in W.run_many(jobs):
    e = r.extra
    results.setdefault(e["target"], {}).setdefault(e["eval"], {})[e["model"]] = {
        "accuracy": round(r.accuracy, 4), "kappa": round(r.kappa, 4), "weighted_f1": round(weighted_f1(r.cm), 4),
        "macro_f1": round(r.macro_f1, 4), "far_rate": round(r.far_rate, 4), "within_1": round(r.within_one, 4),
        "n": r.n, "confusion_matrix": r.cm.astype(int).tolist(), "log": f"docs/weka_runs/final/{r.name}.txt"}

# J48 trees built on all 920 rows (what the Explorer prints as "model on full training set"), text + dot graph
trees = {}
for t, path in TARGETS.items():
    base_cmd = [str(W.JAVA), "--add-opens", "java.base/java.lang=ALL-UNNAMED", "-cp", str(W.WEKA_JAR),
                "weka.classifiers.trees.J48", "-C", "0.25", "-M", "40", "-t", str(path)]
    txt = subprocess.run(base_cmd + ["-x", "5", "-s", "1"], capture_output=True, text=True).stdout
    dot = subprocess.run(base_cmd + ["-g"], capture_output=True, text=True).stdout
    (W.RUN_DIR / f"{t}_J48_tree_full.txt").write_text(txt, encoding="utf-8")
    (W.RUN_DIR / f"{t}_J48_tree_full.dot").write_text(dot, encoding="utf-8")
    body = txt.split("J48 pruned tree", 1)[1].split("Number of Leaves", 1)[0].strip().strip("-").strip()
    trees[t] = {"text": body, "leaves": W.parse_tree(txt)["leaves"], "size": W.parse_tree(txt)["size"],
                "dot": f"docs/weka_runs/final/{t}_J48_tree_full.dot"}

# --------------------------------------------------------------------------------------------- statistics
def share_table(frame, col, order=None):
    vc = frame[col].value_counts()
    if order is not None:
        vc = vc.reindex(order, fill_value=0)
    return [{"value": str(k), "n": int(v), "pct": round(100 * v / len(frame), 1)} for k, v in vc.items()]

def answer_order(item):
    return [a for _, a in sorted({(int(c), a) for c, a in zip(final[item + "_code"], final[item])})]

def assoc(frame, target_col):
    rows = []
    for i in ITEMS:
        rho, p = stats.spearmanr(frame[i + "_code"], frame[target_col])
        rows.append({"question": i, "rho": rho, "p": p})
    t = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    q = t.p * len(t) / (t.index + 1)
    t["q"] = q[::-1].cummin()[::-1].clip(upper=1)
    return {r.question: {"rho": round(r.rho, 4), "p": round(r.p, 5), "q": round(r.q, 5), "reliable": bool(r.q < .05)}
            for r in t.itertuples()}

per_question = {}
for i in ITEMS:
    rows = []
    for code, grp in final.groupby(i + "_code"):
        sh = grp.target_code.value_counts(normalize=True).reindex(range(4), fill_value=0) * 100
        rows.append({"code": int(code), "answer": grp[i].iloc[0], "n": len(grp),
                     "C0": round(sh[0]), "C1": round(sh[1]), "C2": round(sh[2]), "C3": round(sh[3])})
    per_question[i] = rows

gap_stats = {
    "base_rows": len(base), "final_rows": len(final), "dropped_rows": int(base.dropped.sum()),
    "aug67_group_rows": int(base.aug67_group.sum()),
    "aug67_kept": int((base.aug67_group & ~base.dropped).sum()),
    "gap_distribution_all": {int(k): int(v) for k, v in base.gap.value_counts().sort_index().items()},
    "gap_distribution_aug67": {int(k): int(v) for k, v in base[base.aug67_group].gap.value_counts().sort_index().items()},
    "gap_distribution_rest": {int(k): int(v) for k, v in base[~base.aug67_group].gap.value_counts().sort_index().items()},
    "far_gap_rate_aug67": round(float((base[base.aug67_group].gap >= 2).mean()), 4),
    "far_gap_rate_rest": round(float((base[~base.aug67_group].gap >= 2).mean()), 4),
    "far_gap_by_university": base.assign(far=base.gap >= 2).groupby("university").far.agg(["size", "sum"]).astype(int)
                             .reset_index().rename(columns={"size": "n", "sum": "far"}).to_dict("records"),
    "far_gap_by_kuet_semester": base[base.source_form == "kuet"].assign(far=lambda d: d.gap >= 2)
                                .groupby("semester_raw").far.agg(["size", "sum"]).astype(int).reset_index()
                                .rename(columns={"semester_raw": "semester", "size": "n", "sum": "far"}).to_dict("records"),
    "kuet_sem7_by_date": base[(base.source_form == "kuet") & (base.semester_raw == 7)].assign(far=lambda d: d.gap >= 2)
                         .groupby("date").far.agg(["size", "sum"]).astype(int).reset_index()
                         .rename(columns={"size": "n", "sum": "far"}).to_dict("records"),
    "sgpa_x_cgpa_aug67": pd.crosstab(base[base.aug67_group].recent_sgpa_code, base[base.aug67_group].target_code)
                         .reindex(index=range(4), columns=range(4), fill_value=0).astype(int).values.tolist(),
    "sgpa_x_cgpa_rest": pd.crosstab(base[~base.aug67_group].recent_sgpa_code, base[~base.aug67_group].target_code)
                        .reindex(index=range(4), columns=range(4), fill_value=0).astype(int).values.tolist(),
    "sgpa_x_cgpa_final": pd.crosstab(final.recent_sgpa_code, final.target_code)
                         .reindex(index=range(4), columns=range(4), fill_value=0).astype(int).values.tolist(),
    "dropped_by_gap": {int(k): int(v) for k, v in base[base.dropped].gap.value_counts().sort_index().items()},
    "dropped_by_direction": {"SGPA lower than CGPA": int((base.dropped & (base.recent_sgpa_code < base.target_code)).sum()),
                             "SGPA higher than CGPA": int((base.dropped & (base.recent_sgpa_code > base.target_code)).sum())},
    "sgpa_cgpa_rho_aug67": round(float(base[base.aug67_group][["recent_sgpa_code", "target_code"]].corr("spearman").iloc[0, 1]), 3),
    "sgpa_cgpa_rho_rest": round(float(base[~base.aug67_group][["recent_sgpa_code", "target_code"]].corr("spearman").iloc[0, 1]), 3),
    "sgpa_cgpa_rho_final": round(float(final[["recent_sgpa_code", "target_code"]].corr("spearman").iloc[0, 1]), 3),
}

summary = {
    "final_rows": len(final),
    "universities": share_table(final, "university"),
    "cgpa_bands": [{"band": BANDS[k], "n": int(v), "pct": round(100 * v / len(final), 1)}
                   for k, v in final.target_code.value_counts().sort_index().items()],
    "sgpa_bands": [{"band": BANDS[k], "n": int(v), "pct": round(100 * v / len(final), 1)}
                   for k, v in final.recent_sgpa_code.value_counts().sort_index().items()],
    "answers": {i: share_table(final, i, answer_order(i)) for i in ITEMS},
    "association_cgpa": assoc(final, "target_code"),
    "association_sgpa": assoc(final, "recent_sgpa_code"),
    "per_question_cgpa_mix": per_question,
    "gap_analysis": gap_stats,
    "weka": results,
    "j48_trees": trees,
}
(DOCS / "final_results.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

print("\n5-fold CV accuracy (%)")
print(pd.DataFrame({t: {m: v["accuracy"] * 100 for m, v in results[t]["cv5"].items()} for t in TARGETS}).round(2).to_string())
print("\n80/20 split accuracy (%)")
print(pd.DataFrame({t: {m: v["accuracy"] * 100 for m, v in results[t]["split80"].items()} for t in TARGETS}).round(2).to_string())
print("\nJ48 leaves:", {t: v["leaves"] for t, v in trees.items()})
print("universities:", summary["universities"])
print("cgpa bands:", summary["cgpa_bands"])
print("reliable CGPA associations:", [k for k, v in summary["association_cgpa"].items() if v["reliable"]])
print("wrote", FINAL_CSV.relative_to(ROOT), FINAL_ARFF.relative_to(ROOT), "docs/final_results.json")
