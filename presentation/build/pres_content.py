"""Shared content + asset helpers for the three CSE-4112 deck builds.

Every number here is copied from docs/results_summary.json, docs/model_results_*.csv,
docs/DATA_QUALITY_REPORT.md and the executed notebook. Do not edit by hand — if the
notebook is re-run and numbers change, update this file from the new outputs.
"""
import base64, io, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "presentation"

# ---------------------------------------------------------------- assets
_FIGURES = {
    "confound":   "03_semester_confound.png",
    "target":     "04_target_distribution.png",
    "importance": "10_feature_importance_consensus.png",
    "confusion":  "12_confusion_matrices.png",
    "learning":   "13_learning_curve.png",
    "counter":    "17_counterintuitive_effects.png",
}


def data_uri(key, max_w=1250):
    """Downscale a figure and return it as a base64 data URI so the deck is one file."""
    from PIL import Image
    p = FIG / _FIGURES[key]
    im = Image.open(p).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def all_figures():
    return {k: data_uri(k) for k in _FIGURES}


# ---------------------------------------------------------------- deck metadata
META = {
    "course": "CSE-4112",
    "course_full": "Machine Learning Laboratory",
    "title_a": "What actually",
    "title_b": "moves a",
    "title_em": "CGPA",
    "date": "September 2026",
    "subtitle": ("A survey of <strong>230 university students</strong> at BUET, BRAC and CUET — "
                 "cleaned, audited, and modelled as an <strong>ordinal classification</strong> "
                 "of self-reported CGPA bands."),
    "team": [
        ("2107020", "Sheikh Md. Galib Mahim"),
        ("2107030", "Md. Eftakar Jaman Arfan"),
        ("2107007", "Asif Jawad"),
        ("2107008", "Rakibul Islam"),
        ("2107012", "Md Enam E Elahi"),
    ],
    "glance": [
        ("Responses", "230"), ("Survey items", "22"), ("Universities", "03"),
        ("CGPA bands", "04"), ("Rows dropped", "00"),
    ],
}

# ---------------------------------------------------------------- slides
# Each slide is a dict the renderers consume. `kind` selects the layout.

SLIDES = [

    dict(kind="cover", n="01", section="Cover"),

    dict(kind="two-col", n="02", section="The brief",
         kicker="Problem statement",
         head="Two questions, <em>one</em> dataset",
         lead="The course asks for a model. The data asks a harder question first — whether "
              "self-reported habits carry any signal at all.",
         left_title="Pipeline A &mdash; Feature influence",
         left_body="Which of a student's habits, circumstances and attitudes are actually "
                   "associated with academic performance? A ranking problem, answered with "
                   "statistics rather than a black box.",
         right_title="Pipeline B &mdash; CGPA prediction",
         right_body="Given a student's answers, predict their CGPA band. A supervised "
                    "classification problem over four ordered classes.",
         note="The two are kept apart. Previous CGPA dominates any model it sits in, so the "
              "behaviour-only run is what actually answers Pipeline A.",
         table_title="Two experiments, never merged",
         rows=[("Primary", "all features, <em>including</em> previous CGPA band"),
               ("Behaviour-focused", "the same features, <em>minus</em> previous CGPA")]),

    dict(kind="related", n="03", section="Related work",
         kicker="Related work &amp; base paper",
         head="Standing on <em>Cortez &amp; Silva</em>",
         base_title="Cortez &amp; Silva (2008) — <span class='mono'>UCI Student Performance</span>",
         base_body="Predicted secondary-school grades for Portuguese students using demographic, "
                   "social and school features gathered from reports and questionnaires. Compared "
                   "Decision Tree, Random Forest, Neural Network and SVM. The benchmark dataset "
                   "behind 200+ later studies.",
         base_stats=[("~72%", "reported four-class accuracy"),
                     ("33", "attributes"),
                     ("G1, G2", "strongest predictors — prior grades")],
         finding="Their headline caveat is that predictive power collapses once the prior-period "
                 "grades are removed. That is the exact effect we set out to measure.",
         contrib_title="What we contribute",
         contribs=[
             ("New primary data", "230 responses from Bangladeshi university students — a "
                                  "different population, education system and grading culture."),
             ("A quantified leakage audit", "We identify and remove a question that restates the "
                                            "target (&rho; = 0.60) before modelling. The base paper does not."),
             ("Multiple-comparison correction", "16 features tested, Benjamini&ndash;Hochberg applied. "
                                                "Uncorrected significance is reported as a hypothesis, not a result."),
             ("A replicated negative result", "We confirm their caveat in a new setting: prior "
                                              "performance is worth +16.5 accuracy points; behaviour alone is near chance."),
         ]),

    dict(kind="dataset", n="04", section="Dataset",
         kicker="Dataset information",
         head="230 students, <em>four</em> ordered bands",
         lead="A bilingual Google Form (English + Bangla), circulated to students at three "
              "universities. Mostly five-point Likert items plus a banded self-reported GPA.",
         fig="target",
         cap="The form collected GPA as bands, never as a number — so this is ordinal "
             "classification, not regression.",
         unis=[("BUET", "104"), ("BRAC", "101"), ("CUET", "25")],
         bands=[("C0", "below 3.20", "60"), ("C1", "3.20 – 3.49", "52"),
                ("C2", "3.50 – 3.74", "63"), ("C3", "3.75 and above", "55")],
         note="Near-balanced at 23–27% per class. Two consequences: no resampling is needed, "
              "and <strong>accuracy is a fair metric</strong>. Anything we build must beat "
              "<strong>27.4%</strong> (ZeroR)."),

    dict(kind="integrity", n="05", section="Data integrity",
         kicker="Before anything else",
         head="The supplied clean data was <em>not</em> our data",
         lead="Two cleaned CSVs already existed in the repo. Since our whole method rests on "
              "being able to justify the data, we verified them against the raw form rather "
              "than trusting them.",
         cols=("Check", "Our raw form", "The supplied files"),
         rows=[
             ("Row count", "230", "878"),
             ("Semester values", "1 – 12", "S1 – S7 only"),
             ("Departments", "16, all named by respondents", "includes TE, LE, BECM, ESE, IEM, MSE"),
             ("Satisfaction column", "admission satisfaction", "kuet_satisfaction"),
             ("Row alignment", "—", "GPA bands do not match row-for-row"),
         ],
         verdict="Those are KUET's departments, and no KUET student is in our sample. "
                 "The files describe a different, most likely synthetic, dataset.",
         action="We kept the <em>schema</em> — the C0–C3 band scheme, the primary/behaviour "
                "split, the naming — and rebuilt every row from the 230 real responses. "
                "The supplied WEKA tutorial inherits the same error: it says to expect 878 "
                "instances. For us the number is 230."),

    dict(kind="preprocess", n="06", section="Preprocessing",
         kicker="Data preprocessing techniques",
         head="Nine steps, <em>zero</em> rows dropped",
         lead="Every step is recorded as Decision &rarr; Evidence &rarr; Why. The viva asks how "
              "we preprocessed and why; these are the answers.",
         steps=[
             ("Drop <span class='mono'>timestamp</span>",
              "Submission time tracks how the link spread through our networks — a sampling artefact a tree could split on."),
             ("9 university spellings &rarr; 3",
              "<span class='mono'>BRAC / Brac / BRAc / BRAC University</span> are one institution; unmerged, 101 students split into five weak groups."),
             ("Merge 2 department columns &rarr; 1",
              "They are complementary: 174 + 56 = 230. Dropping either deletes a real answer for a quarter of respondents."),
             ("Group departments with &lt; 5 students",
              "A category with two members cannot be learned, only memorised."),
             ("44 blanks &rarr; <span class='mono'>none</span>",
              "The form offered no zero option and blankness is unrelated to GPA (&chi;&sup2; p = 0.372). A blank is an answer."),
             ("6 GPA bands &rarr; 4 classes",
              "Six classes over 230 students leaves ~26 each; per-class F1 would be noise. Four classes come out balanced."),
             ("Likert &rarr; ordinal integers",
              "The categories are ordered. One-hot destroys the order and turns 15 columns into ~70 on 230 rows."),
             ("Drop <span class='mono'>result_satisfaction</span>",
              "Leakage — it restates the target (&rho; = 0.60)."),
             ("Derive <span class='mono'>academic_progress</span>",
              "The raw semester number encodes university, not progress."),
         ],
         stats=[("230", "rows in"), ("230", "rows out"), ("0", "dropped"), ("44", "imputed")]),

    dict(kind="evidence", n="07", section="Preprocessing",
         kicker="Two decisions worth defending",
         head="Leakage, and a <em>confounded</em> variable",
         fig="confound",
         cap="Semester number by university. BRAC runs trimesters, so all 101 of its respondents "
             "sit at 10–11; all 25 CUET respondents are at semester 7.",
         a_title="Leakage &mdash; <span class='mono'>result_satisfaction</span>",
         a_body="“How satisfied were you with your latest semester result?” is asked about the "
                "same semester whose grade is the target. At <strong>&rho; = 0.601</strong> it is "
                "not a cause of the grade, it is a restatement of it.",
         a_action="Excluded from every model. Kept as a data-quality check: satisfaction rises "
                  "monotonically C0&rarr;C3, evidence that the self-reported grades are not random.",
         b_title="Confound &mdash; <span class='mono'>semester</span>",
         b_body="“Semester 11” does not mean further along than “semester 7” — it means BRAC. "
                "A model given the raw number can recover the university and learn per-institution "
                "grading habits while appearing to learn about academic progress.",
         b_action="Replaced with <span class='mono'>academic_progress</span> = semester ÷ programme "
                  "length, which is comparable across the three institutions."),

    dict(kind="importance", n="08", section="Feature influence",
         kicker="Pipeline A &mdash; results",
         head="Four methods. <em>No</em> survivor.",
         lead="No single importance measure is trustworthy on 230 rows, so we ran four and ranked "
              "by consensus. Then corrected for having tested 16 features at once.",
         fig="importance",
         methods=[
             ("Spearman &rho;", "monotonic rank association"),
             ("Mutual information", "catches non-linear structure"),
             ("Kruskal–Wallis H", "distribution differs across bands?"),
             ("Permutation importance", "real out-of-fold accuracy drop"),
         ],
         why="We use permutation importance, not the tree's built-in Gini importance — Gini is "
             "measured on training data and is biased toward features with many distinct values.",
         verdict_head="Zero features survive FDR correction",
         verdict="<span class='mono'>topic_clarity</span> is the only feature significant even "
                 "uncorrected (&rho; = &minus;0.161, p = 0.014, <strong>q = 0.214</strong>). With "
                 "n = 230 across four classes, an effect must reach roughly |&rho;| &gt; 0.18 to be "
                 "detectable; every behavioural effect here is smaller than that.",
         honest="This is the honest answer to the question we asked, and it agrees with the "
                "modelling in the next section. It is a finding, not a failure."),

    dict(kind="counter", n="09", section="Feature influence",
         kicker="Two counterintuitive signs",
         head="Checked, and <em>real</em>",
         lead="The two largest raw correlations both point the wrong way. Before writing either "
              "into a report we re-derived them from the raw form text and stratified by university.",
         fig="counter",
         a_title="Clearer lectures &rarr; slightly lower grades",
         a_body="&rho; = &minus;0.161 overall. Concentrated at <strong>BUET (&rho; = &minus;0.27, "
                "p = 0.006)</strong>, flat at BRAC, slightly positive at CUET.",
         a_read="Most plausible reading is a <strong>self-assessment calibration effect</strong>: "
                "weaker students overestimate how much of a lecture they absorbed, because they do "
                "not yet know what they missed.",
         b_title="More stress &rarr; slightly higher grades",
         b_body="Mean stress rises steadily from <strong>2.07</strong> at C0 to <strong>2.53</strong> at C3.",
         b_read="Most plausible reading is <strong>reverse causation</strong>: students holding a "
                "high CGPA have more to lose and report more pressure.",
         caveat="Neither survives FDR correction. Both are stated as associations with a plausible "
                "mechanism — never as causes. This is cross-sectional, self-reported data."),

    dict(kind="models", n="10", section="Models",
         kicker="Models used &amp; evaluation protocol",
         head="Nine classifiers, <em>one</em> protocol",
         models=[
             ("ZeroR", "majority-class floor"),
             ("OneR-like", "single best feature"),
             ("Naive Bayes", "probabilistic baseline"),
             ("Logistic Regression", "linear, interpretable"),
             ("k-NN (k=15)", "instance-based"),
             ("Decision Tree", "J48 equivalent"),
             ("Random Forest", "bagged ensemble"),
             ("Gradient Boosting", "boosted ensemble"),
             ("SVM (RBF)", "SMO equivalent"),
         ],
         protocol_title="Why this protocol",
         protocol=[
             ("Repeated stratified 10-fold CV",
              "n = 230. A single 20% holdout is 46 rows — far too noisy to rank nine models. Repeated 3&times; to stabilise."),
             ("Accuracy",
              "Fair here <em>only</em> because the classes are balanced (23–27%)."),
             ("Macro-F1",
              "Averages the four classes equally, so a weak class cannot hide behind a strong one."),
             ("Cohen's &kappa;",
              "Corrects for agreement expected by chance. This is the metric that exposes a weak model."),
             ("Within-1-band accuracy",
              "The target is <em>ordinal</em> — predicting C2 when the truth is C3 is a smaller error than predicting C0."),
         ],
         warn="Trap: ZeroR scores 0.739 on within-1-band, higher than any behaviour model — it "
              "always predicts the middle class C2, which is within one band of C1, C2 and C3. "
              "Always read within-1-band next to &kappa;."),

    dict(kind="results", n="11", section="Results",
         kicker="Comparison between models",
         head="Prior CGPA is <em>the</em> signal",
         t1_title="Behaviour-focused &mdash; without previous CGPA",
         t1=[("ZeroR (baseline)", "27.4%", "0.108", "0.000"),
             ("Gradient Boosting", "34.5%", "0.320", "0.093"),
             ("Random Forest", "34.9%", "0.306", "0.104"),
             ("Logistic Regression", "32.2%", "0.293", "0.078"),
             ("Decision Tree (J48)", "31.2%", "0.285", "0.063"),
             ("k-NN (k=15)", "30.3%", "0.274", "0.050"),
             ("SVM (SMO)", "31.9%", "0.272", "0.057"),
             ("Naive Bayes", "28.4%", "0.216", "0.024"),
             ("OneR-like", "31.4%", "0.202", "0.066")],
         t2_title="Primary &mdash; with previous CGPA",
         t2=[("ZeroR (baseline)", "27.4%", "0.108", "0.000"),
             ("Random Forest", "51.0%", "0.499", "0.351"),
             ("Gradient Boosting", "46.7%", "0.471", "0.302"),
             ("SVM (SMO)", "45.2%", "0.428", "0.258"),
             ("Decision Tree (J48)", "43.5%", "0.413", "0.215"),
             ("Logistic Regression", "39.6%", "0.402", "0.219"),
             ("k-NN (k=15)", "38.7%", "0.378", "0.171"),
             ("OneR-like", "39.1%", "0.271", "0.171"),
             ("Naive Bayes", "29.7%", "0.234", "0.042")],
         cols=("Model", "Accuracy", "Macro-F1", "&kappa;"),
         punch_n="+16.5",
         punch_l="accuracy points from one variable",
         read="Behaviour-only &kappa; = 0.09 is <strong>near-chance agreement</strong>. Adding the "
              "student's previous CGPA band lifts accuracy from 34.5% to 51.0% and &kappa; from "
              "0.09 to 0.35. Past performance predicts future performance; self-reported habits "
              "mostly do not."),

    dict(kind="errors", n="12", section="Error analysis",
         kicker="Where the models fail",
         head="Most errors are <em>one</em> band wide",
         fig="confusion",
         fig2="learning",
         gap_title="Gap analysis &mdash; |actual &minus; predicted| band index",
         gap_cols=("Gap", "Meaning", "Primary (RF)", "Behaviour (GB)"),
         gaps=[("0", "correct", "119 (51.7%)", "74 (32.2%)"),
               ("1", "neighbouring band", "68 (29.6%)", "81 (35.2%)"),
               ("2", "two bands off", "23 (10.0%)", "38 (16.5%)"),
               ("3", "three bands off", "20 (8.7%)", "37 (16.1%)")],
         within=("81.3%", "67.4%"),
         lc_title="Would more data help?",
         lc_body="The learning curve is <strong>flat</strong> at n = 230. More responses to the "
                 "same survey would not raise the ceiling — the limit is the features, not the "
                 "sample size.",
         improve_title="What would raise accuracy",
         improve=["Objective grades from the registrar instead of self-reports.",
                  "Previous CGPA — already worth +16.5 points.",
                  "Course-level rather than semester-level records.",
                  "<em>Not</em> more rows of the same questionnaire."]),

    dict(kind="closing", n="13", section="Conclusions",
         kicker="Conclusions &amp; limitations",
         head="What we can <em>honestly</em> claim",
         findings=[
             ("Behaviour alone barely predicts grades",
              "&kappa; = 0.09 against a 27.4% floor, and no feature survives multiple-comparison correction."),
             ("Prior performance carries the signal",
              "One variable moves accuracy 34.5% &rarr; 51.0%. This replicates Cortez &amp; Silva's caveat in a new population."),
             ("The dataset needed rebuilding",
              "The supplied clean files described a different dataset. Ours is rebuilt from 230 raw responses, 0 dropped."),
         ],
         lim_title="Limitations",
         lims=["Self-reported grades and habits — no registrar data to verify against.",
               "n = 230, a convenience sample circulated through our own networks.",
               "University and academic stage are partly confounded.",
               "GPA collected in bands, so no true continuous CGPA exists.",
               "Cross-sectional — every result is an association, never a cause."],
         work_title="Work distribution",
         work_note="To be confirmed by the team before submission.",
         work=[("Sheikh Md. Galib Mahim", "Cleaning pipeline, EDA, modelling notebook, documentation"),
               ("Md. Eftakar Jaman Arfan", "Dataset schema, first cleaning pass"),
               ("Asif Jawad", "Form design &amp; data collection"),
               ("Rakibul Islam", "WEKA experiments &amp; results tables"),
               ("Md Enam E Elahi", "Report writing &amp; related work")],
         repro="Everything is reproducible: <span class='mono'>notebooks/01_cleaning_eda_modeling.ipynb</span> "
               "regenerates every number, figure and export in this deck."),
]
