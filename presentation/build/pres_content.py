"""Shared content + asset helpers for the three CSE-4112 deck builds.

Every number here is copied from docs/results_summary_merged.json,
docs/model_results_merged_*.csv, docs/feature_importance_merged.csv,
docs/replication_n230_vs_n1108.csv, docs/source_form_comparison.csv and
docs/MERGED_DATASET_REPORT.md. Do not edit by hand — if
notebooks/02_merged_cleaning_eda_modeling.ipynb is re-run and numbers change,
update this file from the new outputs and rebuild.
"""
import base64, io, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures" / "merged"
OUT = ROOT / "presentation"

# ---------------------------------------------------------------- assets
_FIGURES = {
    "target":     "m07_target_distribution.png",
    "confound":   "m06_semester_confound.png",
    "srccmp":     "m11_source_comparison.png",
    "importance": "m14_feature_importance_consensus.png",
    "replicate":  "m15_replication_check.png",
    "compare":    "m17_best_models.png",
    "confusion":  "m18_confusion_matrices.png",
    "learning":   "m19_learning_curve.png",
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
    "subtitle": ("Two independent surveys &mdash; <strong>896 KUET students</strong> and "
                 "<strong>230 from BUET, BRAC and CUET</strong> &mdash; merged, cleaned, audited "
                 "and modelled as an <strong>ordinal classification</strong> of self-reported "
                 "CGPA bands."),
    "team": [
        ("2107020", "Sheikh Md. Galib Mahim"),
        ("2107030", "Md. Eftakar Jaman Arfan"),
        ("2107007", "Asif Jawad"),
        ("2107008", "Rakibul Islam"),
        ("2107012", "Md Enam E Elahi"),
    ],
    "glance": [
        ("Raw responses", "1,126"), ("After cleaning", "1,108"), ("Universities", "04"),
        ("Features", "25"), ("CGPA bands", "04"), ("Rows dropped", "18"),
    ],
}

# ---------------------------------------------------------------- slides
# Each slide is a dict the renderers consume. `kind` selects the layout.

SLIDES = [

    dict(kind="cover", n="01", section="Cover"),

    dict(kind="two-col", n="02", section="The brief",
         kicker="Problem statement",
         head="Two questions, <em>one</em> merged dataset",
         lead="The course asks for a model. The data asks a harder question first &mdash; whether "
              "self-reported habits carry any signal at all, and whether two separately collected "
              "surveys can honestly be pooled to find out.",
         left_title="Pipeline A &mdash; Feature influence",
         left_body="Which of a student's habits, circumstances and attitudes are actually "
                   "associated with academic performance? A ranking problem, answered with "
                   "statistics rather than a black box.",
         right_title="Pipeline B &mdash; CGPA prediction",
         right_body="Given a student's answers, predict their CGPA band. A supervised "
                    "classification problem over four ordered classes.",
         note="The two are kept apart. Prior academic performance dominates any model it sits in, "
              "so the behaviour-only run is what actually answers Pipeline A.",
         table_title="Two experiments, never merged",
         rows=[("Primary", "all features, <em>including</em> last semester's SGPA band"),
               ("Behaviour-focused", "the same features, <em>minus</em> prior performance")]),

    dict(kind="related", n="03", section="Related work",
         kicker="Related work &amp; base paper",
         head="Standing on <em>Cortez &amp; Silva</em>",
         base_title="Cortez &amp; Silva (2008) &mdash; <span class='mono'>UCI Student Performance</span>",
         base_body="Predicted secondary-school grades for Portuguese students using demographic, "
                   "social and school features gathered from reports and questionnaires. Compared "
                   "Decision Tree, Random Forest, Neural Network and SVM. The benchmark dataset "
                   "behind 200+ later studies.",
         base_stats=[("~72%", "reported four-class accuracy"),
                     ("33", "attributes"),
                     ("G1, G2", "strongest predictors &mdash; prior grades")],
         finding="Their headline caveat is that predictive power collapses once the prior-period "
                 "grades are removed. That is the exact effect we set out to measure.",
         contrib_title="What we contribute",
         contribs=[
             ("New primary data, and five times more of it",
              "1,126 responses from four Bangladeshi universities &mdash; a different population, "
              "education system and grading culture, at a sample size that can actually detect a "
              "small effect."),
             ("A documented cross-survey merge",
              "Two forms with different wording, different band schemes and a bilingual answer set, "
              "reconciled into one schema with the effect size of every difference measured."),
             ("A replication test, not just a result",
              "Every association is re-run on each sample independently. Two findings from our own "
              "230-response study did <em>not</em> replicate, and we retract them."),
             ("Multiple-comparison correction, applied honestly",
              "15 features tested, Benjamini&ndash;Hochberg applied &mdash; and a bug in our own "
              "first implementation found and fixed."),
         ]),

    dict(kind="two-col", n="04", section="Merging",
         kicker="Step 1 &mdash; how two surveys became one dataset", foot="Merging two surveys",
         head="Same questions, <em>different</em> dialect",
         lead="Two Google Forms, run separately, asking the same 19 substantive questions in a "
              "different order and with different English. Merging them is four passes, each of "
              "which can silently destroy the dataset if skipped.",
         left_title="Source A &mdash; multi-university form",
         left_body="<strong>230 responses</strong> from BUET (104), BRAC (101) and CUET (25). "
                   "English answer values. A department dropdown <em>plus</em> a separate "
                   "&ldquo;if not listed&rdquo; write-in box. GPA collected in <strong>6 "
                   "bands</strong>. No zero option on the commitments question.",
         right_title="Source B &mdash; KUET form",
         right_body="<strong>896 responses</strong>, all KUET, so the form never asked for a "
                    "university. <strong>Bilingual answer values</strong> "
                    "(<span class='mono'>90% or more (৯০% বা তার বেশি)</span>). One free-text "
                    "department box. GPA in <strong>4 bands</strong>, or type a number.",
         note="Pass 1 rejects fuzzy string matching: a threshold loose enough to match "
              "&ldquo;admitted into KUET&rdquo; with &ldquo;admitted into your university&rdquo; also "
              "merges the two <em>different</em> satisfaction questions with each other.<br><br>Pass 2 is the one that makes the merge real. Without it <span class='mono'>90% or more</span> and "
              "<span class='mono'>90% or more (৯০%…)</span> are two unrelated categories, the 1,126 rows "
              "become two disjoint blocks sharing no feature values, and every model learns "
              "<em>which form a row came from</em> instead of anything about students.",
         table_title="Four harmonisation passes",
         rows=[("1 &middot; Columns",
                "21 columns hand-mapped to one schema, with an <span class='mono'>assert</span>"),
               ("2 &middot; Values",
                "Strip the Bangla bracket and emoji out of every answer value"),
               ("3 &middot; Options",
                "Overlap-check all 17 shared columns &mdash; 3 real mismatches found"),
               ("4 &middot; Concatenate",
                "Stack rows, keep <span class='mono'>source_form</span> for auditing, exclude it from every model")]),

    dict(kind="dataset", n="05", section="Dataset",
         kicker="Dataset information",
         head="1,108 students, <em>four</em> ordered bands",
         lead="Bilingual Google Forms (English + Bangla) circulated to students at four "
              "universities. 15 five-point Likert items, a banded SGPA, a banded CGPA, semester "
              "and department.",
         fig="target",
         cap="The forms collected GPA as bands, never as a number &mdash; so this is ordinal "
             "classification, not regression. The band mix is similar in both samples, which is "
             "the first evidence the merge is legitimate.",
         unis=[("KUET", "878"), ("BUET", "104"), ("BRAC", "101"), ("CUET", "25")],
         bands=[("C0", "below 3.20", "236"), ("C1", "3.20 &ndash; 3.49", "321"),
                ("C2", "3.50 &ndash; 3.74", "303"), ("C3", "3.75 and above", "248")],
         note="Near-balanced at 21&ndash;29% per class (ratio 1.36). Two consequences: no "
              "resampling is needed, and <strong>accuracy is a fair metric</strong>. Anything we "
              "build must beat <strong>29.0%</strong> (ZeroR). The 4-class scheme is forced by the "
              "data &mdash; KUET never offered the finer 6-band split."),

    dict(kind="integrity", n="06", section="Raw &rarr; clean",
         kicker="Step 2 &mdash; what cleaning actually did", foot="Raw &rarr; clean", kicker_cls="",
         head="Raw in, <em>clean</em> out &mdash; field by field",
         lead="The clearest single answer to &ldquo;how did you clean it?&rdquo; is to put the two "
              "states of each messy field side by side.",
         cols=("Field", "Raw &mdash; as Google Forms exported it", "Clean &mdash; as the model sees it"),
         rows=[
             ("university", "<span class='mono'>BRAC</span> / <span class='mono'>Brac</span> / <span class='mono'>BRAc</span> / <span class='mono'>BRAC University</span> &nbsp;(10 spellings)",
              "<span class='mono'>BRAC</span> &nbsp;(4 institutions)"),
             ("department", "<span class='mono'>cse</span>, <span class='mono'>Cse</span>, <span class='mono'>Computer Science and Egineering</span>, <span class='mono'>Department of CSE </span> &nbsp;(110 strings)",
              "<span class='mono'>CSE</span> &nbsp;(24 canonical, 18 after rare-grouping)"),
             ("attendance", "<span class='mono'>90% or more (৯০% বা তার বেশি)</span>",
              "<span class='mono'>4</span> &nbsp;(ordinal code, 0&ndash;4)"),
             ("weekly_responsibilities", "<span class='mono'>None (কোনোটিই না)</span> &nbsp;or blank",
              "<span class='mono'>0</span> &nbsp;(blank imputed as <span class='mono'>None</span>)"),
             ("cgpa", "<span class='mono'>3.2 - 3.49</span> / <span class='mono'>3.2+</span> / <span class='mono'>3.71</span> / <span class='mono'>jani na :3</span>",
              "<span class='mono'>C1_3_20_to_3_49</span> &nbsp;(4 ordered classes)"),
         ],
         verdict="1,126 raw rows &rarr; <strong>1,108</strong> clean rows &times; 25 features. "
                 "18 rows dropped, all for the same reason: no usable CGPA answer, so there is no "
                 "label to learn from. That is the only place in the pipeline a row is discarded.",
         action="45 blanks imputed on the commitments question, plus scattered blanks under 1.3% "
                "per column. <strong>Verified, not asserted:</strong> re-running every correlation "
                "on complete cases only moves no coefficient by more than <strong>0.0036</strong>, "
                "so the imputation cannot have manufactured a finding."),

    dict(kind="preprocess", n="07", section="Preprocessing",
         kicker="Data preprocessing techniques",
         head="Ten steps, <em>one</em> reason to drop a row",
         lead="Every step is recorded as Decision &rarr; Evidence &rarr; Why. The viva asks how we "
              "preprocessed and why; these are the answers.",
         steps=[
             ("Drop <span class='mono'>timestamp</span>",
              "Submission time tracks how each link spread &mdash; and because the two forms ran in different windows, it is a perfect proxy for which form a row came from."),
             ("10 university spellings &rarr; 4",
              "<span class='mono'>BRAC / Brac / BRAc / BRAC University</span> are one institution; unmerged, 101 students split into five weak groups."),
             ("110 department strings &rarr; 24",
              "Case, whitespace, <span class='mono'>Department of</span> prefixes, bracketed abbreviations and typos. <span class='mono'>ChE</span> and <span class='mono'>Chemical Engineering</span> are one department."),
             ("Group departments under 15 students",
              "A category with four members cannot be learned, only memorised. Threshold moved from 5 to 15 because the sample is 5&times; larger."),
             ("Strip Bangla + emoji from answers",
              "Otherwise the two samples share no feature values and every model learns the form, not the student."),
             ("2 band schemes + free text &rarr; 4 classes",
              "35 people typed a number instead of picking a band. KUET never offered the 6-band split, so 4 classes is what is recoverable from both."),
             ("45 blanks &rarr; <span class='mono'>None</span>",
              "Source A offered no zero option; KUET did, and 26.5% pressed it. Blankness is independent of GPA (&chi;&sup2; p = 0.688)."),
             ("Likert &rarr; ordinal integers",
              "The categories are ordered. One-hot destroys the order and turns 15 columns into ~72. University and department <em>are</em> nominal, so those do get one-hot &mdash; inside the model only."),
             ("Drop <span class='mono'>result_satisfaction</span>",
              "An outcome, not a behaviour: a student cannot rate a result they have not received."),
             ("Derive <span class='mono'>academic_progress</span>",
              "The raw semester number encodes university, not progress. BRAC runs trimesters; 676 of 896 KUET rows sit at semester 7."),
         ],
         stats=[("1,126", "rows in"), ("1,108", "rows out"),
                ("18", "dropped"), ("45", "imputed")]),

    dict(kind="evidence", n="08", section="Preprocessing",
         kicker="Two decisions worth defending",
         head="An assumption, and a <em>confounded</em> variable",
         fig="confound",
         cap="Semester number by university. BRAC runs trimesters, so all 101 of its respondents "
             "sit at 10&ndash;11; all 25 CUET respondents are at 7; and 676 of 896 KUET "
             "respondents are at 7 because the form went through one batch.",
         a_title="The merge turned an assumption into <em>evidence</em>",
         a_body="Our 230-response study had to <strong>assume</strong> that a blank on the "
                "commitments question meant &ldquo;I have none&rdquo;, because that form offered no "
                "zero option. The KUET form <em>did</em> offer <span class='mono'>None</span>, to a "
                "comparable population answering the same question.",
         a_action="<strong>26.5%</strong> of KUET students actively pressed <span class='mono'>None</span>; "
                  "<strong>19.1%</strong> of Source-A students left it blank when the button did not "
                  "exist. Same behaviour, two expressions. This is the single clearest argument for "
                  "having merged the datasets: an assumption became a measurement.",
         b_title="Confound &mdash; <span class='mono'>semester</span>",
         b_body="&ldquo;Semester 11&rdquo; does not mean further along than &ldquo;semester "
                "7&rdquo; &mdash; it means BRAC. A model given the raw number can recover the "
                "institution and learn per-university grading habits while appearing to learn "
                "about academic progress.",
         b_action="Replaced with <span class='mono'>academic_progress</span> = semester &divide; "
                  "programme length, comparable across all four institutions. Raw "
                  "<span class='mono'>semester_raw</span> never leaves the notebook."),

    dict(kind="evidence", n="09", section="EDA",
         kicker="Exploratory data analysis", foot="Exploratory data analysis",
         head="Is this <em>one</em> dataset, or two?",
         fig="srccmp",
         cap="Cliff's &delta; per item between the two samples &mdash; a non-parametric effect "
             "size for ordinal data. 12 of 16 items are <em>negligible</em>, 3 small, 1 medium.",
         a_title="The merge holds &mdash; with one named exception",
         a_body="KUET students and BUET/BRAC/CUET students report study habits, sleep, stress, "
                "environment, support and distraction at close to the <strong>same rates</strong> "
                "(|&delta;| &lt; 0.15). That is the empirical licence to pool them. The exception "
                "is <span class='mono'>admission_satisfaction</span> (&delta; = &minus;0.34).",
         a_action="Two readings we cannot separate: a real difference, or a response artefact "
                  "&mdash; KUET's <em>Very dissatisfied</em> option drew 40% of answers while "
                  "<em>Dissatisfied</em> drew 5%. We keep the feature and quarantine it: any "
                  "finding resting on it is re-checked within source. It is also why "
                  "<span class='mono'>source_form</span> is excluded from every model.",
         b_title="What else the EDA settled",
         b_body="<strong>No degenerate feature</strong> &mdash; zero items exceed a 60% modal "
                "share, so nothing is dropped for being uninformative. <strong>No "
                "multicollinearity</strong> &mdash; no feature pair exceeds |r| = 0.7, so no "
                "feature needs removing and we are not forced into regularised models only.",
         b_action="<strong>Grading culture does differ by institution</strong> (&chi;&sup2; = 34.6, "
                  "p = 7&times;10&#8315;&#8309;) but weakly: Cram&eacute;r's V = 0.102. And "
                  "<span class='mono'>sleep_duration</span> is confirmed <em>monotonic</em>, not "
                  "inverted-U &mdash; which retroactively justifies encoding it by duration."),

    dict(kind="importance", n="10", section="Feature influence",
         kicker="Pipeline A &mdash; which features matter",
         head="Four methods. <em>Three</em> survivors.",
         lead="No single importance measure is trustworthy on survey data, so we ran four and "
              "ranked by consensus &mdash; then corrected for having tested 15 features at once.",
         fig="importance",
         methods=[
             ("Spearman &rho;", "monotonic rank association"),
             ("Mutual information", "catches non-linear structure"),
             ("Kruskal&ndash;Wallis H", "distribution differs across bands?"),
             ("Permutation importance", "real out-of-fold accuracy drop"),
         ],
         why="Permutation importance, not the tree's built-in Gini importance &mdash; Gini is "
             "measured on training data and is biased toward features with many distinct values.",
         verdict_head="3 of 15 features survive FDR correction",
         verdict="<strong>desired_department_match</strong> &rho; = +0.119, q = 0.001 &nbsp;&middot;&nbsp; "
                 "<strong>attendance</strong> &rho; = +0.090, q = 0.020 &nbsp;&middot;&nbsp; "
                 "<strong>weekly_responsibilities</strong> &rho; = &minus;0.086, q = 0.020. "
                 "The strongest behavioural correlate of CGPA is <em>not a habit at all</em> &mdash; "
                 "it is whether the student got into the subject they wanted.",
         honest="Both facts must be reported together: these effects are <strong>real</strong> "
                "(they survive correction on 1,108 students) and they are <strong>small</strong> "
                "(the largest explains ~1.4% of the variance). They are detectable only because "
                "n = 1,108 reaches |&rho;| &asymp; 0.059, where n = 230 needed |&rho;| &asymp; 0.130."),

    dict(kind="counter", n="11", section="Feature influence",
         kicker="The replication test", foot="Pipeline A &mdash; replication",
         head="What the merge <em>overturned</em>",
         lead="The most instructive result here is not a number &mdash; it is which of our own "
              "earlier findings survived a five-fold increase in sample size. We re-ran every test "
              "on each sample independently.",
         fig="replicate",
         a_title="Two findings we <em>retract</em>",
         a_body="At n = 230 it looked like students who understood lectures <em>better</em> got "
                "<em>worse</em> grades (<span class='mono'>topic_clarity</span> &rho; = &minus;0.161, "
                "the largest effect in that dataset) and that more stress went with better grades "
                "(&rho; = +0.119).",
         a_read="Neither appears in the 896-student KUET sample "
                "(&rho; = +0.053 and &minus;0.016) and neither survives in the merged data. They "
                "were <strong>small-sample noise</strong>. We retract them &mdash; and this is "
                "exactly what merging the datasets was for.",
         b_title="Two findings that <em>replicate</em>",
         b_body="<span class='mono'>desired_department_match</span> is significant in both samples "
                "independently (+0.144 at n = 230, +0.113 at n = 896). "
                "<span class='mono'>weekly_responsibilities</span> shows the same sign and the same "
                "size in both (&minus;0.080 and &minus;0.089) and only becomes detectable when "
                "they are pooled.",
         b_read="Only <strong>6 of 15</strong> features even agree in sign across the two samples "
                "&mdash; about what chance predicts for effects this small, and a standing caution "
                "against reading anything into a single unreplicated survey correlation.",
         caveat="Same test, three samples. Bars are the two samples run separately; diamonds are "
                "the merged estimate. Where the two bars point opposite ways, the n = 230 result "
                "was noise."),

    dict(kind="models", n="12", section="Models",
         kicker="Models used &amp; evaluation protocol",
         head="Nine classifiers, <em>one</em> protocol",
         models=[
             ("ZeroR", "majority-class floor"),
             ("OneR-like", "single best feature"),
             ("Naive Bayes", "probabilistic baseline"),
             ("Logistic Regression", "linear, interpretable"),
             ("k-NN (k=25)", "instance-based"),
             ("Decision Tree", "J48 equivalent"),
             ("Random Forest", "bagged ensemble"),
             ("Gradient Boosting", "boosted ensemble"),
             ("SVM (RBF)", "SMO equivalent"),
         ],
         protocol_title="Why this protocol",
         protocol=[
             ("Repeated stratified 10-fold CV",
              "A single holdout swings several points on the seed alone. 30 fits per model make the ranking stable enough to act on; stratified so every fold keeps the 4-class balance."),
             ("Accuracy",
              "Fair here <em>only</em> because the classes are balanced (21&ndash;29%)."),
             ("Macro-F1 &mdash; what we rank by",
              "Averages the four classes equally, so a weak class cannot hide behind a strong one."),
             ("Cohen's &kappa;",
              "Corrects for agreement expected by chance. The metric that exposes a model which is really ZeroR wearing a hat."),
             ("Within-1-band accuracy",
              "The target is <em>ordinal</em> &mdash; predicting C2 when the truth is C3 is a smaller error than predicting C0."),
         ],
         warn="Trap: ZeroR scores <strong>77.6%</strong> on within-1-band &mdash; higher than any "
              "behaviour model &mdash; without learning anything, because it always predicts a "
              "middle class that is within one band of three of the four. Always read within-1-band "
              "next to &kappa;, which is 0.000 for ZeroR by construction."),

    dict(kind="results", n="13", section="Results",
         kicker="Comparison between results from different models",
         head="Prior performance is <em>the</em> signal",
         t1_title="Behaviour-focused &mdash; without prior SGPA",
         t1=[("ZeroR (baseline)", "29.0%", "0.112", "0.000"),
             ("SVM (RBF)", "31.2%", "0.284", "0.054"),
             ("Random Forest", "31.0%", "0.279", "0.050"),
             ("Logistic Regression", "29.1%", "0.272", "0.025"),
             ("Decision Tree (J48)", "26.6%", "0.264", "0.026"),
             ("Gradient Boosting", "28.0%", "0.258", "0.012"),
             ("k-NN (k=25)", "29.9%", "0.242", "0.035"),
             ("Naive Bayes", "23.8%", "0.201", "0.033"),
             ("OneR-like", "28.2%", "0.180", "0.027")],
         t2_title="Primary &mdash; with last semester's SGPA",
         t2=[("ZeroR (baseline)", "29.0%", "0.112", "0.000"),
             ("Gradient Boosting", "43.4%", "0.432", "0.237"),
             ("Decision Tree (J48)", "43.3%", "0.427", "0.234"),
             ("Random Forest", "43.6%", "0.426", "0.230"),
             ("SVM (RBF)", "41.2%", "0.392", "0.192"),
             ("Logistic Regression", "35.8%", "0.349", "0.128"),
             ("k-NN (k=25)", "35.8%", "0.302", "0.109"),
             ("OneR-like", "34.5%", "0.263", "0.083"),
             ("Naive Bayes", "24.5%", "0.215", "0.049")],
         cols=("Model", "Accuracy", "Macro-F1", "&kappa;"),
         punch_n="+12.2",
         punch_l="accuracy points from one variable",
         read="Behaviour-only &kappa; = 0.054 is <strong>near-chance agreement</strong>. Adding "
              "last semester's SGPA band lifts accuracy 31.2% &rarr; 43.4% and &kappa; 0.054 "
              "&rarr; 0.237 &mdash; more than all 15 behavioural items combined. "
              "<strong>Naive Bayes is the only clear loser</strong>, in both experiments, for an "
              "identifiable reason: it assumes conditional independence, and the Likert items are "
              "mildly but consistently inter-correlated."),

    dict(kind="counter", n="14", section="Results",
         kicker="Reading the comparison honestly", foot="Comparison between models",
         head="A podium is not always a <em>winner</em>",
         lead="Two things the ranking table does not say on its own &mdash; including the check the "
              "primary result had to pass before we were willing to quote it.",
         fig="compare",
         a_title="The top of each table is a statistical <em>tie</em>",
         a_body="The macro-F1 gap between 1st and 2nd on the behaviour experiment is "
                "<strong>0.0049</strong>, against a fold-to-fold accuracy standard deviation of "
                "<strong>&plusmn;0.043</strong>.",
         a_read="So we do <em>not</em> crown a winner on that margin. SVM, Random Forest and "
                "Logistic Regression are indistinguishable here; so are Gradient Boosting, Random "
                "Forest and the Decision Tree in the primary run. Saying so is more useful than "
                "picking one and defending it in the viva.",
         b_title="Is the +12.2 lift real, or <em>arithmetic</em>?",
         b_body="For KUET rows the CGPA target includes the semester whose SGPA is the primary "
                "feature, so part of the lift could be definitional. Source A has no such overlap "
                "&mdash; its CGPA is measured strictly <em>before</em> that semester.",
         b_read="Run separately: <strong>+12.4 points</strong> on the KUET subset, "
                "<strong>+13.5 points</strong> on the no-overlap subset. The lift is if anything "
                "<em>larger</em> where the overlap cannot exist, so the primary result is "
                "prediction, not arithmetic, and needs no discounting.",
         caveat="Best model of each experiment against the ZeroR floor, on all four metrics. Note "
                "how ZeroR's free 77.6% on within-1-band sits above the behaviour model's 71.5%."),

    dict(kind="errors", n="15", section="Error analysis",
         kicker="Where the models fail, and what would fix it",
         head="Most errors are <em>one</em> band wide",
         fig="confusion",
         fig2="learning",
         gap_title="Gap analysis &mdash; |actual &minus; predicted| band index",
         gap_cols=("Gap", "Meaning", "Primary (GB)", "Behaviour (SVM)"),
         gaps=[("0", "correct", "480 (43.3%)", "335 (30.2%)"),
               ("1", "neighbouring band", "385 (34.7%)", "457 (41.2%)"),
               ("2", "two bands off", "176 (15.9%)", "241 (21.8%)"),
               ("3", "three bands off", "67 (6.0%)", "75 (6.8%)")],
         within=("78.1%", "71.5%"),
         lc_title="Would more data help?",
         lc_body="A 10&times; increase in training data bought about <strong>three accuracy "
                 "points</strong> (0.270 at n = 99 &rarr; 0.300 at n = 997), and the curve is flat "
                 "over its final third. The ceiling is the <em>features</em>, not the sample size.",
         improve_title="What would raise accuracy",
         improve=["Registrar grades instead of self-reports.",
                  "Course-level, not semester-level, records.",
                  "Longitudinal data, so habit <em>changes</em> can be measured.",
                  "Prior performance &mdash; already worth +12.2 points.",
                  "<em>Not</em> more rows of the same questionnaire."]),

    dict(kind="integrity", n="16", section="Verdict",
         kicker="Verdict", foot="Verdict", kicker_cls="",
         head="What this dataset <em>can</em> and cannot support",
         lead="Six questions, answered from the computed numbers rather than from what we hoped to "
              "find.",
         cols=("Question", "Answer", "Evidence"),
         rows=[
             ("Can the two surveys honestly be merged?", "<strong>Yes</strong>",
              "12 of 16 items differ by a negligible Cliff's &delta;; band mixes match; the one medium-&delta; exception is named and quarantined"),
             ("Do behaviour features alone predict CGPA?", "<strong>Barely</strong>",
              "31.2% vs a 29.0% floor, &kappa; = 0.054 &mdash; slight agreement at best"),
             ("Does prior performance help?", "<strong>Decisively</strong>",
              "+12.2 accuracy points, &kappa; 0.054 &rarr; 0.237, from one column; verified free of reference-point artefact"),
             ("Are any behavioural associations real?", "<strong>Three are</strong>",
              "desired department, attendance, outside commitments &mdash; all survive FDR at n = 1,108, all small (|&rho;| &le; 0.119)"),
             ("Do the composite &ldquo;trait&rdquo; indices exist?", "<strong>No</strong>",
              "best Cronbach &alpha; = 0.273 at n = 1,108, after failing at n = 230; adding all 7 derived features moves accuracy &minus;0.011"),
             ("Would more of this survey help?", "<strong>No</strong>",
              "learning curve flat: +3 accuracy points across a 10&times; data increase"),
         ],
         verdict="<strong>An interpretable model is worse than guessing here.</strong> The depth-3 "
                 "decision tree we would show a viva panel reaches 27.3% &mdash; <em>below</em> the "
                 "29.0% ZeroR floor. Interpretability and accuracy are in direct conflict on this "
                 "data, and that conflict is a finding, not a bug.",
         action="<strong>And a bug we found in our own statistics.</strong> Our first "
                "Benjamini&ndash;Hochberg implementation took its running minimum in the wrong "
                "direction, collapsing every q-value onto the smallest and reporting 15 of 15 "
                "features as significant. Fixed in both notebooks, both re-run; the corrected count "
                "is 3 of 15. We report it because checking your own statistics is the point."),

    dict(kind="closing", n="17", section="Conclusions",
         kicker="Conclusions &amp; limitations",
         head="What we can <em>honestly</em> claim",
         findings=[
             ("Merging worked &mdash; for the statistics",
              "Five times the data turned a null result into three FDR-surviving associations, and falsified two of our own earlier findings."),
             ("Behaviour alone barely predicts grades",
              "&kappa; = 0.054 against a 29.0% floor; the numeric view gives R&sup2; = &minus;0.001, no better than quoting the class average."),
             ("Prior performance carries the signal",
              "One variable moves accuracy 31.2% &rarr; 43.4%. This replicates Cortez &amp; Silva's caveat in a new population."),
         ],
         lim_title="Limitations",
         lims=["Self-reported grades and habits &mdash; no registrar data to verify against.",
               "Convenience sample: 79% of rows are KUET, and 75% of those sit in one semester.",
               "The CGPA reference point differs by one semester between the two forms.",
               "<span class='mono'>admission_satisfaction</span> may carry a first-option response artefact on the KUET form.",
               "GPA collected in bands, so no true continuous CGPA exists.",
               "Cross-sectional &mdash; every result is an association, never a cause."],
         work_title="Work distribution",
         work_note="To be confirmed by the team before submission.",
         work=[("Sheikh Md. Galib Mahim", "Merge &amp; cleaning pipeline, EDA, modelling notebook, documentation"),
               ("Md. Eftakar Jaman Arfan", "Dataset schema, KUET cleaning pass"),
               ("Asif Jawad", "Form design &amp; data collection"),
               ("Rakibul Islam", "WEKA experiments &amp; results tables"),
               ("Md Enam E Elahi", "Report writing &amp; related work")],
         repro="Everything is reproducible: <span class='mono'>notebooks/02_merged_cleaning_eda_modeling.ipynb</span> "
               "regenerates every number, figure and export in this deck &mdash; 1,108 instances in WEKA, "
               "not the 878 the supplied tutorial mentions."),
]
