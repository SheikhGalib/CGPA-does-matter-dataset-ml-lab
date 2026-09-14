# CSE-4112 — Machine Learning Laboratory

Project context file for AI agents and teammates. Read this first.

---

## 1. Teacher's instructions (verbatim — do not edit)

> Regarding ML Lab Presentation:
>
> Presentation Topics must include:
> 0. Related works
> 1. Dataset information
> 2. Data preprocessing techniques
> 3. Models used
> 4. Comparison between results from different models
> 5. Summary of work distribution among teammates
>
> and one may add any other required sections if needed.
>
> #Bring your own laptop to demonstrate
>
> #You will have to write a final report discussing your work. One report per team.
>
> -Sadi Sir

> ML Lab (viva focus):
> - Full architecture + code dekhe (jei code diya mainly data preprocess, split, train, val, test kora hoise)
> - Base paper er shate comparison. what is your contribution
> - data pre processing kmne kora hoise
> - ki korle accuracy barto ba keno kom hoise accuracy
> - proper evaluation metrics use kora and ken use kora hoise.

**What this implies for us:** every preprocessing decision must have a written *why*. The viva
asks "how did you preprocess and why", so the notebook is the answer sheet — it must show the
decision, the evidence behind it, and the alternative we rejected.

---

## 2. Project scope

**Data sources — TWO forms, merged.**

| | Source A — `University Student Performance Analysis Form` | Source B — `Student Performance Analysis Form` |
|---|---|---|
| Responses | 230 | 896 |
| Universities | BUET (104), BRAC (101), CUET (25) | KUET only |
| Columns | 22 | 21 |
| Answer values | English | **Bilingual** (`90% or more (৯০% বা তার বেশি)`) |
| GPA options | 6 bands | 4 bands, or free-text number |
| Commitments question | no zero option | has an explicit `None` |

Both ask the same 19 substantive questions. **`notebooks/02_...` merges them into 1,126 raw rows
→ 1,108 cleaned rows across four universities.** That is the current dataset for every result we
report. Notebook `01` (Source A alone, n=230) is kept as the *replication baseline*, not as the
headline analysis.

Question text is bilingual in both forms.

**Two deliverable pipelines:**

| # | Pipeline | Question it answers | Framing |
|---|----------|--------------------|---------|
| A | **Feature influence** | Which factors actually move academic performance? | Ranking / inference |
| B | **CGPA prediction** | Given a student's answers, predict their CGPA band | Supervised classification |

**Two experiments** (following the WEKA tutorial in `docs/`):

- **Primary** — all features *including* the student's most recent semester SGPA band.
- **Behaviour-focused** — the same features *minus* prior performance.

They stay separate and are never merged into one ranking. Prior performance dominates any model it
sits in, so the behaviour-only run is what actually answers Pipeline A.

**Target variable.** `cgpa_band` (merged notebook; notebook `01` calls it
`current_cgpa_band`), 4 ordered classes:

| Code | Range | n (merged) | n (notebook 01) |
|------|-------|-----------|-----------------|
| `C0_below_3_20` | below 3.20 | 236 | 60 |
| `C1_3_20_to_3_49` | 3.20 – 3.49 | 321 | 52 |
| `C2_3_50_to_3_74` | 3.50 – 3.74 | 303 | 63 |
| `C3_3_75_plus` | 3.75 and above | 248 | 55 |

Both forms collected GPA as **bands, not numbers**. So this is an **ordinal classification**
problem, not regression. Any "predict the CGPA number" output is a band midpoint and must be
reported as such.

> **The one semantic wart, and it must be stated in the viva.** Source A asked "*before that
> semester*, what was your CGPA?" while Source B asked "what is your *current* CGPA?". The
> reference point differs by one semester, so `recent_sgpa_band` is partially contained in the
> target for KUET rows. §9.3 of the merged notebook bounds the effect by running the primary
> experiment on each source separately: the lift is **+12.4 pts** on KUET and **+13.5 pts** on the
> no-overlap subset, so the headline result is prediction, not arithmetic.

**Tooling.** Cleaning, EDA and model selection in Python (this repo). The final model comparison
is demonstrated in **WEKA**, because that is what the course requires.

---

## 3. Repository map

```
ml-lab/
├── AGENTS.md                          # this file
├── raw-data/                          # SOURCE OF TRUTH. Read-only. Never edit.
│   ├── University Student Performance Analysis Form (Responses)
│   │   - Form Responses 1.csv         # Source A: 230 rows, BUET/BRAC/CUET
│   └── Student Performance Analysis Form (Responses)
│       - Form Responses 1.csv         # Source B: 896 rows, KUET
├── cleaned-dataset/
│   ├── student_performance_*_clean.csv          # Jaman's KUET pass — superseded, see section 4
│   └── ours/
│       ├── primary_clean.csv          # notebook 01 output: 230 rows (baseline only)
│       ├── behavior_clean.csv
│       ├── primary_weka.csv | behavior_weka.csv | primary.arff | behavior.arff
│       ├── merged/                    # notebook 02 output — the cleaned dataset
│       │   ├── merged_primary_clean.csv    # 1108 x 27, text values, with recent SGPA band
│       │   ├── merged_behavior_clean.csv   # 1108 x 26, text values, no prior performance
│       │   ├── merged_full_clean.csv       # every intermediate column, for auditing
│       │   ├── merged_primary_weka.csv | merged_behavior_weka.csv
│       │   ├── merged_primary.arff         # WEKA-ready: 1108 instances
│       │   └── merged_behavior.arff
│       ├── final/                     # *** FINAL DATASET (section 4b) *** — notebooks/final_weka_results.py
│       │   ├── final_920.arff              # load this in WEKA: 12 items + recent SGPA + cgpa_band
│       │   ├── final_920.csv | final_920_with_details.csv | answer_codes.csv
│       │   └── dropped_182_rows.csv        # the removed answers, with university/semester/date/gap
│       └── anomaly/                   # notebook 03 output (investigation stage, superseded by final/)
│           ├── main_cgpa_1102.arff         # the team's WEKA setting: 12 items + recent SGPA (nominal)
│           ├── trusted_main_cgpa.arff      # same, 6-7 Aug KUET sem-7 wave excluded (647 rows)
│           ├── split_{all,trusted}_{train80,test20}.arff   # fixed stratified split, seed 42
│           ├── main_cgpa_1102_with_wave_flag.arff          # for the clean-training-folds demo
│           └── _work/                      # auxiliary ARFFs (git-ignored)
├── notebooks/
│   ├── 01_cleaning_eda_modeling.ipynb # Source A only, n=230 — the replication baseline
│   ├── pipeline_source.py             # its `# %%` source
│   ├── 02_merged_cleaning_eda_modeling.ipynb   # merge + clean + EDA, n=1108 (Python models)
│   ├── pipeline_v2_source.py          # its `# %%` source
│   ├── 03_anomaly_investigation_weka.ipynb     # *** CURRENT *** far-error anomaly, all models in WEKA
│   ├── pipeline_v3_source.py          # its `# %%` source
│   ├── weka_runner.py                 # Python -> WEKA 3.8.7 CLI driver (ARFF, runs, log parsing)
│   ├── final_weka_results.py          # *** FINAL *** builds final_920 + runs every WEKA result in the deck
│   └── py2nb.py                       # `# %%` .py -> .ipynb converter
├── docs/
│   ├── WEKA_Next_Steps_Tutorial.pdf   # given to us
│   ├── WEKA_Next_Steps_Tutorial.md    # markitdown conversion — read this instead
│   ├── FINAL_WEKA_STEPS.md            # *** FINAL *** files, WEKA Explorer steps, expected numbers, screenshot slots
│   ├── final_results.json             # every number in the updated pptx, machine-readable
│   ├── weka_runs/final/               # WEKA logs behind final_results.json
│   ├── figures/final/                 # charts used in the updated pptx
│   ├── ANOMALY_INVESTIGATION_REPORT.md   # checks 1-7, fixes, WEKA steps (investigation stage)
│   ├── anomaly_results.json           # every notebook-03 headline number, machine-readable
│   ├── anomaly_tables/                # notebook-03 result tables (CSV)
│   ├── weka_runs/                     # raw WEKA output for every quoted number (command on line 2)
│   │   └── bulk/                      # permutation / resample / seed-sweep runs
│   ├── MERGED_DATASET_REPORT.md       # notebook-02 report (merge + cleaning)
│   ├── NOTEBOOK_WALKTHROUGH.md        # plain-English guide to notebook 01
│   ├── WEKA_GUIDE.md                  # how to run our data in WEKA
│   ├── DATA_QUALITY_REPORT.md         # notebook-01-era data-quality notes
│   ├── figures/                       # 17 charts from notebook 01
│   │   ├── merged/                    # 22 charts from notebook 02 (m01..m22)
│   │   └── anomaly/                   # charts from notebook 03 (a01..a15)
│   ├── results_summary.json           # notebook 01 headline numbers
│   ├── model_results_primary.csv | model_results_behavior.csv
│   ├── feature_importance_consensus.csv
│   ├── results_summary_merged.json    # notebook 02 headline numbers, machine-readable
│   ├── model_results_merged_primary.csv | model_results_merged_behavior.csv
│   ├── feature_importance_merged.csv  # 4-method consensus with q-values
│   ├── replication_n230_vs_n1108.csv  # which findings replicate
│   ├── source_form_comparison.csv     # Cliff's delta between the two samples
│   └── cleaning_step_log.csv          # the 10-step audit trail
├── presentation/
│   ├── What_Shapes_Student_CGPA_Ready.pptx      # *** THE DECK THE TEAM PRESENTS *** (75 slides)
│   ├── What_Shapes_Student_CGPA_Ready_BEFORE_FINAL_UPDATE.pptx   # untouched copy; update_deck.py reads it
│   ├── final_update/                  # make_figures.py -> docs/figures/final; update_deck.py -> the pptx
│   ├── deck-signal.html               # same 17 slides, three visual styles
│   ├── deck-cobalt.html
│   ├── deck-swiss.html
│   └── build/                         # deck source — see section 8
│       ├── pres_content.py            # ALL slide copy + numbers live here
│       ├── slides_html.py             # shared semantic markup
│       ├── theme_signal|cobalt|swiss.py
│       ├── deck_shell.py              # fixed-stage CSS + controller JS
│       ├── build.py                   # python build.py all
│       └── verify.py                  # python verify.py all
└── venv/                              # Python 3.14, see section 6
```

---

## 4. The `cleaned-dataset/*_clean.csv` files — resolved

An earlier version of this file recorded that `student_performance_primary_clean.csv` /
`_behavior_clean.csv` (878 rows, a `kuet_satisfaction` column, KUET-only departments) were
"a different, most likely synthetic, dataset". **That was wrong, and getting the KUET raw form is
what disproved it.**

Those files are a cleaned export of an *earlier pull* of the KUET form — 878 rows then, 896 now.
Their per-option counts match `raw-data/Student Performance Analysis Form...csv` to within the 18
responses that arrived later. `kuet_satisfaction` was the correct name for that form's question;
`S1–S7` was the correct semester range for a sample sitting overwhelmingly in semester 7.

**What still stands:**

- We rebuild every row from `raw-data/` rather than trusting a derived file. That is a method rule,
  not a judgement about those files.
- We keep that schema's good ideas — the C0–C3 band scheme, the primary/behaviour split, the
  snake_case naming — and `notebooks/02_...` re-derives all of it.
- `docs/WEKA_Next_Steps_Tutorial.pdf` says "check that WEKA shows 878 instances". For our merged
  data the number is **1,108**. Its methodology (ZeroR/OneR baselines, 10-fold CV, macro-F1, kappa,
  confusion matrix, gap analysis) is sound and we follow it.

**If someone asks in the viva why the repo history calls that file synthetic:** we could not
reconcile it with the only raw data we had, we said so with evidence, we got the missing raw file,
and we corrected the record. That sequence is the right one.

---

## 4b. The far-error anomaly and the trusted dataset — must know

The teacher flagged that WEKA confusion matrices put a significant share of predictions **two or
three CGPA bands** away from the truth. `notebooks/03_anomaly_investigation_weka.ipynb` traced it:

- ~22% of students report a recent SGPA and a CGPA **two or more bands apart**. About two thirds of
  all far errors land on them.
- Those contradictions are concentrated in **KUET semester-7 responses submitted on 6-7 August 2026**
  (455 rows). Before 6 Aug the same cohort is consistent; in the wave, SGPA and CGPA look randomly
  chosen, and target-free careless-response checks (satisfaction contradicting own SGPA,
  Mahalanobis outliers) are also elevated.
- Models score at **ZeroR** on that wave even when trained on everyone else; on the remaining
  **647 trusted rows** WEKA SMO reaches ~61% (ZeroR ~31%) with far errors under 10%.

**Rule used, stated exactly:** exclude rows where `source_form == kuet` AND `semester == 7` AND
submission date is 2026-08-06 or 2026-08-07. Full-data numbers are always reported next to trusted
ones — never replace them silently. The exclusion was checked against three guards (random removal
of the same row count, train-trusted/test-wave, and the openly circular "delete contradictory rows"
ceiling); see the report before changing it.

Any WEKA number quoted anywhere must come from a log in `docs/weka_runs/`.

**Final team decision (supersedes the 647-row trusted set for the presentation):**

- Start from 1,102 rows (1,108 minus 6 without a real recent SGPA).
- Of the 455 wave rows, keep those whose recent SGPA band and CGPA band are ≤ 1 apart (273). Drop the
  182 that are 2–3 apart. That leaves **920 rows** (`cleaned-dataset/ours/final/final_920.arff`).
- `result_satisfaction` is **not a feature**. It is subjective.
- Evaluation is **5-fold CV, seed 1**. A WEKA 80/20 percentage split (seed 1) sits on hidden slides only.
- Headline, main CGPA: SMO 55.22%, J48 (`-M 40`) 54.35%, ZeroR 30.87%.
- Known caveat: BUET's 37 big-gap rows are kept, because the rule only targets the wave.

`docs/FINAL_WEKA_STEPS.md` is the reproduction guide.

---

## 5. Preprocessing decisions (each needs a *why* for the viva)

Full narrative in `docs/MERGED_DATASET_REPORT.md`; the notebook states each as Decision → Evidence
→ Why. The merge-specific ones are the ones a panel will probe.

| Decision | Why |
|----------|-----|
| **Hand-map both column sets onto one schema** instead of fuzzy matching | A threshold loose enough to match "admitted into KUET" with "admitted into your university" also merges the two *different* satisfaction questions. 21 columns, mapped by hand, `assert`-guarded. |
| **Strip the Bangla parenthesis + emoji from every answer value** | Without it `90% or more` and `90% or more (৯০%…)` are two categories, the merge is cosmetic, and every model learns *which form a row came from*. Match only brackets containing Bangla, so `(EEE)` survives for the department canonicaliser. |
| **Print the option-set overlap of all 17 shared columns before concatenating** | Caught the three real mismatches: one `study_style` label, the `None` option, the GPA bands. A bad merge otherwise surfaces later as a mystery accuracy drop. |
| **Keep `source_form` for auditing, exclude it from every model** | Every EDA claim is re-checked within source; a model given the column would use it to route around the wording differences instead of learning about students. |
| **110 department strings → 24 canonical, then group under 15 → 18** | `ChE` and `Chemical Engineering` are one department. Threshold moved from 5 to 15 because the sample is 5× larger; the fixed principle is "a level must survive a 10-fold split". `"Mecha"` (n=1) is left `Unknown` rather than guessed between ME and MTE. |
| **2 band schemes + 35 free-text numbers → one 4-class target** | KUET never offered the 6-band split, so 4 classes is what is recoverable from both. It also comes out balanced (21–29%), so accuracy stays honest. |
| **Drop 18 rows with no usable CGPA** | The only rows discarded anywhere. A supervised model cannot learn from an unlabelled row, and imputing the target would invent the answer. |
| Missing "responsibilities hours" → `None`, **not** dropped | **Now evidence, not assumption.** KUET's form offered `None` and 26.5% pressed it; Source A had no zero option and 19.1% left it blank. Same behaviour. Blankness is independent of GPA (χ² p = 0.688). |
| Remaining <1.3% blanks → **within-source median category** | Ordinal data: the mean of `Rarely` and `Often` is not a category. Within-source because the samples differ on some items. Verified: complete-case correlations shift by at most **0.0036**. |
| Likert → **ordinal integer codes**, not one-hot | The categories are ordered. One-hot destroys the order and turns 15 columns into ~72. `university`/`department` *are* nominal and do get one-hot — inside the model pipeline only. |
| `semester` treated as **not comparable across universities** | BRAC runs trimesters (all at 10–11), CUET all at 7, **676 of 896 KUET rows at 7**. The raw number encodes *institution*. Replaced by `academic_progress` = semester ÷ programme length. |
| Drop `result_satisfaction` | An outcome, not a behaviour — a student cannot rate a result they have not received. Note it measures ρ = 0.118 against the *cumulative* target, far below the 0.601 it hit against notebook 01's *semester* target; changing the target largely defused the leak. Report that honestly rather than overstating it. |
| Repeated stratified 10-fold CV, never a single holdout | A single holdout swings several accuracy points on the seed alone; 30 fits per model make the ranking stable. |
| **Benjamini–Hochberg, taken in the right direction** | Our first implementation ran `cummin` forward from the smallest p, collapsing every q onto the minimum and reporting 15/15 significant. Fixed in both notebooks; corrected count is **3 of 15**. Expect to be asked about this — it is in the report on purpose. |

---

## 6. Environment

```powershell
.\venv\Scripts\Activate.ps1        # Python 3.14
```

Installed: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, jupyter, markitdown, pillow,
playwright.

Edit the `# %%` **source**, regenerate the `.ipynb`, then execute it — never hand-edit a notebook's
JSON:

```powershell
.\venv\Scripts\python.exe notebooks\py2nb.py notebooks\pipeline_v2_source.py notebooks\02_merged_cleaning_eda_modeling.ipynb
.\venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 notebooks\02_merged_cleaning_eda_modeling.ipynb
```

Same two commands with `pipeline_source.py` / `01_cleaning_eda_modeling.ipynb` for the baseline
notebook. Executing either regenerates its figures, its `cleaned-dataset/` exports and its `docs/`
result files. Set `PYTHONIOENCODING=utf-8` if printing a Bangla value raises `UnicodeEncodeError`.

nbconvert prints a wall of joblib `resource_tracker` `KeyError` tracebacks on Windows **after** a
successful run. They are shutdown noise, not cell failures. Check for real errors by counting
error outputs in the written notebook rather than by reading the console tail.

### WEKA from the command line (used by notebook 03)

- WEKA 3.8.7 is installed at `C:/Program Files/Weka-3-8-7`. Java is **not on PATH**; WEKA's bundled
  runtime is `jre/jre-25.0.2-full/bin/java.exe`. `notebooks/weka_runner.py` finds it.
- Java 25 needs `--add-opens java.base/java.lang=ALL-UNNAMED`, otherwise every run prints
  reflection warnings (harmless, but noisy). The driver adds it.
- The package manager **cannot download packages** (the repository returns HTTP 403), so
  `ordinalClassClassifier` is unavailable. Use the built-in `meta.CostSensitiveClassifier` for
  ordinal-aware costs.
- Asking for per-row predictions (`-classifications ...CSV`) makes WEKA **omit** the statistics and
  confusion matrix; the driver rebuilds the matrix from the printed predictions.
- `weka_runner.run()` caches by exact command: if `docs/weka_runs/<name>.txt` exists and its command
  line matches, the log is reused. Delete the log to force a fresh WEKA run. A first full execution
  of notebook 03 launches ~1,250 WEKA JVMs and takes 10-20 minutes.
- Nested WEKA options (MultiFilter inside FilteredClassifier, IBk distance settings) are passed as
  one list element each; do not add shell quoting around them.

---

## 7. Conventions for AI agents working in this repo

**Reading PDFs — convert to Markdown first.** Do not read a PDF page-by-page as images; it burns
tokens. Use markitdown (installed in the venv):

```powershell
.\venv\Scripts\python.exe -m markitdown docs\SomeFile.pdf > docs\SomeFile.md
```

Then read the `.md`. Keep the `.md` beside the PDF so the conversion happens only once. If a
`markitdown` **skill** is installed later, prefer the skill; the CLI above is the fallback and is
what currently works in this project.

**Generating slides.** Use the **`frontend-slides`** skill (installed from the
`zarazhangrui/frontend-slides` marketplace). It produces single-file HTML decks on a fixed
1920×1080 stage — not `.pptx`. Three decks already exist in `presentation/`; see §9.

Slides are **generated from data, never hand-written.** All copy and numbers live in
`presentation/build/pres_content.py`, which is transcribed from
`docs/results_summary_merged.json`, `docs/model_results_merged_*.csv`,
`docs/feature_importance_merged.csv` and the notebook. If the notebook is re-run and a number
changes, update `pres_content.py` and rebuild — never edit a generated `.html` by hand.

**Explaining anything to the team.** Use the **`noob-explain`** skill for reports, docs and
walkthrough guides. Everything in `docs/` that a teammate reads should be diagram-first and in
plain English — black box first, then modules. Teammates are being viva'd on this; a doc they
cannot explain out loud is a failed doc.

**Non-negotiables.**
- `raw-data/` is read-only. Every cleaning step lives in the notebook, never in a manual CSV edit.
- No result gets reported that a teammate cannot explain the *why* of.
- Report weak or negative findings honestly. A behaviour-only model that barely beats ZeroR is a
  real finding about self-reported survey data, not a failure to be hidden.
- No causal claims. This is cross-sectional self-reported data — "associated with", never "causes".

---

## 8. Presentation decks

Three complete decks of the same **17 slides**, in three visual styles, for the team to pick from.
Open any of them in a browser — no build step, no internet needed for anything but webfonts.

Slide order: cover · brief · related work · **merging the two surveys** · dataset ·
**raw → clean** · preprocessing · preprocessing evidence · **EDA** · feature ranking ·
**replication test** · models & protocol · results tables · **reading the comparison honestly** ·
error analysis · **verdict** · conclusions.

| File | Style |
|------|-------|
| `presentation/deck-signal.html` | Navy + cream + antique gold, editorial serif |
| `presentation/deck-cobalt.html` | Cream graph-paper, electric cobalt, Newsreader serif |
| `presentation/deck-swiss.html` | White / black / red, Archivo Black, visible grid |

**Controls.** Arrow keys, Space, PageUp/Down, Home/End; mouse wheel; swipe on touch. `E` toggles
inline editing — click any text, edit it, `Ctrl+S` saves to that browser's localStorage. Each
deck deep-links by slide (`#slide-7`).

**Rebuilding.**

```powershell
cd presentation\build
..\..\venv\Scripts\python.exe build.py all     # regenerate all three
..\..\venv\Scripts\python.exe verify.py all    # check every slide fits the 1920x1080 stage
```

`verify.py` renders all 51 slides in headless Chromium and fails on content overflow, escaping
elements, overlapping panels, text under 14px, decorations covering text, and a stage that stops
being 16:9 on a phone. Run it after any content or theme change.

Source layout: `pres_content.py` (all copy and numbers), `slides_html.py` (shared semantic
markup), `theme_*.py` (one stylesheet per style), `deck_shell.py` (fixed-stage CSS + controller),
`build.py`, `verify.py`. Screenshots land in `build/shots/<deck>/`.

> The work-distribution table on slide 17 is a **placeholder** — confirm who did what and edit it
> (press `E` in the deck, or change `pres_content.py` and rebuild).

Figures come from `docs/figures/merged/`. Adding content to an existing slide is the main way to
break a deck: if `verify.py` reports an overflow, **cut copy or split the slide** — never shrink
the type. Several slide `kind`s are reused for more than one slide (`evidence` twice, `integrity`
twice, `counter` twice); pass `foot=` and `kicker_cls=` in `pres_content.py` to relabel them.

---

## 9. Team

| Roll | Name |
|------|------|
| 2107020 | Sheikh Md. Galib Mahim |
| 2107030 | Md. Eftakar Jaman Arfan |
| 2107007 | Asif Jawad |
| 2107008 | Rakibul Islam |
| 2107012 | Md Enam E Elahi |
