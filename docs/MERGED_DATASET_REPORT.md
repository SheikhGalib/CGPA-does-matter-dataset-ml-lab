# Merged Student Performance Dataset — Final Report

**CSE-4112 Machine Learning Laboratory**
Team: 2107020 Sheikh Md. Galib Mahim · 2107030 Md. Eftakar Jaman Arfan · 2107007 Asif Jawad ·
2107008 Rakibul Islam · 2107012 Md Enam E Elahi

Reproduced end-to-end by [`notebooks/02_merged_cleaning_eda_modeling.ipynb`](../notebooks/02_merged_cleaning_eda_modeling.ipynb).
Every number in this report is read from that notebook's outputs
(`docs/results_summary_merged.json`, `docs/model_results_merged_*.csv`,
`docs/feature_importance_merged.csv`, `docs/replication_n230_vs_n1108.csv`).

---

## 0. Executive summary

We merged two independently collected Google Form surveys of Bangladeshi engineering students —
230 responses from BUET / BRAC / CUET and 896 from KUET — into **one cleaned dataset of 1,108
students, 25 features and a 4-class CGPA target**, then re-ran the full cleaning → EDA → feature
analysis → modelling pipeline on it.

| Question | Answer |
|---|---|
| Can the two forms be merged? | **Yes.** 19 questions are the same question; 14 of 16 items differ between the two samples by a *negligible or small* effect size (Cliff's δ). |
| Do behavioural self-reports predict CGPA? | **Barely.** Best behaviour-only model: **31.2%** accuracy vs a **29.0%** ZeroR floor, κ = 0.054. |
| Does prior academic performance help? | **Yes, decisively.** Adding last semester's SGPA takes accuracy to **43.4%** and κ to **0.237** — a **+12.2 point** lift. |
| Are *any* behavioural factors reliably associated with CGPA? | **Yes — 3 of 15**, after Benjamini–Hochberg correction: getting your desired department, class attendance, and outside-commitment load. All are small effects. |
| Did more data change the story? | **It replaced it.** At n = 230 nothing survived correction and two "counterintuitive" findings looked real. At n = 1,108, three real effects appear and **both counterintuitive findings vanish**. |

**Headline for the viva:** the value of this work is not a high accuracy number. It is that a
five-fold increase in sample size turned a null result into three small but reliable findings, and
simultaneously falsified two apparent findings from the smaller sample. That is the whole argument
for merging the datasets.

---

## 1. The two data sources

| | Source A — multi-university form | Source B — KUET form |
|---|---|---|
| File | `raw-data/University Student Performance Analysis Form (Responses)...csv` | `raw-data/Student Performance Analysis Form (Responses)...csv` |
| Responses | 230 | 896 |
| Universities | BUET (104), BRAC (101), CUET (25) | KUET (896) |
| Columns | 22 | 21 |
| Language | Bilingual header, **English answer values** | Bilingual header, **bilingual answer values** |
| Collection window | 2026-08-22 → 2026-09-03 | 2026-07-31 → 2026-08-13 |

Both forms ask the **same 19 substantive questions**: one banded SGPA, one banded CGPA, semester,
department, and 15 five-point Likert items covering attendance, study time, study style, lecture
clarity, sleep, stress, outside commitments, digital distraction, living environment, family
support, routine manageability, admission satisfaction, desired-department match, career
expectation and result satisfaction.

### 1.1 A correction to the record

`AGENTS.md` §4 previously recorded that the teammate-supplied files
`cleaned-dataset/student_performance_*_clean.csv` (878 rows, `kuet_satisfaction` column, KUET-only
department list) were "a different, most likely synthetic, dataset". **That conclusion was wrong,
and this merge is what disproved it.** Those files are a cleaned export of an *earlier* pull of the
KUET form — 878 rows then, 896 now — which the team simply did not have access to at the time.
Their per-option counts match the KUET raw data to within the 18 responses that arrived later.

The methodological point stands: we still rebuild every row from `raw-data/` rather than trusting a
derived file. But the file was real data, and the report says so.

---

## 2. How the two datasets were merged

Four passes, in order. Each is a cell in §3 of the notebook.

### 2.1 Column identity — "the same question written differently"

The two forms ask the same things in a different order and with different English. Examples:

| Meaning | Source A wording | Source B wording |
|---|---|---|
| last semester result | "What **was** your SGPA **in** the most recently completed semester?" | "What **is** your SGPA **of** the most recently completed semester?" |
| admission satisfaction | "…admitted into **your university**?" | "…admitted into **KUET**?" |
| stress | "**How often did** stress make it difficult for you to study" | "Stress made it difficult for you to study" |

**Decision:** hand-map each raw column index onto one `snake_case` schema, and `assert` that the map
covers every column of both files.

**Why not automatic fuzzy matching:** a similarity threshold loose enough to match
"admitted into KUET" with "admitted into your university" is also loose enough to merge the two
*different* satisfaction questions (admission vs result) with each other. With 21 columns, a
hand-written map that fails loudly is safer and auditable by a reader.

**Structural asymmetries and how each was resolved:**

| Column | Present in | Decision | Why |
|---|---|---|---|
| `university` | A only | fill `"KUET"` on every B row | The KUET form did not ask because the sampling frame *is* the university. A known constant, not an imputation. |
| `department_other` (the "if not listed" box) | A only | `NaN` on every B row | B used a single free-text box; the same `fillna` coalesce then handles both shapes. |
| `email` | B only | **drop** | 896 / 896 empty. Zero information, and it would break the "no missing values" export gate. |
| `timestamp` | both | **drop** | Records how the link spread, not a property of the student — and because the two forms ran in different windows it is a perfect proxy for which form a row came from. |

### 2.2 Value text — the single most important line in the merge

The KUET form stores answers **bilingually**:

```
"😠 Very dissatisfied (খুবই অসন্তুষ্ট)"      →  "Very dissatisfied"
"90% or more (৯০% বা তার বেশি)"             →  "90% or more"
"None (কোনোটিই না)"                          →  "None"
```

**Decision:** strip any parenthesis containing a Bangla codepoint (U+0980–U+09FF), and any emoji,
from every categorical answer.

**Why this matters more than it looks:** without it, `"90% or more"` and
`"90% or more (৯০% বা তার বেশি)"` are two unrelated categories. The 1,126-row merge would silently
become two disjoint blocks that share *no feature values at all*, and every model would be learning
"which form did this row come from" rather than anything about students.

The regex deliberately matches **only Bangla-containing** parentheses, so an English clarification
such as `"Electrical and Electronic Engineering (EEE)"` survives for the department canonicaliser
to handle properly instead of being blindly deleted.

### 2.3 Option-set reconciliation

Before concatenating we printed the value overlap of every shared categorical column. Three real
mismatches were found; everything else already agreed exactly.

| # | Mismatch | Resolution |
|---|---|---|
| 1 | `study_style`: A says `Mostly consistent routine`, B says `Consistent routine most of the time` | Renamed to one canonical label. Unmerged, a 5-point ordered scale silently becomes a 6-point one in which KUET and non-KUET students can never share a value. |
| 2 | `weekly_responsibilities`: only B offered a `None` option | See §3.3 — this turned out to be the most useful thing the merge gave us. |
| 3 | GPA bands: A used 6 bands, B used 4 **and** allowed free text | See §3.2. |

One apparent mismatch was **not** a mismatch: `attendance` has 5 options on both forms, but only 3
were ever *chosen* by the 230 Source-A respondents. That is a sampling fact. The merged data now
exercises all 5 levels, so the scale is genuinely 5-point for the first time.

### 2.4 Concatenate, keeping provenance

Rows are stacked with a `source_form` column retained for auditing. It is **excluded from every
model** — a model given it would use it to route around the wording differences documented in
§4.3 — but every EDA claim below is re-checked within each source before being reported.

**Result: 1,126 rows × 22 columns.**

---

## 3. Cleaning and preprocessing

Ten logged steps. Full audit trail in `docs/cleaning_step_log.csv`.

| Step | Decision | Why | Rows after |
|---|---|---|---|
| 5.1 | Drop `timestamp` | sampling artefact; proxy for `source_form` | 1,126 |
| 5.2 | `university`: 10 spellings → 4 | `BRAC / Brac / BRAc / BRAC University` are one institution | 1,126 |
| 5.3 | Department: 2 columns → 1, **110 raw strings → 24** canonical | see §3.1 | 1,126 |
| 5.4 | Reconcile 1 mismatched Likert label | see §2.3 | 1,126 |
| 5.5 | 2 band schemes + free text → **one 4-class target** | see §3.2 | 1,126 |
| 5.6 | **Drop 18 rows** with no usable CGPA answer | cannot train on an unlabelled row | **1,108** |
| 5.7 | Impute 45 blank `weekly_responsibilities` as `None` | see §3.3 | 1,108 |
| 5.8 | Impute remaining < 1.3% scattered blanks (within-source median category) | see §3.4 | 1,108 |
| 5.9 | Ordinal-encode 15 Likert scales + both GPA bands | see §3.5 | 1,108 |
| 5.11 | Derive `academic_progress`, 4 indices, 3 interactions | see §3.6 | 1,108 |

**18 rows dropped out of 1,126 (1.6%). No other row is discarded anywhere in the pipeline.**

### 3.1 Department normalisation — the biggest free-text job

`department` is the only genuinely free-text field, and the KUET form had **no dropdown at all**.
Across both columns of both files there were **110 distinct strings** for **24 real departments**.
Four kinds of variation, each with its own rule, applied in order:

| Kind | Examples | Rule |
|---|---|---|
| Case | `CSE` / `cse` / `Cse` | upper-case |
| Whitespace | `"CSE "` / `" Leather Engineering"` | strip, collapse runs |
| Prefix / suffix noise | `Department of Textile Engineering`, `ELECTRICAL & ELECTRONIC ENGINEERING(EEE)` | drop leading `DEPARTMENT OF`, drop a trailing bracketed abbreviation, `&` → `AND` |
| Abbreviation vs full name, and typos | `ME` / `Mechanical Engineering` / `Mechanical`; `Computer Science and Egineering` | explicit lookup table, then a keyword fallback |

**Why it matters:** `ChE` and `Chemical Engineering` are the same department. Unmapped, one category
splits into two — each too small to learn from, and each contributing a spurious level to the
one-hot encoding.

**What we deliberately did *not* guess.** The canonicaliser prints every string its exact lookup
missed. Exactly one string reached the fallback and could not be resolved: `"Mecha"` (n = 1), which
is ambiguous between KUET's *Mechanical* (ME) and *Mechatronics* (MTE) programmes. It is labelled
`Unknown` rather than guessed. Together with 17 respondents who left both boxes blank, that gives 18
`Unknown` rows before step 5.6 and **16 in the final dataset** — rows whose other ~19 answers remain
perfectly usable, so discarding them would cost more than one extra category does.

**Rare-department grouping.** Seven departments have fewer than 15 respondents (MME, NCE, PMRE,
IPE, NAME, MIE, WRE — all from the smaller Source-A sample). They are pooled into `Other` for
modelling, leaving **18 categories**, while the full 24-value `department` is kept for reporting. A
category with 4 members cannot be *learned*, only memorised: a tree that splits on it has fit four
students. The threshold moved from 5 (notebook `01`) to 15 because the sample is ~5× larger; the
principle — a level must survive a 10-fold split — is what is fixed, not the number.

### 3.2 Building the target from two different band schemes

| | Source A | Source B |
|---|---|---|
| `recent_sgpa_raw` | last completed semester's SGPA | last completed semester's SGPA |
| `cgpa_raw` | "**Before that semester**, what was your CGPA?" | "What is your **current** CGPA?" |
| Options | `Below`, `3.0+`, `3.2+`, `3.5+`, `3.75+`, `3.9+` | `Below`, `3.20 – 3.49`, `3.50 – 3.74`, `3.75+`, **or type a number** |

**Decision:** the target is `cgpa_band`, from the **CGPA** question on both forms. The SGPA question
becomes the prior-performance feature `recent_sgpa_band`.

**Why the CGPA question:** it is the cumulative grade the project set out to predict, and it is the
same quantity on both forms. A single semester's SGPA is noisier, and it is an input the student
already knows about themselves.

**The honest caveat, stated up front:** the *reference point* differs by one semester. KUET's CGPA
includes the most recent semester; Source A's is measured just before it. For a 7th-semester
student that is one semester out of seven — the same variable to within a small drift — but it
means `recent_sgpa_band` is *partially contained in* the target for KUET rows. §5.3 of this report
measures exactly how much that inflates the primary result. (Answer: it does not.)

**Collapsing to 4 classes** is forced by the data, not chosen for convenience — KUET never offered
the finer split, so 6 classes are unrecoverable for 80% of the sample:

| Class | Range | Source A bands | Source B band | n |
|---|---|---|---|---|
| `C0_below_3_20` | below 3.20 | `Below` + `3.0+` | `Below` | 236 (21.3%) |
| `C1_3_20_to_3_49` | 3.20 – 3.49 | `3.2+` | `3.20 – 3.49` | 321 (29.0%) |
| `C2_3_50_to_3_74` | 3.50 – 3.74 | `3.5+` | `3.50 – 3.74` | 303 (27.3%) |
| `C3_3_75_plus` | 3.75 + | `3.75+` + `3.9+` | `3.75+` | 248 (22.4%) |

Imbalance ratio 1.36 — near balanced. Two consequences: no resampling is needed, and **accuracy is
an honest metric**. The ZeroR floor every model must beat is **29.0%**.

**Free text.** 35 KUET respondents typed a number (`3.71`) rather than picking a band; these are
parsed and binned by the same cut-points. Two typed *"jani na"* / *"etao jani na"* ("I don't know") —
these become `NaN` and their rows are dropped in step 5.6.

### 3.3 `weekly_responsibilities` — an assumption that the merge turned into evidence

Notebook `01` had to **assume** that the 44 blanks meant "I have no such commitments", because
Source A's form offered no zero option. It could only defend that with a chi-square test.

The KUET form settles it, because it *did* offer `None` to a comparable population answering the
same question:

| | rate |
|---|---|
| KUET students who **actively chose** `None` (option existed) | **26.5%** |
| Source-A students who **left it blank** (no zero option existed) | **19.1%** |
| KUET students who left it blank anyway (a genuine skip) | 0.3% |

Those are the same behaviour expressed two ways, and the gap is what you would expect from a few
genuine skips. Blankness is also independent of GPA band (χ² = 3.08, dof = 5, **p = 0.688**), so it
is not a response-quality artefact.

**Decision:** fill the blanks with `None`. **Rejected alternative:** dropping them would delete 19%
of the smaller sample *and* bias it toward students who do hold outside commitments — precisely the
wrong direction for a question about how commitments affect grades.

This is the clearest single argument for having merged the datasets: an assumption became a
measurement.

### 3.4 The remaining scattered blanks

Every other column is missing **under 1.3%** of its values, scattered across different rows.

- **Likert items** → imputed with that column's **median category, computed within its own source
  form**. Median, not mean, because these are ordinal codes — the mean of `Rarely` and `Often` is
  not a category. Within-source, because the two samples answer some items differently (§4.3) and
  the pooled median would drag a KUET blank toward Source A's distribution.
- **`recent_sgpa_band`** (24 blanks) → median band *within the student's own CGPA class*; a
  semester result is tightly tied to the cumulative grade, so this is a far better guess than the
  global median.
- **`semester_raw`** (2 blanks) → median semester within the same university, since semester is
  essentially a function of institution (§4.4).

**Verified, not asserted:** re-running every Spearman correlation on complete cases only moves no
coefficient by more than **0.0036**. The imputation cannot have manufactured any finding in §5.

### 3.5 Ordinal encoding, not one-hot

**Decision:** map each ordered scale to `0..k`, low → high.

**Why:** these categories have a real order — `Never < Rarely < Sometimes < Often < Almost always`.
One-hot throws that order away, turns 15 columns into ~72, and makes rank statistics impossible.
Ordinal codes keep the order *and* the dimensionality.

`university` and `department_grouped` genuinely *are* nominal — there is no sense in which CSE is
"more" than EEE — so those **do** get one-hot encoded, but only inside the model pipeline, never in
the exported ordinal matrix.

Two orderings are judgement calls and are flagged as such in the notebook: `study_style` (we rank
`Occasionally when necessary` *below* `Mostly just before exams`, since the latter at least implies
a deliberate if late routine) and `sleep_duration` (encoded by duration, not by "goodness" — §4.2
then confirms the relationship really is monotonic).

A coverage `assert` guards the encoding: if either form ever produces a value not in the explicit
ordering, the pipeline stops rather than silently writing `NaN`.

### 3.6 Hidden / derived features

Two different kinds, built on hypotheses and then **tested** rather than assumed.

**(a) Composite indices** — do several items measure one latent trait?

| Index | Built from | Cronbach's α |
|---|---|---|
| `discipline_index` | attendance, study time, study style, (rev.) distraction | **0.273** |
| `motivation_index` | admission satisfaction, desired-dept match, career expectation | **0.256** |
| `wellbeing_index` | sleep, (rev.) stress, routine manageability | **0.152** |
| `environment_index` | study environment, family/friend support | **0.054** |

**Verdict: they fail, again.** α ≥ 0.7 is strong, 0.5–0.7 acceptable for a short exploratory index,
< 0.5 means the items are not measuring one thing. Notebook `01` found α ≈ 0 at n = 230 and had to
report it as a possible small-sample artefact. **Five times the data did not rescue them**, so it
is not an artefact: there is no single latent trait called "discipline" in these answers. A student
with high attendance is barely more likely than chance to also study many hours.

**(b) Interaction features** — is the *combination* of two answers worth more than the two apart?

| Feature | Definition | Hypothesis |
|---|---|---|
| `effective_study` | study time × topic clarity | An hour you understood is worth more than an hour you didn't. |
| `focus_balance` | study time − distraction | What matters is the *net*, not either alone. |
| `load_pressure` | responsibilities + stress − wellbeing | Commitments hurt most when a student is already stretched. |

**Result: 2 of the 7 derived features beat their own best ingredient**, and adding all seven to the
model changes mean accuracy by **−0.011** (only 1 of 9 models improved). They are kept in the
exported files because they cost nothing and a WEKA user may want them, but **no conclusion in this
report rests on them**, and the modelling section reports the raw items.

Reporting an index built from uncorrelated items as if it measured something is exactly the kind of
unjustified step this pipeline exists to avoid.

### 3.7 Leakage — `result_satisfaction` is removed

*"How satisfied were you with your latest semester result?"* is a **reaction to an academic
outcome**, not a behaviour. A student cannot report satisfaction with a result they have not
received, so a deployed model would never hold this column at prediction time.

**Decision:** excluded from every predictive model.

**But we report the measured number honestly.** Against our target it reaches only ρ = **0.118** —
far weaker than the ρ = 0.601 notebook `01` measured — and the reason is instructive rather than
reassuring. Notebook `01`'s target was *that same semester's SGPA*, which this question directly
restates. Our target is the **cumulative** CGPA, which one semester only partly determines.
Changing the target largely defused the leak. We still exclude the column — an outcome variable
dressed as a feature is a leak waiting for whoever next changes the target back — but we do not
claim it would have wrecked the model, because on these numbers it would not have.

**We keep it for exactly one purpose:** it is an excellent data-quality check. Satisfaction rises
monotonically across the bands **in both samples independently**, which is evidence that
respondents answered the GPA question and the satisfaction question consistently — i.e. that the
self-reported grades are not random button-pressing. On the merged data it does this job better
than it could at n = 230, because it now cross-validates two separately collected samples.

---

## 4. Exploratory data analysis — what it showed

### 4.1 No degenerate feature

The most concentrated item's modal option is well under our 60% threshold, and **zero features
exceed it**. Every scale uses all of its levels. Nothing is dropped for being uninformative.

### 4.2 Feature-vs-target relationships are flat but not empty

Box plots of every item against the band are mostly level — no single Likert item separates the
bands dramatically. But several now show a clean monotone trend that was invisible at n = 230, and
`sleep_duration` is confirmed **monotonic** rather than inverted-U, which retroactively justifies
encoding it by duration.

### 4.3 Does the merge hold up? — Cliff's δ between the two samples

The question that decides whether this dataset is legitimate. For every item we computed Cliff's δ,
a non-parametric effect size appropriate for ordinal data (|δ| < 0.15 negligible, < 0.33 small,
< 0.47 medium).

| Magnitude | Count | Features |
|---|---|---|
| medium | 1 | `admission_satisfaction` (δ = −0.340) |
| small | 3 | `result_satisfaction` (−0.253), `topic_clarity` (−0.204), `academic_progress` (−0.156) |
| negligible | 12 | everything else, including study time, sleep, stress, environment, support, distraction, study style, desired-department match |

**Conclusion: the merge is sound, with one named exception.** KUET students and BUET/BRAC/CUET
students report their study habits, sleep, stress and environment at close to the same rates. That
is the empirical licence to pool them.

The exception is `admission_satisfaction`, where the KUET sample is markedly more negative
(mean code 1.57 vs 2.48). Two readings are possible and this data cannot separate them:

1. a **real** difference in how satisfied the two populations are, or
2. a **response artefact** — the KUET form's `Very dissatisfied` option drew 40% of answers while
   `Dissatisfied` drew 5%, a shape that suggests first-option selection by respondents clicking
   through quickly.

**Consequence, applied throughout:** the feature is kept (dropping a variable because it is
inconvenient is worse than flagging it), but any finding resting on it is re-checked within source
before being reported — and this is exactly why `source_form` is excluded from the model matrix.

### 4.4 The university / semester confound got worse, not better

- BRAC runs **trimesters**, so every BRAC respondent reports semester 10 or 11.
- All 25 CUET respondents sit at semester 7.
- **676 of 896 KUET respondents sit at semester 7** — the form went mainly through one batch.
- Only BUET spreads across 1–12.

"Semester 11" therefore does not mean *further along* than "semester 7" — it means *BRAC*. Feeding
the raw number to a model lets it recover the institution and learn per-university grading habits
while appearing to learn about academic progress.

**Fix:** `academic_progress` = semester ÷ that university's programme length, clipped to 0–1.5.
Raw `semester_raw` never leaves the notebook.

Band mix does differ by institution (χ² = 34.6, p = 7×10⁻⁵) but the association is weak
(**Cramér's V = 0.102**). `university` is kept as a nominal feature — it is a real structural
variable a deployed model would have.

### 4.5 No multicollinearity

Zero feature pairs exceed |r| = 0.7. The strongest pair is `recent_sgpa` with the target
(ρ = 0.327), which is the point of the primary experiment, not a defect. No feature needs dropping
and we are not forced into a regularised-only model set.

---

## 5. Pipeline A — which features actually matter

Four independent methods, ranked by consensus, because no single importance measure is trustworthy
on survey data: correlation misses non-linear effects, mutual information is unstable, and a Random
Forest's built-in Gini importance is biased toward features with many distinct values.

| Method | What it adds |
|---|---|
| Spearman ρ + Benjamini–Hochberg | monotone rank association, corrected for 15 simultaneous tests |
| Mutual information (10 seeds) | catches non-monotone structure |
| Kruskal–Wallis H + BH | distribution differs across bands, no linearity assumed |
| RF **permutation** importance (out-of-fold) | real accuracy drop when a column is shuffled |

Permutation importance is used rather than Gini precisely because Gini is measured on *training*
data and would rank a 5-level scale above a 3-level one for reasons unrelated to predictiveness.

### 5.1 Result — 3 of 15 features survive correction

| Rank | Feature | ρ | q (BH) | Direction |
|---|---|---|---|---|
| 1 | `desired_department_match` | **+0.119** | **0.0010** | got the department you wanted → higher band |
| 3 | `weekly_responsibilities` | **−0.086** | **0.0202** | more tuition/job/club hours → lower band |
| 5 | `attendance` | **+0.090** | **0.0197** | higher attendance → higher band |
| — | `weekly_study_time` | +0.070 | 0.078 | borderline, does not survive |
| — | everything else | \|ρ\| < 0.06 | > 0.15 | not significant |

**Both facts must be reported together: these effects are real, and they are small.** The largest
|ρ| in the whole behavioural set is 0.119 — roughly 1.4% of the variance in band. They survive
correction only because n = 1,108 makes |ρ| ≈ 0.059 detectable, where n = 230 needed |ρ| ≈ 0.130.

Note the one genuinely actionable finding: **the strongest single behavioural correlate of CGPA is
not a habit at all** — it is whether the student got into the subject they wanted. Motivation at
admission outranks attendance and study hours.

### 5.2 The replication check — what the merge overturned

Running the identical test on Source A alone (n = 230), Source B alone (n = 896), and merged:

| Feature | ρ, A only (n=230) | ρ, B only (n=896) | ρ merged | Verdict |
|---|---|---|---|---|
| `desired_department_match` | +0.144 (p=0.029) | +0.113 (p=0.0008) | **+0.119** | **replicates** — significant in both samples independently |
| `weekly_responsibilities` | −0.080 (p=0.229) | −0.089 (p=0.009) | **−0.086** | **replicates** — same sign, same size, now detectable |
| `attendance` | −0.054 (p=0.419) | +0.123 (p=0.0003) | **+0.090** | strong in B; A's sign was noise at n=230 |
| `topic_clarity` | **−0.161 (p=0.014)** | +0.053 (p=0.120) | +0.039 | **does NOT replicate** |
| `stress_frequency` | **+0.119 (p=0.639)** | −0.016 (p=0.646) | −0.005 | **does NOT replicate** |

Only **6 of 15** features even agree in sign across the two independent samples — which is close to
what you would expect by chance for effects this small, and is itself a caution about reading
anything into an unreplicated survey correlation.

**The two headline "counterintuitive findings" of notebook `01` did not survive.** At n = 230 it
looked like students who understood lectures *better* got *worse* grades (ρ = −0.161, the largest
effect in that dataset), and that more stress went with better grades. Neither appears in the
896-student KUET sample, and neither survives in the merged data. They were small-sample noise, and
we retract them.

This is the most important methodological result in the project, and it is exactly what merging the
datasets was for.

---

## 6. Pipeline B — CGPA prediction

### 6.1 Protocol, and why

| Choice | Why |
|---|---|
| **ZeroR** (majority class) floor | An accuracy number means nothing without it. |
| **OneR-like** (depth-1 tree) second floor | Shows how much signal is in a *single* feature; if the ensemble barely beats it, the other 14 are decoration. |
| **Repeated stratified 10-fold CV**, 3 repeats (30 fits/model) | A single holdout swings by several points on the seed alone. Stratified so every fold keeps the 4-class balance. |
| **Accuracy** | Fair here *only* because the classes are balanced (21–29%). |
| **Macro-F1** — what we rank by | Averages the four classes equally; a weak class cannot hide behind a strong one. |
| **Cohen's κ** | Corrects for chance agreement. The metric that exposes a model that is really ZeroR wearing a hat. |
| **Within-1-band accuracy** | The target is *ordinal* — predicting C2 when the truth is C3 is a smaller error than predicting C0. |

Nine models, one per family, so the comparison says something about *model class* rather than
hyper-parameter luck.

### 6.2 Experiment 2 — behaviour-focused (no prior SGPA)

| Model | Accuracy | Macro-F1 | κ | Within-1 |
|---|---|---|---|---|
| **SVM (RBF)** | **31.2% ± 4.3** | **0.284** | **0.054** | 71.5% |
| Random Forest | 31.0% ± 4.2 | 0.279 | 0.050 | 72.7% |
| Logistic Regression | 29.1% ± 3.5 | 0.272 | 0.025 | 74.4% |
| Decision Tree (J48-like) | 26.6% ± 3.8 | 0.264 | 0.026 | 69.8% |
| Gradient Boosting | 28.0% ± 3.9 | 0.258 | 0.012 | 70.0% |
| k-NN (k=25) | 29.9% ± 3.6 | 0.242 | 0.035 | 74.4% |
| Naive Bayes | 23.8% ± 3.4 | 0.201 | 0.033 | 57.6% |
| OneR-like | 28.2% ± 2.8 | 0.180 | 0.027 | 72.2% |
| **ZeroR (baseline)** | 29.0% ± 0.3 | 0.112 | 0.000 | 77.6% |

### 6.3 Experiment 1 — primary (with last semester's SGPA)

| Model | Accuracy | Macro-F1 | κ | Within-1 |
|---|---|---|---|---|
| **Gradient Boosting** | **43.4% ± 4.1** | **0.432** | **0.237** | 78.1% |
| Random Forest | 43.6% ± 4.4 | 0.426 | 0.230 | 79.4% |
| Decision Tree (J48-like) | 43.3% ± 4.5 | 0.427 | 0.234 | 78.2% |
| SVM (RBF) | 41.2% ± 4.5 | 0.392 | 0.192 | 78.2% |
| Logistic Regression | 35.8% ± 3.9 | 0.349 | 0.128 | 78.6% |
| k-NN (k=25) | 35.8% ± 3.8 | 0.302 | 0.109 | 80.2% |
| OneR-like | 34.5% ± 2.6 | 0.263 | 0.083 | 83.6% |
| Naive Bayes | 24.5% ± 3.7 | 0.215 | 0.049 | 58.9% |
| **ZeroR (baseline)** | 29.0% ± 0.3 | 0.112 | 0.000 | 77.6% |

### 6.4 Reading the comparison

**Behaviour alone barely works.** The best behaviour model beats ZeroR by 2.2 accuracy points, with
κ = 0.054 — "slight" agreement on any standard reading of κ. Macro-F1 is the more honest lens: it
rises from 0.112 to 0.284, i.e. the model *does* learn to distribute predictions across all four
classes instead of always guessing C1, but it distributes them barely better than chance.

**Prior academic performance is the signal.** One variable — last completed semester's SGPA band —
moves accuracy from 31.2% → 43.4% and κ from 0.054 → 0.237. That is a **+12.2 point** lift from a
single column, larger than the combined contribution of all 15 behavioural items.

**The top of each table is a statistical tie.** The macro-F1 gap between 1st and 2nd on the
behaviour experiment is **0.0049**, against a fold-to-fold accuracy std of **±0.043**. We therefore
do **not** crown a winner on that margin: SVM, Random Forest and Logistic Regression are
indistinguishable here, and the same is true of Gradient Boosting / Random Forest / Decision Tree in
the primary experiment. Saying so is more useful than picking one.

**Naive Bayes is the only clear loser**, in both experiments, and for an identifiable reason: it
assumes features are conditionally independent given the class, and §4.5 shows the Likert items are
mildly but consistently inter-correlated.

**A trap in the `within_1_band` column.** ZeroR scores **77.6%** on it — higher than any behaviour
model — without learning anything, because it always predicts a middle class that is within one
band of three of the four classes. Read `within_1_band` only alongside κ, which is 0.000 for ZeroR
by construction.

### 6.5 Error analysis (gap = |actual − predicted| band index)

| Gap | Meaning | Primary (GB) | Behaviour (SVM) |
|---|---|---|---|
| 0 | correct | 480 (43.3%) | 335 (30.2%) |
| 1 | neighbouring band | 385 (34.7%) | 457 (41.2%) |
| 2 | two bands off | 176 (15.9%) | 241 (21.8%) |
| 3 | three bands off | 67 (6.0%) | 75 (6.8%) |

**78.1% of primary predictions land within one band.** The errors are overwhelmingly local, which
is what you want from an ordinal model — but it also means the models are compressing toward the
middle of the scale rather than confidently identifying the extremes.

### 6.6 Would more data help? — the learning curve

Cross-validated accuracy moved from **0.270 at n = 99** to **0.300 at n = 997** — a 10× increase in
training data bought roughly three accuracy points, and the curve is flat over its final third.

**Answer to the viva question "ki korle accuracy barto":** not more rows of the same questionnaire.
The ceiling is the *features*, not the sample size. What would actually raise it:

1. **Objective grades from the registrar** instead of self-reports.
2. **Course-level rather than semester-level** records.
3. **Longitudinal** measurement — the same student across semesters, which would let habit *changes*
   be measured rather than habit *levels*.
4. Prior academic performance — already worth +12.2 points, and the one lever proven to work here.

### 6.7 The interpretable model

A depth-3 decision tree on behaviour features alone reaches **27.3%** CV accuracy — **below the
29.0% ZeroR floor**. It splits on `weekly_study_time`, `academic_progress`,
`desired_department_match`, `sleep_duration` and `study_environment`.

We report this rather than hiding it: a readable tree on this data is not merely worse than the
ensemble, it is worse than guessing the majority class. Interpretability and accuracy are in direct
conflict here, and that conflict is a finding about the data, not a bug in the tree.

### 6.8 The numeric-CGPA view

Neither form collected a CGPA *number* from most respondents, so a regression can only predict a
band midpoint. Random Forest regression on the behaviour features gives **MAE = 0.259 GPA points**
against a predict-the-mean baseline of **0.265**, with **R² = −0.001**.

In plain terms: the behavioural model is no better than quoting everyone the class average. This is
the same conclusion as §6.4, expressed on a scale people find intuitive, and it is why the
classification framing is the honest primary result.

---

## 7. Verdict

### 7.1 Is the +12.2 point primary lift real, or an artefact of the KUET reference point?

§3.2 flagged that for KUET rows the CGPA target *includes* the semester whose SGPA is the primary
feature, so some of the lift could be arithmetic rather than prediction. Source A has **no** such
overlap — its CGPA is measured strictly before that semester — so running the experiment separately
on each subset bounds the problem:

| Subset | n | Behaviour acc | Primary acc | Lift |
|---|---|---|---|---|
| KUET only (CGPA includes that semester) | 878 | 31.3% | 43.7% | **+12.4 pts** |
| Source A only (**no overlap**) | 230 | 32.6% | 46.1% | **+13.5 pts** |

**The lift is if anything *larger* on the subset with no overlap.** The primary result is
prediction, not arithmetic, and the headline number does not need discounting.

### 7.2 What we can honestly claim

| Claim | Evidence |
|---|---|
| The two surveys can be merged into one dataset | 14 of 16 items differ by a negligible/small Cliff's δ; band mixes are similar; one medium-δ exception is named and quarantined |
| Behavioural self-reports carry **weak but non-zero** signal about CGPA | best model 31.2% vs 29.0% ZeroR, κ = 0.054; 3 of 15 features survive FDR correction |
| The strongest behavioural correlate is **getting your desired department** | ρ = +0.119, q = 0.001, replicates independently in both samples |
| Outside commitments and attendance also matter, slightly | ρ = −0.086 and +0.090, q ≈ 0.02 |
| **Prior academic performance dominates** | +12.2 accuracy points, +0.18 κ, from a single column; verified free of reference-point artefact |
| Composite "trait" indices do not exist in this data | best Cronbach α = 0.273 at n = 1,108, after failing at n = 230 |
| Two findings from the 230-response study were noise | `topic_clarity` and `stress_frequency` do not replicate in the 896-response sample |
| More of the same survey would not help | learning curve flat: +3 accuracy points across a 10× data increase |

### 7.3 What we cannot claim

- **No causation.** This is cross-sectional self-reported data. Everything above is "associated
  with", never "causes". A student who attends more classes is *observed* in a higher band; nothing
  here shows that raising attendance raises the grade.
- **No claim about students in general.** This is a convenience sample circulated through our own
  networks. 79% of rows are KUET, and 75% of those sit in a single semester.
- **No claim of a deployable predictor.** 31% accuracy on four classes is not a product.

### 7.4 Limitations

1. Self-reported grades and habits — no registrar data to verify against.
2. Convenience sample; 79% KUET, and heavily concentrated in semester 7.
3. The CGPA reference point differs by one semester between the two forms (§3.2, §7.1).
4. `admission_satisfaction` may carry a first-option response artefact on the KUET form (§4.3).
5. GPA was collected in bands, so no true continuous CGPA exists — the regression in §6.8 predicts
   band midpoints, not grades.
6. Cross-sectional design: every result is an association.

---

## 8. Outputs

| Path | Contents |
|---|---|
| `notebooks/02_merged_cleaning_eda_modeling.ipynb` | the executed pipeline, 68 code cells |
| `notebooks/pipeline_v2_source.py` | the same code in `# %%` format, for diffing / regenerating |
| `cleaned-dataset/ours/merged/merged_primary_clean.csv` | 1,108 × 27, readable text values, with prior SGPA |
| `cleaned-dataset/ours/merged/merged_behavior_clean.csv` | 1,108 × 26, without prior SGPA |
| `cleaned-dataset/ours/merged/merged_full_clean.csv` | every intermediate column, for auditing |
| `cleaned-dataset/ours/merged/merged_primary.arff` / `merged_behavior.arff` | WEKA-ready, class attribute last, ordinal codes numeric |
| `docs/figures/merged/m01…m22.png` | 22 charts, regenerated by the notebook |
| `docs/results_summary_merged.json` | every headline number, machine-readable |
| `docs/model_results_merged_primary.csv` / `_behavior.csv` | full 9-model comparison tables |
| `docs/feature_importance_merged.csv` | 4-method consensus ranking with q-values |
| `docs/replication_n230_vs_n1108.csv` | the §5.2 replication table |
| `docs/source_form_comparison.csv` | the §4.3 Cliff's δ table |
| `docs/cleaning_step_log.csv` | the 10-step audit trail |

**In WEKA, expect 1,108 instances** — not the 878 the supplied tutorial PDF mentions, and not
notebook `01`'s 230.

### A note on a bug we found and fixed

While building this notebook we found that the Benjamini–Hochberg implementation inherited from
notebook `01` took its running minimum in the wrong direction (forward from the smallest p-value
instead of backward from the largest). The effect is to collapse every q-value onto the smallest
one, which reported **15 of 15** features as significant. It is fixed in both notebooks and both
have been re-run. Notebook `01`'s published conclusions are unchanged by the fix (0 features
survived correction there either way); this notebook's corrected count is 3 of 15.

We record it because "we checked our own statistics and found an error" is the kind of thing the
viva is testing for.
