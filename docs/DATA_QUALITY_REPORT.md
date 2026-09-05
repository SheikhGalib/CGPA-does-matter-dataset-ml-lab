# Data quality report

What was wrong with the data we were given, how we verified it, and what we did about it.
Written so any teammate can reproduce every claim. Every number comes from
`notebooks/01_cleaning_eda_modeling.ipynb`.

---

## Finding 1 — The existing `cleaned-dataset/` files are not our data

**Severity: blocking.** These files cannot be used for anything we report.

### Evidence

| Check | Our raw form | `student_performance_primary_clean.csv` |
|-------|--------------|------------------------------------------|
| Rows | 230 | **878** |
| Semester values | 1–12 | S1–S7 only |
| Departments | CSE, EEE, ECE, ME, CE, BME, ChE, ARCH, URP, MME, NCE, IPE, PMRE, MIE, NAME, WRE | also contains **TE, LE, BECM, ESE, IEM, MSE** |
| A column named | — | **`kuet_satisfaction`** |
| `university` column | present in source | **dropped** |
| Row alignment | — | GPA bands do not match the raw data row-for-row |

### Why this is conclusive

1. **Row count.** The Google Form collected 230 responses. There is no augmentation step
   documented anywhere that would produce 878.
2. **Impossible departments.** TE (Textile), LE (Leather), BECM, ESE, IEM and MSE appear in the
   file but in **zero** raw responses. Those are KUET departments.
3. **The column name.** `kuet_satisfaction` names an institution none of our 230 respondents
   attend — our sample is BUET (104), BRAC (101), CUET (25).

Taken together this is a different, most likely synthetic, KUET-based dataset.

### Reproduce it

Run §4 of the notebook, or:

```powershell
.\venv\Scripts\python.exe -c "import pandas as pd; print(len(pd.read_csv('raw-data/University Student Performance Analysis Form (Responses) - Form Responses 1.csv')), len(pd.read_csv('cleaned-dataset/student_performance_primary_clean.csv')))"
# -> 230 878
```

### What we did

**Kept the schema, rebuilt the rows.** The schema is genuinely good and matches the WEKA
tutorial — the C0–C3 band scheme, the primary/behaviour split, snake_case naming. We reproduced
it from the 230 real responses in `cleaned-dataset/ours/`, with two fixes:

- `kuet_satisfaction` → **`admission_satisfaction`**
- **`university` added back** — it turned out to be one of the most structurally important
  variables in the dataset (Finding 3)

The old files are left in place, untouched, so the comparison in §4 stays reproducible.

### Knock-on effect

`docs/WEKA_Next_Steps_Tutorial.pdf` was written against these files and tells us to *"check that
WEKA shows 878 instances"*. For our data the correct number is **230**. Its methodology is sound
and we follow it — see `WEKA_GUIDE.md`.

---

## Finding 2 — Two department columns are one column

**Severity: would have cost 24% of a field.**

The form had a dropdown plus an "if not listed, write it here" box.

```
department_listed  filled only : 173
department_other   filled only : 55
both filled                    : 1     ("Already Listed" — a non-answer)
both missing                   : 1
```

173 + 55 = 228, plus the 2 edge cases = 230. In isolation each column looks unusable (24% and
76% missing). Together they are complete.

**What we did.** Coalesced them into one `department`, dropdown taking precedence, then mapped
abbreviations onto canonical names (`ChE` → Chemical Engineering, `NAME` → Naval Architecture &
Marine Engineering, and 14 others). Without that mapping the same department appears as two
unrelated categories and its signal is split.

The single row with neither answered was labelled `Unknown` rather than dropped — its other 19
answers are perfectly usable.

Departments with fewer than 5 respondents were grouped into `Other` for modelling
(`department_grouped`), because a category with 2 members cannot be learned, only memorised. The
ungrouped `department` is retained for reporting.

---

## Finding 3 — `semester` is really `university` in disguise

**Severity: would have produced a misleading model.**

```
          1  2  3  4  5  6  7  8  9  10 11 12
  BRAC    0  0  0  0  0  0  0  0  0  48 53  0
  BUET   10  3  8  8 14  8 17  6  6   7  8  9
  CUET    0  0  0  0  0  0 25  0  0   0  0  0
```

**BRAC runs a trimester system**, so all 101 of its respondents sit at semester 10–11. All 25
CUET respondents are at semester 7. Only BUET spreads across the range.

So "semester 11" does not mean *further along than semester 7* — it means *BRAC student*. A model
given the raw number can recover the university from it and learn per-institution grading habits
while appearing to have learned something about academic progress.

**What we did.** Excluded `semester_raw` from the exports. Derived `academic_progress` =
semester ÷ programme length (BRAC 12, BUET 8, CUET 8), which is comparable across institutions,
plus a coarse `academic_year`.

---

## Finding 4 — 44 blanks that are actually an answer

**Severity: would have cost 19% of the sample.**

`weekly_responsibilities` has 44 blanks (19%). The naive fix is to drop those rows. We tested
the assumption first.

```
Options offered: 1-5 hrs | 6-10 hrs | 11-15 hrs | More than 15 hrs
                 ^ there is no zero option, and the question was not required

Blank rate:  BRAC 21.8%   BUET 15.4%   CUET 24.0%
Chi-square, blankness vs GPA band: chi2 = 5.38, dof = 5, p = 0.372
```

p = 0.372 → blankness is **not** associated with performance, so these are not
low-effort respondents dropping out of a hard question. Combined with the missing zero option,
a blank is best read as *"I have no tuition, job or club load"*.

**What we did.** Imputed as a new lowest category `none`. Zero rows dropped.

---

## Finding 5 — One question leaks the answer

**Severity: would have invalidated the model.**

*"How satisfied were you with your latest semester result?"* is asked about the **same semester**
whose grade is our target.

```
Spearman(result_satisfaction, target) = 0.601   (p = 2e-24)
```

This is not a cause of the grade — it is a restatement of it. Including it produces a model that
scores well and means nothing, and in any real deployment you would not have this field before
knowing the grade.

**What we did.** Excluded from every model. **Kept for one purpose:** mean satisfaction rises
monotonically across C0 → C3, which is positive evidence that respondents answered the GPA and
satisfaction questions consistently, i.e. the self-reported grades are not random noise.

*(The old cleaned files also omitted this column. Our exclusion is deliberate and documented.)*

---

## Finding 6 — Our own hypothesis failed, and we report it

We grouped the Likert items into four composite indices (discipline, wellbeing, environment,
motivation), then tested whether each group actually measures one thing.

| Index | Items | Cronbach's alpha |
|-------|-------|-----------------|
| discipline_index | 4 | 0.022 |
| wellbeing_index | 3 | 0.048 |
| environment_index | 2 | 0.053 |
| motivation_index | 3 | 0.038 |

Alpha ≥ 0.7 is "good". Ours are ≈ 0, meaning the items inside each group are **mutually
uncorrelated** — a student with high attendance is no more likely than chance to also study long
hours. There is no latent "discipline" trait in this data.

**What we did.** Kept the columns in the export (they cost nothing and a WEKA user may want to
test them) but **modelled the raw items instead**. No conclusion in this project rests on an
index.

---

## Summary of actions

| | Count |
|---|---|
| Rows in raw form | 230 |
| Rows dropped | **0** |
| Rows in final dataset | **230** |
| Values imputed | 44 (`weekly_responsibilities` → `none`) |
| Columns dropped | 5 — `timestamp`, `result_satisfaction` (leakage), `semester_raw` (confound), and the two raw GPA strings (replaced by the encoded bands) |
| Columns merged | 2 → 1 (`department`) |
| Columns added | `university` (restored), `academic_progress`, `academic_year`, 4 indices |
| Free-text values canonicalised | 9 university spellings → 3; 31 department strings → 16 |
| Missing values in final export | **0** |

## Limitations that must appear in the report

- **Self-reported grades and habits.** No registrar data to verify against. Finding 3 in the
  notebook (§7.8) shows a self-assessment calibration effect that is visible in the data itself.
- **n = 230**, a convenience sample circulated through the team's own networks — not random.
- **University and academic stage are confounded.** BRAC and CUET respondents cluster in a few
  semesters.
- **GPA collected in bands**, so no true continuous CGPA exists. Any numeric prediction is a band
  midpoint.
- **Cross-sectional data.** Every result is an association. No causal claims anywhere.
