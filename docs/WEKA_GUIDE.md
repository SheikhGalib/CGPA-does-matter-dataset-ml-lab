# WEKA guide — running our data

How to take the notebook's output into WEKA and produce the comparison table the report needs.
This follows `WEKA_Next_Steps_Tutorial.pdf` with **two corrections** (see the box below).

## Read this first — corrections to the supplied tutorial

The tutorial PDF was written against the old `cleaned-dataset/` files, which turned out to
describe a different dataset (see `AGENTS.md` §4). Two things it says are wrong for us:

| Tutorial says | Correct for our data |
|---------------|---------------------|
| "Check that WEKA shows **878 instances**" | **230 instances** |
| Load `student_performance_primary_clean.csv` | Load `cleaned-dataset/ours/primary.arff` |

Everything else in it — ZeroR/OneR baselines, 10-fold CV, macro-F1, kappa, confusion matrix, the
gap analysis — is correct methodology and is what we follow.

## Files to use

```
cleaned-dataset/ours/
  primary.arff     230 instances, 22 attributes + class   <- includes previous CGPA
  behavior.arff    230 instances, 21 attributes + class   <- excludes previous CGPA
```

Work with **one file at a time**. Do not merge their results into a single model ranking — they
are two separate experiments answering two different questions.

Use the `.arff` files rather than the `.csv` ones. ARFF declares each attribute's type and the
exact class order (C0 → C3), so WEKA cannot guess wrong.

---

## Step 1 — Open the primary dataset

1. Launch **WEKA GUI Chooser** → **Explorer**.
2. **Preprocess** tab → **Open file…** → `cleaned-dataset/ours/primary.arff`.
3. Confirm the header reads **Instances: 230**, **Attributes: 23**.
4. In the class selector (bottom left of the Classify tab), confirm **`current_cgpa_band`** — it
   is the last attribute, so WEKA selects it by default.

You should see the class distribution as roughly **60 / 52 / 63 / 55** across C0–C3.

## Step 2 — Set the evaluation method

**Classify** tab → **Test options**:

- Select **Cross-validation**, **Folds = 10**
- Click **More options…** → tick *Output per-class stats*, *Output confusion matrix*, and
  *Output predictions* (needed for Step 6)
- Leave **random seed = 1**

> Never report a number from "Use training set". It is not an honest estimate.

## Step 3 — Run the baselines

Run these two first; every later model is judged against them.

| Classifier | Where | What it tells you |
|-----------|-------|-------------------|
| **ZeroR** | `rules → ZeroR` | Majority-class floor. Ours is **27.4%**. |
| **OneR** | `rules → OneR` | What a single best feature achieves. |

Right-click each result in the result list → **Save result buffer**, naming them
`Primary-ZeroR`, `Primary-OneR`.

## Step 4 — Run the main classifiers

Same 10-fold settings, default parameters:

| WEKA path | Our Python equivalent | Expected (primary) |
|-----------|----------------------|--------------------|
| `trees → J48` | Decision Tree | ~43% |
| `functions → Logistic` | Logistic Regression | ~40% |
| `trees → RandomForest` | Random Forest | **~51%** |
| `functions → SMO` | SVM (RBF) | ~45% |
| `bayes → NaiveBayes` | Naive Bayes | ~30% |

Our Python numbers are in `docs/model_results_primary.csv`. WEKA will not match them exactly —
different default hyper-parameters, different fold assignment — but the **ranking should be
similar**. If WEKA puts Naive Bayes on top, something is misconfigured.

If a classifier complains about missing values (it should not — our export has none), use
`meta → FilteredClassifier` with filter `ReplaceMissingValues`.

## Step 5 — Record the results

Build this table. Copy the values straight from each WEKA result buffer.

| Model | Accuracy | Macro-F1 | Weighted F1 | Kappa | Time (s) |
|-------|----------|----------|-------------|-------|----------|
| ZeroR | | | | | |
| OneR | | | | | |
| J48 | | | | | |
| Logistic | | | | | |
| RandomForest | | | | | |
| SMO | | | | | |

**Where to find each number in the WEKA output:**

```
Correctly Classified Instances     117    50.87 %     <- Accuracy
Kappa statistic                    0.3487              <- Kappa

=== Detailed Accuracy By Class ===
        TP Rate  FP Rate  Precision  Recall  F-Measure  ...  Class
          0.583    0.176      0.545   0.583      0.564  ...  C0_below_3_20
          0.365    0.140      0.404   0.365      0.384  ...  C1_3_20_to_3_49
          0.508    0.180      0.508   0.508      0.508  ...  C2_3_50_to_3_74
          0.564    0.114      0.585   0.564      0.574  ...  C3_3_75_plus
Weighted Avg.  0.509  0.153   0.510   0.509      0.509  <- Weighted F1
```

> **Macro-F1 is not on that screen.** WEKA's "Weighted Avg." row is weighted by class size.
> Compute macro-F1 yourself: average the four per-class F-Measure values.
> For the example above: (0.564 + 0.384 + 0.508 + 0.574) / 4 = **0.508**.

## Step 6 — Repeat with `behavior.arff`

Load `cleaned-dataset/ours/behavior.arff` and repeat Steps 2–5. Title the second table
**"Behaviour-focused model results"**.

Expect these to be much weaker — around **34% accuracy, kappa ≈ 0.09**. That is the finding, not
a mistake. Our Python run reached the same conclusion (`docs/model_results_behavior.csv`).

## Step 7 — Choose the best model

Prefer the highest **macro-F1**, then check kappa, accuracy and the confusion matrix. A good
model must beat ZeroR *and* must not be good at only one class.

If two are close, prefer the one that is easier to explain — **J48 or Logistic** — because you
have to defend it in the viva. A weak behaviour-model result is a valid finding.

## Step 8 — Error analysis (the gap table)

For the selected model, right-click its result → **Visualize classifier errors**, or save the
predictions. Convert class labels to indices:

| Class | C0 | C1 | C2 | C3 |
|-------|----|----|----|----|
| Index | 0 | 1 | 2 | 3 |

For each prediction compute `gap = |actual index − predicted index|`, then summarise counts.
Our Python run produced:

| Gap | Meaning | Primary (RF) | Behaviour (GB) |
|-----|---------|--------------|----------------|
| 0 | correct | 119 (51.7%) | 74 (32.2%) |
| 1 | neighbouring band | 68 (29.6%) | 81 (35.2%) |
| 2 | two bands off | 23 (10.0%) | 38 (16.5%) |
| 3 | three bands off | 20 (8.7%) | 37 (16.1%) |
| | **within one band** | **81.3%** | **67.4%** |

Summarise the counts. Do not comment on individual respondents.

## Step 9 — Optional: WEKA's own feature ranking

Cross-checks our Python consensus ranking (§7 of the notebook).

**Select attributes** tab:

- `InfoGainAttributeEval` + `Ranker` → ranks features individually
- `CfsSubsetEval` + `BestFirst` → picks a non-redundant subset

Use **"Cross-validate"** rather than "Use full training set", so the ranking is not fitted to all
the data. Expect a weak, unstable ranking on `behavior.arff` — consistent with our finding that
no behavioural feature survives FDR correction.

---

## What goes in the final report

- The two experiment definitions: **primary** and **behaviour-focused**
- One comparison table per experiment, including ZeroR and OneR
- The confusion matrix of the selected model
- A short explanation of the most common errors (use the gap table)
- A comparison of performance **with and without previous CGPA** — this is the headline: +16.5
  accuracy points from one variable
- **Limitations:** self-reported data; n = 230 convenience sample; BRAC/CUET respondents cluster
  in a few semesters so university and academic stage are confounded; GPA collected in bands;
  cross-sectional, so no causal claims

> Do not change models repeatedly just to get a higher score. Use the same folds and settings for
> every model, and report weak or negative findings honestly.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| WEKA shows 878 instances | You opened the old `cleaned-dataset/*.csv`. Use `cleaned-dataset/ours/*.arff`. |
| Class attribute is wrong | Classify tab, bottom-left dropdown → `current_cgpa_band`. It is last in the ARFF. |
| Ordinal features read as nominal | Use the `.arff`, not the `.csv`. The ARFF declares them `numeric`. |
| Accuracy looks suspiciously high (>80%) | You are probably on "Use training set", or you rebuilt the ARFF including `result_satisfaction`. That column is leakage and the notebook excludes it. |
| `.arff` will not load | Open it in a text editor — the class attribute must be the last `@attribute` line, before `@data`. |
