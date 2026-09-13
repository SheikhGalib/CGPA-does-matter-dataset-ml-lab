# Final Clean Slide Plan — WEKA Only
## Project title
**What Shapes Student CGPA? An Engineering Student Academic Performance Dataset and Machine Learning Analysis**

Use the same title in the final report.

---

# Global rules

- **Main modelling tool: WEKA only.**
- **Python may be used only for data cleaning / harmonisation / exporting clean CSV or ARFF.**
- Remove Python model results, Python Decision Trees, Python hold-out model comparisons, Python confusion matrices and Python model names from the main deck.
- All Decision Trees shown in the presentation must be generated in **WEKA**.
- All rules shown beside a Decision Tree must be extracted from that exact WEKA tree.
- Every accuracy must show:
  - **model name**
  - **target**
  - **evaluation method**
- If a train/test result is shown, the **same model** must be used for both training and test results.
- Use one fixed WEKA train/test split for teacher-facing J48 analysis.
- Use WEKA 10-fold cross-validation only for the broader model comparison unless a supplied test set is created for every model.
- Every confusion matrix must clearly state the **WEKA model name**.
- Remove repeated explanatory text when the diagram/chart already communicates the idea.
- Use large charts and large screenshots; no tiny tables.
- Previous-work slides move near the end.
- Do not show a separate “Our Contribution” slide.
- Keep limitations near the end.
- Behaviour-only weak rules must not be presented as strong causes.
- No synthetic/modified student records.

---

# Required WEKA reruns before rebuilding slides

## A. Dataset versions

Create/use:

1. **SGPA target dataset**
   - target = recent SGPA class
   - inputs = questionnaire variables
   - exclude CGPA target and leakage-like result-satisfaction variables

2. **CGPA questionnaire dataset**
   - target = CGPA class
   - inputs = questionnaire variables
   - recent SGPA excluded

3. **Main CGPA dataset**
   - target = CGPA class
   - inputs = questionnaire variables + recent SGPA

Use the same cleaned records consistently.

If the six SGPA-imputed rows are considered unsafe for any SGPA-based experiment, exclude them from the relevant WEKA datasets and state the final N once on the train/test slide.

---

## B. WEKA Decision Trees

Run **J48** for:

- SGPA target
- CGPA questionnaire target
- Main CGPA target

For presentation readability:

- use WEKA J48 pruning
- increase minimum instances per leaf if necessary
- keep the tree small enough to read
- do not manually invent branches
- save the WEKA tree output/viewer screenshot
- save the generated textual rules/tree

The model name on slides must be:

> **WEKA J48 Decision Tree**

If a more strongly pruned J48 is used for the slide, report its exact options in backup.

---

## C. WEKA model comparison

Run the same WEKA models for the required target(s):

- J48
- Random Forest
- SMO
- Logistic
- Naive Bayes
- ZeroR baseline

Use the same evaluation scheme for comparisons.

Preferred:
- 10-fold cross-validation for the all-model comparison
- one fixed 80/20 split or supplied test set for the J48 train/test explanation

---

## D. Confusion matrices

Save confusion matrices from WEKA for every model mentioned in the main comparison:

- J48
- Random Forest
- SMO
- Logistic
- Naive Bayes
- ZeroR

Every matrix image must contain a clear heading such as:

> **WEKA SMO — CGPA — 10-fold CV**

Do not show an unlabeled matrix.

---

# Final main presentation

Target: approximately **38–42 visible slides**.

Technical details remain hidden as backup.

---

# SECTION A — Opening

## Slide 1 — Title

**What Shapes Student CGPA?**  
**An Engineering Student Academic Performance Dataset and Machine Learning Analysis**

Visual only:
`Student → Form → Data → Relation → WEKA → CGPA`

Small footer:
- 1,108 usable responses
- BUET · BRAC · CUET · KUET

---

## Slide 2 — What are we trying to understand?

Three questions:

1. What did students answer?
2. How does each answer relate to CGPA?
3. Can WEKA models predict SGPA / CGPA?

Keep this slide very simple.

---

# SECTION B — Forms and responses

## Slide 3 — Survey Form 1

Large screenshot of the multi-university Google Form.

Show:
- form title
- representative questions
- answer options

Do not place long description beside the screenshot.

---

## Slide 4 — Form 1 Responses

Immediately after the form.

Show a large screenshot of the response sheet.

Large number:
> **230 responses**

Add:
> BUET · BRAC · CUET

If the screenshot contains personal information, crop it.

---

## Slide 5 — Survey Form 2 / KUET Form

Large screenshot of the KUET form or its question structure.

Show only the necessary part.

---

## Slide 6 — KUET Form Responses

Large response-sheet screenshot.

Large number:
> **896 responses**

Add:
> KUET

---

## Slide 7 — Total Responses

Large central flow:

**230 + 896 = 1,126 raw responses**

then

**1,108 usable records after cleaning**

Visual:
two source boxes → merged dataset

No extra paragraph.

---

# SECTION C — Form answer statistics

## Slide 8 — Response Categories

Use category blocks only:

- Academic
- Study habit
- Lifestyle
- Stress / responsibility
- Motivation / support
- Environment
- Academic result

This replaces the old long category explanation.

---

## Slide 9 — Form Responses: Academic Questions

Use actual form-question layout or clean Excel-style plots.

For:
- Attendance
- Weekly study time
- Study style

For every option show:
- number of students
- percentage

If the visual uses a form screenshot, add:
> “Counts and percentages added from the cleaned dataset.”

---

## Slide 10 — Form Responses: Academic Understanding + SGPA

Show:
- Topic clarity
- Recent SGPA

Option counts + percentages.

Use large horizontal bars.

---

## Slide 11 — Form Responses: Lifestyle

Make this slide **large** and readable.

Show:
- Sleep
- Phone / social-media distraction

Use graphs generated from the cleaned Excel/CSV data.

Do not use small screenshots.

---

## Slide 12 — Form Responses: Stress and Responsibilities

Show:
- Stress frequency
- Outside commitments

Counts + percentages.

---

## Slide 13 — Form Responses: Motivation and Environment

Show:
- Desired department
- Family/friend support
- Study environment
- Routine manageability

Use two graphs per half-slide or split if readability is poor.

---

# SECTION D — Dataset composition

## Slide 14 — Universities in the Dataset

Show only:

- KUET — 878
- BUET — 104
- BRAC — 101
- CUET — 25

Use a large bar chart.

**Remove the caution/warning text from this slide.**

The sampling limitation will appear later.

---

## Slide 15 — CGPA Classes We Predict

Title:
> **Predicted CGPA Classes**

Show four boxes:

- C0: < 3.20
- C1: 3.20–3.49
- C2: 3.50–3.74
- C3: ≥ 3.75

Counts and percentages below each.

---

## Slide 16 — CGPA Distribution

Large Excel-style bar chart.

Show:
- C0
- C1
- C2
- C3

Include:
> ZeroR majority-class baseline = C1

No discussion of SGPA on this slide.

---

## Slide 17 — Recent SGPA Distribution

This is the second half of the old combined distribution slide.

Large bar chart of SGPA classes.

Use the same class-color order as CGPA.

---

# SECTION E — Every questionnaire variable vs CGPA

Use **100% stacked bar charts** wherever possible.

Each answer option should show the share of:
- C0
- C1
- C2
- C3

This section replaces standalone “response-only” charts later in the old deck.

---

## Slide 18 — Attendance vs CGPA

Title:
> **Attendance and CGPA**

Chart:
CGPA class composition for every attendance option.

No separate bullet explanation unless one short takeaway is necessary.

---

## Slide 19 — Weekly Study Time vs CGPA

Show all study-time options against C0–C3.

---

## Slide 20 — Study Style vs CGPA

Show all study-style options against C0–C3.

---

## Slide 21 — Topic Clarity vs CGPA

Show all topic-clarity options against C0–C3.

---

## Slide 22 — Sleep vs CGPA

Show all sleep-duration options against C0–C3.

---

## Slide 23 — Stress vs CGPA

Show all stress-frequency options against C0–C3.

---

## Slide 24 — Phone / Social-Media Distraction vs CGPA

Show all options against C0–C3.

---

## Slide 25 — Outside Commitments vs CGPA

Show all tuition/job/club-hour options against C0–C3.

---

## Slide 26 — Desired Department vs CGPA

Show all answer levels against C0–C3.

---

## Slide 27 — Support vs CGPA

Show all family/friend-support levels against C0–C3.

---

## Slide 28 — Study Environment vs CGPA

Show all environment levels against C0–C3.

---

## Slide 29 — Routine Manageability vs CGPA

Show all routine levels against C0–C3.

---

## Slide 30 — Summary of Questionnaire–CGPA Relations

One compact chart/table only.

For every questionnaire feature show:
- direction/strength of relationship with CGPA

Highlight the three clearest:
- desired department
- attendance
- outside commitments

Do not repeat all charts in text.

---

# SECTION F — SGPA and CGPA target pattern

## Slide 31 — Do SGPA and CGPA Show the Same Pattern?

Compare each feature's relationship with:
- recent SGPA
- CGPA

Use a paired bar/dot chart.

Highlight only relationships that move in the same direction clearly.

No long explanation.

---

# SECTION G — WEKA model comparison

Move the old target/model comparison here, immediately after the SGPA–CGPA pattern slide.

## Slide 32 — WEKA Target-Specific Model Results

Use **WEKA only**.

Table with model names and accuracies.

Suggested columns:

| WEKA model | SGPA target | CGPA questionnaire target | Main CGPA target |
|---|---:|---:|---:|

Rows:
- J48
- Random Forest
- SMO
- Logistic
- Naive Bayes
- ZeroR

Every value must come from WEKA using the same stated evaluation scheme.

No Python results.

---

# SECTION H — Train/test split

## Slide 33 — Train / Test Split

This replaces the old framework slide.

Diagram only:

**Dataset → 80% Training → WEKA model → 20% Testing**

Show exact counts.

If using 1,102:
- 881 train
- 221 test

State once:
> six SGPA-imputed rows excluded from SGPA-based tree experiments

Do not repeat this explanation later.

---

# SECTION I — WEKA Decision Trees

## Slide 34 — WEKA J48: SGPA Decision Tree

Use the **actual WEKA J48 tree screenshot/viewer**.

Title must include model:
> **WEKA J48 — Recent SGPA**

Use a readable/pruned J48 tree.

If the tree is still large:
- show the top section
- add a small inset of the full tree
- put full tree in backup

Do not use the previous Python shallow tree.

---

## Slide 35 — WEKA J48: SGPA Rules

Rules must come from Slide 34's exact J48 tree.

Show only 2–4 readable rules.

Format:
`IF ... AND ... → SGPA class`

No made-up rule validation badges.

---

## Slide 36 — WEKA J48: CGPA Questionnaire Tree

Target:
CGPA

Inputs:
questionnaire variables only

Show actual WEKA J48 tree.

Title:
> **WEKA J48 — CGPA from Questionnaire Variables**

If performance is weak, state only the score and baseline.

Do not hide the weak result.

---

## Slide 37 — Questionnaire J48 Rules

Extract 2–4 rules from Slide 36.

Do not label them as causes.

Use:
> “tree branch / rule”

not:
> “reason students get good CGPA”

---

## Slide 38 — WEKA J48: Main CGPA Tree

Target:
CGPA

Inputs:
questionnaire variables + recent SGPA

Show actual WEKA J48 tree.

Title:
> **WEKA J48 — Main CGPA Prediction**

Highlight the root split if recent SGPA is selected first.

This is the main Decision Tree slide.

---

## Slide 39 — Main CGPA J48 Rules

Rules must be extracted from Slide 38.

Show the most interpretable:
- higher-CGPA branch
- lower-CGPA branch

Use exact WEKA split wording converted only into readable English.

---

# SECTION J — Accuracy and train/test result

## Slide 40 — WEKA J48 Train vs Test

The model name must be visible:

> **Model: WEKA J48**

Use the same J48 model and same feature set for both train and test numbers.

Show:

- Training accuracy
- Test accuracy
- Baseline

Do not mix J48 training accuracy with another model's test accuracy.

If separate J48 results are shown for SGPA and CGPA, label each target.

---

# SECTION K — Confusion matrices

All confusion matrices must be WEKA outputs.

Every matrix must show:
- model
- target
- evaluation method

## Slide 41 — WEKA J48 Confusion Matrix

Large matrix.

Heading example:
> **WEKA J48 — Main CGPA — Test Set**

Rows = actual  
Columns = predicted

Show C0–C3 meanings once.

---

## Slide 42 — WEKA Random Forest and SMO Confusion Matrices

Two matrices side by side:

- **WEKA Random Forest — Main CGPA**
- **WEKA SMO — Main CGPA**

Use the same evaluation scheme.

---

## Slide 43 — WEKA Logistic and Naive Bayes Confusion Matrices

Two matrices side by side:

- **WEKA Logistic — Main CGPA**
- **WEKA Naive Bayes — Main CGPA**

---

## Slide 44 — WEKA ZeroR Baseline Matrix

Show the baseline confusion matrix.

Purpose:
visually explain what “always predict C1” means.

Keep it simple.

---

# SECTION L — WEKA comparison summary

## Slide 45 — WEKA Model Accuracy Comparison

Bar chart only.

Models:
- ZeroR
- J48
- Random Forest
- SMO
- Logistic
- Naive Bayes

Title must specify target/evaluation:
> **WEKA Main CGPA Accuracy — 10-fold Cross-Validation**

Do not include Python.

---

## Slide 46 — Why Keep Decision Tree?

Simple comparison:

### J48
Readable rules

### Random Forest / SMO
Higher prediction benchmark, no simple readable rule

No repeated accuracy table if Slide 45 already shows it.

---

# SECTION M — Previous work

The old previous-work slide moves near the end.

## Slide 47 — Previous Studies and Models

Use the existing three studies.

Table:
- study
- target/population
- models
- key result

Keep this clean.

---

## Slide 48 — Previous Work vs Our WEKA Models

Only WEKA models for this project.

Do not mention Python models.

Suggested row for this project:

> J48 · Random Forest · SMO · Logistic · Naive Bayes · ZeroR

Add:
> Different datasets and targets; accuracies are not directly comparable.

---

# SECTION N — Findings

## Slide 49 — What Did We Learn?

**Bullet points only. Remove icons.**

Example:

- Student responses vary substantially across academic and lifestyle questions.
- Attendance, desired department and outside commitments show the clearest CGPA relationships.
- Questionnaire variables alone give weak CGPA classification.
- Recent SGPA becomes an important split in the main CGPA Decision Tree.
- WEKA Random Forest / SMO provide stronger prediction benchmarks, while J48 provides readable rules.

Do not repeat numbers already shown in charts unless essential.

---

## Slide 50 — Limitations

Plain bullets:

- self-reported data
- KUET-heavy sample
- one-time survey
- banded SGPA/CGPA
- no verified registrar records

This is where the university imbalance caution belongs, not Slide 14.

---

## Slide 51 — Conclusion

Three short blocks:

### Data
1,108 usable engineering-student responses

### Relationship
Question-by-question CGPA patterns

### WEKA
J48 rules + comparison with standard WEKA classifiers

End:
> Associations and predictions, not causal proof.

---

## Slide 52 — Thank You

Team names and rolls.

---

# Backup slides

Keep hidden unless asked.

## Backup A — Full question list
All form questions/options.

## Backup B — Data cleaning
Raw → cleaned details.

## Backup C — Exact WEKA settings
For every J48 / RF / SMO / Logistic / NB run.

## Backup D — Full J48 textual tree
SGPA.

## Backup E — Full J48 textual tree
CGPA questionnaire.

## Backup F — Full J48 textual tree
Main CGPA.

## Backup G — Full WEKA result table
Accuracy, F1, Kappa.

## Backup H — WEKA screenshots
Classifier output windows.

## Backup I — Team contribution

---

# Delete / hide from the current deck

Remove from the visible presentation:

- old motivation duplicate slides
- old “Our Contribution” slide
- old standalone framework slide
- Python SGPA Decision Tree
- Python rule-validation slides
- Python CGPA shallow-tree slides
- Python Random Forest / SVM hold-out comparison
- Python result table
- duplicate “main prediction result” slide
- separate “accuracy improved because SGPA was added” slide
- icon-heavy “What Did We Learn?” layout

Keep only WEKA modelling.

---

# Important execution rule

Before rebuilding the presentation, the agent must first produce a small audit table:

| Slide result | Target | WEKA model | Evaluation | Dataset N |
|---|---|---|---|---|

No accuracy, rule or confusion matrix may enter the PPT unless this table identifies its exact source.
