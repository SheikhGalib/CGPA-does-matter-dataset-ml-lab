# Latest WEKA operations

Date: 13 September 2026  
WEKA version: 3.8.7

## What was rerun

The final presentation required comparable WEKA results for three targets and real WEKA J48 trees. The latest workflow therefore created three matched datasets from the verified project snapshot.

| Experiment | Target | Inputs | Records |
|---|---|---|---:|
| SGPA | Recent SGPA class | 12 questionnaire variables | 1,102 |
| Questionnaire CGPA | CGPA class | 12 questionnaire variables | 1,102 |
| Main CGPA | CGPA class | 12 questionnaire variables plus recent SGPA | 1,102 |

The 12 questionnaire inputs are attendance, weekly study time, study style, topic clarity, stress frequency, sleep duration, distraction frequency, weekly responsibilities, desired department match, support level, study environment and routine manageability.

University, department, semester, identifiers, CGPA-as-an-SGPA-predictor and result satisfaction were not used as questionnaire model inputs. No synthetic or modified student record was added.

## Data verification and exclusion

The data preparation script reran the existing cleaning and harmonisation source from the two raw CSV files. It compared the rebuilt ordinal modelling columns, target codes, university labels and SGPA-imputation flag against the checked 1,108-row cleaned dataset. The comparison passed.

Six students had a missing recent SGPA that the historical pipeline filled using the most common SGPA band inside their known CGPA class. Those six rows were excluded from all three latest WEKA experiments. This prevents target-conditioned SGPA values from entering an SGPA target experiment or the main CGPA predictor.

Descriptive presentation charts still use all 1,108 usable survey records. WEKA model slides use the same 1,102 eligible records.

## Ten-fold comparison

Each experiment ran the same six WEKA classifiers with stratified 10-fold cross-validation and seed 1.

| WEKA model | Recent SGPA | CGPA questionnaire | Main CGPA |
|---|---:|---:|---:|
| J48 | 30.13% | 26.68% | 38.29% |
| Random Forest | 29.76% | 29.40% | 43.92% |
| SMO | 25.95% | 27.86% | 37.11% |
| Logistic | 24.95% | 29.58% | 36.84% |
| Naive Bayes | 26.68% | 30.04% | 38.11% |
| ZeroR | 26.86% | 29.04% | 29.04% |

All values in a column use the same target, records and evaluation method. The saved log for each run contains the exact WEKA option string and confusion matrix.

The main CGPA comparison shows the clearest lift above ZeroR. Random Forest reaches 43.92%, compared with the 29.04% majority baseline. The questionnaire-only and SGPA-target runs remain weak and should be reported honestly.

## Teacher-facing J48 trees

One existing CGPA-stratified split with seed 42 was reused for every J48 explanation:

- 881 training records;
- 221 held-out test records;
- J48 options `-C 0.25 -M 40`.

The larger minimum leaf size produces smaller trees that remain readable on a slide. It was used for explanation, while the broader 10-fold model table keeps default J48 settings (`-C 0.25 -M 2`).

| Target | Training accuracy | Test accuracy | Test ZeroR |
|---|---:|---:|---:|
| Recent SGPA | 39.39% | 29.86% | 27.15% |
| CGPA questionnaire | 36.55% | 32.13% | 28.96% |
| Main CGPA | 46.20% | 49.77% | 28.96% |

The main CGPA tree first splits on recent SGPA. Its saved graph then uses weekly responsibilities and desired-department match for part of the highest-SGPA branch. The rules shown in the deck were extracted from this exact textual WEKA tree; no branch was invented manually.

## Saved WEKA evidence

`weka-final/evidence/` contains:

- `*_all.arff`, `*_train.arff`, `*_test.arff` for each target;
- `*_cv.txt` for each classifier and target;
- `*_J48_holdout.txt` with tree, training evaluation, test evaluation and ZeroR test baseline;
- `*_tree.txt` and `*_tree.dot` for exact tree definitions;
- `*_J48.model` serialized WEKA models;
- `*_weka_viewer.png` exported by WEKA TreeVisualizer;
- `*_weka_output.png` showing the verbatim WEKA test output in a Java Swing text component;
- `slide_result_audit.csv` and `.md` mapping every reported result to its source;
- `data_provenance.json` and `split_indices.csv`.

The output images are evidence views generated from real WEKA objects and logs. They are not photographs of a manually operated WEKA Explorer window.

## Presentation compatibility checks

The final PowerPoint contains 50 visible slides and 16 hidden backup slides. All editable text uses Times New Roman. It uses standard Office text boxes, rectangles, arrows, native tables and one native editable column chart. Evidence screenshots are embedded PNG images, so the deck has no linked image dependency.

The final deck passed package integrity, slide geometry, font, table and chart checks. Installed Microsoft PowerPoint opened and rendered all 66 slides, and its text-bound audit reported zero clipped text boxes.

## Method limits

The fixed split was originally stratified by CGPA and then reused for SGPA so the teacher-facing examples share the same students. The SGPA split is therefore fixed rather than independently SGPA-stratified.

Questionnaire missing-value modes were calculated in the historical cleaning pipeline before evaluation. That introduces a small preprocessing leakage risk. Results should be described as exploratory comparisons rather than deployment-ready performance estimates.
