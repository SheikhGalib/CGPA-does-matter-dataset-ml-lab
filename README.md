# What Shapes Student CGPA?

This project combines 1,126 survey responses from engineering students at BUET, BRAC, CUET and KUET. The cleaning pipeline keeps 1,108 usable records and predicts four CGPA bands. The final course presentation uses WEKA only for every model result, decision tree, rule and confusion matrix.

## Final presentation

[Download the editable PowerPoint](presentation/What_Shapes_Student_CGPA_WEKA_PowerPoint_2015_Final.pptx)

- 50 visible presentation slides and 16 hidden technical backup slides
- Times New Roman throughout
- Standard PowerPoint text boxes, rectangles, arrows, editable tables and one editable column chart
- Embedded PNG screenshots for Google Form and WEKA evidence
- No online fonts, add-ins, SVG icons, videos or linked external files
- Designed for editing in Microsoft PowerPoint 2015

The deck follows [the final WEKA-only slide plan](docs/CGPA_Final_Clean_Slide_Plan_WEKA_Only.md). The two separate form-introduction slides were consolidated into one form screenshot slide using the layout supplied in the earlier 64-slide deck.

## Updated report

[Download the compiled course report](report/What_Shapes_Student_CGPA_Report.pdf) or edit its source in [report/main.tex](report/main.tex).

The report now uses the same final WEKA evidence as the presentation. Its machine-learning section reports the 1,102-row model population, the seed-1 stratified 10-fold comparison, the fixed 881/221 J48 evaluation, the actual WEKA TreeVisualizer exports, and the corrected 49.77% main-tree test result. The data-article section continues to describe all 1,108 cleaned records.

## Latest release changes - 14 September 2026

- reran and audited all 18 WEKA cross-validation experiments;
- rebuilt the three readable fixed-split J48 models and ZeroR baselines;
- added exact tree text, DOT graphs, serialized models, confusion matrices, TreeVisualizer images and output captures;
- rebuilt the editable PowerPoint with Times New Roman and PowerPoint 2015-supported objects;
- updated `main.tex` and the compiled report to use the final WEKA-only machine-learning results;
- documented the preprocessing and sampling limitations beside the reported scores.

## Latest WEKA operations

The latest run used WEKA 3.8.7 on 1,102 students. Six records whose missing recent SGPA had previously been filled from the CGPA class were excluded from every model experiment.

Three target-specific datasets were used:

1. recent SGPA from 12 questionnaire variables;
2. CGPA from the same 12 questionnaire variables;
3. CGPA from the questionnaire variables plus recent SGPA.

J48, Random Forest, SMO, Logistic, Naive Bayes and ZeroR were compared with stratified 10-fold cross-validation using seed 1. The teacher-facing J48 trees used one fixed 881/221 train/test split, `-C 0.25 -M 40`, to keep the trees readable.

| Target | Best 10-fold model | Accuracy | J48 fixed-test accuracy | ZeroR fixed-test baseline |
|---|---|---:|---:|---:|
| Recent SGPA | J48 | 30.13% | 29.86% | 27.15% |
| CGPA, questionnaire only | Naive Bayes | 30.04% | 32.13% | 28.96% |
| Main CGPA | Random Forest | 43.92% | 49.77% | 28.96% |

The cross-validation J48 uses WEKA's default minimum leaf size of 2. The explanatory fixed-test J48 uses a minimum leaf size of 40. These results answer different questions and should not be mixed.

The complete procedure, exact settings, result interpretation and validation notes are in [docs/WEKA_LATEST_OPERATIONS.md](docs/WEKA_LATEST_OPERATIONS.md).

## Evidence and reproducibility

The [weka-final/evidence](weka-final/evidence) folder contains:

- nine ARFF files covering all/train/test data for the three targets;
- raw WEKA text logs for all 18 cross-validation runs;
- fixed-split J48 logs with training and test confusion matrices;
- exact J48 textual trees, DOT graphs and serialized `.model` files;
- WEKA TreeVisualizer images and output captures;
- the slide-result audit, split indices and raw-to-clean provenance check.

The [weka-final/scripts](weka-final/scripts) folder contains the exact data preparation, WEKA execution, visualization and result-audit code used for this release.

## Main project files

| Path | Purpose |
|---|---|
| `raw-data/` | Original survey exports; treat as read-only |
| `cleaned-dataset/ours/merged/` | Current 1,108-row cleaned dataset and WEKA exports |
| `notebooks/02_merged_cleaning_eda_modeling.ipynb` | Main merge, cleaning and analysis pipeline |
| `notebooks/03_teacher_focused_analysis.ipynb` | Teacher-facing answer, tree and rule analysis |
| `docs/` | Reports, WEKA guides, figures and presentation plans |
| `teacher_focused_final/` | Shared coding helper, requirements and fixed split indices |
| `weka-final/` | Reproducible evidence for the latest WEKA-only presentation |
| `report/` | Updated LaTeX source, figures and compiled course report |

## Run the latest WEKA evidence workflow

From the repository root in PowerShell:

```powershell
.\venv\Scripts\python.exe weka-final\scripts\prepare_data.py

javac -cp "C:\Program Files\Weka-3-8-7\weka.jar" weka-final\scripts\WekaEvidence.java
java --add-opens java.base/java.lang=ALL-UNNAMED `
  -cp "C:\Program Files\Weka-3-8-7\weka.jar;weka-final\scripts" `
  WekaEvidence weka-final\evidence

javac -cp "C:\Program Files\Weka-3-8-7\weka.jar;weka-final\scripts" `
  weka-final\scripts\RenderWeka.java
java --add-opens java.base/java.lang=ALL-UNNAMED `
  -cp "C:\Program Files\Weka-3-8-7\weka.jar;weka-final\scripts" `
  RenderWeka weka-final\evidence

.\venv\Scripts\python.exe weka-final\scripts\assemble_evidence.py
```

The data preparation script first re-executes the repository's cleaning and harmonisation logic against the raw survey files. It stops if the rebuilt modelling columns do not match the checked cleaned dataset.

## Interpretation boundary

The survey is cross-sectional and self-reported, and KUET contributes 878 of the 1,108 usable records. The associations and predictions are exploratory. They do not establish that any questionnaire response causes a higher or lower CGPA. Questionnaire missing-value modes were calculated before cross-validation, so the scores should not be presented as final deployment estimates.
