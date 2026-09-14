# Final WEKA Steps — How to Get Every Result in the Presentation

This is the short, practical guide for checking our final results in WEKA and taking screenshots for the slides.

## 1. The files

| File | What it is |
|---|---|
| `cleaned-dataset/ours/final/final_920.arff` | **Load this one in WEKA.** 920 students, 12 questions, recent SGPA, CGPA group (class). |
| `cleaned-dataset/ours/final/final_920.csv` | The same 920 rows as a CSV, for Excel or a quick look. |
| `cleaned-dataset/ours/final/final_920_with_details.csv` | The same rows plus university, department, semester, submission date and the answer text. |
| `cleaned-dataset/ours/final/answer_codes.csv` | What each number 0–4 means for every question. |
| `cleaned-dataset/ours/final/dropped_182_rows.csv` | The 182 answers we removed, with university, semester, date and band gap. |

Use the **ARFF** for WEKA. A CSV makes WEKA guess the order of the groups (C0, C1, C2, C3), which can swap the rows of the confusion matrix and change some results slightly.

### What is in the file

| Column | Values |
|---|---|
| 12 question columns (`desired_department_match` … `routine_manageability`) | numbers 0–4 (0 = lowest answer, 4 = highest) |
| `recent_sgpa_band` | C0, C1, C2, C3 |
| `cgpa_band` (class, last column) | C0 = below 3.20, C1 = 3.20–3.49, C2 = 3.50–3.74, C3 = 3.75 and above |

The result satisfaction question is not in the file. We do not use it because it depends on each student's own goals.

### How we got 920 rows

1. 1,126 raw responses were cleaned to 1,108 rows (spellings fixed, two forms merged, 18 rows without CGPA removed).
2. 6 rows without a real recent SGPA were removed, which left 1,102.
3. 455 answers came from KUET semester 7 on 6–7 August. In 182 of them the recent SGPA and CGPA were 2 or 3 groups apart, so we removed those 182 and kept the other 273.
4. That leaves **920 rows**.

## 2. Settings used for every run

| Setting | Value |
|---|---|
| WEKA version | 3.8.7 |
| Main test | **Cross-validation, 5 folds** |
| Backup test (hidden slides) | **Percentage split 80%** |
| Random seed (More options…) | **1** |
| J48 | `minNumObj` = **40** (everything else default) |
| RandomForest, SMO, Logistic, NaiveBayes, OneR, ZeroR | default settings |

## 3. Step by step in the WEKA Explorer

### 3.1 Load the data

1. Open WEKA → **Explorer**.
2. **Preprocess** tab → **Open file…** → choose `final_920.arff`.
3. Check the box on the left says **Instances: 920** and **Attributes: 14**.

### 3.2 Main CGPA prediction (12 questions + recent SGPA → CGPA group)

1. Go to the **Classify** tab.
2. Under **Test options**, choose **Cross-validation** and set **Folds = 5**.
3. Click **More options…** and make sure **Random seed for XVal / % Split = 1**. Click OK.
4. In the drop-down above **Start**, choose **(Nom) cgpa_band**.
5. Click **Choose** and pick the classifier:
   - `rules → ZeroR`
   - `trees → J48`, then click the name and set **minNumObj = 40**, click OK
   - `trees → RandomForest`
   - `functions → SMO`
   - `functions → Logistic`
   - `bayes → NaiveBayes`
6. Click **Start**.
7. For a screenshot, scroll the **Classifier output** so the **Summary** and the **Confusion Matrix** are both visible.
8. To save the text: right-click the run in **Result list** → **Save result buffer**.

### 3.3 CGPA from the questions only (no recent SGPA)

1. In **Preprocess**, tick `recent_sgpa_band` and click **Remove**.
2. Go back to **Classify**, keep **Cross-validation, 5 folds, seed 1**, class **(Nom) cgpa_band**, and run the same classifiers.
3. Afterwards click **Undo** in Preprocess (or reload the file) to get `recent_sgpa_band` back.

### 3.4 Recent SGPA from the questions

1. Reload `final_920.arff`.
2. In **Preprocess**, tick `cgpa_band` and click **Remove**.
3. In **Classify**, choose the class **(Nom) recent_sgpa_band**, keep **5 folds, seed 1**, and run the classifiers.

### 3.5 The 80/20 train/test split (hidden backup slides)

Same steps as above, but under **Test options** choose **Percentage split** and set **80 %**. Keep the seed at 1, and leave **Preserve order for % Split** unticked. WEKA then trains on 736 students and tests on 184.

### 3.6 The J48 tree picture

After a J48 run, right-click the run in **Result list** → **Visualize tree**. The tree WEKA shows is built on all 920 students. Right-click inside the tree window → **Fit to screen**, then take the screenshot.

## 4. What you should see

### 4.1 Accuracy, 5-fold cross-validation (main slides)

| Model | Recent SGPA from questions | CGPA from questions | **Main CGPA** (questions + recent SGPA) |
|---|---|---|---|
| ZeroR | 29.24% | 30.87% | 30.87% |
| J48 (minNumObj 40) | 26.30% | 27.83% | 54.35% |
| RandomForest | 32.39% | 30.54% | 54.24% |
| SMO | 27.93% | 28.15% | **55.22%** |
| Logistic | 29.02% | 30.11% | 54.67% |
| NaiveBayes | 30.00% | 31.20% | 54.46% |

### 4.2 Accuracy, 80/20 split (hidden slides)

| Model | Recent SGPA from questions | CGPA from questions | **Main CGPA** |
|---|---|---|---|
| ZeroR | 30.43% | 35.87% | 35.87% |
| J48 (minNumObj 40) | 27.17% | 33.15% | 51.09% |
| RandomForest | 33.15% | 26.63% | 45.65% |
| SMO | 25.54% | 29.35% | 51.09% |
| Logistic | 28.80% | 27.72% | 50.00% |
| NaiveBayes | 26.63% | 28.80% | 52.17% |

The 80/20 test set has only 184 students, so these numbers move around more. The 5-fold results are the main ones.

### 4.3 Confusion matrices for main CGPA, 5-fold (rows = actual group, columns = predicted group)

**SMO — 55.22%**
```
  a   b   c   d   <-- classified as
 102  59  11  10 |   a = C0
  49 146  75  14 |   b = C1
   8  59 131  66 |   c = C2
  12   5  44 129 |   d = C3
```

**J48 (minNumObj 40) — 54.35%**
```
 102  59  11  10
  49 155  66  14
   8  72 119  65
  12  14  40 124
```

**RandomForest — 54.24%**
```
 101  53  20   8
  51 148  72  13
  17  59 132  56
  13  16  43 118
```

**Logistic — 54.67%**
```
 100  60  14   8
  49 151  71  13
   9  62 130  63
  12  10  46 122
```

**NaiveBayes — 54.46%**
```
 102  57  15   8
  54 143  74  13
  12  62 134  56
  11   9  48 122
```

**ZeroR — 30.87%** (every student is predicted as C1)
```
 0 182 0 0
 0 284 0 0
 0 264 0 0
 0 190 0 0
```

### 4.4 Confusion matrix for main CGPA J48, 80/20 split — 51.09%

```
 15 11  4  5
 10 35 20  1
  1 10 18 17
  2  0  9 26
```

### 4.5 The main CGPA J48 tree

```
recent_sgpa_band = C0: C0 (171.0/69.0)
recent_sgpa_band = C1: C1 (269.0/123.0)
recent_sgpa_band = C2: C2 (261.0/130.0)
recent_sgpa_band = C3: C3 (219.0/90.0)
```

With at least 40 students per leaf, J48 keeps only recent SGPA: one rule for each group.

If a number is different, check three things first: folds = 5, seed = 1, and J48 minNumObj = 40.

## 5. Where the screenshots go in the presentation

Slide numbers are for the updated `presentation/What_Shapes_Student_CGPA_Ready.pptx`.

| Slide | What to paste |
|---|---|
| 61 — "We Also Ran the Models in WEKA" | Left: RandomForest, main CGPA, 5-fold output. Right: SMO, main CGPA, 5-fold output. The current pictures show WEKA's output text. Replace them with your Explorer screenshots. |
| 75 (hidden) — "WEKA Output: Random Forest Runs" | Left: RandomForest, main CGPA, 5-fold. Right: RandomForest, CGPA from questions only, 5-fold. |
| 43, 45, 47 — J48 trees | Optional: WEKA **Visualize tree** screenshots for the SGPA, questionnaire CGPA and main CGPA trees. The current pictures are drawn from WEKA's J48 output. |

## 6. Where each number comes from

| Result | File |
|---|---|
| Every WEKA run shown in the slides | `docs/weka_runs/final/` (one text file per run; the command is on line 2) |
| All numbers in one place | `docs/final_results.json` |
| Script that makes the dataset and runs WEKA | `notebooks/final_weka_results.py` |
| Scripts that draw the charts and update the slides | `presentation/final_update/make_figures.py`, `presentation/final_update/update_deck.py` |
| The deck before this update | `presentation/What_Shapes_Student_CGPA_Ready_BEFORE_FINAL_UPDATE.pptx` |

To rebuild everything:

```powershell
.\venv\Scripts\python.exe notebooks\final_weka_results.py
.\venv\Scripts\python.exe presentation\final_update\make_figures.py
.\venv\Scripts\python.exe presentation\final_update\update_deck.py
```

The last script always starts from the backup copy. Any manual edits you make to the presentation will be lost if you run it again.
