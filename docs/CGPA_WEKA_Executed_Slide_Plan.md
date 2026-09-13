# What Shapes Student CGPA? — executed WEKA slide plan

50 visible slides plus hidden technical backup. Form screenshots and their left-plus-four-right composition reuse slide 6 of the user-supplied 64-slide presentation. The Simple Light Mode template supplies the master and slide chrome; all editable text uses Times New Roman for Microsoft PowerPoint 2015 compatibility.

| Slide | Title | Visibility | Source |
|---:|---|---|---|
| 1 | What Shapes Student CGPA? | Visible | Project title supplied in CGPA_Final_Clean_Slide_Plan_WEKA_Only.md |
| 2 | What are we trying to understand? | Visible |  |
| 3 | What did the form look like? | Visible | Reference: user-supplied What_Shapes_Student_CGPA_Final_64_Slides_Ready.pptx, slide 6. Exact embedded screenshot assets reused. |
| 4 | KUET form responses | Visible | Source: teacher_focused_final/assets/dataset_screenshots/raw_responses_preview.png, rendered raw KUET CSV preview. |
| 5 | Total responses | Visible |  |
| 6 | Response categories | Visible |  |
| 7 | Form responses: academic questions | Visible |  |
| 8 | Form responses: understanding and SGPA | Visible |  |
| 9 | Form responses: lifestyle | Visible |  |
| 10 | Form responses: stress and responsibilities | Visible |  |
| 11 | Form responses: motivation and environment | Visible |  |
| 12 | Universities in the dataset | Visible |  |
| 13 | Predicted CGPA classes | Visible |  |
| 14 | CGPA distribution | Visible |  |
| 15 | Recent SGPA distribution | Visible |  |
| 16 | Attendance and CGPA | Visible |  |
| 17 | Weekly study time and CGPA | Visible |  |
| 18 | Study style and CGPA | Visible |  |
| 19 | Topic clarity and CGPA | Visible |  |
| 20 | Sleep and CGPA | Visible |  |
| 21 | Stress and CGPA | Visible |  |
| 22 | Phone distraction and CGPA | Visible |  |
| 23 | Outside commitments and CGPA | Visible |  |
| 24 | Desired department and CGPA | Visible |  |
| 25 | Support and CGPA | Visible |  |
| 26 | Study environment and CGPA | Visible |  |
| 27 | Routine manageability and CGPA | Visible |  |
| 28 | Questionnaire-CGPA relationships | Visible | Source: evidence/data.json, Spearman rank correlations on 1,102 rows with observed SGPA. Signs follow the response order in backup. |
| 29 | Do SGPA and CGPA show the same pattern? | Visible | Source: evidence/data.json, paired Spearman correlations on identical N=1,102. |
| 30 | WEKA target-specific model results | Visible | Sources: sgpa_J48_cv.txt, sgpa_RandomForest_cv.txt, sgpa_SMO_cv.txt, sgpa_Logistic_cv.txt, sgpa_NaiveBayes_cv.txt, sgpa_ZeroR_cv.txt; questionnaire_J48_cv.txt, questionnaire_RandomForest_cv.txt, questionnaire_SMO_cv.txt, questionnaire_Logistic_cv.txt, questionnaire_NaiveBayes_cv.txt, questionnaire_ZeroR_cv.txt; main_J48_cv.txt, main_RandomForest_cv.txt, main_SMO_cv.txt, main_Logistic_cv.txt, main_NaiveBayes_cv.txt, main_ZeroR_cv.txt |
| 31 | Train / test split | Visible | Source: evidence/split_indices.csv and data_provenance.json. CGPA-stratified existing split, seed 42, shared across all targets. |
| 32 | WEKA J48: Recent SGPA | Visible | Sources: sgpa_J48_holdout.txt, sgpa_tree.dot, sgpa_weka_viewer.png. Exact WEKA TreeVisualizer export. Numeric option codes are listed in backup. Leaf (N/errors) denotes training support and training errors. |
| 33 | WEKA J48: SGPA rules | Visible | Exact rules: sgpa_tree.txt and results.json. These are predictions from tree branches, not causal statements. |
| 34 | WEKA J48: CGPA from questionnaire variables | Visible | Sources: questionnaire_J48_holdout.txt, questionnaire_tree.dot, questionnaire_weka_viewer.png. Exact WEKA TreeVisualizer export. Numeric option codes are listed in backup. Leaf (N/errors) denotes training support and training errors. |
| 35 | Questionnaire J48 rules | Visible | Exact rules: questionnaire_tree.txt and results.json. These are predictions from tree branches, not causal statements. |
| 36 | WEKA J48: Main CGPA prediction | Visible | Sources: main_J48_holdout.txt, main_tree.dot, main_weka_viewer.png. Exact WEKA TreeVisualizer export. Numeric option codes are listed in backup. Leaf (N/errors) denotes training support and training errors. |
| 37 | Main CGPA J48 rules | Visible | Exact rules: main_tree.txt and results.json. These are predictions from tree branches, not causal statements. |
| 38 | WEKA J48 training and test accuracy | Visible | Sources: sgpa_J48_holdout.txt, questionnaire_J48_holdout.txt, main_J48_holdout.txt. All J48 models C=.25 M=40. |
| 39 | WEKA J48: main CGPA test confusion matrix | Visible | Source: main_J48_holdout.txt, TEST section, 221 held-out records. |
| 40 | WEKA Random Forest and SMO matrices | Visible | Sources: main_RandomForest_cv.txt, main_SMO_cv.txt. |
| 41 | WEKA Logistic and Naive Bayes matrices | Visible | Sources: main_Logistic_cv.txt, main_NaiveBayes_cv.txt. |
| 42 | WEKA ZeroR baseline matrix | Visible | Source: main_ZeroR_cv.txt. |
| 43 | WEKA main CGPA accuracy: 10-fold CV | Visible | Sources: main_J48_cv.txt, main_RandomForest_cv.txt, main_SMO_cv.txt, main_Logistic_cv.txt, main_NaiveBayes_cv.txt, main_ZeroR_cv.txt |
| 44 | Why keep a decision tree? | Visible |  |
| 45 | Previous studies and models | Visible | Ahmed (2024): https://doi.org/10.1155/2024/4067721 Maruf et al. ICCIT 2024: https://doi.org/10.1109/ICCIT64611.2024.11021990 Albonny & Duru (2026): https://dergipark.org.tr/tr/pub/ijonfest/article/1879280 |
| 46 | Previous work and our WEKA models | Visible | Ahmed (2024): https://doi.org/10.1155/2024/4067721 Maruf et al. ICCIT 2024: https://doi.org/10.1109/ICCIT64611.2024.11021990 Albonny & Duru (2026): https://dergipark.org.tr/tr/pub/ijonfest/article/1879280 Project results: main_J48_cv.txt, main_RandomForest_cv.txt, main_SMO_cv.txt, main_Logistic_cv.txt, main_NaiveBayes_cv.txt, main_ZeroR_cv.txt |
| 47 | What did we learn? | Visible |  |
| 48 | Limitations | Visible | Sources: raw questionnaire, cleaning_verification.log, data_provenance.json. Questionnaire source-mode imputation used full dataset before CV. |
| 49 | Conclusion | Visible |  |
| 50 | Thank you | Visible | Team names and rolls retained from existing deck source. |
| 51 | Backup A: questionnaire options 1 | Hidden backup |  |
| 52 | Backup A: questionnaire options 2 | Hidden backup |  |
| 53 | Backup A: questionnaire options 3 | Hidden backup |  |
| 54 | Backup A: remaining form fields | Hidden backup | Source: raw CSV question headers and data_dictionary.csv. Fields below are excluded from the 12-input questionnaire models. |
| 55 | Backup B: cleaning and exclusions | Hidden backup | Source: cleaning_verification.log and data_provenance.json. |
| 56 | Backup C: exact WEKA settings | Hidden backup | J48: -C 0.25 -M 2 Random Forest: -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 SMO: -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007" -calibrator "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4" Logistic: -R 1.0E-8 -M -1 -num-decimal-places 4 Naive Bayes:  ZeroR:  |
| 57 | Backup D: SGPA full J48 tree | Hidden backup | sgpa_tree.txt. Exact text from WEKA. |
| 58 | Backup E: questionnaire CGPA full J48 tree | Hidden backup | questionnaire_tree.txt. Exact text from WEKA. |
| 59 | Backup F: main CGPA full J48 tree | Hidden backup | main_tree.txt. Exact text from WEKA. |
| 60 | Backup G: sgpa WEKA CV metrics | Hidden backup | Sources: sgpa_J48_cv.txt, sgpa_RandomForest_cv.txt, sgpa_SMO_cv.txt, sgpa_Logistic_cv.txt, sgpa_NaiveBayes_cv.txt, sgpa_ZeroR_cv.txt. Macro F1 from WEKA confusion matrices. |
| 61 | Backup G: questionnaire WEKA CV metrics | Hidden backup | Sources: questionnaire_J48_cv.txt, questionnaire_RandomForest_cv.txt, questionnaire_SMO_cv.txt, questionnaire_Logistic_cv.txt, questionnaire_NaiveBayes_cv.txt, questionnaire_ZeroR_cv.txt. Macro F1 from WEKA confusion matrices. |
| 62 | Backup G: main WEKA CV metrics | Hidden backup | Sources: main_J48_cv.txt, main_RandomForest_cv.txt, main_SMO_cv.txt, main_Logistic_cv.txt, main_NaiveBayes_cv.txt, main_ZeroR_cv.txt. Macro F1 from WEKA confusion matrices. |
| 63 | Backup H: WEKA sgpa output | Hidden backup | sgpa_weka_output.png renders the verbatim WEKA test log in a Swing text component. It is not a screenshot of the Explorer window. |
| 64 | Backup H: WEKA questionnaire output | Hidden backup | questionnaire_weka_output.png renders the verbatim WEKA test log in a Swing text component. It is not a screenshot of the Explorer window. |
| 65 | Backup H: WEKA main output | Hidden backup | main_weka_output.png renders the verbatim WEKA test log in a Swing text component. It is not a screenshot of the Explorer window. |
| 66 | Backup I: team contribution | Hidden backup | Source: existing teacher_focused_final/scripts/build_deck.py. Roles were marked draft in the existing source. |
