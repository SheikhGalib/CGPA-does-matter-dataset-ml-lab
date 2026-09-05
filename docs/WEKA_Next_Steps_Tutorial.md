| Student Performance                       |          | Classification |           |      |                |          |       | WEKA Tutorial |
| ----------------------------------------- | -------- | -------------- | --------- | ---- | -------------- | -------- | ----- | ------------- |
| What                                      |          | to             | Do        | Next | in             | WEKA     |       |               |
| A short                                   | tutorial | for            | comparing | CGPA | classification | models   |       |               |
| Files to                                  | use      |                |           |      |                |          |       |               |
| • student_performance_primary_clean.csv:  |          |                |           |      | includes       | previous | SGPA. |               |
| • student_performance_behavior_clean.csv: |          |                |           |      | excludes       | previous | SGPA. |               |
Work with one file at a time. Do not combine their results into the same model ranking.
1
|     | Open |     | the primary | dataset |     |     |     |     |
| --- | ---- | --- | ----------- | ------- | --- | --- | --- | --- |
Open WEKA GUI Chooser, select Explorer, and go to the Preprocess tab. Click Open file and load the primary
CSV.
Check that WEKA shows 878 instances. In the class selector, choose current_cgpa_band. It should be the last
column.
2
|     | Save | an  | ARFF | copy |     |     |     |     |
| --- | ---- | --- | ---- | ---- | --- | --- | --- | --- |
In the Preprocess tab, click Save and save the loaded data as an ARFF file. Use a clear name such as primary.arff.
| ARFF preserves |     | WEKA’s | attribute | types and | is easier to | reuse. |     |     |
| -------------- | --- | ------ | --------- | --------- | ------------ | ------ | --- | --- |
3
|          | Set      | the  | evaluation | method        |         |     |     |     |
| -------- | -------- | ---- | ---------- | ------------- | ------- | --- | --- | --- |
| Open the | Classify | tab. | Under      | Test options, | select: |     |     |     |
• Cross-validation
• 10 folds
| • Random | seed | 1   |     |     |     |     |     |     |
| -------- | ---- | --- | --- | --- | --- | --- | --- | --- |
OpenMoreoptionsandenabledetailedper-classstatistics,theconfusionmatrix,andpredictionoutputwhenavailable.
| Note: Never | use | “Use training |     | set” for the accuracy | reported | in the project. |     |     |
| ----------- | --- | ------------- | --- | --------------------- | -------- | --------------- | --- | --- |
4
|     | Run | the | baseline | models |     |     |     |     |
| --- | --- | --- | -------- | ------ | --- | --- | --- | --- |
First run ZeroR. This gives the majority-class baseline. Then run OneR as a simple one-feature baseline. Save each
result from the result list using names such as Primary-ZeroR and Primary-OneR.
5
|         | Run       | the         | main  | classifiers |                           |     |     |     |
| ------- | --------- | ----------- | ----- | ----------- | ------------------------- | --- | --- | --- |
| Run the | following | classifiers | using | the same    | 10-fold cross-validation: |     |     |     |
?
| • J48:      | Trees     | J48 |            |     |     |     |     |     |
| ----------- | --------- | --- | ---------- | --- | --- | --- | --- | --- |
| • Logistic: | Functions |     | ? Logistic |     |     |     |     |     |
?
| • RandomForest: |           | Trees |     | RandomForest |     |     |     |     |
| --------------- | --------- | ----- | --- | ------------ | --- | --- | --- | --- |
| • SMO:          | Functions | ?     | SMO |              |     |     |     |     |
?
Start with their default settings. If a classifier has trouble with missing values, choose Meta FilteredClassifier, set
its filter to ReplaceMissingValues, and set the required algorithm as the base classifier.
1

| Student | Performance | Classification |     |     |     |     |     |     |     | WEKA Tutorial |
| ------- | ----------- | -------------- | --- | --- | --- | --- | --- | --- | --- | ------------- |
6
|            |            | Record the | results |            |        |      |            |         |       |      |
| ---------- | ---------- | ---------- | ------- | ---------- | ------ | ---- | ---------- | ------- | ----- | ---- |
| Create one | comparison | table      | and     | copy these | values | from | every WEKA | result: |       |      |
| Model      |            | Accuracy   |         | Macro-F1   |        |      | Weighted   | F1      | Kappa | Time |
ZeroR
OneR
J48
Logistic
RandomForest
SMO
WEKA shows the F1-score for each class. Calculate macro-F1 by averaging the four class F1-scores. Do not confuse
| this with | WEKA’s | weighted | average | F1. |     |     |     |     |     |     |
| --------- | ------ | -------- | ------- | --- | --- | --- | --- | --- | --- | --- |
7
|     |     | Repeat with | the | behaviour |     | dataset |     |     |     |     |
| --- | --- | ----------- | --- | --------- | --- | ------- | --- | --- | --- | --- |
Return to the Preprocess tab and load student_performance_behavior_clean.csv. Confirm that the class is again
| current_cgpa_band, |     | save | an ARFF | copy, and | repeat | Steps | 3–6. |     |     |     |
| ------------------ | --- | ---- | ------- | --------- | ------ | ----- | ---- | --- | --- | --- |
MakeaseparateresultstabletitledBehaviour-focused model results. Thisexperimentshowshowwellthenon-SGPA
| variables | work | by themselves. |     |     |     |     |     |     |     |     |
| --------- | ---- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
8
|     |     | Choose the | best | model |     |     |     |     |     |     |
| --- | --- | ---------- | ---- | ----- | --- | --- | --- | --- | --- | --- |
Prefer the model with the strongest macro-F1, then check its kappa, accuracy, and confusion matrix. A good model
| should beat | ZeroR | and should | not | perform well | for | only one | CGPA | category. |     |     |
| ----------- | ----- | ---------- | --- | ------------ | --- | -------- | ---- | --------- | --- | --- |
If two models are very close, prefer the easier model to explain, such as J48 or Logistic. A weak behaviour-model result
| is still a | valid | finding. |     |     |     |     |     |     |     |     |
| ---------- | ----- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
9
|     |     | Examine | the errors |     |     |     |     |     |     |     |
| --- | --- | ------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
For the selected model, save the cross-validation predictions. Convert the class labels to numbers:
|          |            |            |     |            | Class | C0              | C1 C2 | C3     |     |     |
| -------- | ---------- | ---------- | --- | ---------- | ----- | --------------- | ----- | ------ | --- | --- |
|          |            |            |     |            | Index | 0               | 1     | 2 3    |     |     |
| For each | prediction | calculate: |     |            |       |                 |       |        |     |     |
|          |            |            |     | gap=actual |       | index?predicted |       | index. |     |     |
A gap of 0 is correct, |gap|=1 is a neighbouring-category error, and |gap|?2 is a larger error. Summarize the counts
| instead | of making | personal | judgments | about | individual | respondents. |     |     |     |     |
| ------- | --------- | -------- | --------- | ----- | ---------- | ------------ | --- | --- | --- | --- |
10
|                |                | Finish the      | report       |                  |                        |           |       |     |     |     |
| -------------- | -------------- | --------------- | ------------ | ---------------- | ---------------------- | --------- | ----- | --- | --- | --- |
| Your final     | report         | should contain: |              |                  |                        |           |       |     |     |     |
| • The          | two experiment | definitions:    |              | primary          | and behaviour-focused. |           |       |     |     |     |
| • A table      | comparing      | all classifiers |              | against ZeroR    |                        | and OneR. |       |     |     |     |
| • The          | confusion      | matrix of       | the selected | model.           |                        |           |       |     |     |     |
| • A short      | explanation    | of              | the most     | common           | errors.                |           |       |     |     |     |
| • A comparison |                | of performance  |              | with and without |                        | previous  | SGPA. |     |     |     |
• Limitations: self-reported data, Semester 7 concentration, and no causal claims.
Note: Do not change models repeatedly just to obtain a higher score. Use the same folds and settings for a fair comparison,
| and report | weak | or negative | findings | honestly. |     |     |     |     |     |     |
| ---------- | ---- | ----------- | -------- | --------- | --- | --- | --- | --- | --- | --- |
TheexactpositionofabuttonmayvaryslightlybetweenWEKAversions,butthetabandclassifiernamesshouldremainsimilar.
2
