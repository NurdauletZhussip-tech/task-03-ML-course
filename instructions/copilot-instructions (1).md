# Copilot instructions — MOP 3231, Task 3 (Midterm Project, Bank Marketing)

The full handout is in `docs/TASK3.md` (paste the Canvas Week 07 text there). Read the section a request names
and follow it literally. If these instructions and my message disagree, stop and ask me.
RULE ZERO: if `docs/TASK3.md` is missing or empty, say so and stop. Do not guess the metric, the output format or the model-saving format.

## How we work
- Chat with me in **Russian**. Code, comments, docstrings, notebook markdown are in **English**.
- ONE sub-step per request. Before code, restate in 2–4 lines what the step requires. Then write one notebook cell.
- Never invent numbers or conclusions. Numbers come only from outputs I ran and pasted.
- Interpretations, model choice justification and the defence notes are **mine**: leave a markdown `TODO (student): ...`
  with the facts to cover. You may only check my text against my numbers.
- After the code, tell me in 1–2 lines how I can verify the result.

## Fixed facts about the data (checked)
- `data/train.csv`: 32,940 rows, columns `id`, 19 features, `duration`, target `y` (`yes`/`no`). `data/test.csv`: 8,236 rows, `id` + 19 features; **no `y`, no `duration`**.
- Positive class `yes` = 11.3% (imbalanced). Never use plain accuracy as the main metric. Use the metric named in `docs/TASK3.md`.
- Map target explicitly: `y_bin = (y == "yes").astype(int)`. Positive class = 1 (`yes`).
- `id` is an identifier, never a feature. Keep it only to build the submission file.
- `duration` exists only in train. A model that needs it cannot score `test.csv`, so **drop `duration` from the features**
  (call length is also unknown before the call). Do not use it, not even "for comparison", unless the handout says so.
- Missing values are encoded as the string `unknown` (job, marital, education, default, housing, loan). Treat `unknown` as its own category. Do not drop those rows.
- `pdays == 999` means "never contacted before" (96% of rows). Do not treat 999 as a number: add a flag `was_contacted = (pdays != 999)` and handle the value inside the pipeline.
- Social-economic columns (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`) are strongly correlated with each other and with `month`. Mention it, do not silently drop them.

## Hard rules
- `SEED` is defined once (my student-ID value, same as Task 2 unless the handout says otherwise) and used for every `random_state`. Never 42.
- Library versions are **pinned** in `requirements.txt` (numpy 2.3.4, pandas 2.3.3, scikit-learn 1.7.2, joblib 1.5.2, matplotlib 3.10.7, scipy 1.16.3). Do not change them and do not add other libraries (no xgboost, lightgbm, imblearn).
- The test file is scored exactly once, at the end. Never fit or tune anything on `test.csv` (it has no labels anyway).
- All preprocessing (imputation, encoding, scaling, feature flags) is **inside one sklearn `Pipeline` / `ColumnTransformer`** and is fitted inside each CV fold. No `fit`/`fit_transform` on the full train before cross-validation.
- Validation: stratified split or `StratifiedKFold(5, shuffle=True, random_state=SEED)`. Keep a held-out validation part for the final honest estimate if the handout asks for it.
- Choose the decision threshold on validation/CV predictions, never on test. State the threshold in the notebook.
- Class imbalance: first try `class_weight="balanced"` and threshold tuning. Resampling libraries are not allowed.
- Do not suppress warnings globally.
- The saved model loads with exactly the pinned versions: save the **whole pipeline** (preprocessing + model) with `joblib.dump`, in the file name and place the handout requires. After saving, reload it in a fresh cell and check that its predictions equal the in-memory ones.
- The output file for the test set (name, columns, label format) is taken from `docs/TASK3.md` word for word.

## Do NOT
- use `duration`, `id` or any column absent from `test.csv` as a feature;
- one-hot encode inside a manual loop outside the pipeline, or fit encoders on train+test together;
- report only accuracy, or only a score from a single split without a spread;
- tune on the test predictions, or select a model by looking at the leaderboard-like test output;
- rename the files or functions the handout prescribes; commit the virtual environment;
- edit `data/*.csv` or `requirements.txt`.

## Done means
The cell runs without errors, output is visible, the notebook passes **Restart & Run All**,
and the saved model + predictions file are reproducible from the notebook alone.
