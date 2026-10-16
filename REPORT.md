# Bank Marketing: subscription ranking

## Objective and data

The bank can call only 10% of the clients, so the goal is to rank clients by their likelihood of subscribing and
prioritise the highest-scoring tenth. The primary metric is Precision@10%: the share of subscribers in that call
list. Overall accuracy is secondary because the target is imbalanced.

`data/train.csv` has 32,940 rows and `data/test.csv` has 8,236. A stratified 80/20 split is made immediately after
loading the labelled training file, before supervised EDA, preprocessing comparisons or model selection. The
development part has 26,352 rows: 23,383 `no` and 2,969 `yes` (11.27% positive); the 6,588-row holdout is reserved
until final evaluation. Train has `y` and `duration`, while test has neither: `y` is the label, and `duration` is
only known after a call. The model therefore excludes both from predictors; `id` is used only to match submission
rows.

EDA below uses development data only. Its predictors have no actual NaN values. Literal `unknown` counts are job
216, marital 54, education 1,138, default 5,477, housing 625 and loan 625. `pdays == 999` occurs in 25,367 of
26,352 development rows (96.26%); this sentinel means no previous contact and is treated as a category.

The observed IDs increase monotonically within each file (train 0–32,939; test 32,940–41,175), but `id` is a
synthetic identifier, not a date. There are no date/time/year columns, so row order cannot be interpreted as
chronology or used for a train-on-past/test-on-future split. The train/test schema difference is `y` and
`duration`, both absent from test; no hidden test labels or test-based model selection are used.

In development, cellular's observed subscription rate is 14.64%, versus 5.38% for telephone (association, not
causation). Duration is strongly associated with outcome, but is post-call information: median duration is 163
seconds for `no` and 450 seconds for `yes`; the observed yes rate ranges from 0.04% in the shortest duration decile
to 46.15% in the longest. As a leakage diagnostic only, adding duration to the same development folds raises Precision@10%
from 0.4880 ± 0.0236 to 0.6086 ± 0.0202 and ROC AUC from 0.7890 ± 0.0084 to 0.9363 ± 0.0043. These inflated
scores are not candidate results: duration is unavailable at decision time and absent from test.

## Preprocessing and model selection

All fitted preprocessing is inside the sklearn pipeline and therefore learned within each training fold. Numeric
columns use median imputation and standard scaling; categorical columns use most-frequent imputation and one-hot
encoding. Literal `unknown` is retained as a category. In five stratified development folds, treating `unknown`
as missing and imputing the mode gives Precision@10% 0.4888 ± 0.0235 versus 0.4880 ± 0.0236 when retained;
the 0.0008 difference is small relative to fold variability. `pdays` is categorical, preserving 999 as its own
one-hot category; no separate contact flag is used because it duplicates that signal. The saved pipeline accepts
the raw test DataFrame and ignores `id`.

Candidate comparisons use the same five stratified outer folds; hyperparameters are searched only by three-fold CV
inside each outer training fold. The prior majority classifier is the ranking baseline and minimum bar for every
candidate. Its high accuracy (0.8873) is misleading: Precision@10% is only 0.1069, near prevalence, and tied scores
cannot meaningfully rank clients.

| Model | Precision@10% | ROC AUC | Average precision | Accuracy |
|---|---:|---:|---:|---:|
| Majority prior baseline | 0.1069 ± 0.0075 | 0.5000 ± 0.0000 | 0.1127 ± 0.0001 | 0.8873 ± 0.0001 |
| Logistic regression, balanced | 0.4877 ± 0.0240 | 0.7889 ± 0.0085 | 0.4362 ± 0.0177 | 0.8288 ± 0.0109 |
| Logistic regression, unweighted | 0.4907 ± 0.0154 | 0.7891 ± 0.0088 | 0.4389 ± 0.0167 | 0.8987 ± 0.0030 |
| Decision tree, balanced | 0.4945 ± 0.0152 | 0.7805 ± 0.0082 | 0.4132 ± 0.0176 | 0.8394 ± 0.0229 |
| Random forest, unweighted | 0.5040 ± 0.0166 | 0.7874 ± 0.0059 | 0.4547 ± 0.0129 | 0.8986 ± 0.0022 |
| Random forest, balanced | 0.4994 ± 0.0188 | 0.7914 ± 0.0110 | 0.4530 ± 0.0214 | 0.8654 ± 0.0041 |

Search spaces: logistic regression `C ∈ {0.1, 1, 10}` with and without class weights; balanced decision tree
`max_depth ∈ {6, 10, None}`, `min_samples_leaf=5`; random forest with 60 trees and
`min_samples_leaf ∈ {3, 5, 10}`, both balanced and unweighted. Weighting does not show a clear benefit:
logistic balanced versus unweighted is 0.4877 versus 0.4907; random forest is 0.4994 versus 0.5040. The best
mean is the unweighted forest, but its 0.0163 Precision@10% gain over balanced logistic regression is smaller
than the combined fold spread (~0.0292). There is no robust winner; the simpler balanced logistic model
(`C=1.0`) is selected.

## Final evaluation and submission

After model choices, the reserved 6,588-row holdout was evaluated once. It contains 5,846 `no` and 742 `yes`.
ROC AUC is 0.8036, average precision 0.4884,
Precision@10% 0.5341, recall 0.4744 and lift over random calling 4.7425. The 10% holdout rule selected 659
clients. Of these, 352 subscribed and 307 were false-positive calls; a further 390 subscribers were not reached.
The random stratified split may flatter future performance because train and holdout can share campaign periods
and economic conditions; a future campaign may differ.

The final pipeline is refit on all labelled training data and saved as `model.joblib`. In a fresh Python process,
it loads and scores the raw `data/test.csv` DataFrame. Root `predictions.csv` has columns `id, y_pred, y_score`,
all 8,236 test IDs, model-matching scores and exactly 824 positive call labels (ceil of 10% capacity). Hidden-test
AUC cannot be calculated locally because test labels are withheld.

## Reproducibility and limitations

The notebook prints Python and library versions, sets `SEED = 2729`, and runs top-to-bottom with saved outputs.
`requirements.txt` pins dependencies. The repository includes the notebook, report, requirements, model, train/test
data and root submission file; git history contains at least three commits. The latest notebook and submission
updates must be included in the final submitted commit.

Performance may degrade under campaign/economic drift, changed client mix, or changes to the input schema and
`unknown` rates. Monitor those distributions, the `pdays` sentinel share, score distribution and realised call-list
size; once outcomes arrive, track Precision@10%, lift, ROC AUC and average precision over time.
