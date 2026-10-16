# Model card: Bank Marketing subscription ranking

## Purpose

Rank clients by estimated probability of subscribing to a term deposit before a campaign call. Campaign capacity is
limited to the top 10% of the client list; Precision@10% is the primary selection metric.

The practical objective is to use the limited call list on the clients most likely to subscribe. Since only one client
in ten can be called, ranking quality in that top decile matters more operationally than overall accuracy; hence
Precision@10% is the primary metric.

## Data

The model is developed with `data/train.csv`. The target is `y` (`yes` is positive). `id` is retained only for
submission matching and is not a predictor. `duration` is excluded because it is unavailable before a call and absent
from `test.csv`. The pipeline receives the raw test predictors without manual preprocessing.

The training set has 32,940 rows and the test set has 8,236. There are 29,229 `no` and 3,711 `yes` outcomes
(positive share 0.1127). Train has `duration` and `y`, which test lacks: `y` supplies training labels, while
`duration` cannot be used for pre-call scoring and is excluded to avoid leakage. `id` is kept for output matching,
not as a predictor. Literal `unknown` counts are: job 275, marital 64, education 1,401, default 6,871, housing
794, and loan 794; the predictors contain zero actual NaNs. The `pdays == 999` sentinel appears in 31,694 training
rows (0.9622 share).

Duration is recorded during the call, so it is not available when deciding whom to call. It is strongly associated
with the eventual outcome: median duration is 164 seconds for `no` and 449 seconds for `yes`. Across duration
deciles from shortest to longest, the observed `yes` rate rises from 0.03% to 46.03%. This is post-outcome
information, not a usable pre-call signal.

The `unknown` values represent unavailable or unspecified client information, not invalid observations, so rows are
kept and `unknown` remains its own category. `pdays == 999` is a sentinel for no prior contact, not a real elapsed-day
measurement; it is one-hot encoded as its own category. A separate `was_contacted` flag was omitted because it
duplicates the information already represented by this category. The observed positive rate is 14.70% for
cellular and 5.26% for telephone contacts, an association in this dataset rather than evidence that channel causes
the difference. A random split can put records from similar campaign periods and macroeconomic conditions on both
sides; a future campaign may have a different channel mix, response prevalence, or economic environment, so random
holdout results may be optimistic for deployment.

## Pipeline and evaluation

Numeric features use median imputation and standard scaling. Categorical features use most-frequent imputation and
one-hot encoding; literal `unknown` values remain categories. `pdays` is treated categorically, so its 999 sentinel
is represented as its own category. All transformations are fitted inside the scikit-learn pipeline; no project
Python module is required to load the saved model. A CV comparison of retaining `unknown` as a category versus
imputing it to the most frequent category gave Precision@10% of 0.4880 ± 0.0236 and 0.4888 ± 0.0235, respectively.

To quantify why `duration` must be excluded, the same five development folds were also run as a leakage diagnostic
with `duration` added as a numeric input. Precision@10% rose from 0.4880 ± 0.0236 to 0.6086 ± 0.0202, and ROC AUC
from 0.7890 ± 0.0084 to 0.9363 ± 0.0043. These inflated figures are not candidate-model results: this feature is
unavailable at decision time and absent from test.

The notebook compares the prior majority baseline, logistic regression, decision tree, and random forest on the same
five stratified outer folds; small parameter grids are tuned inside each outer training fold. Results:

The majority-prior baseline is the reference floor for every ranking candidate, not an accuracy target: its scores are tied,
so Precision@10% is near prevalence. Search spaces: logistic regression `C ∈ {0.1, 1.0, 10.0}`, both balanced
and unweighted; balanced decision tree `max_depth ∈ {6, 10, None}` with `min_samples_leaf=5`; random forest
`n_estimators=60`, `min_samples_leaf ∈ {3, 5, 10}`, compared with and without balanced class weights. Inner tuning
uses three stratified folds; outer comparisons use five.

| Model | Precision@10% | ROC AUC | Average precision | Accuracy |
|---|---:|---:|---:|---:|
| Majority baseline (prior) | 0.1069 ± 0.0075 | 0.5000 ± 0.0000 | 0.1127 ± 0.0001 | 0.8873 ± 0.0001 |
| Logistic regression (balanced) | 0.4877 ± 0.0240 | 0.7889 ± 0.0085 | 0.4362 ± 0.0177 | 0.8288 ± 0.0109 |
| Logistic regression (no class weights) | 0.4907 ± 0.0154 | 0.7891 ± 0.0088 | 0.4389 ± 0.0167 | 0.8987 ± 0.0030 |
| Decision tree (balanced) | 0.4945 ± 0.0152 | 0.7805 ± 0.0082 | 0.4132 ± 0.0176 | 0.8394 ± 0.0229 |
| Random forest (no class weights) | 0.5040 ± 0.0166 | 0.7874 ± 0.0059 | 0.4547 ± 0.0129 | 0.8986 ± 0.0022 |
| Random forest (balanced) | 0.4994 ± 0.0188 | 0.7914 ± 0.0110 | 0.4530 ± 0.0214 | 0.8654 ± 0.0041 |

The majority baseline reaches 0.8873 accuracy by predicting the common class, but its Precision@10% is only 0.1069,
close to the positive prevalence; its tied scores cannot rank clients meaningfully. This makes its Precision@10%
the baseline floor for the other models. For logistic regression, balanced weighting gives 0.4877 ± 0.0240 versus
0.4907 ± 0.0154 without weights. The random-forest comparison is 0.4994 ± 0.0188 (balanced) versus
0.5040 ± 0.0166 (unweighted). These differences are small relative to the combined fold spreads and do not show a
clear benefit from class weighting. The highest mean is random forest without class weights; its Precision@10%
advantage over balanced logistic regression is 0.0163, below their combined fold spread of about 0.0292. There is
no robust winner on these folds. The notebook selects the simpler logistic model rather than claiming a robust
nonlinear-model win. Its selected regularization parameter is `C=1.0`.

For unknown handling, keeping `unknown` as a category and imputing it to the most frequent category give nearly
identical mean Precision@10% (0.4880 vs 0.4888), with fold spreads overlapping substantially (0.0236 and 0.0235).
Precision@10% depends only on the ordering of the top tenth, so this close result means the two treatments produced
similar top-decile rankings in these folds; it does not imply identical probabilities or features. Retaining
`unknown` is a reasonable semantic choice because unavailable information may itself carry signal.

The reserved holdout was scored once after model selection. Results: ROC AUC 0.8036, average precision 0.4884,
10%-capacity threshold 0.7851, Precision@10% 0.5341, recall 0.4744, and lift over random calls 4.7425. The confusion
matrix (true rows 0/1, predicted columns 0/1) is `[[5539, 307], [390, 352]]`. The final pipeline is refit on all
labelled training rows and saved as `model.joblib`. `predictions.csv` has columns `id, y_pred, y_score` for all
8,236 test rows; exactly 824 rows (ceil of 10% capacity) are marked `yes` by ranking the full-training model's
scores. The holdout-derived threshold is reported for holdout analysis only and is not used to form the test call
list. Of the 659 holdout calls, 307 were false-positive calls that did not result in a subscription; 390 subscribers
were not reached. The test labels are not supplied, so hidden-test ROC-AUC cannot be calculated locally.

## Reproducibility and repository contents

The notebook prints Python, NumPy, pandas, scikit-learn and joblib versions and sets `SEED = 2729`; pinned
dependencies are in `requirements.txt`. The executed notebook contains saved outputs, and the fitted artifact is
loadable in a fresh Python process without project-specific helper modules. The repository inventory includes
`zhussip_task3.ipynb`, `REPORT.md`, `requirements.txt`, `model.joblib`, `data/train.csv`, `data/test.csv`, and
`predictions.csv`. At review time, `git log` showed at least three commits (latest: `f86a491`, `9178216`,
`7d43707`), meeting the handout's minimum-history requirement. The latest notebook and regenerated root
`predictions.csv` changes still need to be included in the final submitted commit.

## Limitations, failure modes, and monitoring

The ranking may degrade if campaign timing, macroeconomic conditions, channel mix, or client composition changes; it
may also fail if input definitions or the prevalence/meaning of `unknown` values change. Validate feature schemas
and monitor feature drift, unknown rates, `pdays` sentinel share, score distributions, and the realised top-decile
call-list size. Once outcomes mature, monitor response prevalence, Precision@10%, recall, lift, ROC AUC, and average
precision by campaign period and channel; review drift before retraining or recalibrating. The supplied test set has
no labels, so its ROC AUC cannot be calculated locally.

## Defence preparation

Walk through the pipeline without reading cells: reserve a stratified holdout; fit imputation, scaling and encoding
inside CV; tune models only inside outer training folds; compare top-decile Precision@10%; select the simpler model
when gains are within fold variability; evaluate once on holdout; refit on all labelled rows; rank test rows within
the 10% call capacity. The saved pipeline accepts the unmodified `test.csv` DataFrame and ignores `id`.

The `unknown` alternative changed mean Precision@10% by 0.0008 against a combined fold spread of 0.0333. The top
random forest led balanced logistic regression by 0.0163 against a combined spread of about 0.0292. For a live
change, reducing logistic `C` from 1.0 to 0.1 should shrink coefficients but may alter the call ranking; predict
that the metric can move either way, then compare with the same CV rather than claiming improvement in advance.
Deployment failure modes include campaign/economic drift and changed input schemas or `unknown` rates; monitor those
and the `pdays` sentinel share.
