# Order within seven days

The prediction is made immediately after an order: will this user place another
order within seven days (including day seven)? Product rows are aggregated into
orders. The next observed order supplies the label through its
`days_since_prior_order` field. Same-day orders count as positive.

The last observed order for each user supplies the test outcome; the
second-to-last supplies validation. Earlier outcomes supply training. All input
features are shifted back at least one order. The outcome order's basket, date,
gap, IDs, and split indicators are excluded from predictors. First orders have
no preceding prediction origin and are excluded. Consecutive order numbers are
checked before labeling.

Features describe the preceding order's timing, basket size, reorder fraction,
aisle/department breadth, produce/dairy shares, and historical/recent behavior.
Missing historical gaps are left missing for XGBoost. User and product IDs are
not predictors. Historical features update as orders become observed, including
the validation order when forecasting the test outcome. This is a rolling
per-user evaluation, not a shared calendar cutoff or an unseen-user evaluation.

CPU histogram training uses four threads and validation log loss for early
stopping. The test set is evaluated after fitting; it is not used to select
parameters or stopping iteration. Baselines are a constant training positive
rate and the user's historical fraction of gaps at most seven days. Accuracy
uses a fixed 0.5 threshold. No threshold optimization is performed.

The data contain relative order timing, not absolute dates or a known observation
end date. Evaluation therefore covers observed next-order transitions; it does
not establish performance on permanent churn or customers with no recorded
subsequent order. The training and holdout positive rates differ substantially;
review probability calibration before operational use.

Run from the Instacart project root:

```sh
.venv/bin/python -u codex-xgboost-experiment/models/weekly_reorder/train.py
```

Outputs: `model.ubj`, `metrics.json`, `learning_curve.json`, and
`test_predictions.parquet`. `training.log` captures the original training run.
`requirements.txt` records package versions. The script re-creates aggregate
tables in memory and uses local temporary disk storage when necessary.

To load the model:

```python
import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model('models/weekly_reorder/model.ubj')
# X must contain historical features in the order listed in metrics.json.
probability = model.predict_proba(X)[:, 1]
```
