"""Predict next-order arrival within seven days using prior-order information only."""
from pathlib import Path
import json
import time
import duckdb
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, accuracy_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
start = time.time()
con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='3GB'")
con.execute(f"SET temp_directory='{OUT / 'tmp'}'")
print('Aggregating product rows into orders...', flush=True)
order_query = f"""
SELECT user_id, order_id, min(order_number) AS order_number,
 min(order_dow) AS dow, min(order_hour_of_day) AS hour,
 min(days_since_prior_order) AS gap,
 count(*) AS basket_size, avg(reordered) AS reorder_fraction,
 count(DISTINCT aisle_id) AS aisle_count,
 count(DISTINCT department_id) AS department_count,
 avg(CASE WHEN department='produce' THEN 1.0 ELSE 0.0 END) AS produce_fraction,
 avg(CASE WHEN department='dairy eggs' THEN 1.0 ELSE 0.0 END) AS dairy_fraction
FROM read_parquet('{ROOT / 'data/final.parquet'}') WHERE user_id % 16 = {{partition}} GROUP BY user_id, order_id
"""
for partition in range(16):
    prefix = 'CREATE TABLE orders AS ' if partition == 0 else 'INSERT INTO orders '
    con.execute(prefix + order_query.format(partition=partition))
bad = con.execute("""SELECT count(*) FROM (
 SELECT order_number, lag(order_number) OVER (PARTITION BY user_id ORDER BY order_number) AS prev FROM orders
) WHERE prev IS NOT NULL AND order_number <> prev+1""").fetchone()[0]
if bad:
    raise ValueError(f'{bad} nonconsecutive order transitions; investigate before labeling')
print('Building strictly historical features...', flush=True)
con.execute("""
CREATE TABLE features AS
SELECT *, CASE WHEN reverse_rank=1 THEN 'test' WHEN reverse_rank=2 THEN 'validation' ELSE 'train' END AS split
FROM (
 SELECT user_id, order_id, order_number AS outcome_order_number,
 row_number() OVER (PARTITION BY user_id ORDER BY order_number DESC) AS reverse_rank,
 CASE WHEN gap<=7 THEN 1 ELSE 0 END AS target,
 gap AS outcome_gap,
 lag(order_number) OVER w AS prior_order_number,
 lag(dow) OVER w AS prior_dow, lag(hour) OVER w AS prior_hour,
 lag(gap) OVER w AS prior_gap,
 lag(basket_size) OVER w AS prior_basket_size,
 lag(reorder_fraction) OVER w AS prior_reorder_fraction,
 lag(aisle_count) OVER w AS prior_aisle_count,
 lag(department_count) OVER w AS prior_department_count,
 lag(produce_fraction) OVER w AS prior_produce_fraction,
 lag(dairy_fraction) OVER w AS prior_dairy_fraction,
 avg(gap) OVER hist AS historical_mean_gap,
 stddev_samp(gap) OVER hist AS historical_std_gap,
 avg(CASE WHEN gap IS NULL THEN NULL WHEN gap<=7 THEN 1.0 ELSE 0.0 END) OVER hist AS historical_weekly_rate,
 avg(basket_size) OVER hist AS historical_mean_basket,
 avg(reorder_fraction) OVER hist AS historical_mean_reorder_fraction,
 avg(gap) OVER recent AS recent_mean_gap,
 avg(basket_size) OVER recent AS recent_mean_basket
 FROM orders
 WINDOW w AS (PARTITION BY user_id ORDER BY order_number),
 hist AS (PARTITION BY user_id ORDER BY order_number ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
 recent AS (PARTITION BY user_id ORDER BY order_number ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING)
) WHERE outcome_gap IS NOT NULL AND prior_order_number IS NOT NULL
""")
checks = con.execute("SELECT split,count(*),count(DISTINCT user_id),avg(target) FROM features GROUP BY split ORDER BY split").fetchall()
print('Split counts/users/positive rates:', checks, flush=True)
assert con.execute("SELECT count(*) FROM features WHERE prior_order_number>=outcome_order_number").fetchone()[0] == 0
assert con.execute("SELECT count(*) FROM (SELECT user_id FROM features WHERE split='test' GROUP BY user_id HAVING count(*)<>1)").fetchone()[0] == 0
features = [r[0] for r in con.execute('DESCRIBE features').fetchall() if r[0] not in {'user_id','order_id','outcome_order_number','reverse_rank','target','outcome_gap','split'}]
def load(split):
    df = con.execute(f"SELECT {','.join(features)},target FROM features WHERE split='{split}' ORDER BY user_id,outcome_order_number").df()
    y = df.pop('target').to_numpy(dtype=np.int32)
    return df.astype('float32'), y

X_train,y_train=load('train')
X_valid,y_valid=load('validation')
params = dict(tree_method='hist', device='cpu', n_jobs=4, n_estimators=1000,
              learning_rate=0.05, max_depth=5, min_child_weight=50,
              subsample=0.8, colsample_bytree=0.9, reg_lambda=10,
              objective='binary:logistic', eval_metric='logloss',
              early_stopping_rounds=50, random_state=42)
model=xgb.XGBClassifier(**params)
print('Training XGBoost...', flush=True)
model.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],verbose=50)
model.save_model(OUT/'model.ubj')
def metrics(y,p):
    return dict(roc_auc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),
                log_loss=float(log_loss(y,p)),brier_score=float(brier_score_loss(y,p)),
                accuracy_at_0_5=float(accuracy_score(y,p>=0.5)))
results={'split_summary': [dict(split=s,orders=n,users=u,positive_rate=r) for s,n,u,r in checks],
         'features':features,'parameters':params,'best_iteration':model.best_iteration,
         'validation':metrics(y_valid,model.predict_proba(X_valid)[:,1])}
del X_train,X_valid
X_test,y_test=load('test')
p=model.predict_proba(X_test)[:,1]
results['test']=metrics(y_test,p)
results['test_constant_baseline']=metrics(y_test,np.full(len(y_test),y_train.mean()))
hist=X_test['historical_weekly_rate'].fillna(float(y_train.mean())).to_numpy()
results['test_history_baseline']=metrics(y_test,np.clip(hist,1e-6,1-1e-6))
ids=con.execute("SELECT user_id,order_id FROM features WHERE split='test' ORDER BY user_id,outcome_order_number").df()
ids['actual']=y_test;ids['predicted_probability']=p
ids.to_parquet(OUT/'test_predictions.parquet',index=False)
results['feature_importance']=dict(sorted(zip(features,map(float,model.feature_importances_)),key=lambda t:-t[1]))
results['elapsed_seconds']=time.time()-start
(OUT/'metrics.json').write_text(json.dumps(results,indent=2))
(OUT/'learning_curve.json').write_text(json.dumps(model.evals_result(),indent=2))
print(json.dumps(results,indent=2),flush=True)
