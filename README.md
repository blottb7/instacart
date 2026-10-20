# Instacart order analysis and modeling

Portfolio project exploring the [Instacart Online Grocery Shopping Dataset 2017](https://www.kaggle.com/competitions/instacart-market-basket-analysis/data). The notebooks cover data preparation, exploratory analysis, order-level feature engineering, and modeling whether a customer places another observed order within seven days.

## Project layout

- `notebooks/01_initial/`: initial data preparation, analysis, and modeling.
- `notebooks/02_full_data/`: analysis and modeling on the full order history.
- `notebooks/archive/`: earlier modeling work.
- `codex-xgboost-experiment/models/weekly_reorder/`: reproducible XGBoost training script, requirements, and result summaries.
- `reference/`: earlier SQL and notebook analysis, with its own README.

## Data and setup

The dataset is not included in this repository. Download it from the Kaggle link above and place the source CSV files in `data/raw/`. Generated tables and local data files also belong under `data/`. Notebook paths and dependencies vary across the exploratory work; check each notebook's first cells before running it.

For the XGBoost experiment, see [`codex-xgboost-experiment/models/weekly_reorder/README.md`](codex-xgboost-experiment/models/weekly_reorder/README.md) and its pinned `requirements.txt`. Run the training script from the project root with:

```sh
python codex-xgboost-experiment/models/weekly_reorder/train.py
```

The script regenerates model files and prediction outputs locally. They are excluded from Git; the small metrics and learning curve summaries are included.
