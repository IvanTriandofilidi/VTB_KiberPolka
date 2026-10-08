# Data contract

The CLI accepts Parquet files with `customer_id`, `cat_feature_*` and `num_feature_*` columns. Feature descriptions and product names are not available; the package preserves the anonymized schema.

## Identity and joins

Every table must have unique, non-null IDs. Main and extra tables must contain exactly the same customer set with disjoint feature names. Extra columns and targets are reordered by ID before use. No labels are matched by position, and `customer_id` is excluded from model inputs.

The target table requires exactly 41 complete binary `target_*` columns. Target column order is captured in the training artifact and reused at inference. `predict_*` columns follow that order; output rows follow the input main table.

## Feature treatment

Numerical columns are stored as float32; missing values remain explicit for the tree branches. Five row-level features capture standardized missing count and numerical mean, standard deviation, minimum and maximum. Statistics use only the original input columns, so engineered features do not recursively contribute to themselves.

Missing-count mean and scale are fitted separately inside each training fold. Categorical codes must be integral. CatBoost receives string categories, including an explicit missing token. XGBoost receives pandas categories with a training-fitted dictionary; unseen inference categories become missing. No category dictionary is fitted on the test set.

## Files

```text
data/
  train_main_features.parquet
  train_extra_features.parquet
  train_target.parquet
  test_main_features.parquet
  test_extra_features.parquet
```

The dataset is obtained through the competition provider. Customer records, predictions and trained artifacts are ignored by Git. The synthetic demo has 41 binary targets and deliberately shuffled tables to exercise the join contract. Its target names and prevalence are illustrative, not the competition distribution.
