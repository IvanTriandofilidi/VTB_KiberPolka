# Research experiments

The source notebook explores several approaches. The maintained package implements the final two-branch stack; this page preserves the context without presenting every exploratory cell as a tested production path.

| Experiment | Source approach | Role |
|---|---|---|
| CatBoost baseline | MultiLogloss; 3,000 iterations, depth 8, learning rate 0.01 | Initial baseline |
| CatBoost with extra features | MultiLogloss; 3,500 iterations, depth 7, learning rate 0.03, L2 12, Bernoulli subsample 0.66 | Final stack branch |
| XGBoost | Binary logistic objective, one output per tree, 1,000 estimators, depth 3, max_bin 64 | Final stack branch |
| Independent LightGBM | One binary GBDT per target, imbalance weighting | Alternative base approach |
| LightGBM / logistic hybrid | Targets with fewer than 500 training positives use balanced logistic regression on 300 correlation-ranked numerical features; other targets use LightGBM with prevalence-dependent leaf settings | Rare-product experiment |
| CatBoost classifier chain | 41 sequential binary models; order selected by per-target validation AUC | Label-dependence experiment |
| Neural stacking | 82 probability inputs → quantile normalization → 128 → 256 → 41 logits | Final combination |

The source focal objective is `0.25 * (1 - p_t)^2 * BCE`, averaged across rows and targets. Its alpha multiplies all examples equally; it is not positive/negative class balancing. The package keeps this formula explicit.

Classifier-chain training uses observed preceding labels while inference uses predicted labels, creating exposure to accumulated errors. Chain order chosen with validation labels also belongs to model selection. A future reproduction should select order inside an inner split and assess the chain on untouched data.

The hybrid's correlation ranking and scaler must be fitted on training rows only. The maintained final stack does not depend on externally ranked features, feature-importance CSVs or positional prediction CSVs.
