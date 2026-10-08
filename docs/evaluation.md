# Evaluation protocol

## Reproducible package

1. Reserve 20% of labeled customers as an outer holdout using iterative multilabel stratification.
2. Split the remaining development customers into seeded folds. Fit preprocessing and both tree models on each fold's training rows. Generate one held-out prediction per development customer and branch, with ID and target order retained.
3. Split the OOF predictions into 80% meta-training and 20% meta-validation. Fit the QuantileTransformer on meta-training only. Select the neural checkpoint by meta-validation macro ROC-AUC.
4. Refit the tree branches on the development data. Evaluate the selected stacker on the outer holdout. Holdout labels do not participate in preprocessing, base fitting, epoch selection, or quantile fitting.

OOF meta-validation is used for checkpoint selection, not reported as an independent final estimate. Cross-fitted training predictions and full-development base predictions have different training sizes; their distribution shift should be checked on real data. Do not repeatedly tune against the outer holdout. For a final experiment, use a separately frozen customer/session/time holdout where applicable.

Base tree iteration counts are fixed by configuration: no fold-validation labels are used for early stopping. This is a deliberate departure from the original experiments. The stacker uses the source architecture and objective but an epoch-length cosine schedule rather than a ten-epoch schedule cycled across a hundred epochs.

Macro ROC-AUC is the arithmetic mean across all 41 targets. A single-class target makes the full macro score undefined; the report lists affected targets and separately reports the defined-target mean. It never silently assigns AUC 0.5. Rare-label fold training with only one class fails explicitly.

Output values are ranking logits, not calibrated product-opening probabilities. Class imbalance weighting and focal loss do not establish calibration. Calibration and recommendation-policy evaluation require separate data and business criteria.

## Original recorded experiment

The source contains 100 printed neural-training epochs. The largest recorded meta-validation macro ROC-AUC is **0.83423**, at epoch 33. This is a checkpoint-selection result from saved output, not a verified competition leaderboard score or a newly reproduced full-data result.

The original final stack combines CatBoost and XGBoost probability tables from one 80/20 base split. Those CSVs are called `oof` in the source but are single-holdout predictions, not complete cross-validation OOF matrices. They omit customer IDs. Source stacking reconstructs the split and relies on positional correspondence. It also fits quantile normalization before the meta split, includes customer_id during one stratification call, and substitutes 0.5 for undefined AUC. These details limit direct comparison with the refactored pipeline.

Comments mention CatBoost 0.8211 and 0.8394; an XGBoost filename includes 0.837. These annotations are not treated as independently verified scores. An external feature-importance table supplies top 800 / 1,600 features in different experiments; its generation scope cannot be audited from the notebook alone. The package uses all supplied features rather than reusing that selection.

LightGBM, rare-target logistic regression and classifier-chain experiments are described in [the experiment notes](experiments.md). They are not additional inputs to the original final neural stack.
