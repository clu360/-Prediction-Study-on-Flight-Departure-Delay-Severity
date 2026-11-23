# Source-input MLP results

The fixed-setting comparison is verified. Development tuning and tuned holdout scoring are running.

## Fixed settings

All rows average three matched seeds. Each seed is evaluated on 519,971 November and 532,053 December flights. Pooled scores are computed from the combined predictions for each seed. The original MLP settings use one epoch.

| Arm | Inputs and loss | November macro-F1 | December macro-F1 | Pooled macro-F1 | Change vs one-hot | Pooled log loss | Severe precision | Severe recall | Severe AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP-A | G1–G3, standard | 0.6155 | 0.6274 | 0.6232 | +0.0014 | 0.4714 | 80.37% | 39.38% | 0.5529 |
| MLP-B | G1–G3, weighted | 0.5136 | 0.5370 | 0.5275 | +0.0824 | 0.8581 | 29.36% | 56.71% | 0.5490 |
| MLP-C | G1–G4, standard | 0.6138 | 0.6252 | 0.6212 | +0.0125 | 0.4682 | 81.90% | 38.97% | 0.5562 |
| MLP-D | G1–G4, weighted | 0.5167 | 0.5300 | 0.5255 | +0.0806 | 0.8633 | 28.02% | 57.04% | 0.5506 |

Without weather, the representation increased standard MLP macro-F1 by 0.0014 and severe precision by 14.76 percentage points. Severe recall fell by 3.66 points. The weather-input standard MLP gained 0.0125 macro-F1. Both standard embedded MLPs trailed their matched Phase IV XGBoost models, with macro-F1 0.6467 and 0.6482, respectively.

The weighted source-input models still flag severe outcomes more often. Without G4, the weighted MLP flagged 12.84% of flights as severe, compared with 3.17% for standard loss and 6.42% observed severe prevalence. The class weights therefore still trade precision and overall performance for recall.

The fixed comparison estimates the combined effect of learned categorical embeddings, fixed numeric unit scaling, and numeric missing-value indicators. The same approved 20 or 25 original predictors and the same causal prediction instant apply. The reused holdout makes the comparisons exploratory.

[Per-seed and monthly results](metrics_by_seed.csv) · [Seed means](metrics_summary.csv) · [Fit and cohort verification](manifest.json) · [Protocol and progress](progress.md)
