# Source-input MLP results

All 24 real-data fits and both development sweeps completed. Each row averages seeds 6740, 6741 and 6742. Each seed scored 519,971 November and 532,053 December flights using the same model and adapter fitted on 1,663,534 August–October flights. Pooled metrics use all 1,052,024 saved predictions per fit.

## Two comparisons

The fixed runs measure the combined representation change under the original one-epoch settings. The tuned runs measure the effect of selecting settings for the new representation, using only the April–October development folds. Both regimes start with the same approved 20 or 25 predictors and prediction instant. The new adapter uses category embeddings, fixed numerical unit conversions and numeric missingness indicators.

## Fixed Phase IV settings

| Arm | Inputs and loss | Nov macro-F1 | Dec macro-F1 | Pooled macro-F1 | Change vs Phase IV MLP | Pooled log loss | Severe precision | Severe recall | Severe AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP-A | G1–G3, standard | 0.6155 | 0.6274 | 0.6232 | +0.0014 | 0.4714 | 80.37% | 39.38% | 0.5529 |
| MLP-B | G1–G3, weighted | 0.5136 | 0.5370 | 0.5275 | +0.0824 | 0.8581 | 29.36% | 56.71% | 0.5490 |
| MLP-C | G1–G4, standard | 0.6138 | 0.6252 | 0.6212 | +0.0125 | 0.4682 | 81.90% | 38.97% | 0.5562 |
| MLP-D | G1–G4, weighted | 0.5167 | 0.5300 | 0.5255 | +0.0806 | 0.8633 | 28.02% | 57.04% | 0.5506 |

## Development-tuned settings

| Arm | Inputs and loss | Nov macro-F1 | Dec macro-F1 | Pooled macro-F1 | Change vs Phase IV MLP | Pooled log loss | Severe precision | Severe recall | Severe AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP-A | G1–G3, standard | 0.6454 | 0.6536 | 0.6509 | +0.0291 | 0.4420 | 83.03% | 40.58% | 0.5734 |
| MLP-B | G1–G3, weighted | 0.5157 | 0.5408 | 0.5304 | +0.0853 | 0.8232 | 26.40% | 58.02% | 0.5416 |
| MLP-C | G1–G4, standard | 0.6534 | 0.6637 | 0.6601 | +0.0513 | 0.4402 | 78.48% | 41.86% | 0.5788 |
| MLP-D | G1–G4, weighted | 0.4665 | 0.4814 | 0.4760 | +0.0311 | 0.9785 | 23.29% | 64.58% | 0.5718 |

## Locked tuning decisions

| View | Hidden widths | Dropout | Learning rate | Epochs | Mean development log loss |
| --- | --- | ---: | ---: | ---: | ---: |
| G1–G3 | 256, 128 | 0.1 | 0.001 | 19 | 0.432721 |
| G1–G4 | 256, 128 | 0.3 | 0.0003 | 29 | 0.426911 |

Selection was unweighted. Each loss arm reused its view’s selected settings. The fixed settings were [256, 128], dropout 0.3, learning rate 0.0003 and one epoch. G4 retained the original architecture and optimizer settings and changed its epoch count to 29.

## Interpretation

- At fixed settings, the representation raised standard macro-F1 by 0.0014 without G4 and 0.0125 with G4. Tuning the source-input MLP raised it a further 0.0277 and 0.0388, respectively. The same-width embedded network uses roughly one fifth the learned parameters of the matched one-hot MLP.
- The tuned standard MLP exceeded matched XGBoost on pooled macro-F1 on both views. The gains were 0.0042 without G4 and 0.0119 with G4. XGBoost retained lower log loss (0.4394 and 0.4366) and higher severe precision (87.74% and 87.12%). The tuned MLP caught a modestly larger share of severe flights at the default argmax rule.
- Inverse-frequency weighting still traded severe precision and overall macro-F1 for recall. In the tuned G4 arm it increased severe recall from 41.86% to 64.58% and reduced precision from 78.48% to 23.29%. The weighted G4 model flagged 18.02% of flights as severe versus 6.42% observed severe prevalence.
- The matched holdout was previously examined, so these are exploratory performance differences. The fixed comparison changes embeddings, numerical scaling and missingness handling together. The tuned comparisons also reflect the settings selected for each view.

[Per-seed and monthly results](metrics_by_seed.csv) · [Seed means and variation](metrics_summary.csv) · [Fit and cohort verification](manifest.json) · [Development verification](development_verification.json) · [Holdout replay](holdout_verification.json) · [Protocol and progress](progress.md)

The [complete analysis notebook](../../notebooks/04_weather_mlp_analysis.ipynb) now saves both fixed and tuned comparisons, all 72 per-seed/month outcomes, both tuning sweeps and all three classes. [Detailed class/risk metrics](metric_details.json) were derived from the existing frozen predictions and checked against the original headline scores: 24 fits, 25,248,576 prediction rows and 216 class metric records. [Export verification](detail_export_verification.json) records the hashes and scope. No fitted state, model selection or existing headline performance changed during this notebook completion.
