# Analysis notebooks

| Notebook | Completed study coverage |
|---|---|
| [03 classical models](03_classical_models_analysis.ipynb) | All Phase III classical configurations, screening and selected development results, monthly/pooled holdouts, class/airport/calibration/importance results, the CDF baseline revision and verification scope |
| [04 weather and MLP](04_weather_mlp_analysis.ipynb) | Original Phase IV weather/weighting comparisons and the completed source-input MLP follow-up, including fixed and tuned regimes, all seeds/months, three-class metrics and development locks |

The saved Markdown tables contain every final fit/seed/month comparison and class result for these studies, so they remain readable without running a kernel. The code cells expose the complete public aggregate CSVs, subgroup results, curves and verification evidence. Numerical precision is rounded for saved display; report CSV/JSON files retain their original precision. Source-input subgroup analyses that were not run are not claimed.

Rebuild and validate the two source notebooks from the repository root:

```sh
python scripts/build_phase3_notebook.py
python scripts/phase4_build_notebook.py
python scripts/validate_analysis_notebooks.py
```

Validation runs from an isolated checkout containing public aggregates only. Rendered copies, including figures, are saved under ignored `data/model_outputs/analysis_notebook_checks/`; tracked notebooks retain cleared execution outputs and no private data. [Coverage and execution evidence](../reports/analysis_notebook_coverage.json) records input/source/output hashes and completed-run counts.

The source-input detail aggregate can be regenerated from existing private frozen predictions using `python scripts/export_source_mlp_details.py`. It checks the existing 24 fits and 72 per-seed metric records, without training or changing headline results. Weather, weighting, representation and the CDF baseline amendments retain their documented exploratory/post-holdout scope.
