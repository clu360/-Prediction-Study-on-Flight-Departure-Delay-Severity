# Source-input MLP follow-up

**Status** In progress.

## Predeclared comparison

This study uses the existing 20 G1–G3 predictors and their 25-column G4 extension at the same causal prediction instant. It tests an input representation, using the established source feature lists as-is. It replaces one-hot encoding with learned eight-dimensional category embeddings and replaces train-fitted numeric medians and standardization with fixed unit conversions, zero padding for missing numerical values, and missingness indicators. Category vocabularies are fitted inside the training fold, with distinct missing and unseen representations. Raw outcomes, IDs and timestamps remain outside the predictor allowlists.

Fixed fits reuse the Phase IV hidden widths [256, 128], learning rate 0.0003, dropout 0.3 and one epoch. Tuned fits screen the same six candidates on April, July and October, then evaluate the top two on all seven development folds independently for each view. Inner stopping uses the last month of each three-month training window. The winning configuration minimizes equal-month log loss, using the existing tie-break rules, and the final epoch count is the median of its seven inner selections. Selection uses standard cross-entropy. Both loss arms subsequently share each view's fixed or tuned settings. Optimization, class-weight formula, batch size and final seeds are reused from Phase IV.

The original Phase IV one-hot fits are matched references. Two representations, two configuration policies and the weighted/unweighted comparison separate the practical effects. The representation change combines categorical and numerical changes, so their individual contributions cannot be inferred from this experiment. The previously examined November–December 2025 holdout is exploratory. Final fitting uses August–October and freezes all fitted state for both holdout months.

## Execution

From the repository root, run `python scripts/source_mlp.py all`. On the current Windows environment use `python -X pycache_prefix=.venv/pycache -u scripts/source_mlp.py all`. The `tune`, `fit` and `publish` stages can also run separately. Verified candidate folds, final fits and scored months are resumable under ignored private data. If a process was terminated, inspect the owner PID before clearing its stale writer lock.

- 2026-11-23. Started the follow-up on `feat/mlp-source-inputs`, based on Phase IV commit `4dfae3a`. The pre-existing local Phase III notebook edit is outside the branch changes.
